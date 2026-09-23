import json
import math
import os

import numpy as np

from .checkpoints import write_adapter
from .finetune import (
    DEFAULT_BASE, _calls_of, _training_rng, augment_jsonl, fit_max_len, init_lora,
    load_jsonl, lora_target_paths, read_examples, render_example,
)
from .tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID, TOOL_CALL_START, TOOL_CALL_END

MLX_INSTALL = 'pip install "cactus-needle[train,mlx]"'


def warmup_cosine(peak, warmup, total):
    """optax.warmup_cosine_decay_schedule(0, peak, warmup, total) as an MLX schedule."""
    import mlx.core as mx

    decay = total - warmup

    def schedule(step):
        step = step.astype(mx.float32)
        warm = peak * step / warmup if warmup > 0 else mx.array(peak, mx.float32)
        t = mx.clip(step - warmup, 0, decay) / decay
        return mx.where(step < warmup, warm, 0.5 * peak * (1 + mx.cos(math.pi * t)))

    return schedule


def clip_by_global_norm(grads, max_norm):
    """optax.clip_by_global_norm: rescale only when the norm reaches max_norm."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
    return tree_map(lambda g: mx.where(norm < max_norm, g, g / norm * max_norm), grads)


def adamw(schedule):
    import mlx.optimizers as optim

    # optax.adamw's defaults; MLX defaults to weight_decay=0.01 and no bias correction.
    return optim.AdamW(learning_rate=schedule, betas=[0.9, 0.999], eps=1e-8,
                       weight_decay=1e-4, bias_correction=True)


def _generate(logits_fn, tokenizer, config, prompt, max_new_tokens):
    """run.generate at temperature 0, without streaming."""
    import mlx.core as mx

    prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
    buf_len = min(config.max_seq_len, len(prompt_ids) + max_new_tokens)
    if len(prompt_ids) >= buf_len:
        raise ValueError(f"Prompt ({len(prompt_ids)} tokens) does not fit in max_seq_len={config.max_seq_len}")
    buffer = np.full((1, buf_len), PAD_ID, np.int32)
    buffer[0, :len(prompt_ids)] = prompt_ids
    generated = []
    for pos in range(len(prompt_ids) - 1, buf_len - 1):
        next_token = int(mx.argmax(logits_fn(mx.array(buffer))[0, pos]))
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        buffer[0, pos + 1] = next_token
    return tokenizer.decode(generated)


def _score_quantised(logits_fn, tokenizer, config, data_path, n_val, rng_seed=0):
    """finetune._score_quantised on MLX: exact-call accuracy of the held-out split."""
    examples = list(read_examples(data_path))
    if not examples:
        return 0, 0
    order = np.random.default_rng(rng_seed).permutation(len(examples))
    held = [examples[i] for i in order[:n_val]]
    correct = 0
    for example in held:
        prompt, _ = render_example({**example, "answers": []})
        text = _generate(logits_fn, tokenizer, config, prompt, max_new_tokens=96)
        want = _calls_of(TOOL_CALL_START + json.dumps(
            example.get("answers", []), separators=(",", ":")) + TOOL_CALL_END)
        correct += _calls_of(text) == want
    return correct, len(held)


def finetune_mlx(args, progress=None):
    """finetune_local on MLX: same data, LoRA, schedule, QAT numerics and adapter."""
    try:
        import mlx.core as mx
    except ImportError as err:
        raise SystemExit(f"--backend mlx needs MLX on Apple silicon: {MLX_INSTALL}") from err
    import jax
    from .run import load_checkpoint
    from .architecture_mlx import cq_ste_leaf, forward, is_quant_leaf, to_mlx
    from .quantize import WEIGHT_BITS

    def emit(msg):
        print(msg, flush=True)
        if progress:
            progress(msg)

    base_path = args.checkpoint or DEFAULT_BASE
    data_path = args.jsonl_path
    if getattr(args, "generate", 0):
        data_path = augment_jsonl(data_path, args.generate, model=getattr(args, "model", None),
                                  workers=getattr(args, "workers", 8))

    params, config = load_checkpoint(base_path)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    device = "gpu" if mx.default_device().type == mx.gpu else "cpu"
    emit(f"  {'backend':<9} mlx {device}  float32")
    emit(f"  {'depth':<9} {config.num_layers} layers")
    tokenizer = get_tokenizer(config.vocab_size)
    max_len = fit_max_len(data_path, tokenizer, args.max_len)
    seqs, masks = load_jsonl(data_path, tokenizer, max_len)
    if len(seqs) == 0:
        raise SystemExit("no usable examples in " + data_path)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}  cap {args.max_len}")
    emit(f"  {'numerics':<9} CQ W{WEIGHT_BITS} STE + A8 (matches export)")

    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    seed = int(getattr(args, "seed", 0))
    rng = _training_rng(seed)
    lora = {"/".join(path): {"A": mx.array(np.asarray(v["A"])), "B": mx.array(np.asarray(v["B"]))}
            for path, v in init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(seed)).items()}
    emit(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups")

    # Frozen leaves quantize to the same values every step, so they are done once,
    # one leaf at a time to bound peak memory; only the LoRA-merged targets go
    # through the STE inside the loss.
    flat = to_mlx(params)
    del params
    base = {name: flat.pop(name) for name in lora}
    frozen = {}
    for name in list(flat):
        value = flat.pop(name)
        frozen[name] = cq_ste_leaf(name, value, WEIGHT_BITS) if is_quant_leaf(name, value) else value
        mx.eval(frozen[name])

    def merged(lora):
        out = dict(frozen)
        for name, adapter in lora.items():
            out[name] = cq_ste_leaf(name, base[name] + scale * (adapter["A"] @ adapter["B"]),
                                    WEIGHT_BITS)
        return out

    n_val = min(int(len(seqs) * getattr(args, "val_split", 0.1)), len(seqs) - 1)
    if n_val > 0:
        order = rng.permutation(len(seqs))
        seqs, masks = seqs[order], masks[order]
        val_seqs, val_masks = seqs[:n_val], masks[:n_val]
        seqs, masks = seqs[n_val:], masks[n_val:]
        emit(f"  {'holdout':<9} {n_val} examples for validation")

    batch, count = args.batch_size, len(seqs)
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), total_steps - 1)
    optimizer = adamw(warmup_cosine(args.lr, warmup, total_steps))
    optimizer.init(lora)
    emit(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  cosine decay  clip 1.0  (compiling...)")

    def loss_fn(lora, ids, mask):
        logits = forward(merged(lora), config, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = (mx.logsumexp(logits, axis=-1)
              - mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0])
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = mx.value_and_grad(loss_fn)
    state = [optimizer.state]

    def step(lora, ids, mask):
        loss, grads = loss_and_grad(lora, ids, mask)
        return optimizer.apply_gradients(clip_by_global_norm(grads, 1.0), lora), loss

    train_step = mx.compile(step, inputs=state, outputs=state)
    eval_step = mx.compile(loss_fn)

    every = max(1, total_steps // 50)
    step_i = 0
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        last = 0.0
        for start in range(0, count, batch):
            idx = order[start:start + batch]
            lora, loss = train_step(lora, mx.array(seqs[idx]), mx.array(masks[idx]))
            mx.eval(lora, state)
            last = float(loss)
            step_i += 1
            if step_i % every == 0:
                emit(f"  {'step':<9} {step_i}/{total_steps}  loss {last:.4f}")
        if n_val > 0:
            val = np.mean([float(eval_step(lora, mx.array(val_seqs[i:i + batch]),
                                           mx.array(val_masks[i:i + batch])))
                           for i in range(0, n_val, batch)])
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}  val {val:.4f}")
        else:
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}")

    if n_val > 0 and getattr(args, "score", True):
        tuned = merged(lora)
        mx.eval(tuned)
        logits_fn = mx.compile(lambda tokens: forward(tuned, config, tokens))
        correct, total = _score_quantised(logits_fn, tokenizer, config, data_path, n_val,
                                          rng_seed=seed)
        if total:
            emit(f"  {'accuracy':<9} {correct}/{total} held-out calls exact, scored on the "
                 f"W{WEIGHT_BITS} weights the archive ships")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    out = args.out or os.path.join(args.checkpoint_dir, "needle_lora.safetensors")
    write_adapter(out, {
        "lora": {name: {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])}
                 for name, v in lora.items()},
        "scale": float(scale),
        "base": base_path,
        "rank": args.lora_rank,
        "seed": seed,
    })
    print(f"  {'adapter':<9} {out}")
    print(f"  {'next':<9} needle build {base_path} --lora {out}")
    print(f"  {'note':<9} local tuning leaves the confidence head untrained; needle build drops it and confidence reports None")
