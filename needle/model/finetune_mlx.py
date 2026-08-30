"""LoRA finetuning on MLX (Apple GPU) — `needle finetune --backend mlx`.

Same data format, LoRA placement (the five stacked attention kernels),
schedule (AdamW, warmup + cosine decay, global-norm clip 1.0) and masked-CE
loss as the JAX path in finetune.py; the adapter .pkl it writes is loadable
by `needle build --lora` unchanged. jax-metal is numerically broken for this
architecture (loss goes NaN on the first steps), so on macOS this is the
only GPU training path.
"""

import os
import pickle

import numpy as np

from .finetune import DEFAULT_BASE, fit_max_len, load_jsonl
from .tokenizer import get_tokenizer

# keep in sync with run.CHECKPOINT_FORMAT_VERSION (run.py imports jax at
# module level, which must not happen on the mlx-only path)
CHECKPOINT_FORMAT_VERSION = 2


def _load_checkpoint(path):
    """pickle load + format check; downloads from HF like run.load_checkpoint
    but without importing jax."""
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        from .tokenizer import HF_REPO

        name = path if path.startswith("checkpoints/") else "checkpoints/" + os.path.basename(path)
        print(f"  {'fetch':<9} {path}  downloading from Hugging Face", flush=True)
        path = hf_hub_download(HF_REPO, name, repo_type="model", local_dir=".")
    with open(path, "rb") as handle:
        ckpt = pickle.load(handle)
    if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"{path} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint")
    return ckpt, path


def finetune_mlx(args, progress=None):
    import mlx.core as mx
    import mlx.optimizers as optim

    from .architecture_mlx import Config, LORA_TARGETS, flatten_params, forward

    def emit(msg):
        print(msg, flush=True)
        if progress:
            progress(msg)

    dtype = getattr(mx, getattr(args, "dtype", None) or "float32")
    mx.set_cache_limit(2 << 30)

    if getattr(args, "generate", 0):
        from .finetune import augment_jsonl

        args.jsonl_path = augment_jsonl(
            args.jsonl_path, args.generate, model=getattr(args, "model", None),
            workers=getattr(args, "workers", 8))

    ckpt, base_path = _load_checkpoint(args.checkpoint or DEFAULT_BASE)
    params = flatten_params(ckpt["params"])
    cfg = Config(**ckpt["config"])
    emit(f"  {'backend':<9} mlx-gpu  {getattr(args, 'dtype', None) or 'float32'}")

    tokenizer = get_tokenizer(cfg.vocab_size)
    max_len = fit_max_len(args.jsonl_path, tokenizer, args.max_len)
    seqs, masks = load_jsonl(args.jsonl_path, tokenizer, max_len)
    if len(seqs) == 0:
        raise SystemExit("no usable examples in " + args.jsonl_path)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}  cap {args.max_len}")

    paths = [k for k in params
             if k.endswith("kernel") and "stack" in k and "layers" in k
             and any(t in k for t in LORA_TARGETS)
             and float(mx.abs(params[k]).max()) > 1e-6]
    scale = args.lora_alpha / args.lora_rank
    rng = np.random.default_rng(0)
    lora = {}
    for path in paths:
        w = params[path]
        lead, in_dim, out_dim = w.shape[:-2], w.shape[-2], w.shape[-1]
        lora[path] = {
            "A": mx.array(rng.normal(size=(*lead, in_dim, args.lora_rank)).astype(np.float32)
                          / args.lora_rank),
            "B": mx.zeros((*lead, args.lora_rank, out_dim)),
        }
    emit(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups")

    n_val = min(int(len(seqs) * getattr(args, "val_split", 0.1)), len(seqs) - 1)
    if n_val > 0:
        order = np.random.default_rng(0).permutation(len(seqs))
        seqs, masks = seqs[order], masks[order]
        val_seqs, val_masks = seqs[:n_val], masks[:n_val]
        seqs, masks = seqs[n_val:], masks[n_val:]
        emit(f"  {'holdout':<9} {n_val} examples for validation")

    batch, count = args.batch_size, len(seqs)
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), total_steps - 1)
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, warmup),
         optim.cosine_decay(args.lr, total_steps - warmup)],
        [warmup])
    opt = optim.AdamW(learning_rate=schedule)
    emit(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  cosine decay  clip 1.0  (compiling...)")

    def merged(lora):
        out = dict(params)
        for path, ab in lora.items():
            out[path] = params[path] + scale * (ab["A"] @ ab["B"])
        return out

    def loss_fn(lora, ids, mask):
        logits = forward(merged(lora), ids, cfg, dtype=dtype)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = (mx.logsumexp(logits, axis=-1)
              - mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0])
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    grad_fn = mx.value_and_grad(loss_fn)

    def train_step(lora, ids, mask):
        loss, grads = grad_fn(lora, ids, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        return loss, opt.apply_gradients(grads, lora)

    # cache the step graph: 27 layers x sinkhorn iterations are thousands of
    # ops that would otherwise be rebuilt on every step
    train_step = mx.compile(train_step, inputs=[opt.state], outputs=[opt.state])

    def eval_loss():
        tot, n = 0.0, 0
        for i in range(0, n_val, batch):
            l = loss_fn(lora, mx.array(val_seqs[i:i + batch]), mx.array(val_masks[i:i + batch]))
            tot, n = tot + float(l), n + 1
        return tot / max(n, 1)

    every = max(1, total_steps // 50)
    step_i = 0
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        last = 0.0
        for start in range(0, count, batch):
            idx = order[start:start + batch]
            if len(idx) < batch:  # keep a single compiled shape
                idx = order[-batch:]
            loss, lora = train_step(lora, mx.array(seqs[idx]), mx.array(masks[idx]))
            mx.eval(loss, lora, opt.state)
            last = float(loss)
            step_i += 1
            if step_i % every == 0:
                emit(f"  {'step':<9} {step_i}/{total_steps}  loss {last:.4f}")
        if n_val > 0:
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}  val {eval_loss():.4f}")
        else:
            emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    out = args.out or os.path.join(args.checkpoint_dir, "needle_lora.pkl")
    with open(out, "wb") as handle:
        pickle.dump({
            "lora": {p: {"A": np.asarray(ab["A"]), "B": np.asarray(ab["B"])}
                     for p, ab in lora.items()},
            "scale": float(scale),
            "base": base_path,
            "rank": args.lora_rank,
        }, handle)
    print(f"  {'adapter':<9} {out}")
    print(f"  {'next':<9} needle build {base_path} --lora {out}")
    print(f"  {'note':<9} confidence reports None with tuned weights; the head is not tuned")
