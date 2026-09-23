import json
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.slow

TOOLS = [{"name": "send_email", "parameters": {"type": "object", "properties": {
    "to": {"type": "string"}, "subject": {"type": "string"}}, "required": ["to"]}}]


def _write_data(path):
    rows = [
        {"tools": TOOLS, "query": "email a@b.com about lunch",
         "reasoning": "to from query", "answers": [
             {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "lunch"}}]},
        {"tools": TOOLS, "query": "nothing actionable here",
         "reasoning": "off-topic", "answers": []},
    ]
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _finetune_args(data, checkpoint, out, ckpt_dir):
    return types.SimpleNamespace(
        jsonl_path=str(data), checkpoint=checkpoint, epochs=1, batch_size=2,
        lr=1e-3, lora_rank=4, lora_alpha=8.0, max_len=64, generate=0,
        model=None, checkpoint_dir=str(ckpt_dir), out=str(out))


def test_lora_loss_and_grads_match_jax(deploy_numerics, perturbed_model):
    """The finetune_local loss and its LoRA gradient, computed by both backends."""
    import jax.numpy as jnp
    import optax
    from needle.model.architecture_mlx import cq_ste_leaf, cq_ste_params, forward, to_mlx
    from needle.model.finetune import init_lora, lora_target_paths, merge_lora
    from needle.model.quantize import WEIGHT_BITS, cq_ste_params as jax_cq_ste_params

    model, config, params, tokens = perturbed_model()
    deploy_numerics(config)
    mask = np.ones(tokens.shape, np.float32)
    mask[:, :6] = 0
    paths = lora_target_paths(params)
    lora = init_lora(params, paths, 4, jax.random.PRNGKey(0))
    lora = {p: {"A": v["A"], "B": v["B"] + 0.01 * jax.random.normal(jax.random.PRNGKey(i), v["B"].shape)}
            for i, (p, v) in enumerate(lora.items())}
    scale = 2.0

    def jax_loss(lora):
        merged = jax_cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)
        logits = model.apply({"params": merged}, jnp.asarray(tokens), quant=True)[:, :-1]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(tokens[:, 1:]))
        return (ce * mask[:, 1:]).sum() / mask[:, 1:].sum()

    ref_loss, ref_grads = jax.value_and_grad(jax_loss)(lora)

    flat = to_mlx(params)
    base = {"/".join(p): flat.pop("/".join(p)) for p in paths}
    frozen = cq_ste_params(flat, WEIGHT_BITS)

    def mlx_loss(lora):
        weights = dict(frozen)
        for name, adapter in lora.items():
            weights[name] = cq_ste_leaf(name, base[name] + scale * (adapter["A"] @ adapter["B"]),
                                        WEIGHT_BITS)
        logits = forward(weights, config, mx.array(tokens), quant=True)[:, :-1]
        targets = mx.array(tokens[:, 1:])
        ce = (mx.logsumexp(logits, axis=-1)
              - mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0])
        return (ce * mx.array(mask[:, 1:])).sum() / float(mask[:, 1:].sum())

    mlx_lora = {"/".join(p): {k: mx.array(np.asarray(v[k])) for k in ("A", "B")}
                for p, v in lora.items()}
    got_loss, got_grads = mx.value_and_grad(mlx_loss)(mlx_lora)

    np.testing.assert_allclose(float(got_loss), float(ref_loss), rtol=1e-5)
    for path, grads in ref_grads.items():
        for matrix in ("A", "B"):
            np.testing.assert_allclose(np.asarray(got_grads["/".join(path)][matrix]),
                                       np.asarray(grads[matrix]), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("total", [1, 2, 40])
def test_optimizer_matches_optax(total):
    import jax.numpy as jnp
    import optax
    from needle.model.finetune_mlx import adamw, clip_by_global_norm, warmup_cosine

    warmup = min(max(1, total // 20), total - 1)
    rng = np.random.default_rng(total)
    w0 = rng.standard_normal((6, 5)).astype(np.float32)
    grads = [rng.standard_normal((6, 5)).astype(np.float32) * (3.0 if i % 2 else 0.1)
             for i in range(total)]

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
        optax.warmup_cosine_decay_schedule(0.0, 1e-2, warmup, total)))
    ref = {"w": jnp.asarray(w0)}
    state = tx.init(ref)
    opt = adamw(warmup_cosine(1e-2, warmup, total))
    got = {"w": mx.array(w0)}
    opt.init(got)
    for g in grads:
        updates, state = tx.update({"w": jnp.asarray(g)}, state, ref)
        ref = optax.apply_updates(ref, updates)
        got = opt.apply_gradients(clip_by_global_norm({"w": mx.array(g)}, 1.0), got)
    np.testing.assert_allclose(np.asarray(got["w"]), np.asarray(ref["w"]), rtol=1e-5, atol=1e-6)


def test_finetune_mlx_writes_an_adapter_build_merges(tiny_checkpoint, tmp_path, published_base):
    from needle.model.checkpoints import read_adapter
    from needle.model.export import read_export
    from needle.model.finetune import build_main
    from needle.model.finetune_mlx import finetune_mlx

    data = tmp_path / "data.jsonl"
    _write_data(data)
    out = tmp_path / "adapter.safetensors"
    args = _finetune_args(data, tiny_checkpoint, out, tmp_path / "ck")
    args.seed = 5
    args.val_split = 0.5
    progress = []
    finetune_mlx(args, progress=progress.append)

    assert any(m.split()[:2] == ["backend", "mlx"] for m in progress)
    assert any("CQ W4 STE + A8" in m for m in progress)
    assert any("accuracy" in m for m in progress)
    adapter = read_adapter(str(out))
    assert adapter["rank"] == 4 and abs(adapter["scale"] - 2.0) < 1e-6
    assert adapter["base"] == tiny_checkpoint and adapter["seed"] == 5
    assert adapter["lora"] and all(np.isfinite(v["B"]).all() and np.abs(v["B"]).max() > 0
                                   for v in adapter["lora"].values())

    merged = str(tmp_path / "merged.cact")
    build_main(types.SimpleNamespace(checkpoint=tiny_checkpoint, lora=str(out), out=merged,
                                     upload=False))
    header, _ = read_export(merged)
    assert header["num_tensors"] > 0


def test_finetune_mlx_adapter_matches_the_jax_layout(tiny_checkpoint, tmp_path):
    from needle.model.checkpoints import read_adapter
    from needle.model.finetune import finetune_local
    from needle.model.finetune_mlx import finetune_mlx

    data = tmp_path / "data.jsonl"
    _write_data(data)
    adapters = {}
    for name, train in (("jax", finetune_local), ("mlx", finetune_mlx)):
        args = _finetune_args(data, tiny_checkpoint, tmp_path / f"{name}.safetensors", tmp_path)
        args.score = False
        train(args)
        adapters[name] = read_adapter(str(args.out))["lora"]
    assert adapters["jax"].keys() == adapters["mlx"].keys()
    for key, value in adapters["jax"].items():
        np.testing.assert_array_equal(adapters["mlx"][key]["A"].shape, value["A"].shape)
        np.testing.assert_array_equal(adapters["mlx"][key]["B"].shape, value["B"].shape)
