import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.slow

REAL_CHECKPOINT = os.environ.get("NEEDLE_MLX_PARITY_CHECKPOINT", "checkpoints/needle3.safetensors")


def _jax_logits(model, params, tokens, quant=False, **kw):
    from needle.model.quantize import WEIGHT_BITS, cq_ste_params

    weights = cq_ste_params(params, WEIGHT_BITS) if quant else params
    return model.apply({"params": weights}, jax.numpy.asarray(tokens), quant=quant, **kw)


def _mlx_logits(config, params, tokens, quant=False, **kw):
    from needle.model.architecture_mlx import cq_ste_params, forward, to_mlx
    from needle.model.quantize import WEIGHT_BITS

    weights = to_mlx(params)
    if quant:
        weights = cq_ste_params(weights, WEIGHT_BITS)
    return forward(weights, config, mx.array(tokens), quant=quant, **kw)


def _assert_close_up_to_rounding_flips(got, ref, atol=5e-5):
    """Under quant, an ulp-level difference can move a value across a rounding boundary;
    that shows up as a few whole positions off, never as a drift across all of them."""
    exact = np.all(np.abs(got - ref) <= atol, axis=-1)
    assert exact.mean() >= 0.9
    assert np.abs(got - ref).max() < 1e-2


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("kv_bits", [8, 4])
def test_forward_matches_jax(quant, flash, kv_bits, deploy_numerics, perturbed_model):
    model, config, params, tokens = perturbed_model(flash=flash, kv_bits=kv_bits)
    deploy_numerics(config)
    ref = np.asarray(_jax_logits(model, params, tokens, quant))
    got = np.asarray(_mlx_logits(config, params, tokens, quant))
    assert got.shape == ref.shape == (2, 24, 500)
    if quant:
        _assert_close_up_to_rounding_flips(got, ref)
    else:
        np.testing.assert_allclose(got, ref, atol=5e-5)


def test_bfloat16_forward_is_as_close_to_float32_as_jax(perturbed_model):
    model, config, params, tokens = perturbed_model()
    ref32 = np.asarray(_jax_logits(model, params, tokens))
    bf16_model, bf16_config, _, _ = perturbed_model(dtype="bfloat16")
    jax_err = np.abs(np.asarray(_jax_logits(bf16_model, params, tokens)) - ref32).mean()
    mlx_err = np.abs(np.asarray(_mlx_logits(bf16_config, params, tokens)) - ref32).mean()
    assert 0 < mlx_err < 2 * jax_err


def test_split_hadamard_perms_match_jax(perturbed_model):
    model, config, params, tokens = perturbed_model(ladder_widths=(24,))
    np.testing.assert_allclose(np.asarray(_mlx_logits(config, params, tokens)),
                               np.asarray(_jax_logits(model, params, tokens)), atol=5e-5)


def test_exit_depth_matches_jax(perturbed_model):
    model, config, params, tokens = perturbed_model()
    ref_full, ref_exit = _jax_logits(model, params, tokens, exit_depth=3)
    got_full, got_exit = _mlx_logits(config, params, tokens, exit_depth=3)
    np.testing.assert_allclose(np.asarray(got_full), np.asarray(ref_full), atol=5e-5)
    np.testing.assert_allclose(np.asarray(got_exit), np.asarray(ref_exit), atol=5e-5)

    ref_sub = _jax_logits(model, params, tokens, exit_depth=3, subnetwork_only=True)
    got_sub = _mlx_logits(config, params, tokens, exit_depth=3, subnetwork_only=True)
    np.testing.assert_allclose(np.asarray(got_sub), np.asarray(ref_sub), atol=5e-5)


def test_padding_mask_matches_jax(perturbed_model):
    from needle.model.architecture import make_causal_mask, make_padding_mask
    from needle.model.architecture_mlx import forward, to_mlx

    model, config, params, tokens = perturbed_model()
    tokens[:, -5:] = config.pad_token_id
    mask = make_causal_mask(tokens.shape[1]) & make_padding_mask(tokens, config.pad_token_id)
    ref = model.apply({"params": params}, jax.numpy.asarray(tokens), mask=mask)
    got = forward(to_mlx(params), config, mx.array(tokens), mask=mx.array(np.asarray(mask)))
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=5e-5)


@pytest.mark.parametrize("bits", [2, 4])
def test_cq_quantize_matches_jax(bits):
    from needle.model import architecture_mlx, quantize

    w = np.random.default_rng(bits).standard_normal((3, 200, 96)).astype(np.float32)
    ref = np.asarray(quantize.cq_quantize(jax.numpy.asarray(w), bits))
    got = np.asarray(architecture_mlx.cq_quantize(mx.array(w), bits))
    assert np.mean(np.abs(got - ref) > 1e-5) < 1e-3
    np.testing.assert_allclose(np.median(np.abs(got - ref)), 0, atol=1e-6)


@pytest.mark.skipif(not os.path.exists(REAL_CHECKPOINT),
                    reason=f"no Needle 3 checkpoint at {REAL_CHECKPOINT}")
def test_published_checkpoint_matches_jax(deploy_numerics):
    """Float forward on the published checkpoint agrees to rounding. Under QAT the A8 and
    W4 rounding flips on ulp-level differences, so that path is compared by loss."""
    import optax
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.run import load_checkpoint

    params, config = load_checkpoint(REAL_CHECKPOINT)
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
    model = SimpleAttentionNetwork(config)
    deploy_numerics(config)
    tokens = np.random.default_rng(0).integers(14, config.vocab_size, (2, 96)).astype(np.int32)

    ref = np.asarray(_jax_logits(model, params, tokens))
    got = np.asarray(_mlx_logits(config, params, tokens))
    np.testing.assert_allclose(got, ref, atol=5e-4)
    assert (got.argmax(-1) == ref.argmax(-1)).all()

    def loss(logits):
        return float(optax.softmax_cross_entropy_with_integer_labels(
            np.asarray(logits)[:, :-1], tokens[:, 1:]).mean())

    ref_q = _jax_logits(model, params, tokens, quant=True)
    got_q = _mlx_logits(config, params, tokens, quant=True)
    assert abs(loss(got_q) - loss(ref_q)) < 1e-2 * loss(ref_q)
    assert (np.asarray(got_q).argmax(-1) == np.asarray(ref_q).argmax(-1)).mean() > 0.9
