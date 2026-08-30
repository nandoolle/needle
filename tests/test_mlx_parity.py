"""Forward parity: MLX training backend vs the reference JAX model.

Skipped unless both jax and mlx are importable and a format-v2 checkpoint is
available (checkpoints/needle2.pkl or $NEEDLE_PARITY_CHECKPOINT). In float32
the two backends must agree on every argmax; measured max-abs logit
difference on the published needle2 checkpoint is ~3e-3.
"""

import os
import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mx = pytest.importorskip("mlx.core")

CKPT = os.environ.get("NEEDLE_PARITY_CHECKPOINT", "checkpoints/needle2.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason=f"no checkpoint at {CKPT}")


def _tokens(vocab_size):
    rng = np.random.default_rng(0)
    tokens = rng.integers(14, vocab_size, size=(2, 48)).astype(np.int32)
    tokens[0, :3] = [2, 4, 8]  # BOS, im_start, tools_start
    return tokens


def test_forward_parity_f32():
    import jax.numpy as jnp

    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.architecture_mlx import Config, flatten_params, forward

    ckpt = pickle.load(open(CKPT, "rb"))
    config = TransformerConfig(**ckpt["config"])
    config.dtype = "float32"
    config.flash = False
    config.remat = False
    params32 = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), ckpt["params"])
    tokens = _tokens(config.vocab_size)

    ref = np.asarray(
        SimpleAttentionNetwork(config).apply({"params": params32}, jnp.asarray(tokens)),
        np.float32)
    got = np.asarray(
        forward(flatten_params(ckpt["params"]), mx.array(tokens), Config(**ckpt["config"])),
        np.float32)

    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 2e-2
    assert (got.argmax(-1) == ref.argmax(-1)).all()
