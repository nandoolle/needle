"""MLX (Apple GPU) port of the training-path forward pass.

Covers exactly what `needle finetune` exercises: SimpleAttentionNetwork
__call__ with quant=False (fake-quant is a no-op there: KV_BITS=0 and quant
is the Python literal False on the training path). The confidence and
contrastive heads, the MTP block and KV-window decoding are out of scope —
they are not touched by LoRA finetuning.

Params are a flat {"a/b/c": mx.array} dict mirroring the flax checkpoint
tree; scan-stacked leaves keep their leading num_layers axis and are sliced
per layer. dtype=bfloat16 mirrors the flax dtype semantics (dense/attention
compute in bf16; norms, MHC mixing, sinkhorn internals and logits in f32).

Memory: each MHC layer runs under mx.checkpoint — the flax side uses
nn.remat for the same reason.
"""

import math

import mlx.core as mx
import numpy as np

ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")


class Config:
    def __init__(self, **kw):
        self.vocab_size = kw.get("vocab_size", 8192)
        self.d_model = kw.get("d_model", 512)
        self.attn_dim = kw.get("attn_dim", 0) or self.d_model
        self.num_heads = kw.get("num_heads", 8)
        self.num_kv_heads = kw.get("num_kv_heads", 4)
        self.num_layers = kw.get("num_layers", 27)
        self.rope_theta = kw.get("rope_theta", 100000.0)
        self.engram_orders = tuple(kw.get("engram_orders", (2, 3)))
        self.engram_heads = kw.get("engram_heads", 0)
        self.engram_slots = kw.get("engram_slots", 8192)
        self.engram_layers = tuple(kw.get("engram_layers", (2, 15)))
        self.mhc_lanes = kw.get("mhc_lanes", 4)
        self.pad_token_id = kw.get("pad_token_id", 0)


def flatten_params(nested):
    """Nested checkpoint pytree -> flat {"a/b/c": mx.array} in float32."""
    flat = {}

    def walk(t, prefix=""):
        for k, v in t.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}/")
            else:
                flat[prefix + k] = mx.array(np.asarray(v, dtype=np.float32))

    walk(nested)
    return flat


def engram_geometry(cfg):
    orders = cfg.engram_orders
    heads = cfg.engram_heads or max(1, cfg.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = cfg.d_model // (len(orders) * heads)
    return orders, heads, sub_dim


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return mx.pad(x, pad)[:, : x.shape[1]]


def _valid_from(T, offset):
    """_mask_diag of a pure causal mask: position t is valid iff t >= offset."""
    return (mx.arange(T) >= offset).astype(mx.float32)[None, :]


def engram_indices(tokens, orders, heads, slots):
    u = tokens.astype(mx.uint32)
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = mx.full(u.shape, seed, dtype=mx.uint32)
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * mx.array(_ENGRAM_PRIME, dtype=mx.uint32)
            acc = acc ^ (acc >> mx.array(15, dtype=mx.uint32))
            idx.append((acc % mx.array(slots, dtype=mx.uint32)).astype(mx.int32))
    return mx.stack(idx, axis=-1)  # (B, T, tables)


def _rms_unit(x, epsilon=1e-6):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + epsilon)


def zc_rmsnorm(x, scale, epsilon=1e-6):
    """f32 internals, output cast to the compute dtype (flax ZCRMSNorm)."""
    xf = x.astype(mx.float32)
    rms = mx.sqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + epsilon)
    return ((1 + scale) * xf / rms).astype(_GEOM["dtype"])


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - mx.logsumexp(log_K, axis=-1, keepdims=True)
        log_K = log_K - mx.logsumexp(log_K, axis=-2, keepdims=True)
    return mx.exp(log_K)


def rope_tables(head_dim, seq_len, theta):
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    angles = t[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


_WALSH_CACHE = {}


def walsh_matrix(n):
    if n not in _WALSH_CACHE:
        H = np.array([[1.0]], dtype=np.float32)
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        _WALSH_CACHE[n] = mx.array(H / np.sqrt(n))
    return _WALSH_CACHE[n]


# --- per-layer computation, checkpoint-friendly: everything it reads arrives
# --- through `lp` (this layer's param slices) or explicit array arguments.

# static ints + compute dtype for the current forward, set by stack()
_GEOM = {"dtype": mx.float32}


def attention(lp, x, cos, sin):
    B, T, _ = x.shape
    H, KV, D = _GEOM["heads"], _GEOM["kv_heads"], _GEOM["head_dim"]
    dt = _GEOM["dtype"]

    q = (x @ lp["q_proj"].astype(dt)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = (x @ lp["k_proj"].astype(dt)).reshape(B, T, KV, D).transpose(0, 2, 1, 3)
    v = (x @ lp["v_proj"].astype(dt)).reshape(B, T, KV, D).transpose(0, 2, 1, 3)

    q = zc_rmsnorm(q, lp["q_norm"])
    k = zc_rmsnorm(k, lp["k_norm"])
    q = apply_rope(q, cos, sin).astype(dt)
    k = apply_rope(k, cos, sin).astype(dt)

    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(D), mask="causal")
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    out = out * mx.sigmoid(x @ lp["gate_proj"].astype(dt))
    return out @ lp["out_proj"].astype(dt)


def hadamard_mlp(lp, x):
    n = x.shape[-1]  # d_model == power of two here (512)
    dt = _GEOM["dtype"]
    H = walsh_matrix(n).astype(dt)
    z = (lp["d1"].astype(dt) * x) @ H
    a = lp["d2"].astype(dt) * z
    z = (a * mx.sigmoid(a)) @ H
    return lp["d3"].astype(dt) * z


def block(lp, x, cos, sin, ek, ev):
    d_model = x.shape[-1]
    alpha = mx.sigmoid(
        mx.einsum("btd,sbtd->sbt", _rms_unit(x), _rms_unit(ek)) / math.sqrt(d_model))
    x = x + mx.einsum("s,sbt,sbtd->btd", lp["site_flags"], alpha,
                      ev.astype(mx.float32)).astype(x.dtype)

    skip = x
    x = zc_rmsnorm(x, lp["pre_attn_norm"])
    x = attention(lp, x, cos, sin)
    x = zc_rmsnorm(x, lp["post_attn_norm"])
    x = skip + mx.sigmoid(lp["attn_gate"]) * x

    skip = x
    x = zc_rmsnorm(x, lp["pre_hada_norm"])
    x = hadamard_mlp(lp, x)
    return skip + x


def mhc_layer(x, lp, cos, sin, ek, ev):
    """One scan step: MHC pre-mix -> block -> MHC post-mix. x: (B, T, n, C).
    Mixing math stays in f32 (upstream _ScanBody uses xf = x.astype(float32));
    only the block input/output live in the compute dtype."""
    B, T, n, C = x.shape
    dt = _GEOM["dtype"]
    xf = x.astype(mx.float32)
    nx = _rms_unit(x.reshape(B, T, n * C))
    hpre = mx.sigmoid(lp["a_pre"] * (nx @ lp["phi_pre"]) + lp["b_pre"] + lp["pre_off"])
    u = mx.einsum("btn,btnc->btc", hpre, xf).astype(dt)
    y = block(lp, u, cos, sin, ek, ev) - u
    hpost = 2 * mx.sigmoid(lp["a_post"] * (nx @ lp["phi_post"]) + lp["b_post"] + lp["post_off"])
    res = (nx @ lp["phi_res"]).reshape(B, T, n, n)
    hres = _sinkhorn(lp["a_res"] * res + lp["b_res"])
    return (mx.einsum("btij,btjc->btic", hres, xf)
            + hpost[..., None] * y.astype(mx.float32)[:, :, None, :]).astype(dt)


_ckpt_layer = mx.checkpoint(mhc_layer)


def _layer_params(p, l, pre_off, post_off, site_flags):
    a = "stack/layers/block/self_attn/"
    b = "stack/layers/block/"
    return {
        "q_proj": p[a + "q_proj/kernel"][l], "k_proj": p[a + "k_proj/kernel"][l],
        "v_proj": p[a + "v_proj/kernel"][l], "gate_proj": p[a + "gate_proj/kernel"][l],
        "out_proj": p[a + "out_proj/kernel"][l],
        "q_norm": p[a + "q_norm/scale"][l], "k_norm": p[a + "k_norm/scale"][l],
        "pre_attn_norm": p[b + "ZCRMSNorm_0/scale"][l],
        "post_attn_norm": p[b + "post_attn_norm/scale"][l],
        "pre_hada_norm": p[b + "pre_hada_norm/scale"][l],
        "attn_gate": p[b + "attn_gate"][l],
        "d1": p[b + "hadamard_mlp/d1"][l], "d2": p[b + "hadamard_mlp/d2"][l],
        "d3": p[b + "hadamard_mlp/d3"][l],
        "phi_pre": p["stack/mhc_phi_pre"][l], "phi_post": p["stack/mhc_phi_post"][l],
        "phi_res": p["stack/mhc_phi_res"][l],
        "b_pre": p["stack/mhc_b_pre"][l], "b_post": p["stack/mhc_b_post"][l],
        "b_res": p["stack/mhc_b_res"][l],
        "a_pre": p["stack/mhc_a_pre"][l], "a_post": p["stack/mhc_a_post"][l],
        "a_res": p["stack/mhc_a_res"][l],
        "pre_off": pre_off[l], "post_off": post_off[l], "site_flags": site_flags[l],
    }


def stack(p, x, cos, sin, cfg, ekv, checkpoint=True, dtype=mx.float32):
    n, L = cfg.mhc_lanes, cfg.num_layers
    B, T, C = x.shape
    _GEOM.update(heads=cfg.num_heads, kv_heads=cfg.num_kv_heads,
                 head_dim=cfg.attn_dim // cfg.num_heads, dtype=dtype)
    x = x.astype(dtype)
    lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
    pre_off = mx.array(8 * lane - 4)     # (L, n)
    post_off = mx.array(-4 * (1 - lane))

    site_flags = np.zeros((L, len(cfg.engram_layers)), np.float32)
    for s, layer in enumerate(cfg.engram_layers):
        site_flags[layer, s] = 1.0
    site_flags = mx.array(site_flags)

    ek, ev = ekv
    step = _ckpt_layer if checkpoint else mhc_layer
    x = mx.broadcast_to(x[:, :, None, :], (B, T, n, C))
    for l in range(L):
        x = step(x, _layer_params(p, l, pre_off, post_off, site_flags), cos, sin, ek, ev)

    x = mx.mean(x.astype(mx.float32), axis=2).astype(dtype)
    return zc_rmsnorm(x, p["stack/final_norm/scale"])


def engram_kv(p, tokens, cfg):
    orders, heads, _ = engram_geometry(cfg)
    T = tokens.shape[1]
    indices = engram_indices(tokens, orders, heads, cfg.engram_slots)
    ngram_ok = mx.stack([mx.broadcast_to(_valid_from(T, o - 1), tokens.shape)
                         for o in orders for _ in range(heads)], axis=-1)
    tap_ok = mx.stack([mx.broadcast_to(_valid_from(T, j * max(orders)), tokens.shape)
                       for j in range(ENGRAM_CONV_TAPS)])
    dt = _GEOM["dtype"]
    ks, vs = [], []
    for site in range(len(cfg.engram_layers)):
        tables = p[f"engrams_{site}/embedding"]  # (tables, slots, sub)
        fetched = mx.stack([tables[i][indices[..., i]] for i in range(tables.shape[0])], axis=2)
        fetched = fetched * ngram_ok[..., None]
        e = fetched.reshape(*tokens.shape, -1).astype(dt)
        k = e @ p[f"engrams_{site}/key_proj/kernel"].astype(dt)
        v = e @ p[f"engrams_{site}/value_proj/kernel"].astype(dt)
        taps = p[f"engrams_{site}/taps"].astype(dt)
        v = sum(taps[j] * _shift_right(v, j * max(orders)) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        ks.append(k)
        vs.append(v)
    return mx.stack(ks), mx.stack(vs)


def forward(p, tokens, cfg, checkpoint=True, dtype=mx.float32):
    """tokens (B, T) int32 -> logits (B, T, vocab) in f32. Causal mask only,
    no quant — the exact configuration `needle finetune`'s loss_fn exercises.
    dtype=bfloat16 mirrors the upstream flax dtype semantics: dense/attention
    compute in bf16, norms/MHC/sinkhorn internals and logits in f32."""
    _GEOM["dtype"] = dtype
    E = p["embedding/embedding"]
    x = E[tokens] * math.sqrt(cfg.d_model)
    head_dim = cfg.attn_dim // cfg.num_heads
    cos, sin = rope_tables(head_dim, tokens.shape[1], cfg.rope_theta)
    ekv = engram_kv(p, tokens, cfg)
    x = stack(p, x, cos, sin, cfg, ekv, checkpoint=checkpoint, dtype=dtype)
    return x.astype(mx.float32) @ E.T
