import jax
import jax.numpy as jnp
import numpy as np
import chex
from utils.ops import (
    rms_norm,
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
    feed_forward,
    grouped_query_attention,
    AttentionParams,
    FeedForwardParams,
)
from utils.kvcache import KVCache

# General parameters for tests
BSZ = 2
SEQLEN = 64
DIM = 128
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = DIM // N_HEADS
MAX_SEQLEN = 256
DTYPE = jnp.float32


def test_jit_rms_norm():
    """Verify JIT correctness and no re-tracing for rms_norm."""
    jitted_rms_norm = jax.jit(chex.assert_max_traces(rms_norm, n=1))

    # First call to check correctness and compile
    key = jax.random.PRNGKey(0)
    key_x, key_w = jax.random.split(key)
    x = jax.random.normal(key_x, (BSZ, SEQLEN, DIM), dtype=DTYPE)
    weight = jax.random.normal(key_w, (DIM,), dtype=DTYPE)

    output_eager = rms_norm(x, weight)
    output_jitted = jitted_rms_norm(x, weight)
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=1e-5
    )

    # Subsequent calls to check for re-traces
    for i in range(1, 3):
        key = jax.random.PRNGKey(i)
        key_x, key_w = jax.random.split(key)
        x = jax.random.normal(key_x, (BSZ, SEQLEN, DIM), dtype=DTYPE)
        weight = jax.random.normal(key_w, (DIM,), dtype=DTYPE)
        jitted_rms_norm(x, weight).block_until_ready()


def test_jit_apply_rotary_emb():
    """Verify JIT correctness and no re-tracing for apply_rotary_emb."""
    freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQLEN)
    jitted_apply_rotary = jax.jit(chex.assert_max_traces(apply_rotary_emb, n=1))

    # First call
    x = jax.random.normal(
        jax.random.PRNGKey(0), (BSZ, SEQLEN, N_HEADS, HEAD_DIM), dtype=DTYPE
    )
    output_eager = apply_rotary_emb(x, freqs_cis[:SEQLEN])
    output_jitted = jitted_apply_rotary(x, freqs_cis[:SEQLEN])
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=5e-4
    )

    # Subsequent calls
    for i in range(1, 3):
        x = jax.random.normal(
            jax.random.PRNGKey(i), (BSZ, SEQLEN, N_HEADS, HEAD_DIM), dtype=DTYPE
        )
        jitted_apply_rotary(x, freqs_cis[:SEQLEN]).block_until_ready()


def test_jit_repeat_kv():
    """Verify JIT correctness and no re-tracing for repeat_kv."""
    n_rep = N_HEADS // N_KV_HEADS
    jitted_repeat_kv = jax.jit(
        chex.assert_max_traces(repeat_kv, n=1), static_argnums=(1,)
    )

    # First call
    x = jax.random.normal(
        jax.random.PRNGKey(0), (BSZ, SEQLEN, N_KV_HEADS, HEAD_DIM), dtype=DTYPE
    )
    output_eager = repeat_kv(x, n_rep)
    output_jitted = jitted_repeat_kv(x, n_rep)
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=1e-5
    )

    # Subsequent calls
    for i in range(1, 3):
        x = jax.random.normal(
            jax.random.PRNGKey(i), (BSZ, SEQLEN, N_KV_HEADS, HEAD_DIM), dtype=DTYPE
        )
        jitted_repeat_kv(x, n_rep).block_until_ready()


def test_jit_feed_forward():
    """Verify JIT correctness and no re-tracing for feed_forward."""
    hidden_dim = 256
    activation_fn = "silu"
    jitted_ff = jax.jit(
        chex.assert_max_traces(feed_forward, n=1), static_argnames=("activation_fn",)
    )

    key = jax.random.PRNGKey(0)
    key_x, key_p1, key_p2, key_p3 = jax.random.split(key, 4)
    params = FeedForwardParams(
        w1_gate=jax.random.normal(key_p1, (DIM, hidden_dim), dtype=DTYPE),
        w2_up=jax.random.normal(key_p2, (DIM, hidden_dim), dtype=DTYPE),
        w3_down=jax.random.normal(key_p3, (hidden_dim, DIM), dtype=DTYPE),
    )
    x = jax.random.normal(key_x, (BSZ, SEQLEN, DIM), dtype=DTYPE)

    # First call
    output_eager = feed_forward(x, params, activation_fn)
    output_jitted = jitted_ff(x, params, activation_fn)
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=1e-5, atol=1e-5
    )

    # Subsequent calls
    for i in range(1, 3):
        loop_key = jax.random.PRNGKey(i)
        x = jax.random.normal(loop_key, (BSZ, SEQLEN, DIM), dtype=DTYPE)
        jitted_ff(x, params, activation_fn).block_until_ready()


def test_jit_attention_prefill():
    """Verify JIT correctness and no re-tracing for GQA in prefill mode."""
    layer_idx = 0
    start_pos = 0
    freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQLEN)

    key = jax.random.PRNGKey(0)
    key_x, key_wq, key_wk, key_wv, key_wo = jax.random.split(key, 5)
    params = AttentionParams(
        wq=jax.random.normal(key_wq, (DIM, N_HEADS, HEAD_DIM), dtype=DTYPE),
        wk=jax.random.normal(key_wk, (DIM, N_KV_HEADS, HEAD_DIM), dtype=DTYPE),
        wv=jax.random.normal(key_wv, (DIM, N_KV_HEADS, HEAD_DIM), dtype=DTYPE),
        wo=jax.random.normal(key_wo, (N_HEADS * HEAD_DIM, DIM), dtype=DTYPE),
    )
    x = jax.random.normal(key_x, (BSZ, SEQLEN, DIM), dtype=DTYPE)
    kv_cache = KVCache.new(1, BSZ, MAX_SEQLEN, N_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    prefill_mask = jnp.ones((BSZ, SEQLEN), dtype=jnp.bool_)

    jitted_gqa = jax.jit(chex.assert_max_traces(grouped_query_attention, n=1))

    # First call
    output_eager, cache_eager = grouped_query_attention(
        x, freqs_cis, params, kv_cache, layer_idx, start_pos, prefill_mask
    )
    output_jitted, cache_jitted = jitted_gqa(
        x, freqs_cis, params, kv_cache, layer_idx, start_pos, prefill_mask
    )
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=5e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(cache_eager.k), np.array(cache_jitted.k), rtol=5e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(cache_eager.v), np.array(cache_jitted.v), rtol=5e-3, atol=1e-3
    )

    # Subsequent calls
    for i in range(1, 3):
        loop_key = jax.random.PRNGKey(i)
        x_new = jax.random.normal(loop_key, (BSZ, SEQLEN, DIM), dtype=DTYPE)
        _, _ = jitted_gqa(
            x_new, freqs_cis, params, kv_cache, layer_idx, start_pos, prefill_mask
        )


def test_jit_attention_decode():
    """Verify JIT correctness and no re-tracing for GQA in decode mode."""
    decode_seqlen = 1
    start_pos = SEQLEN
    layer_idx = 0
    freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQLEN)

    key = jax.random.PRNGKey(0)
    key_x, key_wq, key_wk, key_wv, key_wo, key_cache = jax.random.split(key, 6)
    params = AttentionParams(
        wq=jax.random.normal(key_wq, (DIM, N_HEADS, HEAD_DIM), dtype=DTYPE),
        wk=jax.random.normal(key_wk, (DIM, N_KV_HEADS, HEAD_DIM), dtype=DTYPE),
        wv=jax.random.normal(key_wv, (DIM, N_KV_HEADS, HEAD_DIM), dtype=DTYPE),
        wo=jax.random.normal(key_wo, (N_HEADS * HEAD_DIM, DIM), dtype=DTYPE),
    )
    x = jax.random.normal(key_x, (BSZ, decode_seqlen, DIM), dtype=DTYPE)

    # Pre-fill the cache with some data
    kv_cache = KVCache.new(1, BSZ, MAX_SEQLEN, N_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    prefill_k = jax.random.normal(
        key_cache, (1, BSZ, start_pos, N_KV_HEADS, HEAD_DIM), dtype=DTYPE
    )
    prefill_v = jax.random.normal(
        key_cache, (1, BSZ, start_pos, N_KV_HEADS, HEAD_DIM), dtype=DTYPE
    )
    k_updated = kv_cache.k.at[0, :, :start_pos, :, :].set(prefill_k[0])
    v_updated = kv_cache.v.at[0, :, :start_pos, :, :].set(prefill_v[0])
    prefilled_cache = KVCache(k=k_updated, v=v_updated)

    jitted_gqa = jax.jit(chex.assert_max_traces(grouped_query_attention, n=1))

    # First call
    output_eager, cache_eager = grouped_query_attention(
        x, freqs_cis, params, prefilled_cache, layer_idx, start_pos, prefill_mask=None
    )
    output_jitted, cache_jitted = jitted_gqa(
        x, freqs_cis, params, prefilled_cache, layer_idx, start_pos, prefill_mask=None
    )
    np.testing.assert_allclose(
        np.array(output_eager), np.array(output_jitted), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.array(cache_eager.k), np.array(cache_jitted.k), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.array(cache_eager.v), np.array(cache_jitted.v), rtol=1e-5, atol=1e-5
    )

    # Subsequent calls
    for i in range(1, 3):
        loop_key = jax.random.PRNGKey(i)
        x_new = jax.random.normal(loop_key, (BSZ, decode_seqlen, DIM), dtype=DTYPE)
        _, _ = jitted_gqa(
            x_new,
            freqs_cis,
            params,
            prefilled_cache,
            layer_idx,
            start_pos,
            prefill_mask=None,
        )
