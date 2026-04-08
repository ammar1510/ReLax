"""
Gemma 4-specific JAX operations.

Contains ops that differ from the shared implementations in utils/ops.py:
- build_sliding_attn_mask: Causal mask with sliding window constraint
- rms_norm_no_scale: RMSNorm without learnable scale (for V norm)
- logit_softcap: Logit soft-capping via tanh
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial

from utils.kvcache import KVCache
from utils.qwen_ops import apply_partial_rotary_emb_batch  # noqa: F401 — re-export


@partial(jit, static_argnames=["seqlen", "sliding_window"])
def build_sliding_attn_mask(
    seqlen: int,
    kv_cache: KVCache,
    true_len: jax.Array,
    sliding_window: int,
) -> jax.Array:
    """
    Build causal attention mask with a sliding window constraint.

    Tokens can only attend to keys within the last `sliding_window` positions.

    Args:
        seqlen: Input sequence length (static for JIT).
        kv_cache: KV cache with positions.
        true_len: Actual (non-padded) input lengths [bsz].
        sliding_window: Maximum number of past positions to attend to.

    Returns:
        Boolean mask [bsz, seqlen, max_seqlen].
    """
    max_seqlen = kv_cache.k.shape[3]
    cache_positions = kv_cache.seq_positions

    def build_mask_for_sequence(true_length, cache_pos):
        query_offsets = jnp.arange(seqlen)
        key_positions = jnp.arange(max_seqlen)
        query_positions = cache_pos + query_offsets

        # Causal: query can attend to keys at positions <= query position
        causal_mask = query_positions[:, None] >= key_positions[None, :]

        # Sliding window: query can only attend to keys within the window
        window_mask = (query_positions[:, None] - key_positions[None, :]) < sliding_window

        # Valid query mask: only non-padded queries
        valid_query_mask = query_offsets < true_length

        return causal_mask & window_mask & valid_query_mask[:, None]

    return jax.vmap(build_mask_for_sequence)(true_len, cache_positions)


@jit
def rms_norm_no_scale(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm without a learnable scale parameter.

    Used for Gemma 4's V normalization (with_scale=False).
    """
    return x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


@jit
def logit_softcap(logits: jax.Array, cap: float = 30.0) -> jax.Array:
    """Apply logit soft-capping: tanh(logits / cap) * cap.

    Prevents logit magnitudes from growing unboundedly.
    """
    return jnp.tanh(logits / cap) * cap
