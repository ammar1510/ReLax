# Placeholder for shared, reusable Flax modules (RMSNorm, FFN, Attention, etc.)
import logging
import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.lax as lax
from jax import jit
from flax import struct
from jax.experimental.pallas.ops.tpu import flash_attention
from .kvcache import KVCache
from typing import Optional
from functools import partial
import numpy as np
import sys

logger = logging.getLogger(__name__)

@partial(jit, static_argnames=["head_dim", "end", "use_scaled", "dtype"])
def precompute_freqs_cis(
    head_dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Precompute the rotational frequency embeddings.

    Args:
        head_dim: Dimension of each attention head.
        end: Maximum sequence length supported by the model.
        theta: Base parameter for frequency calculation.
        use_scaled: Whether to apply scaling to frequencies. Not implemented.

    Returns:
        A JAX array of shape for the rotary embeddings.
        The shape is `[end, head_dim // 2, 2]` containing the cosine and sine components.
    """
    freqs = 1.0 / (
        theta
        ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)].astype(dtype) / head_dim)
    )  # Shape: (head_dim // 2,)
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)  # Shape: (end,)
    freqs = jnp.outer(t, freqs)  # Shape: (end, head_dim // 2)

    # In JAX, torch.polar(torch.ones_like(freqs), freqs) is equivalent to jnp.cos(freqs) + 1j * jnp.sin(freqs)
    freqs_cos = jnp.cos(freqs)  # Shape: (end, head_dim // 2)
    freqs_sin = jnp.sin(freqs)  # Shape: (end, head_dim // 2)

    # Stack on the last dimension to create a shape of [end, head_dim // 2, 2]
    freqs_cis = jnp.stack(
        [freqs_cos, freqs_sin], axis=-1
    )  # Shape: (end, head_dim // 2, 2)
    return freqs_cis


@partial(
    jit,
    static_argnames=[
        "scale_factor",
        "low_freq_factor",
        "high_freq_factor",
        "old_context_len",
    ],
    donate_argnums=[0],
)
def apply_scaling(
    freqs: jax.Array,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: float = 8192.0,
) -> jax.Array:
    """
    Apply RoPE scaling to frequencies based on Llama 3 implementation.
    The scaling is done to extend the context length.

    Args:
        freqs: The original frequencies.
        scale_factor: The scaling factor.
        low_freq_factor: The factor for the low frequency.
        high_freq_factor: The factor for the high frequency.
        old_context_len: The original context length.

    Returns:
        The scaled frequencies.
    """
    # RoPE scaling (values obtained from grid search)
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / freqs

    # This is the smooth transition part from the original implementation.
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    freqs_for_mid_range = (1 - smooth) * freqs / scale_factor + smooth * freqs

    # Conditions to select which scaling to apply.
    # 1. if wavelen > low_freq_wavelen, new_freqs = freqs / scale_factor
    # 2. if wavelen < high_freq_wavelen, new_freqs = freqs
    # 3. otherwise, new_freqs = freqs_for_mid_range
    new_freqs = jnp.where(
        wavelen > low_freq_wavelen,
        freqs / scale_factor,
        jnp.where(wavelen < high_freq_wavelen, freqs, freqs_for_mid_range),
    )
    return new_freqs


@struct.dataclass
class AttentionParams:
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array


@struct.dataclass
class FeedForwardParams:
    w_gate: jax.Array  # Corresponds to gate_proj
    w_up: jax.Array  # Corresponds to up_proj
    w_down: jax.Array  # Corresponds to down_proj


@jit
def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    """
    Apply Root Mean Square Normalization.

    Args:
        x: Input tensor.
        weight: Weight tensor.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    output = x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return output * weight


@partial(jit, donate_argnums=[0])
def apply_rotary_emb(x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
    """
    Apply Rotary Positional Embeddings (RoPE) to a tensor.

    Args:
        x: Input tensor, shape [bsz, seqlen, n_heads, head_dim] or [bsz, seqlen, n_kv_heads, head_dim].
        freqs_cis: Precomputed rotary frequency embeddings. A JAX array of shape
                   `[seqlen, head_dim // 2, 2]`, where the last dimension contains
                   the cosine and sine components.

    Returns:
        The transformed tensor with RoPE applied.
    """
    # x: [..., head_dim] -> [..., head_dim//2, 2]
    x_shaped = x.reshape(*x.shape[:-1], -1, 2)
    x_r, x_i = x_shaped[..., 0], x_shaped[..., 1]

    # freqs_cis: [seqlen, head_dim//2, 2] -> [1, seqlen, 1, head_dim//2, 2]
    # This reshapes freqs_cis to be broadcastable with the x tensor.
    freqs_cis = freqs_cis[None, :, None, :, :]
    freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]

    # Apply the rotation using complex number multiplication logic.
    # (v_r + i*v_i) * (cos + i*sin) = (v_r*cos - v_i*sin) + i*(v_r*sin + v_i*cos)
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Combine the real and imaginary parts back into a single tensor.
    # [..., head_dim//2, 2] -> [..., head_dim]
    x_out = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(x.shape)

    return x_out


@partial(jit, donate_argnums=[0])
def apply_rotary_emb_batch(x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
    """
    Apply Rotary Positional Embeddings (RoPE) to a tensor with per-batch-item frequencies.

    Args:
        x: Input tensor, shape [bsz, seqlen, n_heads, head_dim] or [bsz, seqlen, n_kv_heads, head_dim].
        freqs_cis: Precomputed rotary frequency embeddings per batch item. A JAX array of shape
                   `[bsz, seqlen, head_dim // 2, 2]`, where the last dimension contains
                   the cosine and sine components.

    Returns:
        The transformed tensor with RoPE applied.
    """
    # x: [..., head_dim] -> [..., head_dim//2, 2]
    x_shaped = x.reshape(*x.shape[:-1], -1, 2)
    x_r, x_i = x_shaped[..., 0], x_shaped[..., 1]

    # freqs_cis: [bsz, seqlen, head_dim//2, 2] -> [bsz, seqlen, 1, head_dim//2, 2]
    # This reshapes freqs_cis to be broadcastable with the x tensor.
    freqs_cis = freqs_cis[:, :, None, :, :]
    freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]

    # Apply the rotation using complex number multiplication logic.
    # (v_r + i*v_i) * (cos + i*sin) = (v_r*cos - v_i*sin) + i*(v_r*sin + v_i*cos)
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Combine the real and imaginary parts back into a single tensor.
    # [..., head_dim//2, 2] -> [..., head_dim]
    x_out = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(x.shape)

    return x_out



@jit
def build_attn_mask(
    x: jax.Array,  # [bsz, seqlen, dim] - input tensor
    kv_cache: KVCache,  # KV cache with k, v, and positions
    true_len: jax.Array,  # [bsz] - actual (non-padded) sequence lengths
) -> jax.Array:
    """
    Build attention mask for variable-length sequences with KV caching.

    Args:
        x: Input tensor of shape [bsz, seqlen, dim] (padded to max length in batch).
        kv_cache: KV cache containing keys, values, and positions.
        true_len: Actual (non-padded) input lengths for each sequence [bsz].

    Returns:
        Boolean attention mask of shape [bsz, seqlen, max_seqlen] where True means attend
        and False means don't attend.
    """
    bsz, seqlen, _ = x.shape
    
    max_seqlen = kv_cache.k.shape[3] 
    
    cache_positions = kv_cache.positions 
    
    def build_mask_for_sequence(true_length, cache_pos):
        """Build attention mask for one sequence.

        Args:
            true_length: Actual input length (non-padded)
            cache_pos: Cache position (start_pos) for this sequence
        """
        query_offsets = jnp.arange(seqlen)  # [seqlen] - Static: [0, 1, 2, ..., seqlen-1]
        key_positions = jnp.arange(max_seqlen)  # [max_seqlen] - Static

        query_positions = cache_pos  + query_offsets  # [seqlen]

        # Causal mask: query at pos i can only attend to keys at pos <= i
        causal_mask = query_positions[:, None] >= key_positions[None, :]  # [seqlen, max_seqlen]

        # Valid query mask: only first true_length queries are real (rest are padding)
        valid_query_mask = query_offsets < true_length  # [seqlen]

        mask = causal_mask & valid_query_mask[:, None]  # [seqlen, max_seqlen]
        
        return mask  # [seqlen, max_seqlen] 

    mask = jax.vmap(build_mask_for_sequence)(true_len, cache_positions) # [bsz, seqlen, max_seqlen]

    return mask


# @partial(jit, static_argnames=["layer_idx"], donate_argnums=[0, 1, 3])
def grouped_query_attention(
    x: jax.Array,
    freqs_cis: jax.Array,  # Precomputed freqs for max_seqlen
    params: AttentionParams,
    kv_cache: KVCache,
    layer_idx: int,
    mask: jax.Array,  # [bsz, seqlen, max_seqlen] - boolean attention mask (True = attend, False = don't attend)
) -> tuple[jax.Array, KVCache]:
    """
    Compute Grouped Query Attention with variable-length sequences and per-sequence positions.

    Args:
        x: Input tensor of shape [bsz, seqlen, dim] (padded to max length in batch).
        freqs_cis: Precomputed rotary frequency embeddings [max_seqlen, head_dim//2, 2].
        params: Dataclass containing weight matrices (wq, wk, wv, wo).
        kv_cache: The current KV Cache (tracks positions per sequence internally).
        layer_idx: The index of the current layer.
        mask: Boolean attention mask of shape [bsz, seqlen, max_seqlen] where True means attend
            and False means don't attend.

    Returns:
        Tuple of (Output tensor after attention, updated KVCache).
    """
    bsz, seqlen, dim = x.shape

    start_positions = kv_cache.positions  # [bsz] - can be different per sequence

    # logger.debug(f"Start positions: {start_positions}, Sequence length: {seqlen}")

    xq = jnp.einsum("bsd,dhc->bshc", x, params.wq)  # [bsz, seqlen, n_heads, head_dim]
    xk = jnp.einsum("bsd,dkc->bskc", x, params.wk)  # [bsz, seqlen, n_kv_heads, head_dim]
    xv = jnp.einsum("bsd,dvc->bsvc", x, params.wv)  # [bsz, seqlen, n_kv_heads, head_dim]

    # Debug: After Q, K, V projection (commented out for performance)
    # xq_np = np.array(xq[0, 0, 0, :10], dtype=np.float32)
    # xk_np = np.array(xk[0, 0, 0, :10], dtype=np.float32)
    # xv_np = np.array(xv[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After Q/K/V projection: Q={xq_np}, K={xk_np}, V={xv_np}")

    position_offsets = jnp.arange(seqlen)[None, :]  # [1, seqlen]
    absolute_positions = start_positions[:, None] + position_offsets  # [bsz, seqlen]

    batch_freqs_cis = freqs_cis[absolute_positions]  # [bsz, seqlen, head_dim//2, 2]

    xq = apply_rotary_emb_batch(xq, batch_freqs_cis)
    xk = apply_rotary_emb_batch(xk, batch_freqs_cis)

    # Debug: After RoPE (commented out for performance)
    # xq_rope_np = np.array(xq[0, 0, 0, :10], dtype=np.float32)
    # xk_rope_np = np.array(xk[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After RoPE: Q={xq_rope_np}, K={xk_rope_np}")

    xk_transposed = xk.transpose(0, 2, 1, 3)  # [bsz, n_kv_heads, seqlen, head_dim]
    xv_transposed = xv.transpose(0, 2, 1, 3)  # [bsz, n_kv_heads, seqlen, head_dim]

    # Get cached keys/values for this layer (these will be updated later in the model)
    updated_cache = kv_cache.update(xk_transposed, xv_transposed, layer_idx)
    keys, values = updated_cache.get_layer(layer_idx)

    # Debug: After cache retrieval (commented out for performance)
    # keys_np = np.array(keys[0, 0, 0, :10], dtype=np.float32)
    # values_np = np.array(values[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - Cached Keys={keys_np}, Values={values_np}")

    _, _, n_heads, head_dim = xq.shape
    n_kv_heads = keys.shape[1]
    n_rep = n_heads // n_kv_heads

    if n_rep != 1:
        # Repeat along head dimension: [bsz, kv_heads, max_seqlen, head_dim] -> [bsz, n_heads, max_seqlen, head_dim]
        keys = jnp.repeat(keys, n_rep, axis=1)
        values = jnp.repeat(values, n_rep, axis=1)

    xq = xq.transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhqd,bhkd->bhqk", xq, keys) / jnp.sqrt(head_dim)

    # Debug: After attention scores (commented out for performance)
    # scores_np = np.array(scores[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After attention scores: {scores_np}")

    mask = mask[:, None, :, :]  # [bsz, 1, seqlen, max_seqlen] - boolean mask

    scores = nn.softmax(scores.astype(jnp.float32), where=mask, axis=-1).astype(x.dtype)

    # Debug: After softmax (commented out for performance)
    # scores_softmax_np = np.array(scores[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After softmax: {scores_softmax_np}")

    attn_output = jnp.einsum("bhqk,bhkd->bhqd", scores, values) # [bsz, n_heads, seqlen, head_dim]

    # Debug: After attention output (commented out for performance)
    # attn_output_np = np.array(attn_output[0, 0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After attention output: {attn_output_np}")

    attn_output = attn_output.transpose(0, 2, 1, 3) # [bsz, seqlen, n_heads, head_dim]
    attn_output = attn_output.reshape(bsz, seqlen, -1) # [bsz, seqlen, n_heads * head_dim]

    output = jnp.einsum("bsd,do->bso", attn_output, params.wo) # [bsz, seqlen, n_heads * head_dim]

    # Debug: After output projection (commented out for performance)
    # output_np = np.array(output[0, 0, :10], dtype=np.float32)
    # logger.debug(f"Layer {layer_idx} - After output projection: {output_np}")

    return output, updated_cache


@partial(jit, static_argnames=["activation_fn"], donate_argnums=[0])
def feed_forward(
    x: jax.Array,
    params: FeedForwardParams,
    activation_fn: str,  # Added activation function name
) -> jax.Array:
    """
    Compute FeedForward network (MLP) using a configurable activation function (like SwiGLU).

    Args:
        x: Input tensor of shape [batch_size, seqlen, dim].
        params: Dataclass containing weight matrices (w_gate, w_up, w_down).
        activation_fn: Name of the activation function ('silu', 'relu', 'gelu').

    Returns:
        Output tensor after MLP computation.
    """

    # Project input: x -> gate, up
    # x: [bs, seqlen, dim], w_gate: [dim, hidden_dim], w_up: [dim, hidden_dim]
    gate = jnp.einsum("bsd,dh->bsh", x, params.w_gate)
    up = jnp.einsum("bsd,dh->bsh", x, params.w_up)

    # Apply the specified activation function (SwiGLU style)
    if activation_fn == "silu":
        activated_gate = nn.silu(gate)
    elif activation_fn == "relu":
        activated_gate = nn.relu(gate)
    elif activation_fn == "gelu":
        # Use approximate=False for exact GELU, True for faster approximation
        activated_gate = nn.gelu(gate, approximate=False)
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")
        # replace error handling with chex

    fused_activation = activated_gate * up

    # Project down
    # fused_swiglu: [bs, seqlen, hidden_dim], w_down: [hidden_dim, dim]
    output = jnp.einsum("bsh,hd->bsd", fused_activation, params.w_down)

    return output
