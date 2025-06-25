# Placeholder for shared, reusable Flax modules (RMSNorm, FFN, Attention, etc.) 
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

@partial(jit, static_argnames=['head_dim', 'end', 'theta', 'use_scaled', 'dtype'])
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
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
    if use_scaled:
        raise NotImplementedError("`use_scaled` is not implemented.")

    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)].astype(dtype) / head_dim)) # Shape: (head_dim // 2,)
    t = jnp.arange(end, dtype=dtype) # Shape: (end,)
    freqs = jnp.outer(t, freqs) # Shape: (end, head_dim // 2)

    # In JAX, torch.polar(torch.ones_like(freqs), freqs) is equivalent to jnp.cos(freqs) + 1j * jnp.sin(freqs)
    freqs_cos = jnp.cos(freqs) # Shape: (end, head_dim // 2)
    freqs_sin = jnp.sin(freqs) # Shape: (end, head_dim // 2)

    # Stack on the last dimension to create a shape of [end, head_dim // 2, 2]
    freqs_cis = jnp.stack([freqs_cos, freqs_sin], axis=-1) # Shape: (end, head_dim // 2, 2)
    return freqs_cis

@struct.dataclass
class AttentionParams:
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array

@struct.dataclass
class FeedForwardParams:
    w1_gate: jax.Array # Corresponds to gate_proj
    w2_up: jax.Array   # Corresponds to up_proj
    w3_down: jax.Array # Corresponds to down_proj


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


@jit
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

@partial(jit, static_argnames=['n_rep'])
def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    """
    Repeat Key/Value heads for Grouped Query Attention.

    Args:
        x: Input tensor (keys or values) with shape [bs, seqlen, n_kv_heads, head_dim].
        n_rep: Number of times to repeat the KV heads.

    Returns:
        Tensor with repeated KV heads, shape [bs, seqlen, n_q_heads, head_dim].
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # equivalent to torch.repeat_interleave(x, repeats=n_rep, dim=2)
    return jnp.broadcast_to(x[:, :, :, None, :], (bs, slen, n_kv_heads, n_rep, head_dim)).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

@jit
def grouped_query_attention(
    x: jax.Array,
    freqs_cis: jax.Array, # Precomputed freqs for max_seqlen
    params: AttentionParams,
    kv_cache: KVCache,
    layer_idx: int,
    start_pos: int,
    prefill_mask: Optional[jax.Array] = None, # padding mask during prefill stage
) -> tuple[jax.Array, KVCache]: # Return output and updated cache
    """
    Compute Grouped Query Attention with KV Caching.

    Args:
        x: Input tensor of shape [batch_size, seqlen, dim] (seqlen=1 for decoding).
        freqs_cis: Precomputed rotary frequency embeddings (complex format, shape like [max_seqlen, head_dim//2, 2]).
        params: Dataclass containing weight matrices (wq, wk, wv, wo).
        kv_cache: The current KV Cache.
        layer_idx: The index of the current layer.
        start_pos: The starting position index for the current computation.
        prefill_mask: Optional padding mask for the prefill stage.

    Returns:
        Tuple of (Output tensor after attention, Updated KVCache).
    """
    bsz, seqlen, dim = x.shape

    # Project inputs to queries, keys, values for the current token(s)
    xq = jnp.einsum('bsd,dhc->bshc', x, params.wq)
    xk = jnp.einsum('bsd,dkc->bskc', x, params.wk)
    xv = jnp.einsum('bsd,dvc->bsvc', x, params.wv)

    # Apply rotary positional embeddings
    current_freqs_cis = lax.dynamic_slice_in_dim(freqs_cis, start_pos, seqlen, axis=0)
    xq = apply_rotary_emb(xq, freqs_cis=current_freqs_cis) # Shape: (bsz, seqlen, n_heads, head_dim)
    xk = apply_rotary_emb(xk, freqs_cis=current_freqs_cis) # Shape: (bsz, seqlen, n_kv_heads, head_dim)

    # During prefill, mask out padding tokens before caching
    if prefill_mask is not None:
        prefill_mask_expanded = prefill_mask[:, :, None, None]
        xk = xk * prefill_mask_expanded
        xv = xv * prefill_mask_expanded

    # Update the KV cache
    updated_cache = kv_cache.update(xk, xv, layer_idx, start_pos)
    keys, values = updated_cache.get_layer(layer_idx)

    max_seqlen = keys.shape[1]

    # Build attention mask
    query_positions = jnp.arange(seqlen) + start_pos
    key_positions = jnp.arange(max_seqlen)

    mask = (query_positions[:, None] >= key_positions[None, :])[None, :, :] # [1, seqlen, max_seqlen] 

    if prefill_mask is not None:
        # Mask out queries at padding positions
        query_mask = prefill_mask[:, :, None] # [bsz, seqlen, 1]
        mask = mask & query_mask

    # Add head dimension for broadcasting
    mask = mask[:, None, :, :]

    # Perform attention using JAX's optimized dot-product attention
    attn_output = nn.dot_product_attention(
        query=xq,
        key=keys,
        value=values,
        mask=mask,
    )

    # Zero out outputs for padded tokens to prevent NaNs
    if prefill_mask is not None:
        attn_output = jnp.where(prefill_mask[:, :, None, None], attn_output, 0)

    attn_output = attn_output.reshape(bsz, seqlen, -1)
    output = jnp.einsum('bsd,do->bso', attn_output, params.wo)

    return output, updated_cache


@partial(jit, static_argnames=['activation_fn'])
def feed_forward(
    x: jax.Array,
    params: FeedForwardParams,
    activation_fn: str, # Added activation function name
) -> jax.Array:
    """
    Compute FeedForward network (MLP) using a configurable activation function (like SwiGLU).

    Args:
        x: Input tensor of shape [batch_size, seqlen, dim].
        params: Dataclass containing weight matrices (w1_gate, w2_up, w3_down).
        activation_fn: Name of the activation function ('silu', 'relu', 'gelu').

    Returns:
        Output tensor after MLP computation.
    """

    # Project input: x -> gate, up
    # x: [bs, seqlen, dim], w1_gate: [dim, hidden_dim], w2_up: [dim, hidden_dim]
    gate = jnp.einsum('bsd,dh->bsh', x, params.w1_gate)
    up = jnp.einsum('bsd,dh->bsh', x, params.w2_up)

    # Apply the specified activation function (SwiGLU style)
    if activation_fn == 'silu':
        activated_gate = nn.silu(gate)
    elif activation_fn == 'relu':
        activated_gate = nn.relu(gate)
    elif activation_fn == 'gelu':
        # Use approximate=False for exact GELU, True for faster approximation
        activated_gate = nn.gelu(gate, approximate=False)
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")
        # replace error handling with chex 
   
    fused_activation = activated_gate * up

    # Project down
    # fused_swiglu: [bs, seqlen, hidden_dim], w3_down: [hidden_dim, dim]
    output = jnp.einsum('bsh,hd->bsd', fused_activation, params.w3_down)

    return output 
