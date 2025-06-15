# Placeholder for shared, reusable Flax modules (RMSNorm, FFN, Attention, etc.) 
import jax
import jax.numpy as jnp
import jax.nn as nn 
import jax.lax as lax
from flax import struct
from jax.experimental.pallas.ops.tpu import flash_attention
from .kvcache import KVCache 

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False) -> jax.Array:
    """
    Precompute the rotational frequency embeddings.
    This function is a JAX implementation of the PyTorch code snippet provided by the user.

    Args:
        dim: Dimension of each attention head.
        end: Maximum sequence length supported by the model.
        theta: Base parameter for frequency calculation.
        use_scaled: Whether to apply scaling to frequencies. Not implemented.

    Returns:
        A JAX array of shape for the rotary embeddings.
        The shape is `[end, dim // 2, 2]` containing the cosine and sine components.
    """
    if use_scaled:
        raise NotImplementedError("`use_scaled` is not implemented.")

    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)

    # In JAX, torch.polar(torch.ones_like(freqs), freqs) is equivalent to jnp.cos(freqs) + 1j * jnp.sin(freqs)
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)

    # Stack on the last dimension to create a shape of [end, dim // 2, 2]
    freqs_cis = jnp.stack([freqs_cos, freqs_sin], axis=-1)
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


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
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


def apply_rotary_emb(x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
    """
    Apply Rotary Positional Embeddings (RoPE) to a tensor.

    This function is a JAX implementation of the PyTorch code snippet provided by the user.

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
    freqs_cis = jnp.expand_dims(freqs_cis, axis=(0, 2))
    freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]

    # Apply the rotation using complex number multiplication logic.
    # (v_r + i*v_i) * (cos + i*sin) = (v_r*cos - v_i*sin) + i*(v_r*sin + v_i*cos)
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Combine the real and imaginary parts back into a single tensor.
    # [..., head_dim//2, 2] -> [..., head_dim]
    x_out = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(x.shape)

    return x_out

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


def grouped_query_attention(
    x: jnp.ndarray,
    freqs_cis: jnp.ndarray, # Precomputed freqs for max_seqlen
    # mask: jnp.ndarray | None, # Mask is implicitly causal up to start_pos+seqlen
    params: AttentionParams,
    kv_cache: KVCache,
    layer_idx: int,
    start_pos: int,
    n_heads: int,
    n_kv_heads: int,
) -> tuple[jax.Array, KVCache]: # Return output and updated cache
    """
    Compute Grouped Query Attention with KV Caching.

    Args:
        x: Input tensor of shape [batch_size, seqlen, dim] (seqlen=1 for decoding).
        freqs_cis: Precomputed rotary frequency embeddings (complex format, shape like [2, max_seqlen, head_dim//2]).
        params: Dataclass containing weight matrices (wq, wk, wv, wo).
        kv_cache: The current KV Cache.
        layer_idx: The index of the current layer.
        start_pos: The starting position index for the current computation.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.

    Returns:
        Tuple of (Output tensor after attention, Updated KVCache).
    """
    bsz, seqlen, dim = x.shape
    # Assuming parameters are already shaped correctly.
    # wq: [dim, n_heads, head_dim]
    # wk: [dim, n_kv_heads, head_dim]
    # wv: [dim, n_kv_heads, head_dim]
    head_dim = params.wq.shape[-1]

    # Project inputs to queries, keys, values for the current token(s)
    # Shapes: [bsz, seqlen, n_heads, head_dim] for xq
    # Shapes: [bsz, seqlen, n_kv_heads, head_dim] for xk, xv
    xq = jnp.einsum('bsd,dhc->bshc', x, params.wq)
    xk = jnp.einsum('bsd,dkc->bskc', x, params.wk)
    xv = jnp.einsum('bsd,dvc->bsvc', x, params.wv)

    # Apply rotary positional embeddings to the new queries and keys
    # Slice freqs_cis for the current position(s)
    # Assuming freqs_cis has shape [2, max_seqlen, head_dim//2]
    current_freqs_cis = (
        lax.dynamic_slice_in_dim(freqs_cis[0], start_pos, seqlen, axis=0),
        lax.dynamic_slice_in_dim(freqs_cis[1], start_pos, seqlen, axis=0)
    )
    # Pass the sliced freqs to RoPE
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=current_freqs_cis)

    # Update the KV cache
    updated_cache = kv_cache.update(xk, xv, layer_idx, start_pos)

    # Retrieve the full key/value sequences for this layer from cache
    # keys/values shape: [bsz, current_seqlen, n_kv_heads, head_dim]
    current_seqlen = start_pos + seqlen
    keys, values = updated_cache.get_layer(layer_idx, current_seqlen)


    # Repeat KV heads if n_kv_heads < n_heads (GQA)
    n_rep = n_heads // n_kv_heads
    keys = repeat_kv(keys, n_rep) # Shape: [bsz, current_seqlen, n_heads, head_dim]
    values = repeat_kv(values, n_rep) # Shape: [bsz, current_seqlen, n_heads, head_dim]

    # Transpose query: [bsz, n_heads, seqlen, head_dim]
    xq = xq.transpose(0, 2, 1, 3)
    # Transpose keys/values: [bsz, n_heads, current_seqlen, head_dim]
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # Calculate attention scores
    # xq: [bsz, n_heads, seqlen, head_dim]
    # keys: [bsz, n_heads, current_seqlen, head_dim] -> keys.transpose: [bsz, n_heads, head_dim, current_seqlen]
    scores = jnp.matmul(xq, keys.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

    # Apply causal mask (implicit via sequence lengths during decoding)
    # During decoding (seqlen=1), scores shape is [bsz, n_heads, 1, current_seqlen].
    # Softmax over the last dim (current_seqlen) naturally handles causality
    # as keys/values only contain past and current tokens.
    # For prefill (seqlen > 1), an explicit mask might be needed if not handled upstream.
    # Let's add an explicit causal mask for generality, assuming needed for prefill.
    if seqlen > 1:
      # Adjust mask based on start_pos. This assumes query positions are [start_pos..start_pos+seqlen-1]
      # and key positions are [0..current_seqlen-1].
      # A query at pos q_idx = start_pos + i can attend to key at pos k_idx if k_idx <= q_idx.
      # The mask needs to be [bsz, n_heads, seqlen, current_seqlen]
      query_indices = jnp.arange(seqlen) + start_pos
      key_indices = jnp.arange(current_seqlen)
      causal_mask = key_indices <= query_indices[:, None]
      scores = jnp.where(causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)

    # Softmax scores
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = attn_weights.astype(values.dtype) # Ensure consistent dtype

    # Apply attention weights to values
    # attn_weights: [bsz, n_heads, seqlen, current_seqlen]
    # values: [bsz, n_heads, current_seqlen, head_dim] -> output: [bsz, n_heads, seqlen, head_dim]
    attn_output = jnp.matmul(attn_weights, values)

    # Concatenate heads and project output
    # attn_output: [bsz, n_heads, seqlen, head_dim] -> [bsz, seqlen, n_heads, head_dim]
    attn_output = attn_output.transpose(0, 2, 1, 3)
    # attn_output: -> [bsz, seqlen, n_heads * head_dim]
    attn_output = attn_output.reshape(bsz, seqlen, -1)

    # Final linear projection
    # attn_output: [bsz, seqlen, dim], params.wo: [n_heads * head_dim, dim]
    output = jnp.einsum('bsd,do->bso', attn_output, params.wo)

    return output, updated_cache


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