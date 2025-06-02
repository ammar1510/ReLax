# Placeholder for shared, reusable Flax modules (RMSNorm, FFN, Attention, etc.) 
import jax
import jax.numpy as jnp
import jax.nn as nn # Ensure nn is imported
import jax.lax as lax
from flax import struct
from jax.experimental.pallas.ops.tpu import flash_attention
from .kvcache import KVCache # Import KVCache

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> jnp.ndarray:
    """
    Precompute the rotational frequency embeddings (cis = complex numbers).

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length supported by the model.
        theta: Base parameter for frequency calculation.

    Returns:
        A JAX array of shape [2, max_seq_len, head_dim // 2] containing
        the cosine [0, :, :] and sine [1, :, :] components.
    """
    # Calculate frequencies: theta^( -2(i-1) / dim ) for i = 1..dim/2
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)].astype(jnp.float32) / head_dim))

    # Create position indices: [0, 1, ..., max_seq_len - 1]
    t = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product to get frequencies for each position: shape [max_seq_len, head_dim // 2]
    freqs = jnp.outer(t, freqs)

    # Calculate cosine and sine components
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)

    # Stack cos and sin: shape [2, max_seq_len, head_dim // 2]
    freqs_cis = jnp.stack([freqs_cos, freqs_sin], axis=0)

    return freqs_cis

@struct.dataclass
class AttentionParams:
    wq: jnp.ndarray
    wk: jnp.ndarray
    wv: jnp.ndarray
    wo: jnp.ndarray

@struct.dataclass
class FeedForwardParams:
    w1_gate: jnp.ndarray # Corresponds to gate_proj
    w2_up: jnp.ndarray   # Corresponds to up_proj
    w3_down: jnp.ndarray # Corresponds to down_proj


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Apply Root Mean Square Normalization.

    Args:
        x: Input tensor.
        weight: Weight tensor for scaling.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    output = x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return output * weight

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply Rotary Positional Embeddings to query and key tensors.

    Args:
        xq: Query tensor.
        xk: Key tensor.
        freqs_cis: Precomputed rotary frequency embeddings.

    Returns:
        Tuple of query and key tensors with applied RoPE.
    """
    # xq_ and xk_ have shape [batch_size, seq_len, n_heads, head_dim]
    # freqs_cis has shape [seq_len, head_dim // 2 * 2] or similar complex format

    # Reshape xq and xk to view the last dimension as pairs of real/imaginary parts
    xq_r, xq_i = jnp.split(xq, 2, axis=-1)
    xk_r, xk_i = jnp.split(xk, 2, axis=-1)

    # Reshape freqs_cis for broadcasting
    # Input freqs_cis shape: (2, seq_len_slice, head_dim // 2)
    # Need to broadcast to: (bs, seq_len_slice, n_heads, head_dim // 2)
    # Add batch and head dims, expand last dim
    freqs_cos = freqs_cis[0][None, :, None, :].repeat(xq_r.shape[-1] // freqs_cis[0].shape[-1], axis=-1)
    freqs_sin = freqs_cis[1][None, :, None, :].repeat(xq_r.shape[-1] // freqs_cis[1].shape[-1], axis=-1)

    freqs_cos = jnp.broadcast_to(freqs_cos, xq_r.shape) # Real part
    freqs_sin = jnp.broadcast_to(freqs_sin, xq_r.shape) # Imaginary part


    # Apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine back
    xq_out = jnp.concatenate([xq_out_r, xq_out_i], axis=-1)
    xk_out = jnp.concatenate([xk_out_r, xk_out_i], axis=-1)

    return xq_out, xk_out

def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    Repeat Key/Value heads for Grouped Query Attention.

    Args:
        x: Input tensor (keys or values) with shape [bs, seq_len, n_kv_heads, head_dim].
        n_rep: Number of times to repeat the KV heads.

    Returns:
        Tensor with repeated KV heads, shape [bs, seq_len, n_q_heads, head_dim].
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # equivalent to torch.repeat_interleave(x, repeats=n_rep, dim=2)
    return jnp.broadcast_to(x[:, :, :, None, :], (bs, slen, n_kv_heads, n_rep, head_dim)).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


def grouped_query_attention(
    x: jnp.ndarray,
    freqs_cis: jnp.ndarray, # Precomputed freqs for max_seq_len
    # mask: jnp.ndarray | None, # Mask is implicitly causal up to start_pos+seqlen
    params: AttentionParams,
    kv_cache: KVCache,
    layer_idx: int,
    start_pos: int,
    n_heads: int,
    n_kv_heads: int,
) -> tuple[jnp.ndarray, KVCache]: # Return output and updated cache
    """
    Compute Grouped Query Attention with KV Caching.

    Args:
        x: Input tensor of shape [batch_size, seq_len, dim] (seq_len=1 for decoding).
        freqs_cis: Precomputed rotary frequency embeddings (complex format, shape like [2, max_seq_len, head_dim//2]).
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
    head_dim = params.wq.shape[-1] // n_heads # Assuming wq shape [dim, n_heads * head_dim]

    # Project inputs to queries, keys, values for the current token(s)
    # Shapes: [bsz, seqlen, n_heads, head_dim] for xq
    # Shapes: [bsz, seqlen, n_kv_heads, head_dim] for xk, xv
    xq = jnp.einsum('bsd,dhc->bshc', x, params.wq.reshape(dim, n_heads, head_dim))
    xk = jnp.einsum('bsd,dkc->bskc', x, params.wk.reshape(dim, n_kv_heads, head_dim))
    xv = jnp.einsum('bsd,dvc->bsvc', x, params.wv.reshape(dim, n_kv_heads, head_dim))

    # Apply rotary positional embeddings to the new queries and keys
    # Slice freqs_cis for the current position(s)
    # Assuming freqs_cis has shape [2, max_seq_len, head_dim//2]
    current_freqs_cis = (
        lax.dynamic_slice_in_dim(freqs_cis[0], start_pos, seqlen, axis=0),
        lax.dynamic_slice_in_dim(freqs_cis[1], start_pos, seqlen, axis=0)
    )
    # Pass the sliced freqs to RoPE
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=current_freqs_cis)

    # Update the KV cache
    updated_cache = kv_cache.update(xk, xv, layer_idx, start_pos)

    # Retrieve the full key/value sequences for this layer from cache
    # keys/values shape: [bsz, current_seq_len, n_kv_heads, head_dim]
    current_seq_len = start_pos + seqlen
    keys, values = updated_cache.get_layer(layer_idx, start_pos, seqlen)


    # Repeat KV heads if n_kv_heads < n_heads (GQA)
    n_rep = n_heads // n_kv_heads
    keys = repeat_kv(keys, n_rep) # Shape: [bsz, current_seq_len, n_heads, head_dim]
    values = repeat_kv(values, n_rep) # Shape: [bsz, current_seq_len, n_heads, head_dim]

    # Transpose for score calculation: [bsz, n_heads, seqlen, head_dim]
    xq = xq.transpose(0, 2, 1, 3)
    # Transpose keys/values: [bsz, n_heads, current_seq_len, head_dim]
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # Calculate attention scores
    # xq: [bsz, n_heads, seqlen, head_dim]
    # keys: [bsz, n_heads, current_seq_len, head_dim] -> keys.transpose: [bsz, n_heads, head_dim, current_seq_len]
    scores = jnp.matmul(xq, keys.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

    # Apply causal mask (implicit via sequence lengths during decoding)
    # During decoding (seqlen=1), scores shape is [bsz, n_heads, 1, current_seq_len].
    # Softmax over the last dim (current_seq_len) naturally handles causality
    # as keys/values only contain past and current tokens.
    # For prefill (seqlen > 1), an explicit mask might be needed if not handled upstream.
    # Let's add an explicit causal mask for generality, assuming needed for prefill.
    if seqlen > 1:
      # Adjust mask based on start_pos. This assumes query positions are [start_pos..start_pos+seqlen-1]
      # and key positions are [0..current_seq_len-1].
      # A query at pos q_idx = start_pos + i can attend to key at pos k_idx if k_idx <= q_idx.
      # The mask needs to be [bsz, n_heads, seqlen, current_seq_len]
      query_indices = jnp.arange(seqlen) + start_pos
      key_indices = jnp.arange(current_seq_len)
      causal_mask = key_indices <= query_indices[:, None]
      scores = jnp.where(causal_mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)

    # Softmax scores
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = attn_weights.astype(values.dtype) # Ensure consistent dtype

    # Apply attention weights to values
    # attn_weights: [bsz, n_heads, seqlen, current_seq_len]
    # values: [bsz, n_heads, current_seq_len, head_dim] -> output: [bsz, n_heads, seqlen, head_dim]
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
    x: jnp.ndarray,
    params: FeedForwardParams,
    activation_fn: str, # Added activation function name
) -> jnp.ndarray:
    """
    Compute FeedForward network (MLP) using a configurable activation function (like SwiGLU).

    Args:
        x: Input tensor of shape [batch_size, seq_len, dim].
        params: Dataclass containing weight matrices (w1_gate, w2_up, w3_down).
        activation_fn: Name of the activation function ('silu', 'relu', 'gelu').

    Returns:
        Output tensor after MLP computation.
    """

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
    
    # Project input: x -> gate, up
    # x: [bs, seqlen, dim], w1_gate: [dim, hidden_dim], w2_up: [dim, hidden_dim]
    gate = jnp.einsum('bsd,dh->bsh', x, params.w1_gate)
    up = jnp.einsum('bsd,dh->bsh', x, params.w2_up)

   
    fused_activation = activated_gate * up

    # Project down
    # fused_swiglu: [bs, seqlen, hidden_dim], w3_down: [hidden_dim, dim]
    output = jnp.einsum('bsh,hd->bsd', fused_activation, params.w3_down)

    return output 