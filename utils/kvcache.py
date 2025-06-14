import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class KVCache:
  k: jax.Array
  v: jax.Array

  @classmethod
  # @functools.partial(jax.jit, static_argnums=(0,1,2,3,4,5)) # Jitting classmethod might be tricky, usually jit the function calling this
  def new(cls, n_layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int, dtype=jnp.bfloat16) -> 'KVCache':
    return cls(
        k=jnp.zeros((n_layers, bsz, max_seq_len, kv_heads, head_dim), dtype=dtype),
        v=jnp.zeros((n_layers, bsz, max_seq_len, kv_heads, head_dim), dtype=dtype)
    )

  def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, start_pos: int):
    """Updates the Key and Value cache for a specific layer.

    This method takes the newly computed key and value tensors for a single layer
    and updates the cache at the correct position. This is used during the forward
    pass of the transformer to store the context for subsequent generation steps.

    Args:
      xk: The new key tensor to be added to the cache.
          Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
      xv: The new value tensor to be added to the cache.
          Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
      layer_idx: The index of the layer to update.
      start_pos: The starting position in the sequence dimension where the update
                 should be applied. This is used to correctly place the new keys
                 and values in the cache.

    Returns:
      A new KVCache object with the updated key and value tensors.
    """
    # xk, xv shapes: [bsz, seqlen, n_kv_heads, head_dim]
    # self.k, self.v shapes: [layers, bsz, max_seq_len, n_kv_heads, head_dim]

    # Ensure xk/xv have the same dtype as cache
    xk = xk.astype(self.k.dtype)
    xv = xv.astype(self.v.dtype)

    # Prepare update tensor shape to match the slice rank: [1, bsz, seqlen, n_kv_heads, head_dim]
    # The leading dimension corresponds to the layer axis.
    k_update = xk[None, ...]
    v_update = xv[None, ...]

    # Define start indices for the update slice in the cache
    # Indices correspond to: (layer, batch, sequence, kv_head, head_dim)
    start_indices = (layer_idx, 0, start_pos, 0, 0)

    # Update the cache using dynamic_update_slice
    k = jax.lax.dynamic_update_slice(self.k, k_update, start_indices)
    v = jax.lax.dynamic_update_slice(self.v, v_update, start_indices)

    return KVCache(k=k, v=v)

  def get_layer(self, layer_idx: int, start_pos: int, seqlen: int):
        """Retrieves K/V for a specific layer up to the current sequence length."""
        # Fetch slice up to start_pos + seqlen
        current_seq_len = start_pos + seqlen
        # Use dynamic_slice for retrieval. Indices: (layer, batch, seq, head, dim)
        # Slice shape: [1, bsz, current_seq_len, n_kv_heads, head_dim]
        start_indices_k = (layer_idx, 0, 0, 0, 0)
        slice_sizes_k = (1, self.k.shape[1], current_seq_len, self.k.shape[3], self.k.shape[4])
        start_indices_v = (layer_idx, 0, 0, 0, 0)
        slice_sizes_v = (1, self.v.shape[1], current_seq_len, self.v.shape[3], self.v.shape[4])

        keys = jax.lax.dynamic_slice(self.k, start_indices_k, slice_sizes_k)[0] # Remove leading layer dim
        values = jax.lax.dynamic_slice(self.v, start_indices_v, slice_sizes_v)[0] # Remove leading layer dim

        # keys/values shape: [bsz, current_seq_len, n_kv_heads, head_dim]
        return keys, values 