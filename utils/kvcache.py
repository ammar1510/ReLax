import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class KVCache:
  k: jax.Array
  v: jax.Array

  @classmethod
  # @functools.partial(jax.jit, static_argnums=(0,1,2,3,4,5)) # Jitting classmethod might be tricky, usually jit the function calling this
  def new(cls, n_layers: int, bsz: int, max_seqlen: int, kv_heads: int, head_dim: int, dtype=jnp.bfloat16) -> 'KVCache':
    return cls(
        k=jnp.zeros((n_layers, bsz, max_seqlen, kv_heads, head_dim), dtype=dtype),
        v=jnp.zeros((n_layers, bsz, max_seqlen, kv_heads, head_dim), dtype=dtype)
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
    # self.k, self.v shapes: [layers, bsz, max_seqlen, n_kv_heads, head_dim]

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

  def get_layer(self, layer_idx: int):
        """Retrieves K/V for a specific layer.
        
        Note: This returns the full cache up to max_seqlen. The caller is
        responsible for masking for attention computation.
        """
        # self.k, self.v shapes: [layers, bsz, max_seqlen, n_kv_heads, head_dim]
        keys = self.k[layer_idx]
        values = self.v[layer_idx]

        # keys/values shape: [bsz, max_seqlen, n_kv_heads, head_dim]
        return keys, values 