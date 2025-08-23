import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class KVCache:
    k: jax.Array
    v: jax.Array
    positions: jax.Array  # [bsz] - tracks current filled position for each sequence

    @classmethod
    # @functools.partial(jax.jit, static_argnums=(0,1,2,3,4,5)) 
    def new(
        cls,
        n_layers: int,
        bsz: int,
        max_seqlen: int,
        kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> "KVCache":
        return cls(
            k=jnp.zeros((n_layers, bsz, max_seqlen, kv_heads, head_dim), dtype=dtype),
            v=jnp.zeros((n_layers, bsz, max_seqlen, kv_heads, head_dim), dtype=dtype),
            positions=jnp.zeros(bsz, dtype=jnp.int32),
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

        return KVCache(k=k, v=v, positions=self.positions)

    def _update_single_sequence(self, cache_k_seq, cache_v_seq, position, xk_seq, xv_seq, seq_len):
        """Update cache for a single sequence.
        
        Args:
            cache_k_seq: Current cache keys for one sequence [max_seqlen, n_kv_heads, head_dim]
            cache_v_seq: Current cache values for one sequence [max_seqlen, n_kv_heads, head_dim]  
            position: Current cache position for this sequence (scalar)
            xk_seq: New keys for this sequence [seqlen, n_kv_heads, head_dim]
            xv_seq: New values for this sequence [seqlen, n_kv_heads, head_dim]
            seq_len: Actual length of new tokens (scalar)
            
        Returns:
            Tuple of (updated_cache_k, updated_cache_v, new_position)
        """        
        # Only update the valid portion of the sequence
        # No need to mask - just slice to the actual length
        xk_valid = xk_seq[:seq_len]  # [seq_len, n_kv_heads, head_dim]
        xv_valid = xv_seq[:seq_len]  # [seq_len, n_kv_heads, head_dim]
        
        # Update cache at current position with only valid tokens
        new_cache_k = jax.lax.dynamic_update_slice(
            cache_k_seq, xk_valid, (position, 0, 0)
        )
        new_cache_v = jax.lax.dynamic_update_slice(
            cache_v_seq, xv_valid, (position, 0, 0)
        )
        
        # New position after adding valid tokens
        new_position = position + seq_len
        
        return new_cache_k, new_cache_v, new_position

    def update_batch(self, xk: jax.Array, xv: jax.Array, layer_idx: int, seq_lengths: jax.Array):
        """Updates the Key and Value cache for a specific layer with variable-length sequences.

        This method handles updating the cache when sequences in the batch have different lengths.
        Each sequence writes tokens starting at its current cached position.

        Args:
          xk: The new key tensor to be added to the cache.
              Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
          xv: The new value tensor to be added to the cache.
              Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
          layer_idx: The index of the layer to update.
          seq_lengths: Actual sequence lengths for each batch item in the current input.
                      Shape: `(bsz,)` - each element indicates length of new tokens to add

        Returns:
          A new KVCache object with the updated key and value tensors and positions.
        """
        # Ensure xk/xv have the same dtype as cache
        xk = xk.astype(self.k.dtype)
        xv = xv.astype(self.v.dtype)

        # Get current cache layer
        cache_k_layer = self.k[layer_idx]  # [bsz, max_seqlen, n_kv_heads, head_dim]
        cache_v_layer = self.v[layer_idx]  # [bsz, max_seqlen, n_kv_heads, head_dim]
        
        # Vectorize the single-sequence update function
        vmapped_update = jax.vmap(
            self._update_single_sequence,
            in_axes=(0, 0, 0, 0, 0, 0),  # All inputs have batch dimension at axis 0
            out_axes=(0, 0, 0)           # All outputs have batch dimension at axis 0
        )
        
        # Apply to entire batch
        new_k_layer, new_v_layer, new_positions = vmapped_update(
            cache_k_layer, cache_v_layer, self.positions, xk, xv, seq_lengths
        )
        
        # Update the full cache tensor
        new_k = self.k.at[layer_idx].set(new_k_layer)
        new_v = self.v.at[layer_idx].set(new_v_layer)

        return KVCache(k=new_k, v=new_v, positions=new_positions)

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
