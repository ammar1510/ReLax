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
            k=jnp.zeros((n_layers, bsz, kv_heads, max_seqlen, head_dim), dtype=dtype),
            v=jnp.zeros((n_layers, bsz, kv_heads, max_seqlen, head_dim), dtype=dtype),
            positions=jnp.zeros(bsz, dtype=jnp.int32),
        )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int):
        """Updates the Key and Value cache for all sequences, each at their own position.

        Each sequence in the batch can be at a different cache position. This method
        writes each sequence's keys/values at its corresponding position tracked in
        self.positions.

        Args:
          xk: The new key tensor to be added to the cache.
              Shape: `(bsz, n_kv_heads, seqlen, head_dim)` - already transposed
          xv: The new value tensor to be added to the cache.
              Shape: `(bsz, n_kv_heads, seqlen, head_dim)` - already transposed
          layer_idx: The index of the layer to update.

        Returns:
          A new KVCache object with updated keys, values, and incremented positions.

        Note:
          Padding tokens in the input are written to cache but will be masked
          during attention based on positions parameter.
        """
        # xk, xv shapes: [bsz, n_kv_heads, seqlen, head_dim] - already transposed
        # self.k, self.v shapes: [layers, bsz, kv_heads, max_seqlen, head_dim]

        bsz, n_kv_heads, seqlen, head_dim = xk.shape

        # Ensure xk/xv have the same dtype as cache
        xk = xk.astype(self.k.dtype)
        xv = xv.astype(self.v.dtype)

        # Start with current cache
        new_k = self.k
        new_v = self.v

        # Update each sequence at its own position
        # Loop over batch dimension - unrolls at compile time (bsz is static)
        for i in range(bsz):
            # Get this sequence's current cache position
            start_pos = self.positions[i]  # Dynamic value, different per sequence

            # Extract this sequence's keys/values: [n_kv_heads, seqlen, head_dim]
            xk_i = xk[i]
            xv_i = xv[i]

            # Reshape for dynamic_update_slice: [1, 1, n_kv_heads, seqlen, head_dim]
            # Leading dims: (layer, batch_idx)
            xk_update = xk_i[None, None, :, :, :]
            xv_update = xv_i[None, None, :, :, :]

            # Update cache at this sequence's position
            # Indices: (layer, batch_idx, kv_head, seq_pos, head_dim)
            new_k = jax.lax.dynamic_update_slice(
                new_k,
                xk_update,
                (layer_idx, i, 0, start_pos, 0)
            )
            new_v = jax.lax.dynamic_update_slice(
                new_v,
                xv_update,
                (layer_idx, i, 0, start_pos, 0)
            )
        return KVCache(k=new_k, v=new_v, positions=self.positions)


    def update_positions(self, true_len: jax.Array):
        """Updates the positions of the cache.

        Args:
            true_len: Actual (non-padded) sequence lengths for each sequence.
                Shape: `(bsz,)`
        """
        return KVCache(k=self.k, v=self.v, positions=self.positions + true_len)
    def get_layer(self, layer_idx: int):
        """Retrieves K/V for a specific layer.

        Note: This returns the full cache up to max_seqlen. The caller is
        responsible for masking for attention computation.
        """
        return self.k[layer_idx], self.v[layer_idx] # [bsz, kv_heads, max_seqlen, head_dim]