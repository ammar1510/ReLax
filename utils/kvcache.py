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

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int):
        """Updates the Key and Value cache for all sequences, each at their own position.

        Each sequence in the batch can be at a different cache position. This method
        writes each sequence's keys/values at its corresponding position tracked in
        self.positions, then increments all positions by seqlen.

        Args:
          xk: The new key tensor to be added to the cache.
              Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
          xv: The new value tensor to be added to the cache.
              Shape: `(bsz, seqlen, n_kv_heads, head_dim)`
          layer_idx: The index of the layer to update.

        Returns:
          A new KVCache object with updated keys, values, and incremented positions.

        Note:
          Padding tokens in the input are written to cache but will be masked
          during attention based on true_lengths parameter.
        """
        # xk, xv shapes: [bsz, seqlen, n_kv_heads, head_dim]
        # self.k, self.v shapes: [layers, bsz, max_seqlen, n_kv_heads, head_dim]

        bsz, seqlen, n_kv_heads, head_dim = xk.shape

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

            # Extract this sequence's keys/values: [seqlen, n_kv_heads, head_dim]
            xk_i = xk[i]
            xv_i = xv[i]

            # Reshape for dynamic_update_slice: [1, 1, seqlen, n_kv_heads, head_dim]
            # Leading dims: (layer, batch_idx)
            xk_update = xk_i[None, None, :, :, :]
            xv_update = xv_i[None, None, :, :, :]

            # Update cache at this sequence's position
            # Indices: (layer, batch_idx, seq_pos, kv_head, head_dim)
            new_k = jax.lax.dynamic_update_slice(
                new_k,
                xk_update,
                (layer_idx, i, start_pos, 0, 0)
            )
            new_v = jax.lax.dynamic_update_slice(
                new_v,
                xv_update,
                (layer_idx, i, start_pos, 0, 0)
            )

        # Update positions: all sequences increment by seqlen (the padded length)
        new_positions = self.positions + seqlen

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
