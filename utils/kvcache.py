import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class TempKV:
    """Temporary K/V pairs during a forward pass."""
    k: jax.Array  # [n_layers, bsz, n_kv_heads, seqlen, head_dim]
    v: jax.Array  # [n_layers, bsz, n_kv_heads, seqlen, head_dim]

    @classmethod
    def new(
        cls,
        n_layers: int,
        bsz: int,
        seqlen: int,
        kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> "TempKV":
        """Create temporary storage for new K/V pairs."""
        return cls(
            k=jnp.zeros((n_layers, bsz, kv_heads, seqlen, head_dim), dtype=dtype),
            v=jnp.zeros((n_layers, bsz, kv_heads, seqlen, head_dim), dtype=dtype),
        )

    def set_layer(self, layer_idx: int, xk: jax.Array, xv: jax.Array) -> "TempKV":
        """Store K/V tensors for a specific layer.

        Args:
            layer_idx: Index of the layer
            xk: Keys for this layer [bsz, n_kv_heads, seqlen, head_dim]
            xv: Values for this layer [bsz, n_kv_heads, seqlen, head_dim]

        Returns:
            New TempKV with the layer set.
        """
        new_k = self.k.at[layer_idx].set(xk)
        new_v = self.v.at[layer_idx].set(xv)
        return TempKV(k=new_k, v=new_v)


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

    def update(self, temp_kv: TempKV, true_len: jax.Array):
        """Updates the Key and Value cache for all layers and sequences.

        Each sequence in the batch can be at a different cache position. This method
        writes each sequence's keys/values at its corresponding position tracked in
        self.positions, then increments each position by its actual (non-padded) length.

        Args:
          temp_kv: TempKV containing new K/V pairs for all layers.
              Shapes: k/v are [n_layers, bsz, n_kv_heads, seqlen, head_dim]
          true_len: Actual (non-padded) sequence lengths for each sequence.
              Shape: `(bsz,)`

        Returns:
          A new KVCache object with updated keys, values, and incremented positions.

        Note:
          Padding tokens in the input are written to cache but will be masked
          during attention based on true_lengths parameter.
        """
        # temp_kv.k/v shapes: [n_layers, bsz, n_kv_heads, seqlen, head_dim]
        # self.k, self.v shapes: [n_layers, bsz, kv_heads, max_seqlen, head_dim]

        n_layers, bsz, n_kv_heads, seqlen, head_dim = temp_kv.k.shape

        # Ensure temp kv has the same dtype as cache
        new_k_data = temp_kv.k.astype(self.k.dtype)
        new_v_data = temp_kv.v.astype(self.v.dtype)

        # Start with current cache
        new_k = self.k
        new_v = self.v

        # Update each sequence at its own position
        # Loop over batch dimension - unrolls at compile time (bsz is static)
        for i in range(bsz):
            # Get this sequence's current cache position
            start_pos = self.positions[i]  # Dynamic value, different per sequence

            # Extract this sequence's keys/values across all layers: [n_layers, n_kv_heads, seqlen, head_dim]
            xk_i = new_k_data[:, i, :, :, :]
            xv_i = new_v_data[:, i, :, :, :]

            # Reshape for dynamic_update_slice: [n_layers, 1, n_kv_heads, seqlen, head_dim]
            xk_update = xk_i[:, None, :, :, :]
            xv_update = xv_i[:, None, :, :, :]

            # Update cache at this sequence's position for all layers
            # Indices: (layer, batch_idx, kv_head, seq_pos, head_dim)
            new_k = jax.lax.dynamic_update_slice(
                new_k,
                xk_update,
                (0, i, 0, start_pos, 0)
            )
            new_v = jax.lax.dynamic_update_slice(
                new_v,
                xv_update,
                (0, i, 0, start_pos, 0)
            )

        # Update positions: each sequence increments by its actual (non-padded) length
        new_positions = self.positions + true_len

        return KVCache(k=new_k, v=new_v, positions=new_positions)

    def get_layer(self, layer_idx: int):
        """Retrieves K/V for a specific layer.

        Note: This returns the full cache up to max_seqlen. The caller is
        responsible for masking for attention computation.
        """
        return self.k[layer_idx], self.v[layer_idx] # [bsz, kv_heads, max_seqlen, head_dim]
