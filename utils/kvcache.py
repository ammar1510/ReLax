import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class KVCache:
    k: jax.Array
    v: jax.Array
    seq_positions: jax.Array  # [bsz] - tracks current filled position for each sequence

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
            seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
        )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int):
        """Updates the Key and Value cache for all sequences, each at their own position.

        Uses vmap to vectorize across the batch dimension — a single batched
        scatter replaces bsz sequential dynamic_update_slice calls.

        Args:
          xk: The new key tensor to be added to the cache.
              Shape: `(bsz, n_kv_heads, seqlen, head_dim)` - already transposed
          xv: The new value tensor to be added to the cache.
              Shape: `(bsz, n_kv_heads, seqlen, head_dim)` - already transposed
          layer_idx: The index of the layer to update.

        Returns:
          A new KVCache object with updated keys, values, and incremented positions.
        """
        xk = xk.astype(self.k.dtype)
        xv = xv.astype(self.v.dtype)

        def _update_single(cache_slice, new_kv, pos):
            """Update one sequence's cache at its position.

            Args:
                cache_slice: [kv_heads, max_seqlen, head_dim]
                new_kv: [kv_heads, seqlen, head_dim]
                pos: scalar — this sequence's current cache position
            """
            return jax.lax.dynamic_update_slice(cache_slice, new_kv, (0, pos, 0))

        # Extract this layer: [bsz, kv_heads, max_seqlen, head_dim]
        layer_k = self.k[layer_idx]
        layer_v = self.v[layer_idx]

        # vmap over batch — all sequences updated in one batched op
        updated_k = jax.vmap(_update_single)(layer_k, xk, self.seq_positions)
        updated_v = jax.vmap(_update_single)(layer_v, xv, self.seq_positions)

        # Write back to full cache
        new_k = self.k.at[layer_idx].set(updated_k)
        new_v = self.v.at[layer_idx].set(updated_v)

        return KVCache(k=new_k, v=new_v, seq_positions=self.seq_positions)


    def update_positions(self, true_len: jax.Array):
        """Updates the positions of the cache.

        Args:
            true_len: Actual (non-padded) sequence lengths for each sequence.
                Shape: `(bsz,)`
        """
        return KVCache(k=self.k, v=self.v, seq_positions=self.seq_positions + true_len)
    def get_layer(self, layer_idx: int):
        """Retrieves K/V for a specific layer.

        Note: This returns the full cache up to max_seqlen. The caller is
        responsible for masking for attention computation.
        """
        return self.k[layer_idx], self.v[layer_idx] # [bsz, kv_heads, max_seqlen, head_dim]