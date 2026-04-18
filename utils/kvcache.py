import jax
import jax.numpy as jnp
from flax import struct
from functools import partial


@partial(jax.jit, static_argnames=['prefill_seqlen'])
def _jitted_batch_insert(k, v, seq_positions, stacked_k, stacked_v, slot_idxs, lens, next_tokens, curr_tokens, prefill_seqlen):
    # stacked_k: [n_entries, n_layers, n_kv_heads, prefill_seqlen, head_dim]
    # k:         [n_layers, bsz, n_kv_heads, max_seqlen, head_dim]
    # target:    k[:, slot_idxs, :, :prefill_seqlen, :] -> [n_layers, n_entries, n_kv_heads, prefill_seqlen, head_dim]
    new_k = k.at[:, slot_idxs, :, :prefill_seqlen, :].set(stacked_k.transpose(1, 0, 2, 3, 4))
    new_v = v.at[:, slot_idxs, :, :prefill_seqlen, :].set(stacked_v.transpose(1, 0, 2, 3, 4))
    new_positions = seq_positions.at[slot_idxs].set(lens)
    new_tokens = curr_tokens.at[slot_idxs, 0].set(next_tokens)
    return new_k, new_v, new_positions, new_tokens


@struct.dataclass
class KVCache:
    k: jax.Array
    v: jax.Array
    seq_positions: jax.Array  # [bsz] - tracks current filled position for each sequence

    @classmethod
    def new(
        cls,
        config,
        bsz: int,
        max_seqlen: int,
        dtype=None,
        mesh=None,
    ) -> "KVCache":
        """Allocate an empty KVCache from a model config.

        Args:
            config: Any model config with n_layers, n_kv_heads, head_dim, dtype fields.
            bsz: Batch size.
            max_seqlen: Maximum sequence length.
            dtype: Data type. Defaults to config.dtype if not provided.
            mesh: If provided, allocate shards directly on each device without
                  an intermediate full-sized host or device allocation.
        """
        dt = jnp.dtype(dtype if dtype is not None else config.dtype)
        k_shape = (config.n_layers, bsz, config.n_kv_heads, max_seqlen, config.head_dim)
        if mesh is not None:
            from utils.mesh_helpers import MeshHelper
            from jax.sharding import PartitionSpec as PS
            dp = "dp" if "dp" in mesh.axis_names else None
            tp = MeshHelper.get_tp_axis(mesh)
            if dp is not None:
                k_spec = PS(None, dp, tp, None, None)
                pos_spec = PS(dp)
            else:
                k_spec = MeshHelper.batch_axis_spec(mesh, rank=5, batch_axis=1)
                pos_spec = MeshHelper.batch_axis_spec(mesh, rank=1, batch_axis=0)
            return cls(
                k=MeshHelper.create_sharded_zeros(k_shape, dt, mesh, k_spec),
                v=MeshHelper.create_sharded_zeros(k_shape, dt, mesh, k_spec),
                seq_positions=MeshHelper.create_sharded_zeros((bsz,), jnp.int32, mesh, pos_spec),
            )
        return cls(
            k=jnp.zeros(k_shape, dtype=dt),
            v=jnp.zeros(k_shape, dtype=dt),
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

    def slice(self, idx: int) -> "KVCache":
        """Extract a single-sequence cache at batch index idx.

        Args:
            idx: Batch index to extract.
        """
        return KVCache(
            k=self.k[:, idx : idx + 1, :, :, :],
            v=self.v[:, idx : idx + 1, :, :, :],
            seq_positions=self.seq_positions[idx : idx + 1],
        )

    def place_on_mesh(self, mesh) -> "KVCache":
        """Re-shard KVCache onto mesh, preserving existing data.

        Use this after mutating the cache to restore correct sharding
        before the next JIT-compiled decode step.

        Args:
            mesh: JAX device mesh.
        """
        from utils.mesh_helpers import MeshHelper
        return MeshHelper.place_kv_cache(self, mesh)

    def batch_insert(self, entries, slot_idxs, lens, next_tokens, curr_tokens):
        """Scatter prefill results into decode slots.

        Args:
            entries: List of single-sequence KVCaches from prefill (via slice()).
            slot_idxs: Decode slot indices to insert into.
            lens: Prefill sequence lengths for each entry.
            next_tokens: First generated token for each entry.
            curr_tokens: Current token array for the decode batch [batch, 1].

        Returns:
            Tuple of (updated KVCache, updated curr_tokens).
        """
        prefill_seqlen = entries[0].k.shape[3]
        # Stack entries: [n_entries, n_layers, n_kv_heads, prefill_seqlen, head_dim]
        stacked_k = jnp.stack([e.k[:, 0, :, :, :] for e in entries])
        stacked_v = jnp.stack([e.v[:, 0, :, :, :] for e in entries])
        new_k, new_v, new_positions, new_tokens = _jitted_batch_insert(
            self.k, self.v, self.seq_positions,
            stacked_k, stacked_v,
            jnp.array(slot_idxs), jnp.array(lens), jnp.array(next_tokens),
            curr_tokens, prefill_seqlen,
        )
        return KVCache(k=new_k, v=new_v, seq_positions=new_positions), new_tokens
