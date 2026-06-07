"""
KV cache for Gemma 4 models with hybrid sliding/global attention.

Combines two independent KVCaches:
- sliding_cache: for local sliding-window attention layers
- global_cache: for full-context attention layers
"""

import jax
import jax.numpy as jnp
from flax import struct

from utils.kvcache import KVCache


@struct.dataclass
class GemmaCache:
    """Dual KV cache for Gemma's hybrid sliding/global attention.

    Attributes:
        sliding_cache: KVCache for sliding-window attention layers.
            Shape: [n_sliding_layers, bsz, n_kv_heads, max_seqlen, head_dim]
        global_cache: KVCache for full-context attention layers.
            Shape: [n_global_layers, bsz, n_global_kv_heads, max_seqlen, global_head_dim]
    """

    sliding_cache: KVCache
    global_cache: KVCache

    @classmethod
    def new(
        cls,
        config,
        bsz: int,
        max_seqlen: int,
        dtype=jnp.bfloat16,
        mesh=None,
    ) -> "GemmaCache":
        """Allocate an empty GemmaCache from a GemmaConfig.

        Args:
            config: GemmaConfig instance.
            bsz: Batch size.
            max_seqlen: Maximum sequence length for both sub-caches.
            dtype: Data type for cache arrays.
            mesh: If provided, allocate shards directly on each device without
                  an intermediate full-sized allocation.
        """
        sliding_shape = (config.n_sliding_layers, bsz, config.n_kv_heads, max_seqlen, config.head_dim)
        global_shape = (config.n_global_layers, bsz, config.n_global_kv_heads, max_seqlen, config.global_head_dim)
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
                sliding_cache=KVCache(
                    k=MeshHelper.create_sharded_zeros(sliding_shape, dtype, mesh, k_spec),
                    v=MeshHelper.create_sharded_zeros(sliding_shape, dtype, mesh, k_spec),
                    seq_positions=MeshHelper.create_sharded_zeros((bsz,), jnp.int32, mesh, pos_spec),
                ),
                global_cache=KVCache(
                    k=MeshHelper.create_sharded_zeros(global_shape, dtype, mesh, k_spec),
                    v=MeshHelper.create_sharded_zeros(global_shape, dtype, mesh, k_spec),
                    seq_positions=MeshHelper.create_sharded_zeros((bsz,), jnp.int32, mesh, pos_spec),
                ),
            )
        return cls(
            sliding_cache=KVCache(
                k=jnp.zeros(sliding_shape, dtype=dtype),
                v=jnp.zeros(sliding_shape, dtype=dtype),
                seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
            ),
            global_cache=KVCache(
                k=jnp.zeros(global_shape, dtype=dtype),
                v=jnp.zeros(global_shape, dtype=dtype),
                seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
            ),
        )

    def update_positions(self, true_len: jax.Array) -> "GemmaCache":
        """Advance seq_positions in both sub-caches by true_len.

        Args:
            true_len: Actual (non-padded) sequence lengths [bsz].
        """
        return GemmaCache(
            sliding_cache=self.sliding_cache.update_positions(true_len),
            global_cache=self.global_cache.update_positions(true_len),
        )

    def slice(self, idx: int) -> "GemmaCache":
        """Extract a single-sequence cache at batch index idx.

        Used by the engine to pull an individual prefill result out of a
        batched prefill before inserting it into a decode slot.

        Args:
            idx: Batch index to extract.
        """
        return GemmaCache(
            sliding_cache=KVCache(
                k=self.sliding_cache.k[:, idx : idx + 1, :, :, :],
                v=self.sliding_cache.v[:, idx : idx + 1, :, :, :],
                seq_positions=self.sliding_cache.seq_positions[idx : idx + 1],
            ),
            global_cache=KVCache(
                k=self.global_cache.k[:, idx : idx + 1, :, :, :],
                v=self.global_cache.v[:, idx : idx + 1, :, :, :],
                seq_positions=self.global_cache.seq_positions[idx : idx + 1],
            ),
        )

    def place_on_mesh(self, mesh) -> "GemmaCache":
        """Re-shard GemmaCache onto mesh, preserving existing data.

        Use this after mutating the cache (e.g. inserting a prefill result
        into a decode slot) to restore the correct sharding before the next
        JIT-compiled decode step.

        Args:
            mesh: JAX device mesh.
        """
        from utils.mesh_helpers import MeshHelper

        return GemmaCache(
            sliding_cache=MeshHelper.place_kv_cache(self.sliding_cache, mesh),
            global_cache=MeshHelper.place_kv_cache(self.global_cache, mesh),
        )

    def batch_insert(self, entries, slot_idxs, lens, next_tokens, curr_tokens):
        """Scatter prefill results into decode slots.

        Both sliding and global sub-caches are updated independently since
        they have different layer counts and head dimensions.

        Args:
            entries: List of single-sequence GemmaCaches from prefill (via slice()).
            slot_idxs: Decode slot indices to insert into.
            lens: Prefill sequence lengths for each entry.
            next_tokens: First generated token for each entry.
            curr_tokens: Current token array for the decode batch [batch, 1].

        Returns:
            Tuple of (updated GemmaCache, updated curr_tokens).
        """
        idx = jnp.array(slot_idxs)
        new_sliding_positions = self.sliding_cache.seq_positions.at[idx].set(jnp.array(lens))
        new_global_positions = self.global_cache.seq_positions.at[idx].set(jnp.array(lens))
        new_tokens = curr_tokens.at[idx, 0].set(jnp.array(next_tokens))
        new_sk = self.sliding_cache.k
        new_sv = self.sliding_cache.v
        new_gk = self.global_cache.k
        new_gv = self.global_cache.v
        for entry, slot_idx in zip(entries, slot_idxs):
            s_seqlen = entry.sliding_cache.k.shape[3]
            g_seqlen = entry.global_cache.k.shape[3]
            new_sk = new_sk.at[:, slot_idx, :, :s_seqlen, :].set(entry.sliding_cache.k[:, 0, :, :, :])
            new_sv = new_sv.at[:, slot_idx, :, :s_seqlen, :].set(entry.sliding_cache.v[:, 0, :, :, :])
            new_gk = new_gk.at[:, slot_idx, :, :g_seqlen, :].set(entry.global_cache.k[:, 0, :, :, :])
            new_gv = new_gv.at[:, slot_idx, :, :g_seqlen, :].set(entry.global_cache.v[:, 0, :, :, :])
        new_sliding = KVCache(k=new_sk, v=new_sv, seq_positions=new_sliding_positions)
        new_global = KVCache(k=new_gk, v=new_gv, seq_positions=new_global_positions)
        return GemmaCache(sliding_cache=new_sliding, global_cache=new_global), new_tokens
