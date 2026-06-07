"""
Hybrid cache for Qwen3.5 models with mixed attention types.

Combines KV cache (for full attention layers) with recurrent state
(for Gated DeltaNet linear attention layers).
"""

import jax
import jax.numpy as jnp
from flax import struct
from .kvcache import KVCache


@struct.dataclass
class DeltaNetState:
    """Recurrent state for all linear attention (Gated DeltaNet) layers.

    Attributes:
        state: Recurrent state matrices.
            Shape: [n_linear_layers, bsz, n_v_heads, k_head_dim, v_head_dim]
        conv_state: Rolling buffer for causal conv1d.
            Shape: [n_linear_layers, bsz, conv_dim, kernel_size - 1]
    """

    state: jax.Array
    conv_state: jax.Array

    @classmethod
    def new(
        cls,
        n_linear_layers: int,
        bsz: int,
        n_v_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        conv_dim: int,
        kernel_size: int,
        dtype=jnp.bfloat16,
    ) -> "DeltaNetState":
        return cls(
            state=jnp.zeros(
                (n_linear_layers, bsz, n_v_heads, k_head_dim, v_head_dim), dtype=dtype
            ),
            conv_state=jnp.zeros(
                (n_linear_layers, bsz, conv_dim, kernel_size - 1), dtype=dtype
            ),
        )

    def get_layer(self, linear_layer_idx: int):
        """Returns (state, conv_state) for a given linear attention layer."""
        return self.state[linear_layer_idx], self.conv_state[linear_layer_idx]

    def update(self, new_state, new_conv_state, linear_layer_idx: int):
        """Returns a new DeltaNetState with one layer's state updated."""
        return DeltaNetState(
            state=self.state.at[linear_layer_idx].set(new_state),
            conv_state=self.conv_state.at[linear_layer_idx].set(new_conv_state),
        )


@struct.dataclass
class HybridCache:
    """Combined cache for hybrid attention models.

    Full attention layers use a standard KV cache (indexed 0..n_full-1).
    Linear attention layers use recurrent DeltaNet state (indexed 0..n_linear-1).
    """

    kv_cache: KVCache
    deltanet_state: DeltaNetState

    @classmethod
    def new(cls, config, bsz: int, max_cache_seqlen: int = None, dtype=jnp.bfloat16, mesh=None):
        """Create an empty hybrid cache from a QwenConfig.

        Args:
            config: QwenConfig instance.
            bsz: Batch size.
            max_cache_seqlen: Max sequence length for KV cache. Defaults to config.max_seqlen.
            dtype: Data type for cache arrays.
            mesh: If provided, allocate shards directly on each device without
                  an intermediate full-sized allocation.
        """
        if max_cache_seqlen is None:
            max_cache_seqlen = config.max_seqlen

        k_shape = (config.n_full_attn_layers, bsz, config.n_kv_heads, max_cache_seqlen, config.head_dim)
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
            kv = KVCache(
                k=MeshHelper.create_sharded_zeros(k_shape, dtype, mesh, k_spec),
                v=MeshHelper.create_sharded_zeros(k_shape, dtype, mesh, k_spec),
                seq_positions=MeshHelper.create_sharded_zeros((bsz,), jnp.int32, mesh, pos_spec),
            )
        else:
            kv = KVCache(
                k=jnp.zeros(k_shape, dtype=dtype),
                v=jnp.zeros(k_shape, dtype=dtype),
                seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
            )

        delta = DeltaNetState.new(
            n_linear_layers=config.n_linear_attn_layers,
            bsz=bsz,
            n_v_heads=config.linear_num_value_heads,
            k_head_dim=config.linear_key_head_dim,
            v_head_dim=config.linear_value_head_dim,
            conv_dim=config.linear_conv_dim,
            kernel_size=config.linear_conv_kernel_dim,
            dtype=dtype,
        )

        return cls(kv_cache=kv, deltanet_state=delta)

    def update_positions(self, true_len: jax.Array):
        """Advance KV cache positions. DeltaNet state is stateful and needs no position tracking."""
        return HybridCache(
            kv_cache=self.kv_cache.update_positions(true_len),
            deltanet_state=self.deltanet_state,
        )

    def slice(self, idx: int) -> "HybridCache":
        """Extract a single-sequence cache at batch index idx.

        Args:
            idx: Batch index to extract.
        """
        return HybridCache(
            kv_cache=self.kv_cache.slice(idx),
            deltanet_state=DeltaNetState(
                state=self.deltanet_state.state[:, idx : idx + 1, :, :, :],
                conv_state=self.deltanet_state.conv_state[:, idx : idx + 1, :, :],
            ),
        )

    def place_on_mesh(self, mesh) -> "HybridCache":
        """Re-shard HybridCache onto mesh, preserving existing data.

        Use this after mutating the cache to restore correct sharding
        before the next JIT-compiled decode step.

        Args:
            mesh: JAX device mesh.
        """
        from utils.mesh_helpers import MeshHelper
        return MeshHelper.place_hybrid_cache(self, mesh)

    def batch_insert(self, entries, slot_idxs, lens, next_tokens, curr_tokens):
        """Scatter prefill results into decode slots.

        Args:
            entries: List of single-sequence HybridCaches from prefill (via slice()).
            slot_idxs: Decode slot indices to insert into.
            lens: Prefill sequence lengths for each entry.
            next_tokens: First generated token for each entry.
            curr_tokens: Current token array for the decode batch [batch, 1].

        Returns:
            Tuple of (updated HybridCache, updated curr_tokens).
        """
        import jax.numpy as jnp
        idx = jnp.array(slot_idxs)
        new_positions = self.kv_cache.seq_positions.at[idx].set(jnp.array(lens))
        new_tokens = curr_tokens.at[idx, 0].set(jnp.array(next_tokens))
        new_k = self.kv_cache.k
        new_v = self.kv_cache.v
        new_state = self.deltanet_state.state
        new_conv_state = self.deltanet_state.conv_state
        for entry, slot_idx in zip(entries, slot_idxs):
            prefill_seqlen = entry.kv_cache.k.shape[3]
            new_k = new_k.at[:, slot_idx, :, :prefill_seqlen, :].set(entry.kv_cache.k[:, 0, :, :, :])
            new_v = new_v.at[:, slot_idx, :, :prefill_seqlen, :].set(entry.kv_cache.v[:, 0, :, :, :])
            new_state = new_state.at[:, slot_idx, ...].set(entry.deltanet_state.state[:, 0, ...])
            new_conv_state = new_conv_state.at[:, slot_idx, ...].set(entry.deltanet_state.conv_state[:, 0, ...])
        new_kv = KVCache(k=new_k, v=new_v, seq_positions=new_positions)
        new_delta = DeltaNetState(state=new_state, conv_state=new_conv_state)
        return HybridCache(kv_cache=new_kv, deltanet_state=new_delta), new_tokens
