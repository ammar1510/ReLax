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
    def new(cls, config, bsz: int, max_cache_seqlen: int = None, dtype=jnp.bfloat16):
        """Create an empty hybrid cache from a QwenConfig.

        Args:
            config: QwenConfig instance.
            bsz: Batch size.
            max_cache_seqlen: Max sequence length for KV cache. Defaults to config.max_seqlen.
            dtype: Data type for cache arrays.
        """
        if max_cache_seqlen is None:
            max_cache_seqlen = config.max_seqlen

        kv = KVCache.new(
            n_layers=config.n_full_attn_layers,
            bsz=bsz,
            max_seqlen=max_cache_seqlen,
            kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtype,
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
