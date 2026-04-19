"""
TurboQuant-compressed KV caches.

Drop-in replacements for `utils.kvcache.KVCache` that store K/V vectors as
compact quantized representations.  Compatible with the existing attention
interface (`grouped_query_attention`, `build_attn_mask`) because they expose
the same `.k`, `.v`, `.seq_positions` attributes with the same leading shape
`[n_layers, bsz, n_kv_heads, max_seqlen, ...]`.

Two variants:

* `TurboQuantMSEKVCache`   — MSE-optimal (Algorithm 1).  Simplest, ~2x memory
  compression (uint8 indices + bf16 scales).  Best for bulk memory savings.
* `TurboQuantProdKVCache`  — inner-product-optimal (Algorithm 2).  Adds int8
  QJL sign bits + residual norms.  Unbiased attention scores (important for
  long-context accuracy), but needs bit-packing to actually shrink vs. bf16.
"""

import jax
import jax.numpy as jnp
from flax import struct

from .turboquant import (
    TurboQuantMSEParams,
    TurboQuantProdParams,
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
)


# ---------------------------------------------------------------------------
# MSE-compressed KV cache
# ---------------------------------------------------------------------------

@struct.dataclass
class TurboQuantMSEKVCache:
    """KV cache storing K/V as MSE-quantized uint8 indices + bfloat16 scales.

    The `k` and `v` fields intentionally share names with `KVCache` so that
    `build_attn_mask` (which reads `kv_cache.k.shape[3]` for max_seqlen) and
    `grouped_query_attention` (which calls `update` / `get_layer`) work with
    zero modifications.

    Attributes:
        k, v:                 uint8 centroid indices,
                              shape [n_layers, bsz, n_kv_heads, max_seqlen, head_dim].
        k_scale, v_scale:     bfloat16 per-vector L2 norms,
                              shape [n_layers, bsz, n_kv_heads, max_seqlen].
        seq_positions:        int32 per-sequence cache position, shape [bsz].
        tq_params:            Shared TurboQuantMSEParams (rotation + codebook).
    """
    k: jax.Array          # uint8  [n_layers, bsz, n_kv_heads, max_seqlen, head_dim]
    v: jax.Array          # uint8  same
    k_scale: jax.Array    # bf16   [n_layers, bsz, n_kv_heads, max_seqlen]
    v_scale: jax.Array    # bf16   same
    seq_positions: jax.Array   # int32 [bsz]
    tq_params: TurboQuantMSEParams

    @classmethod
    def new(
        cls,
        config,
        bsz: int,
        max_seqlen: int,
        tq_params: TurboQuantMSEParams,
    ) -> "TurboQuantMSEKVCache":
        """Allocate an empty compressed KV cache.

        Args:
            config:     ModelConfig with n_layers, n_kv_heads, head_dim.
            bsz:        Batch size.
            max_seqlen: Maximum sequence length.
            tq_params:  Pre-computed TurboQuantMSEParams (from `create_mse_params`).
        """
        idx_shape = (config.n_layers, bsz, config.n_kv_heads, max_seqlen, config.head_dim)
        scale_shape = (config.n_layers, bsz, config.n_kv_heads, max_seqlen)
        return cls(
            k=jnp.zeros(idx_shape, dtype=jnp.uint8),
            v=jnp.zeros(idx_shape, dtype=jnp.uint8),
            k_scale=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            v_scale=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
            tq_params=tq_params,
        )

    def update(
        self,
        xk: jax.Array,   # [bsz, n_kv_heads, seqlen, head_dim]
        xv: jax.Array,   # [bsz, n_kv_heads, seqlen, head_dim]
        layer_idx: int,
    ) -> "TurboQuantMSEKVCache":
        """Quantize new K/V and scatter into each sequence's cache slot.

        Mirrors `KVCache.update` but quantizes before storing.

        Args:
            xk, xv:    Incoming K/V, shape [bsz, n_kv_heads, seqlen, head_dim].
            layer_idx: Transformer layer index.
        """
        k_idx, k_sc = quantize_mse(self.tq_params, xk)   # [bsz,kv,s,d], [bsz,kv,s]
        v_idx, v_sc = quantize_mse(self.tq_params, xv)

        layer_k       = self.k[layer_idx]         # [bsz, kv, max_seqlen, head_dim]
        layer_v       = self.v[layer_idx]
        layer_k_scale = self.k_scale[layer_idx]   # [bsz, kv, max_seqlen]
        layer_v_scale = self.v_scale[layer_idx]

        def _insert_idx(slot, new_idx, pos):
            # slot: [kv, max_seqlen, head_dim], new_idx: [kv, seqlen, head_dim]
            return jax.lax.dynamic_update_slice(slot, new_idx, (0, pos, 0))

        def _insert_scale(slot, new_sc, pos):
            # slot: [kv, max_seqlen], new_sc: [kv, seqlen]
            return jax.lax.dynamic_update_slice(slot, new_sc, (0, pos))

        # vmap across batch dim — per-sequence positions
        new_lk = jax.vmap(_insert_idx)(layer_k, k_idx, self.seq_positions)
        new_lv = jax.vmap(_insert_idx)(layer_v, v_idx, self.seq_positions)
        new_lks = jax.vmap(_insert_scale)(layer_k_scale, k_sc, self.seq_positions)
        new_lvs = jax.vmap(_insert_scale)(layer_v_scale, v_sc, self.seq_positions)

        return TurboQuantMSEKVCache(
            k=self.k.at[layer_idx].set(new_lk),
            v=self.v.at[layer_idx].set(new_lv),
            k_scale=self.k_scale.at[layer_idx].set(new_lks),
            v_scale=self.v_scale.at[layer_idx].set(new_lvs),
            seq_positions=self.seq_positions,
            tq_params=self.tq_params,
        )

    def get_layer(self, layer_idx: int) -> tuple[jax.Array, jax.Array]:
        """Dequantize and return full-precision K/V for a layer.

        Returns:
            (keys, values), each float32, shape [bsz, n_kv_heads, max_seqlen, head_dim].
        """
        keys   = dequantize_mse(self.tq_params, self.k[layer_idx], self.k_scale[layer_idx])
        values = dequantize_mse(self.tq_params, self.v[layer_idx], self.v_scale[layer_idx])
        return keys, values

    def update_positions(self, true_len: jax.Array) -> "TurboQuantMSEKVCache":
        """Advance per-sequence cache positions by `true_len` tokens."""
        return TurboQuantMSEKVCache(
            k=self.k, v=self.v,
            k_scale=self.k_scale, v_scale=self.v_scale,
            seq_positions=self.seq_positions + true_len,
            tq_params=self.tq_params,
        )

    def slice(self, idx: int) -> "TurboQuantMSEKVCache":
        """Extract a single-sequence view at batch index `idx`."""
        return TurboQuantMSEKVCache(
            k=self.k[:, idx:idx + 1, :, :, :],
            v=self.v[:, idx:idx + 1, :, :, :],
            k_scale=self.k_scale[:, idx:idx + 1, :, :],
            v_scale=self.v_scale[:, idx:idx + 1, :, :],
            seq_positions=self.seq_positions[idx:idx + 1],
            tq_params=self.tq_params,
        )


# ---------------------------------------------------------------------------
# Inner-product-compressed KV cache
# ---------------------------------------------------------------------------

@struct.dataclass
class TurboQuantProdKVCache:
    """KV cache using inner-product-optimal TurboQuant (Algorithm 2).

    Stores a (b-1)-bit MSE index + 1-bit QJL sign + scale + residual norm
    per vector.  Provides unbiased attention score estimation: the dequantized
    K/V vectors satisfy E[<Q, K̃>] = <Q, K>.

    Without bit-packing, this layout is the same size as bf16 — the memory
    gain requires packing uint8 indices to (b-1) bits and int8 QJL bits
    to 1 bit (future work).  Until then, use this cache for quality
    (unbiased attention) rather than memory.
    """
    k: jax.Array                 # uint8  [n_layers, bsz, kv, max_seqlen, head_dim]
    v: jax.Array                 # uint8
    k_qjl: jax.Array             # int8   ±1
    v_qjl: jax.Array             # int8   ±1
    k_scale: jax.Array           # bf16   [n_layers, bsz, kv, max_seqlen]
    v_scale: jax.Array           # bf16
    k_residual_norm: jax.Array   # bf16
    v_residual_norm: jax.Array   # bf16
    seq_positions: jax.Array     # int32  [bsz]
    tq_params: TurboQuantProdParams

    @classmethod
    def new(
        cls,
        config,
        bsz: int,
        max_seqlen: int,
        tq_params: TurboQuantProdParams,
    ) -> "TurboQuantProdKVCache":
        """Allocate an empty inner-product-quantized KV cache."""
        idx_shape = (config.n_layers, bsz, config.n_kv_heads, max_seqlen, config.head_dim)
        scale_shape = (config.n_layers, bsz, config.n_kv_heads, max_seqlen)
        return cls(
            k=jnp.zeros(idx_shape, dtype=jnp.uint8),
            v=jnp.zeros(idx_shape, dtype=jnp.uint8),
            k_qjl=jnp.zeros(idx_shape, dtype=jnp.int8),
            v_qjl=jnp.zeros(idx_shape, dtype=jnp.int8),
            k_scale=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            v_scale=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            k_residual_norm=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            v_residual_norm=jnp.zeros(scale_shape, dtype=jnp.bfloat16),
            seq_positions=jnp.zeros(bsz, dtype=jnp.int32),
            tq_params=tq_params,
        )

    def update(
        self,
        xk: jax.Array,
        xv: jax.Array,
        layer_idx: int,
    ) -> "TurboQuantProdKVCache":
        """Quantize (MSE + QJL residual) and scatter into sequence slots."""
        k_idx, k_qjl, k_sc, k_rn = quantize_prod(self.tq_params, xk)
        v_idx, v_qjl, v_sc, v_rn = quantize_prod(self.tq_params, xv)

        def _insert_idx(slot, new_idx, pos):
            return jax.lax.dynamic_update_slice(slot, new_idx, (0, pos, 0))

        def _insert_scale(slot, new_sc, pos):
            return jax.lax.dynamic_update_slice(slot, new_sc, (0, pos))

        def _scatter_idx(full, new_layer):
            return full.at[layer_idx].set(new_layer)

        # Keys
        layer_k   = self.k[layer_idx]
        layer_kq  = self.k_qjl[layer_idx]
        layer_ks  = self.k_scale[layer_idx]
        layer_krn = self.k_residual_norm[layer_idx]
        new_k   = jax.vmap(_insert_idx)(layer_k, k_idx, self.seq_positions)
        new_kq  = jax.vmap(_insert_idx)(layer_kq, k_qjl, self.seq_positions)
        new_ks  = jax.vmap(_insert_scale)(layer_ks, k_sc, self.seq_positions)
        new_krn = jax.vmap(_insert_scale)(layer_krn, k_rn, self.seq_positions)

        # Values
        layer_v   = self.v[layer_idx]
        layer_vq  = self.v_qjl[layer_idx]
        layer_vs  = self.v_scale[layer_idx]
        layer_vrn = self.v_residual_norm[layer_idx]
        new_v   = jax.vmap(_insert_idx)(layer_v, v_idx, self.seq_positions)
        new_vq  = jax.vmap(_insert_idx)(layer_vq, v_qjl, self.seq_positions)
        new_vs  = jax.vmap(_insert_scale)(layer_vs, v_sc, self.seq_positions)
        new_vrn = jax.vmap(_insert_scale)(layer_vrn, v_rn, self.seq_positions)

        return TurboQuantProdKVCache(
            k=self.k.at[layer_idx].set(new_k),
            v=self.v.at[layer_idx].set(new_v),
            k_qjl=self.k_qjl.at[layer_idx].set(new_kq),
            v_qjl=self.v_qjl.at[layer_idx].set(new_vq),
            k_scale=self.k_scale.at[layer_idx].set(new_ks),
            v_scale=self.v_scale.at[layer_idx].set(new_vs),
            k_residual_norm=self.k_residual_norm.at[layer_idx].set(new_krn),
            v_residual_norm=self.v_residual_norm.at[layer_idx].set(new_vrn),
            seq_positions=self.seq_positions,
            tq_params=self.tq_params,
        )

    def get_layer(self, layer_idx: int) -> tuple[jax.Array, jax.Array]:
        """Dequantize via DEQUANT_prod and return full-precision K/V."""
        keys = dequantize_prod(
            self.tq_params,
            self.k[layer_idx],
            self.k_qjl[layer_idx],
            self.k_scale[layer_idx],
            self.k_residual_norm[layer_idx],
        )
        values = dequantize_prod(
            self.tq_params,
            self.v[layer_idx],
            self.v_qjl[layer_idx],
            self.v_scale[layer_idx],
            self.v_residual_norm[layer_idx],
        )
        return keys, values

    def update_positions(self, true_len: jax.Array) -> "TurboQuantProdKVCache":
        """Advance per-sequence cache positions by `true_len` tokens."""
        return TurboQuantProdKVCache(
            k=self.k, v=self.v,
            k_qjl=self.k_qjl, v_qjl=self.v_qjl,
            k_scale=self.k_scale, v_scale=self.v_scale,
            k_residual_norm=self.k_residual_norm,
            v_residual_norm=self.v_residual_norm,
            seq_positions=self.seq_positions + true_len,
            tq_params=self.tq_params,
        )

    def slice(self, idx: int) -> "TurboQuantProdKVCache":
        """Extract a single-sequence view at batch index `idx`."""
        return TurboQuantProdKVCache(
            k=self.k[:, idx:idx + 1, :, :, :],
            v=self.v[:, idx:idx + 1, :, :, :],
            k_qjl=self.k_qjl[:, idx:idx + 1, :, :, :],
            v_qjl=self.v_qjl[:, idx:idx + 1, :, :, :],
            k_scale=self.k_scale[:, idx:idx + 1, :, :],
            v_scale=self.v_scale[:, idx:idx + 1, :, :],
            k_residual_norm=self.k_residual_norm[:, idx:idx + 1, :, :],
            v_residual_norm=self.v_residual_norm[:, idx:idx + 1, :, :],
            seq_positions=self.seq_positions[idx:idx + 1],
            tq_params=self.tq_params,
        )
