"""
Qwen3.5 MoE model implementation in Flax.

Supports the hybrid attention architecture:
- Full attention layers (GQA with gated output + partial RoPE)
- Linear attention layers (Gated DeltaNet recurrence)
- All layers use Mixture-of-Experts FFN with a shared expert
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn

from utils.ops import (
    precompute_freqs_cis,
    build_attn_mask,
    FeedForwardParams,
)
from utils.kvcache import KVCache
from utils.hybrid_cache import HybridCache, DeltaNetState
from utils.qwen_ops import (
    qwen_rms_norm,
    apply_partial_rotary_emb_batch,
    gated_deltanet_step,
    gated_deltanet_prefill,
    moe_forward,
    DeltaNetParams,
    MoEParams,
)
from .config import QwenConfig


_init = nn.initializers.normal(stddev=0.02)
_zeros = nn.initializers.zeros


class FullAttentionBlock(nn.Module):
    """Transformer block with Gated GQA + MoE FFN."""

    args: QwenConfig

    def setup(self):
        cfg = self.args
        dt = cfg.dtype

        # --- Gated attention weights ---
        # Q projects to n_heads * head_dim * 2 (query + gate)
        self.wq = self.param(
            "wq", _init, (cfg.dim, cfg.n_heads, cfg.head_dim * 2), dt
        )
        self.wk = self.param(
            "wk", _init, (cfg.dim, cfg.n_kv_heads, cfg.head_dim), dt
        )
        self.wv = self.param(
            "wv", _init, (cfg.dim, cfg.n_kv_heads, cfg.head_dim), dt
        )
        self.wo = self.param(
            "wo", _init, (cfg.n_heads * cfg.head_dim, cfg.dim), dt
        )

        # Per-head QK norms
        self.q_norm = self.param("q_norm", _zeros, (cfg.head_dim,), dt)
        self.k_norm = self.param("k_norm", _zeros, (cfg.head_dim,), dt)

        # --- MoE weights ---
        self.moe = MoEParams(
            router_weight=self.param(
                "router_weight", _init, (cfg.num_experts, cfg.dim), dt
            ),
            expert_gate_up=self.param(
                "expert_gate_up",
                _init,
                (cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.dim),
                dt,
            ),
            expert_down=self.param(
                "expert_down",
                _init,
                (cfg.num_experts, cfg.dim, cfg.moe_intermediate_size),
                dt,
            ),
            shared_gate=self.param(
                "shared_gate",
                _init,
                (cfg.dim, cfg.shared_expert_intermediate_size),
                dt,
            ),
            shared_up=self.param(
                "shared_up",
                _init,
                (cfg.dim, cfg.shared_expert_intermediate_size),
                dt,
            ),
            shared_down=self.param(
                "shared_down",
                _init,
                (cfg.shared_expert_intermediate_size, cfg.dim),
                dt,
            ),
            shared_expert_gate=self.param(
                "shared_expert_gate", _init, (cfg.dim, 1), dt
            ),
        )

        # --- Layer norms ---
        self.attention_norm_weight = self.param(
            "attention_norm_weight", _zeros, (cfg.dim,), dt
        )
        self.ffn_norm_weight = self.param(
            "ffn_norm_weight", _zeros, (cfg.dim,), dt
        )

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        kv_cache: KVCache,
        kv_layer_idx: int,
        mask: jax.Array,
    ) -> tuple:
        """
        Args:
            x: [bsz, seqlen, dim]
            freqs_cis: Precomputed RoPE frequencies [max_seqlen, rotary_dim//2, 2]
            kv_cache: KVCache for full attention layers.
            kv_layer_idx: Index into KV cache (0..n_full_attn-1).
            mask: Boolean attention mask [bsz, seqlen, max_cache_seqlen].

        Returns:
            (output [bsz, seqlen, dim], updated KVCache)
        """
        cfg = self.args
        bsz, seqlen, dim = x.shape

        # Pre-norm
        h = qwen_rms_norm(x, self.attention_norm_weight, eps=cfg.rms_norm_eps)

        # Q/K/V projections
        xq = jnp.einsum("bsd,dhc->bshc", h, self.wq)  # [bsz, seqlen, n_heads, head_dim*2]
        xk = jnp.einsum("bsd,dkc->bskc", h, self.wk)  # [bsz, seqlen, n_kv_heads, head_dim]
        xv = jnp.einsum("bsd,dvc->bsvc", h, self.wv)  # [bsz, seqlen, n_kv_heads, head_dim]

        # Split Q into query and gate
        xq_query = xq[..., : cfg.head_dim]  # [bsz, seqlen, n_heads, head_dim]
        xq_gate = xq[..., cfg.head_dim :]  # [bsz, seqlen, n_heads, head_dim]

        # Per-head QK norms (Qwen uses (1+weight) variant)
        xq_query = qwen_rms_norm(xq_query, self.q_norm, eps=cfg.rms_norm_eps)
        xk = qwen_rms_norm(xk, self.k_norm, eps=cfg.rms_norm_eps)

        # Partial RoPE: compute per-batch position frequencies
        start_positions = kv_cache.seq_positions  # [bsz]
        position_offsets = jnp.arange(seqlen)[None, :]  # [1, seqlen]
        absolute_positions = start_positions[:, None] + position_offsets  # [bsz, seqlen]

        rotary_dim = cfg.rotary_dim
        batch_freqs_cis = freqs_cis[absolute_positions]  # [bsz, seqlen, rotary_dim//2, 2]

        xq_query = apply_partial_rotary_emb_batch(xq_query, batch_freqs_cis, rotary_dim)
        xk = apply_partial_rotary_emb_batch(xk, batch_freqs_cis, rotary_dim)

        # Update KV cache
        xk_t = xk.transpose(0, 2, 1, 3)  # [bsz, n_kv_heads, seqlen, head_dim]
        xv_t = xv.transpose(0, 2, 1, 3)
        updated_cache = kv_cache.update(xk_t, xv_t, kv_layer_idx)
        keys, values = updated_cache.get_layer(kv_layer_idx)

        # GQA: repeat KV heads
        n_rep = cfg.n_heads // cfg.n_kv_heads
        if n_rep != 1:
            keys = jnp.repeat(keys, n_rep, axis=1)
            values = jnp.repeat(values, n_rep, axis=1)

        # Attention scores
        xq_t = xq_query.transpose(0, 2, 1, 3)  # [bsz, n_heads, seqlen, head_dim]
        scores = jnp.einsum("bhqd,bhkd->bhqk", xq_t, keys) / jnp.sqrt(
            float(cfg.head_dim)
        )

        attn_mask = mask[:, None, :, :]  # [bsz, 1, seqlen, max_cache_seqlen]
        scores = jnn.softmax(scores.astype(jnp.float32), where=attn_mask, axis=-1).astype(
            x.dtype
        )

        attn_output = jnp.einsum(
            "bhqk,bhkd->bhqd", scores, values
        )  # [bsz, n_heads, seqlen, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)  # [bsz, seqlen, n_heads, head_dim]

        # Gated attention output
        if cfg.attn_output_gate:
            attn_output = attn_output * jnn.sigmoid(xq_gate)

        # Reshape and project
        attn_output = attn_output.reshape(bsz, seqlen, -1)
        attn_output = jnp.einsum("bsd,do->bso", attn_output, self.wo)

        # Residual
        x = x + attn_output

        # MoE FFN
        h_ffn = qwen_rms_norm(x, self.ffn_norm_weight, eps=cfg.rms_norm_eps)
        moe_out = moe_forward(
            h_ffn, self.moe, cfg.num_experts_per_tok, cfg.activation_fn
        )
        x = x + moe_out

        return x, updated_cache


class LinearAttentionBlock(nn.Module):
    """Transformer block with Gated DeltaNet linear attention + MoE FFN."""

    args: QwenConfig

    def setup(self):
        cfg = self.args
        dt = cfg.dtype
        key_dim = cfg.linear_key_dim
        value_dim = cfg.linear_value_dim
        conv_dim = cfg.linear_conv_dim

        # --- DeltaNet weights ---
        self.deltanet = DeltaNetParams(
            in_proj_qkv=self.param(
                "in_proj_qkv", _init, (cfg.dim, key_dim * 2 + value_dim), dt
            ),
            in_proj_z=self.param("in_proj_z", _init, (cfg.dim, value_dim), dt),
            in_proj_a=self.param(
                "in_proj_a", _init, (cfg.dim, cfg.linear_num_value_heads), dt
            ),
            in_proj_b=self.param(
                "in_proj_b", _init, (cfg.dim, cfg.linear_num_value_heads), dt
            ),
            conv1d_weight=self.param(
                "conv1d_weight", _init, (conv_dim, cfg.linear_conv_kernel_dim), dt
            ),
            dt_bias=self.param(
                "dt_bias", nn.initializers.ones, (cfg.linear_num_value_heads,), dt
            ),
            A_log=self.param(
                "A_log", nn.initializers.zeros, (cfg.linear_num_value_heads,), dt
            ),
            norm_weight=self.param("norm_weight", _zeros, (value_dim,), dt),
            out_proj=self.param("out_proj", _init, (value_dim, cfg.dim), dt),
        )

        # --- MoE weights (same structure as FullAttentionBlock) ---
        self.moe = MoEParams(
            router_weight=self.param(
                "router_weight", _init, (cfg.num_experts, cfg.dim), dt
            ),
            expert_gate_up=self.param(
                "expert_gate_up",
                _init,
                (cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.dim),
                dt,
            ),
            expert_down=self.param(
                "expert_down",
                _init,
                (cfg.num_experts, cfg.dim, cfg.moe_intermediate_size),
                dt,
            ),
            shared_gate=self.param(
                "shared_gate",
                _init,
                (cfg.dim, cfg.shared_expert_intermediate_size),
                dt,
            ),
            shared_up=self.param(
                "shared_up",
                _init,
                (cfg.dim, cfg.shared_expert_intermediate_size),
                dt,
            ),
            shared_down=self.param(
                "shared_down",
                _init,
                (cfg.shared_expert_intermediate_size, cfg.dim),
                dt,
            ),
            shared_expert_gate=self.param(
                "shared_expert_gate", _init, (cfg.dim, 1), dt
            ),
        )

        # --- Layer norms ---
        self.attention_norm_weight = self.param(
            "attention_norm_weight", _zeros, (cfg.dim,), dt
        )
        self.ffn_norm_weight = self.param(
            "ffn_norm_weight", _zeros, (cfg.dim,), dt
        )

    def __call__(
        self,
        x: jax.Array,
        deltanet_state: DeltaNetState,
        linear_layer_idx: int,
    ) -> tuple:
        """
        Args:
            x: [bsz, seqlen, dim]
            deltanet_state: DeltaNetState for all linear layers.
            linear_layer_idx: Index into DeltaNet state (0..n_linear-1).

        Returns:
            (output [bsz, seqlen, dim], updated DeltaNetState)
        """
        cfg = self.args
        bsz, seqlen, dim = x.shape

        # Pre-norm
        h = qwen_rms_norm(x, self.attention_norm_weight, eps=cfg.rms_norm_eps)

        # Get this layer's state
        state, conv_state = deltanet_state.get_layer(linear_layer_idx)

        # DeltaNet forward
        if seqlen == 1:
            # Decode: single step
            attn_output, new_state, new_conv_state = gated_deltanet_step(
                h.squeeze(1),  # [bsz, dim]
                state,
                conv_state,
                self.deltanet,
                cfg.linear_num_key_heads,
                cfg.linear_key_head_dim,
                cfg.linear_num_value_heads,
                cfg.linear_value_head_dim,
            )
            attn_output = attn_output[:, None, :]  # [bsz, 1, dim]
        else:
            # Prefill: scan over sequence
            attn_output, new_state, new_conv_state = gated_deltanet_prefill(
                h,
                state,
                conv_state,
                self.deltanet,
                cfg.linear_num_key_heads,
                cfg.linear_key_head_dim,
                cfg.linear_num_value_heads,
                cfg.linear_value_head_dim,
            )

        updated_deltanet = deltanet_state.update(
            new_state, new_conv_state, linear_layer_idx
        )

        # Residual
        x = x + attn_output

        # MoE FFN
        h_ffn = qwen_rms_norm(x, self.ffn_norm_weight, eps=cfg.rms_norm_eps)
        moe_out = moe_forward(
            h_ffn, self.moe, cfg.num_experts_per_tok, cfg.activation_fn
        )
        x = x + moe_out

        return x, updated_deltanet


class Qwen(nn.Module):
    """Qwen3.5 MoE model with hybrid attention."""

    args: QwenConfig

    def setup(self):
        cfg = self.args

        self.tok_embeddings = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=cfg.dtype,
        )

        # Create layers based on layer_types
        layers = []
        for i in range(cfg.n_layers):
            if cfg.layer_types[i] == "full_attention":
                layers.append(FullAttentionBlock(cfg, name=f"layer_{i}"))
            else:
                layers.append(LinearAttentionBlock(cfg, name=f"layer_{i}"))
        self.layers = layers

        self.norm_weight = self.param(
            "norm_weight", _zeros, (cfg.dim,), cfg.dtype
        )

        self.output = self.param(
            "output",
            nn.initializers.normal(stddev=0.02),
            (cfg.dim, cfg.vocab_size),
            cfg.dtype,
        )

        # Precompute RoPE frequencies for partial rotation
        rotary_dim = cfg.rotary_dim  # head_dim * partial_rotary_factor = 64
        self.freqs_cis = precompute_freqs_cis(
            rotary_dim,
            cfg.max_seqlen * 2,
            cfg.rope_theta,
            dtype=jnp.float32,
        ).astype(cfg.dtype)

    def __call__(
        self,
        tokens: jax.Array,
        true_lengths: jax.Array,
        hybrid_cache: HybridCache,
    ):
        """
        Args:
            tokens: [bsz, seqlen] token IDs.
            true_lengths: [bsz] actual non-padded lengths.
            hybrid_cache: HybridCache containing KV cache and DeltaNet state.

        Returns:
            (logits [bsz, seqlen, vocab_size], updated HybridCache)
        """
        cfg = self.args
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        kv_cache = hybrid_cache.kv_cache
        deltanet_state = hybrid_cache.deltanet_state

        mask = build_attn_mask(seqlen, kv_cache, true_lengths)

        full_attn_idx = 0
        linear_attn_idx = 0

        for i, layer in enumerate(self.layers):
            if cfg.layer_types[i] == "full_attention":
                h, kv_cache = layer(
                    h, self.freqs_cis, kv_cache, full_attn_idx, mask
                )
                full_attn_idx += 1
            else:
                h, deltanet_state = layer(h, deltanet_state, linear_attn_idx)
                linear_attn_idx += 1

        h = qwen_rms_norm(h, self.norm_weight, eps=cfg.rms_norm_eps)
        logits = jnp.einsum("bsd,dv->bsv", h, self.output)

        hybrid_cache = HybridCache(
            kv_cache=kv_cache.update_positions(true_lengths),
            deltanet_state=deltanet_state,
        )

        return logits, hybrid_cache
