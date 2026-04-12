"""
Gemma 4 model implementation in Flax.

Supports the hybrid attention architecture:
- Sliding attention layers (local window, head_dim=256, 16 KV heads)
- Global attention layers (full context, head_dim=512, 4 KV heads, K=V)
- Sandwich norms (pre + post norm around attention and FFN)
- Logit soft-capping
- Tied word embeddings with sqrt(dim) scaling
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn

from utils.ops import (
    precompute_freqs_cis,
    rms_norm,
    apply_rotary_emb_batch,
    build_attn_mask,
    FeedForwardParams,
    feed_forward,
)
from utils.kvcache import KVCache
from utils.gemma_cache import GemmaCache
from utils.gemma_ops import (
    build_sliding_attn_mask,
    rms_norm_no_scale,
    logit_softcap,
    apply_partial_rotary_emb_batch,
)
from .config import GemmaConfig


_init = nn.initializers.normal(stddev=0.02)


# ---------------------------------------------------------------------------
# Sliding Attention Block
# ---------------------------------------------------------------------------


class SlidingAttentionBlock(nn.Module):
    """Transformer block with sliding-window GQA + Gated FFN.

    Uses head_dim=256, 16 KV heads, full RoPE, QK norm + V norm (no scale).
    Sandwich norms: input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm.
    """

    args: GemmaConfig

    def setup(self):
        cfg = self.args
        dt = cfg.dtype
        hd = cfg.head_dim

        # --- Attention projections ---
        self.wq = self.param("wq", _init, (cfg.dim, cfg.n_heads, hd), dt)
        self.wk = self.param("wk", _init, (cfg.dim, cfg.n_kv_heads, hd), dt)
        self.wv = self.param("wv", _init, (cfg.dim, cfg.n_kv_heads, hd), dt)
        self.wo = self.param("wo", _init, (cfg.n_heads * hd, cfg.dim), dt)

        # QK norms (with learnable scale)
        self.q_norm = self.param("q_norm", nn.initializers.ones, (hd,), dt)
        self.k_norm = self.param("k_norm", nn.initializers.ones, (hd,), dt)

        # --- FFN ---
        self.feed_forward_params = FeedForwardParams(
            w_gate=self.param("w_gate", _init, (cfg.dim, cfg.ffn_hidden_dim), dt),
            w_up=self.param("w_up", _init, (cfg.dim, cfg.ffn_hidden_dim), dt),
            w_down=self.param("w_down", _init, (cfg.ffn_hidden_dim, cfg.dim), dt),
        )

        # --- 4 layer norms (sandwich pattern) ---
        self.input_layernorm = self.param(
            "input_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.post_attention_layernorm = self.param(
            "post_attention_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.pre_feedforward_layernorm = self.param(
            "pre_feedforward_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.post_feedforward_layernorm = self.param(
            "post_feedforward_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )

        # Per-layer residual scalar
        self.layer_scalar = self.param(
            "layer_scalar", nn.initializers.ones, (1,), dt
        )

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        kv_cache: KVCache,
        sliding_layer_idx: int,
        mask: jax.Array,
    ) -> tuple:
        cfg = self.args
        bsz, seqlen, dim = x.shape
        hd = cfg.head_dim

        # --- Attention ---
        residual = x
        h = rms_norm(x, self.input_layernorm, eps=cfg.rms_norm_eps)

        xq = jnp.einsum("bsd,dhc->bshc", h, self.wq)
        xk = jnp.einsum("bsd,dkc->bskc", h, self.wk)
        xv = jnp.einsum("bsd,dvc->bsvc", h, self.wv)

        # QK norm (with scale), V norm (no scale)
        xq = rms_norm(xq, self.q_norm, eps=cfg.rms_norm_eps)
        xk = rms_norm(xk, self.k_norm, eps=cfg.rms_norm_eps)
        xv = rms_norm_no_scale(xv, eps=cfg.rms_norm_eps)

        # Full RoPE
        start_positions = kv_cache.seq_positions
        position_offsets = jnp.arange(seqlen)[None, :]
        absolute_positions = start_positions[:, None] + position_offsets
        batch_freqs_cis = freqs_cis[absolute_positions]

        xq = apply_rotary_emb_batch(xq, batch_freqs_cis)
        xk = apply_rotary_emb_batch(xk, batch_freqs_cis)

        # Update KV cache
        xk_t = xk.transpose(0, 2, 1, 3)
        xv_t = xv.transpose(0, 2, 1, 3)
        updated_cache = kv_cache.update(xk_t, xv_t, sliding_layer_idx)
        keys, values = updated_cache.get_layer(sliding_layer_idx)

        # GQA: repeat KV heads
        n_rep = cfg.n_heads // cfg.n_kv_heads
        if n_rep != 1:
            keys = jnp.repeat(keys, n_rep, axis=1)
            values = jnp.repeat(values, n_rep, axis=1)

        # Attention scores
        xq_t = xq.transpose(0, 2, 1, 3)
        # QK norm replaces traditional 1/sqrt(d) scaling
        scores = jnp.einsum("bhqd,bhkd->bhqk", xq_t, keys)

        attn_mask = mask[:, None, :, :]
        scores = jnn.softmax(scores.astype(jnp.float32), where=attn_mask, axis=-1).astype(
            x.dtype
        )

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", scores, values)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, seqlen, -1)
        attn_output = jnp.einsum("bsd,do->bso", attn_output, self.wo)

        # Post-attention norm + residual
        attn_output = rms_norm(attn_output, self.post_attention_layernorm, eps=cfg.rms_norm_eps)
        x = residual + attn_output

        # --- FFN ---
        residual = x
        h_ffn = rms_norm(x, self.pre_feedforward_layernorm, eps=cfg.rms_norm_eps)
        ffn_output = feed_forward(h_ffn, self.feed_forward_params, cfg.activation_fn)
        ffn_output = rms_norm(ffn_output, self.post_feedforward_layernorm, eps=cfg.rms_norm_eps)
        x = residual + ffn_output

        # Per-layer scalar
        x = x * self.layer_scalar

        return x, updated_cache


# ---------------------------------------------------------------------------
# Global Attention Block
# ---------------------------------------------------------------------------


class GlobalAttentionBlock(nn.Module):
    """Transformer block with full-context GQA + Gated FFN.

    Uses global_head_dim=512, 4 KV heads, partial RoPE (25%), K=V (no v_proj).
    Sandwich norms: input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm.
    """

    args: GemmaConfig

    def setup(self):
        cfg = self.args
        dt = cfg.dtype
        ghd = cfg.global_head_dim

        # --- Attention projections (no V proj — K=V) ---
        self.wq = self.param("wq", _init, (cfg.dim, cfg.n_heads, ghd), dt)
        self.wk = self.param("wk", _init, (cfg.dim, cfg.n_global_kv_heads, ghd), dt)
        self.wo = self.param("wo", _init, (cfg.n_heads * ghd, cfg.dim), dt)

        # QK norms (with learnable scale)
        self.q_norm = self.param("q_norm", nn.initializers.ones, (ghd,), dt)
        self.k_norm = self.param("k_norm", nn.initializers.ones, (ghd,), dt)

        # --- FFN ---
        self.feed_forward_params = FeedForwardParams(
            w_gate=self.param("w_gate", _init, (cfg.dim, cfg.ffn_hidden_dim), dt),
            w_up=self.param("w_up", _init, (cfg.dim, cfg.ffn_hidden_dim), dt),
            w_down=self.param("w_down", _init, (cfg.ffn_hidden_dim, cfg.dim), dt),
        )

        # --- 4 layer norms (sandwich pattern) ---
        self.input_layernorm = self.param(
            "input_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.post_attention_layernorm = self.param(
            "post_attention_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.pre_feedforward_layernorm = self.param(
            "pre_feedforward_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )
        self.post_feedforward_layernorm = self.param(
            "post_feedforward_layernorm", nn.initializers.ones, (cfg.dim,), dt
        )

        # Per-layer residual scalar
        self.layer_scalar = self.param(
            "layer_scalar", nn.initializers.ones, (1,), dt
        )

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        kv_cache: KVCache,
        global_layer_idx: int,
        mask: jax.Array,
    ) -> tuple:
        cfg = self.args
        bsz, seqlen, dim = x.shape
        ghd = cfg.global_head_dim
        rotary_dim = cfg.global_rotary_dim

        # --- Attention ---
        residual = x
        h = rms_norm(x, self.input_layernorm, eps=cfg.rms_norm_eps)

        xq = jnp.einsum("bsd,dhc->bshc", h, self.wq)
        xk = jnp.einsum("bsd,dkc->bskc", h, self.wk)

        # QK norm (with learned scale); V norm (no scale) on same raw projection
        xq = rms_norm(xq, self.q_norm, eps=cfg.rms_norm_eps)
        xv = rms_norm_no_scale(xk, eps=cfg.rms_norm_eps)
        xk = rms_norm(xk, self.k_norm, eps=cfg.rms_norm_eps)

        # Partial RoPE (25% of global_head_dim)
        start_positions = kv_cache.seq_positions
        position_offsets = jnp.arange(seqlen)[None, :]
        absolute_positions = start_positions[:, None] + position_offsets
        batch_freqs_cis = freqs_cis[absolute_positions]

        xq = apply_partial_rotary_emb_batch(xq, batch_freqs_cis, rotary_dim)
        xk = apply_partial_rotary_emb_batch(xk, batch_freqs_cis, rotary_dim)

        # Update KV cache
        xk_t = xk.transpose(0, 2, 1, 3)
        xv_t = xv.transpose(0, 2, 1, 3)
        updated_cache = kv_cache.update(xk_t, xv_t, global_layer_idx)
        keys, values = updated_cache.get_layer(global_layer_idx)

        # GQA: repeat KV heads
        n_rep = cfg.n_heads // cfg.n_global_kv_heads
        if n_rep != 1:
            keys = jnp.repeat(keys, n_rep, axis=1)
            values = jnp.repeat(values, n_rep, axis=1)

        # Attention scores
        xq_t = xq.transpose(0, 2, 1, 3)
        # QK norm replaces traditional 1/sqrt(d) scaling
        scores = jnp.einsum("bhqd,bhkd->bhqk", xq_t, keys)

        attn_mask = mask[:, None, :, :]
        scores = jnn.softmax(scores.astype(jnp.float32), where=attn_mask, axis=-1).astype(
            x.dtype
        )

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", scores, values)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, seqlen, -1)
        attn_output = jnp.einsum("bsd,do->bso", attn_output, self.wo)

        # Post-attention norm + residual
        attn_output = rms_norm(attn_output, self.post_attention_layernorm, eps=cfg.rms_norm_eps)
        x = residual + attn_output

        # --- FFN ---
        residual = x
        h_ffn = rms_norm(x, self.pre_feedforward_layernorm, eps=cfg.rms_norm_eps)
        ffn_output = feed_forward(h_ffn, self.feed_forward_params, cfg.activation_fn)
        ffn_output = rms_norm(ffn_output, self.post_feedforward_layernorm, eps=cfg.rms_norm_eps)
        x = residual + ffn_output

        # Per-layer scalar
        x = x * self.layer_scalar

        return x, updated_cache


# ---------------------------------------------------------------------------
# Gemma 4 Model
# ---------------------------------------------------------------------------


class Gemma(nn.Module):
    """Gemma 4 model with hybrid sliding/global attention."""

    args: GemmaConfig

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
            if cfg.layer_types[i] == "sliding_attention":
                layers.append(SlidingAttentionBlock(cfg, name=f"layer_{i}"))
            else:
                layers.append(GlobalAttentionBlock(cfg, name=f"layer_{i}"))
        self.layers = layers

        self.norm_weight = self.param(
            "norm_weight", nn.initializers.ones, (cfg.dim,), cfg.dtype
        )

        # Precompute RoPE frequencies for both attention types
        # Sliding: full rotation with head_dim=256, theta=10000
        self.sliding_freqs_cis = precompute_freqs_cis(
            cfg.head_dim,
            cfg.max_seqlen * 2,
            cfg.sliding_rope_theta,
            dtype=jnp.float32,
        ).astype(cfg.dtype)

        # Global: partial rotation with rotary_dim, theta=1e6
        self.global_freqs_cis = precompute_freqs_cis(
            cfg.global_rotary_dim,
            cfg.max_seqlen * 2,
            cfg.global_rope_theta,
            dtype=jnp.float32,
        ).astype(cfg.dtype)

    def __call__(
        self,
        tokens: jax.Array,
        true_lengths: jax.Array,
        cache: GemmaCache,
    ):
        """
        Args:
            tokens: [bsz, seqlen] token IDs.
            true_lengths: [bsz] actual non-padded lengths.
            cache: GemmaCache containing sliding and global KV caches.

        Returns:
            (logits [bsz, seqlen, vocab_size], updated GemmaCache)
        """
        cfg = self.args

        # Embed with sqrt(dim) scaling
        h = self.tok_embeddings(tokens) * jnp.sqrt(float(cfg.dim))

        sliding_cache = cache.sliding_cache
        global_cache = cache.global_cache

        # Build sliding window mask from the sliding cache positions
        _, seqlen = tokens.shape
        sliding_mask = build_sliding_attn_mask(
            seqlen, sliding_cache, true_lengths, cfg.sliding_window
        )
        # Global mask uses the standard causal mask
        global_mask = build_attn_mask(seqlen, global_cache, true_lengths)

        sliding_idx = 0
        global_idx = 0

        for i, layer in enumerate(self.layers):
            if cfg.layer_types[i] == "sliding_attention":
                h, sliding_cache = layer(
                    h, self.sliding_freqs_cis, sliding_cache, sliding_idx, sliding_mask
                )
                sliding_idx += 1
            else:
                h, global_cache = layer(
                    h, self.global_freqs_cis, global_cache, global_idx, global_mask
                )
                global_idx += 1

        h = rms_norm(h, self.norm_weight, eps=cfg.rms_norm_eps)

        # Tied embeddings: reuse embedding weight as output projection
        embed_weight = self.tok_embeddings.embedding  # [vocab_size, dim]
        logits = jnp.einsum("bsd,vd->bsv", h, embed_weight)

        # Logit soft-capping
        if cfg.final_logit_softcapping is not None:
            logits = logit_softcap(logits, cfg.final_logit_softcapping)

        updated_cache = GemmaCache(
            sliding_cache=sliding_cache.update_positions(true_lengths),
            global_cache=global_cache.update_positions(true_lengths),
        )

        return logits, updated_cache
