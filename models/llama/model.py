from functools import partial
import logging
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax import jit

from utils.ops import (
    rms_norm,
    grouped_query_attention,
    feed_forward,
    precompute_freqs_cis,
    AttentionParams,
    FeedForwardParams,
)
from utils.kvcache import KVCache
from .config import ModelConfig

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    args: ModelConfig

    def setup(self):
        self.attention = AttentionParams(
            wq=self.param(
                "wq",
                nn.initializers.normal(stddev=0.02),
                (self.args.dim, self.args.n_heads, self.args.head_dim),
                dtype=self.args.dtype,
            ),
            wk=self.param(
                "wk",
                nn.initializers.normal(stddev=0.02),
                (self.args.dim, self.args.n_kv_heads, self.args.head_dim),
                dtype=self.args.dtype,
            ),
            wv=self.param(
                "wv",
                nn.initializers.normal(stddev=0.02),
                (self.args.dim, self.args.n_kv_heads, self.args.head_dim),
                dtype=self.args.dtype,
            ),
            wo=self.param(
                "wo",
                nn.initializers.normal(stddev=0.02),
                (self.args.n_heads * self.args.head_dim, self.args.dim),
                dtype=self.args.dtype,
            ),
        )
        self.feed_forward = FeedForwardParams(
            w_gate=self.param(
                "w_gate",
                nn.initializers.normal(stddev=0.02),
                (self.args.dim, self.args.ffn_hidden_dim),
                dtype=self.args.dtype,
            ),
            w_up=self.param(
                "w_up",
                nn.initializers.normal(stddev=0.02),
                (self.args.dim, self.args.ffn_hidden_dim),
                dtype=self.args.dtype,
            ),
            w_down=self.param(
                "w_down",
                nn.initializers.normal(stddev=0.02),
                (self.args.ffn_hidden_dim, self.args.dim),
                dtype=self.args.dtype,
            ),
        )
        self.attention_norm_weight = self.param(
            "attention_norm_weight",
            nn.initializers.ones,
            (self.args.dim,),
            dtype=self.args.dtype,
        )
        self.ffn_norm_weight = self.param(
            "ffn_norm_weight",
            nn.initializers.ones,
            (self.args.dim,),
            dtype=self.args.dtype,
        )

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        kv_cache: KVCache,
        layer_idx: int,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        h_norm = rms_norm(x, self.attention_norm_weight, eps=self.args.rms_norm_eps)
        attn_output, updated_cache = grouped_query_attention(
            h_norm,
            freqs_cis=freqs_cis,
            params=self.attention,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            mask=mask,
        )
        x = x + attn_output

        h_ffn_norm = rms_norm(x, self.ffn_norm_weight, eps=self.args.rms_norm_eps)
        ffn_output = feed_forward(
            h_ffn_norm,
            params=self.feed_forward,
            activation_fn=self.args.activation_fn,
        )
        x = x + ffn_output
        return x, updated_cache


class LLaMa(nn.Module):
    args: ModelConfig

    def setup(self):
        self.tok_embeddings = nn.Embed(
            num_embeddings=self.args.vocab_size,
            features=self.args.dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.args.dtype,
        )

        self.layers = [
            TransformerBlock(self.args, name=f"layer_{i}")
            for i in range(self.args.n_layers)
        ]

        self.norm_weight = self.param(
            "norm_weight", nn.initializers.ones, (self.args.dim,), dtype=self.args.dtype
        )

        self.output = self.param(
            "output",
            nn.initializers.normal(stddev=0.02),
            (self.args.dim, self.args.vocab_size),
            self.args.dtype,
        )

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        self.freqs_cis = precompute_freqs_cis(
            self.args.head_dim,
            self.args.max_seqlen * 2,
            self.args.rope_theta,
            dtype=jnp.float64,
            use_scaled=self.args.use_scaled_rope,
        ).astype(self.args.dtype)
        jax.config.update("jax_enable_x64", False)

    def __call__(
        self,
        tokens: jax.Array,
        true_lengths: jax.Array,
        kv_cache: KVCache,
        mask: jax.Array,  # [bsz, seqlen, max_seqlen] - attention mask
    ):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        for layer_idx, layer in enumerate(self.layers):
            h, kv_cache = layer(h, self.freqs_cis, kv_cache, layer_idx, mask)

        h = rms_norm(h, self.norm_weight, eps=self.args.rms_norm_eps)

        logits = jnp.einsum("bsd,dv->bsv", h, self.output)
        kv_cache = kv_cache.update_positions(true_lengths)

        return logits, kv_cache
