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
from utils.kvcache import KVCache, TempKV
from .config import ModelConfig

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    args: ModelConfig

    def setup(self):
        # Attention parameters
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
        # Feed-forward parameters
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
        # Normalization layer parameters
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
        mask: jax.Array,  # [bsz, seqlen, max_seqlen] - attention mask
        true_len: jax.Array,  # [bsz] - actual (non-padded) sequence lengths
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Attention block
        h_norm = rms_norm(x, self.attention_norm_weight, eps=self.args.rms_norm_eps)
        attn_output, xk, xv = grouped_query_attention(
            h_norm,
            freqs_cis=freqs_cis,
            params=self.attention,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            mask=mask,
            true_len=true_len,
        )
        x = x + attn_output  # Residual connection
        # Debug: After attention
        h_np = np.array(x, dtype=np.float32)
        logger.debug(f"Layer {layer_idx} - After attention: Sample values (first batch, first position, first 10 dims): {h_np[0, 0, :10]}")

        # Feed-forward block
        h_ffn_norm = rms_norm(x, self.ffn_norm_weight, eps=self.args.rms_norm_eps)
        ffn_output = feed_forward(
            h_ffn_norm,
            params=self.feed_forward,
            activation_fn=self.args.activation_fn,
        )
        x = x + ffn_output  # Residual connection
        # Debug: After feedforward (commented out for performance)
        h_np = np.array(x, dtype=np.float32)
        logger.debug(f"Layer {layer_idx} - After feedforward: Sample values (first batch, first position, first 10 dims): {h_np[0, 0, :10]}")

        return x, xk, xv


class LLaMa(nn.Module):
    args: ModelConfig

    def setup(self):
        # Token embeddings
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

        # Final normalization weight
        self.norm_weight = self.param(
            "norm_weight", nn.initializers.ones, (self.args.dim,), dtype=self.args.dtype
        )

        # Final output layer
        self.output = self.param(
            "output",
            nn.initializers.normal(stddev=0.02),
            (self.args.dim, self.args.vocab_size),
            self.args.dtype,
        )

        # Precompute RoPE frequencies
        # Used float64 to match the precision of the torch/numpy implementation
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
        # jax.config.update("jax_default_matmul_precision", "default")

    def __call__(
        self,
        tokens: jax.Array,
        true_lengths: jax.Array,  # [bsz] - actual (non-padded) sequence lengths
        kv_cache: KVCache,
        mask: jax.Array,  # [bsz, seqlen, max_seqlen] - attention mask
    ):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Debug: After embeddings
        h_np = np.array(h, dtype=np.float32)
        logger.debug(f"After embeddings: Sample values (first batch, first position, first 10 dims): {h_np[0, 0, :10]}")

        temp_kv = TempKV.new(
            n_layers=self.args.n_layers,
            bsz=bsz,
            seqlen=seqlen,
            kv_heads=self.args.n_kv_heads,
            head_dim=self.args.head_dim,
            dtype=self.args.dtype,
        )

        for layer_idx, layer in enumerate(self.layers):
            h, xk, xv = layer(h, self.freqs_cis, kv_cache, layer_idx, mask, true_lengths)
            temp_kv = temp_kv.set_layer(layer_idx, xk, xv)

        # Update the KV cache with all layers at once
        updated_kv_cache = kv_cache.update(temp_kv, true_lengths)

        # Final normalization and output projection
        h = rms_norm(h, self.norm_weight, eps=self.args.rms_norm_eps)
        # Debug: After final norm
        h_np = np.array(h, dtype=np.float32)
        logger.debug(f"After final norm: Sample values (first batch, first position, first 10 dims): {h_np[0, 0, :10]}")
        # Tie weights: use the token embedding matrix for the final linear layer
        logits = jnp.einsum("bsd,dv->bsv", h, self.output)

        return logits, updated_kv_cache  # Return logits and the updated KVCache