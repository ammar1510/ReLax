import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass

from utils.ops import rms_norm, apply_rotary_emb, repeat_kv, grouped_query_attention, feed_forward, precompute_freqs_cis, AttentionParams, FeedForwardParams
from utils.kvcache import KVCache 
from .config import ModelConfig

class TransformerBlock(nn.Module):
    args: ModelConfig

    def setup(self):
        # Attention parameters
        self.attention = AttentionParams(
            wq=self.param('wq', nn.initializers.normal(stddev=0.02), (self.args.dim, self.args.n_heads, self.args.head_dim)),
            wk=self.param('wk', nn.initializers.normal(stddev=0.02), (self.args.dim, self.args.n_kv_heads, self.args.head_dim)),
            wv=self.param('wv', nn.initializers.normal(stddev=0.02), (self.args.dim, self.args.n_kv_heads, self.args.head_dim)),
            wo=self.param('wo', nn.initializers.normal(stddev=0.02), (self.args.n_heads * self.args.head_dim, self.args.dim)),
        )
        # Feed-forward parameters
        self.feed_forward = FeedForwardParams(
            w1_gate=self.param('w1_gate', nn.initializers.normal(stddev=0.02), (self.args.dim, self.args.ffn_hidden_dim)),
            w2_up=self.param('w2_up', nn.initializers.normal(stddev=0.02), (self.args.dim, self.args.ffn_hidden_dim)),
            w3_down=self.param('w3_down', nn.initializers.normal(stddev=0.02), (self.args.ffn_hidden_dim, self.args.dim)),
        )
        # Normalization layer parameters
        self.attention_norm_weight = self.param('attention_norm_weight', nn.initializers.ones, (self.args.dim,))
        self.ffn_norm_weight = self.param('ffn_norm_weight', nn.initializers.ones, (self.args.dim,))

    def __call__(self, x: jax.Array, freqs_cis: jax.Array, kv_cache: KVCache, layer_idx: int, start_pos: int) -> tuple[jax.Array, KVCache]:
        # Attention block
        h_norm = rms_norm(x, self.attention_norm_weight, eps=self.args.rms_norm_eps)
        attn_output, kv_cache = grouped_query_attention(
            h_norm,
            freqs_cis=freqs_cis,
            params=self.attention,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            n_heads=self.args.n_heads,
            n_kv_heads=self.args.n_kv_heads,
        )
        x = x + attn_output  # Residual connection

        # Feed-forward block
        h_ffn_norm = rms_norm(x, self.ffn_norm_weight, eps=self.args.rms_norm_eps)
        ffn_output = feed_forward(
            h_ffn_norm,
            params=self.feed_forward,
            activation_fn=self.args.activation_fn,
        )
        x = x + ffn_output  # Residual connection

        return x, kv_cache


class LLaMA(nn.Module):
    args: ModelConfig

    def setup(self):
        # Token embeddings
        self.tok_embeddings = nn.Embed(
            num_embeddings=self.args.vocab_size,
            features=self.args.dim,
            embedding_init=nn.initializers.normal(stddev=0.02) # Common initialization
        )

        # Initialize transformer blocks
        self.layers = [TransformerBlock(self.args, name=f'layer_{i}') for i in range(self.args.n_layers)]

        # Final normalization weight
        self.norm_weight = self.param('norm_weight', nn.initializers.ones, (self.args.dim,))

        # Output layer (Language Model head)
        self.output = nn.Dense(
            features=self.args.vocab_size,
            use_bias=False, # LLaMA usually doesn't use bias in the LM head
            kernel_init=nn.initializers.normal(stddev=0.02) # Consistent init
        )

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.args.head_dim, # Use head_dim here
            self.args.max_seq_len,
            self.args.rope_theta
        )

    def __call__(self, tokens: jax.Array, start_pos: int, kv_cache: KVCache):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            h, kv_cache = layer(h, self.freqs_cis, kv_cache, layer_idx, start_pos)

        # Final normalization and output projection
        h = rms_norm(h, self.norm_weight, eps=self.args.rms_norm_eps)
        logits = self.output(h)

        return logits, kv_cache # Return logits and the updated KVCache 