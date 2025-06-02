import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass

# Import the building blocks and KVCache
from utils.ops import rms_norm, apply_rotary_emb, repeat_kv, grouped_query_attention, feed_forward, precompute_freqs_cis, AttentionParams, FeedForwardParams
from utils.kvcache import KVCache # Assuming kvcache.py exists

@dataclass
class ModelArgs:
    dim: int = 3072
    n_layers: int = 32
    n_heads: int = 24
    n_kv_heads: int | None = 8 # Use None for MHA, specify for GQA/MQA
    vocab_size: int = 128256 
    ffn_hidden_dim: int = 8192 
    activation_fn: str = 'silu' # Added: 'silu', 'relu', 'gelu', etc.
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 2048 # Max sequence length for precomputed freqs & KV cache

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.dim % self.n_heads == 0
        self.head_dim = self.dim // self.n_heads

class LLaMA(nn.Module):
    args: ModelArgs

    def setup(self): 
        # Token embeddings
        self.tok_embeddings = nn.Embed(
            num_embeddings=self.args.vocab_size,
            features=self.args.dim,
            embedding_init=nn.initializers.normal(stddev=0.02) # Common initialization
        )

        # Initialize transformer block parameters
        self.layers = []
        for _ in range(self.args.n_layers):
            self.layers.append({
                'attention': AttentionParams(
                    dim=self.args.dim,
                    n_heads=self.args.n_heads,
                    n_kv_heads=self.args.n_kv_heads,
                    head_dim=self.args.head_dim
                ),
                'feed_forward': FeedForwardParams(
                    dim=self.args.dim,
                    hidden_dim=self.args.ffn_hidden_dim,
                    activation_fn=self.args.activation_fn
                ),
                'attention_norm': rms_norm(self.args.dim, eps=self.args.rms_norm_eps),
                'ffn_norm': rms_norm(self.args.dim, eps=self.args.rms_norm_eps),
            })

        # Final normalization
        self.norm = rms_norm(self.args.dim, eps=self.args.rms_norm_eps)

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
        # pass # TODO: Remove pass once other sections are filled -> Removing this

    def __call__(self, tokens: jax.Array, start_pos: int, kv_cache: KVCache):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Get the RoPE frequencies for the current sequence segment
        freqs_cis_slice = jax.lax.dynamic_slice(
            self.freqs_cis,
            (start_pos, 0),
            (seqlen, self.freqs_cis.shape[-1])
        )

        # Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Attention block
            h_norm = layer['attention_norm'](h)
            attn_output, kv_cache = grouped_query_attention(
                h_norm,
                freqs_cis_slice,
                layer['attention'],
                layer_idx, # Pass layer index
                kv_cache, # Pass the cache object to be updated
                start_pos
            )
            h = h + attn_output # Residual connection

            # Feed-forward block
            h_ffn_norm = layer['ffn_norm'](h)
            ffn_output = feed_forward(
                h_ffn_norm,
                layer['feed_forward']
            )
            h = h + ffn_output # Residual connection


        # Final normalization and output projection
        h = self.norm(h)
        logits = self.output(h)

        return logits, kv_cache # Return logits and the updated KVCache 