import numpy as np
import jax
import jax.numpy as jnp
import os

# JAX/Flax model components
from models.llama.model import LLaMa as Llama_jax
from models.llama.config import ModelConfig
from utils.kvcache import KVCache as KVCache_jax

def generate_jax_output():
    # 1. Shared Configuration
    dim = 3072
    hidden_dim = 8192
    n_layers = 28
    n_heads = 24
    n_kv_heads = 8
    vocab_size = 128256
    max_seqlen = 8192
    batch_size = 4 # increased batch size
    seqlen = 64
    start_pos = 10 # this will be the size of our pre-filled cache
    activation_fn = 'silu'
    dtype = np.float64

    
    jax_args = ModelConfig(
        dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, 
        vocab_size=vocab_size, ffn_hidden_dim=hidden_dim, max_seqlen=max_seqlen,
        rope_theta=500000.0, rms_norm_eps=1e-5,
        activation_fn=activation_fn
    )

    head_dim = dim // n_heads

    # 2. Weight Generation
    np.random.seed(0)
    jax_weights = {'params': {}}
    tok_embeddings_np = np.random.randn(vocab_size, dim).astype(dtype)
    jax_weights['params']['tok_embeddings'] = {'embedding': tok_embeddings_np}

    for i in range(n_layers):
        layer_key = f'layer_{i}'
        # This structure must be flat to match how Flax registers params with self.param(), fixing the ScopeParamNotFoundError
        jax_weights['params'][layer_key] = {
            'wq': np.random.randn(dim, n_heads, head_dim).astype(dtype),
            'wk': np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            'wv': np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            'wo': np.random.randn(n_heads * head_dim, dim).astype(dtype),
            'w1_gate': np.random.randn(dim, hidden_dim).astype(dtype),
            'w2_up': np.random.randn(dim, hidden_dim).astype(dtype),
            'w3_down': np.random.randn(hidden_dim, dim).astype(dtype),
            'attention_norm_weight': np.random.randn(dim).astype(dtype),
            'ffn_norm_weight': np.random.randn(dim).astype(dtype),
        }
    
    jax_weights['params']['norm_weight'] = np.random.randn(dim).astype(dtype)
    jax_weights['params']['output'] = {'kernel': np.random.randn(dim, vocab_size).astype(dtype)}

    # 3. JAX Model Initialization
    jax_model = Llama_jax(jax_args)
    
    # 4. Input token generation
    tokens_np = np.random.randint(0, vocab_size, (batch_size, seqlen))
    tokens_jax = jnp.array(tokens_np)
    
    # 5. Pre-fill KV Caches to be non-empty
    prefill_len = start_pos
    prefill_k = np.random.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim).astype(dtype)
    prefill_v = np.random.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim).astype(dtype)

    # JAX KV Cache
    k_init_jax = jnp.zeros((n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float64)
    v_init_jax = jnp.zeros((n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float64)
    k_updated_jax = k_init_jax.at[:, :, :prefill_len, :, :].set(jnp.array(prefill_k))
    v_updated_jax = v_init_jax.at[:, :, :prefill_len, :, :].set(jnp.array(prefill_v))
    kv_cache_jax = KVCache_jax(k=k_updated_jax, v=v_updated_jax)

    # 6. JAX execution
    logits_jax, _ = jax_model.apply({'params': jax_weights['params']}, tokens=tokens_jax, start_pos=start_pos, kv_cache=kv_cache_jax)

    # 7. Save output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "jax_logits.npy"), logits_jax)
    print(f"JAX logits saved to {os.path.join(output_dir, 'jax_logits.npy')}")

if __name__ == "__main__":
    generate_jax_output() 