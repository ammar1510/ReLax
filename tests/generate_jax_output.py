import numpy as np
import jax
import jax.numpy as jnp
import os
import torch
from jax.dlpack import from_dlpack

# JAX/Flax model components
from models.llama.model import LLaMa as Llama_jax
from models.llama.config import ModelConfig
from utils.kvcache import KVCache as KVCache_jax

def torch_to_jax(torch_tensor):
    """Converts a PyTorch tensor to a JAX array without copying data."""
    dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
    return from_dlpack(dlpack_capsule)

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
    dtype_torch = torch.float32

    
    jax_args = ModelConfig(
        dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, 
        vocab_size=vocab_size, ffn_hidden_dim=hidden_dim, max_seqlen=max_seqlen,
        rope_theta=500000.0, rms_norm_eps=1e-5,
        activation_fn=activation_fn
    )

    head_dim = dim // n_heads

    # 2. Weight Generation
    print("Generating Torch weights for JAX...")
    torch.cuda.manual_seed(0)
    jax_weights = {'params': {}}

    tok_embeddings_torch = torch.randn(vocab_size, dim, device='cuda', dtype=dtype_torch)
    jax_weights['params']['tok_embeddings'] = {'embedding': torch_to_jax(tok_embeddings_torch)}
    print(f"  tok_embeddings shape: {tok_embeddings_torch.shape}")

    for i in range(n_layers):
        layer_key = f'layer_{i}'
        # This structure must be flat to match how Flax registers params with self.param(), fixing the ScopeParamNotFoundError
        jax_weights['params'][layer_key] = {
            'wq': torch_to_jax(torch.randn(dim, n_heads, head_dim, device='cuda', dtype=dtype_torch)),
            'wk': torch_to_jax(torch.randn(dim, n_kv_heads, head_dim, device='cuda', dtype=dtype_torch)),
            'wv': torch_to_jax(torch.randn(dim, n_kv_heads, head_dim, device='cuda', dtype=dtype_torch)),
            'wo': torch_to_jax(torch.randn(n_heads * head_dim, dim, device='cuda', dtype=dtype_torch)),
            'w1_gate': torch_to_jax(torch.randn(dim, hidden_dim, device='cuda', dtype=dtype_torch)),
            'w2_up': torch_to_jax(torch.randn(dim, hidden_dim, device='cuda', dtype=dtype_torch)),
            'w3_down': torch_to_jax(torch.randn(hidden_dim, dim, device='cuda', dtype=dtype_torch)),
            'attention_norm_weight': torch_to_jax(torch.randn(dim, device='cuda', dtype=dtype_torch)),
            'ffn_norm_weight': torch_to_jax(torch.randn(dim, device='cuda', dtype=dtype_torch)),
        }
    
    jax_weights['params']['norm_weight'] = torch_to_jax(torch.randn(dim, device='cuda', dtype=dtype_torch))
    jax_weights['params']['output'] = {'kernel': torch_to_jax(torch.randn(dim, vocab_size, device='cuda', dtype=dtype_torch))}
    print("JAX weights generated.")

    # 3. JAX Model Initialization
    print("\nInitializing JAX model...")
    jax_model = Llama_jax(jax_args)
    print("JAX model initialized.")
    
    # 4. Input token generation
    print("\nGenerating input tokens...")
    tokens_torch = torch.randint(0, vocab_size, (batch_size, seqlen), device='cuda')
    tokens_jax = torch_to_jax(tokens_torch.to(torch.int32)) # JAX requires int32 for tokens
    print(f"  Input tokens shape: {tokens_jax.shape}")
    
    # 5. Pre-fill KV Caches to be non-empty
    print("\nPre-filling KV cache...")
    prefill_len = start_pos

    prefill_k_torch = torch.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim, device='cuda', dtype=dtype_torch)
    prefill_v_torch = torch.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim, device='cuda', dtype=dtype_torch)
    prefill_k_jax = torch_to_jax(prefill_k_torch)
    prefill_v_jax = torch_to_jax(prefill_v_torch)

    print(f"  Prefill K cache shape: {prefill_k_jax.shape}")
    print(f"  Prefill V cache shape: {prefill_v_jax.shape}")

    # JAX KV Cache
    k_init_jax = jnp.zeros((n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float32)
    v_init_jax = jnp.zeros((n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float32)
    k_updated_jax = k_init_jax.at[:, :, :prefill_len, :, :].set(prefill_k_jax)
    v_updated_jax = v_init_jax.at[:, :, :prefill_len, :, :].set(prefill_v_jax)
    kv_cache_jax = KVCache_jax(k=k_updated_jax, v=v_updated_jax)
    print("KV cache pre-filled.")

    # 6. JAX execution
    print("\nExecuting JAX model...")
    logits_jax, _ = jax_model.apply({'params': jax_weights['params']}, tokens=tokens_jax, start_pos=start_pos, kv_cache=kv_cache_jax)
    print(f"  Output logits shape: {logits_jax.shape}")
    print("JAX model execution complete.")

    # 7. Save output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "jax_logits.npy"), logits_jax)
    print(f"\nJAX logits saved to {os.path.join(output_dir, 'jax_logits.npy')}")

if __name__ == "__main__":
    generate_jax_output() 