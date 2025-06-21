import numpy as np
import torch
import os

# PyTorch model components
from tests.torch_ops import Transformer as Llama_torch, ModelArgs, KVCache as KVCache_torch

def generate_torch_output():
    # 1. Shared Configuration
    dim = 3072
    hidden_dim = 8192
    n_layers = 28
    n_heads = 24
    n_kv_heads = 8
    vocab_size = 128256
    max_seqlen = 8192
    batch_size = 4 # Increased batch size
    seqlen = 64
    start_pos = 10 # This will be the size of our pre-filled cache
    dtype = np.float64

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    # PyTorch config
    torch_args = ModelArgs(
        dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, 
        vocab_size=vocab_size, max_seqlen=max_seqlen,
        flash=False, max_batch_size=batch_size, device=torch_device
    )

    head_dim = dim // n_heads

    # 2. Weight Generation
    np.random.seed(0)
    # Note: The weight generation logic is duplicated from the JAX script to ensure consistency.
    # In a real-world scenario, you might save and load weights from a file.
    tok_embeddings_np = np.random.randn(vocab_size, dim).astype(dtype)
    
    # Generate JAX-structured weights first to ensure they match
    jax_weights = {'params': {}}
    for i in range(n_layers):
        layer_key = f'layer_{i}'
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

    # 3. PyTorch Model Initialization and Weight Loading
    torch_model = Llama_torch(torch_args).to(torch_device)
    torch_model.tok_embeddings.weight = torch.nn.Parameter(torch.tensor(tok_embeddings_np, device=torch_device))
    
    for i in range(n_layers):
        wq = jax_weights['params'][f'layer_{i}']['wq'].reshape(dim, n_heads * head_dim)
        wk = jax_weights['params'][f'layer_{i}']['wk'].reshape(dim, n_kv_heads * head_dim)
        wv = jax_weights['params'][f'layer_{i}']['wv'].reshape(dim, n_kv_heads * head_dim)
        wo = jax_weights['params'][f'layer_{i}']['wo']
        torch_model.layers[i].attention.wq.weight = torch.nn.Parameter(torch.tensor(wq.T, device=torch_device))
        torch_model.layers[i].attention.wk.weight = torch.nn.Parameter(torch.tensor(wk.T, device=torch_device))
        torch_model.layers[i].attention.wv.weight = torch.nn.Parameter(torch.tensor(wv.T, device=torch_device))
        torch_model.layers[i].attention.wo.weight = torch.nn.Parameter(torch.tensor(wo.T, device=torch_device))
        
        w1, w3, w2 = (jax_weights['params'][f'layer_{i}'][k] for k in ['w1_gate', 'w2_up', 'w3_down'])
        torch_model.layers[i].feed_forward.w1.weight = torch.nn.Parameter(torch.tensor(w1.T, device=torch_device))
        torch_model.layers[i].feed_forward.w3.weight = torch.nn.Parameter(torch.tensor(w3.T, device=torch_device))
        torch_model.layers[i].feed_forward.w2.weight = torch.nn.Parameter(torch.tensor(w2.T, device=torch_device))

        torch_model.layers[i].attention_norm.weight = torch.nn.Parameter(torch.tensor(jax_weights['params'][f'layer_{i}']['attention_norm_weight'], device=torch_device))
        torch_model.layers[i].ffn_norm.weight = torch.nn.Parameter(torch.tensor(jax_weights['params'][f'layer_{i}']['ffn_norm_weight'], device=torch_device))

    torch_model.norm.weight = torch.nn.Parameter(torch.tensor(jax_weights['params']['norm_weight'], device=torch_device))
    torch_model.output.weight = torch.nn.Parameter(torch.tensor(jax_weights['params']['output']['kernel'].T, device=torch_device))

    # 4. Input token generation
    tokens_np = np.random.randint(0, vocab_size, (batch_size, seqlen))
    tokens_torch = torch.tensor(tokens_np, device=torch_device)
    
    # 5. Pre-fill KV Caches to be non-empty
    prefill_len = start_pos
    prefill_k = np.random.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim).astype(dtype)
    prefill_v = np.random.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim).astype(dtype)
    
    for i in range(n_layers):
        layer_cache = KVCache_torch(batch_size, max_seqlen, n_kv_heads, head_dim, dtype=torch.float64, device=torch_device)
        layer_cache.cache_k[:, :prefill_len, :, :] = torch.tensor(prefill_k[i], device=torch_device)
        layer_cache.cache_v[:, :prefill_len, :, :] = torch.tensor(prefill_v[i], device=torch_device)
        torch_model.layers[i].attention.cache = layer_cache

    # 6. PyTorch execution
    logits_torch = torch_model.forward_inference(tokens_torch, start_pos)
    
    # 7. Save output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "torch_logits.npy"), logits_torch.cpu().detach().numpy())
    print(f"PyTorch logits saved to {os.path.join(output_dir, 'torch_logits.npy')}")

if __name__ == "__main__":
    generate_torch_output() 