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
    dtype_torch = torch.float32

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.manual_seed(0)

    # PyTorch config
    torch_args = ModelArgs(
        dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, 
        vocab_size=vocab_size, max_seqlen=max_seqlen,
        flash=False, max_batch_size=batch_size, device=torch_device
    )

    head_dim = dim // n_heads

    # 2. PyTorch Model Initialization and Weight Loading
    torch_model = Llama_torch(torch_args).to(torch_device)
    
    # Generate weights with torch and load them into the model
    # Note: The weight generation logic is duplicated from the JAX script to ensure consistency.
    tok_embeddings = torch.randn(vocab_size, dim, dtype=dtype_torch, device=torch_device)
    torch_model.tok_embeddings.weight = torch.nn.Parameter(tok_embeddings)
    
    for i in range(n_layers):
        # Attention weights
        wq = torch.randn(dim, n_heads * head_dim, dtype=dtype_torch, device=torch_device)
        wk = torch.randn(dim, n_kv_heads * head_dim, dtype=dtype_torch, device=torch_device)
        wv = torch.randn(dim, n_kv_heads * head_dim, dtype=dtype_torch, device=torch_device)
        wo = torch.randn(n_heads * head_dim, dim, dtype=dtype_torch, device=torch_device)
        torch_model.layers[i].attention.wq.weight = torch.nn.Parameter(wq.T)
        torch_model.layers[i].attention.wk.weight = torch.nn.Parameter(wk.T)
        torch_model.layers[i].attention.wv.weight = torch.nn.Parameter(wv.T)
        torch_model.layers[i].attention.wo.weight = torch.nn.Parameter(wo.T)
        
        # Feed-forward weights
        w1 = torch.randn(dim, hidden_dim, dtype=dtype_torch, device=torch_device)
        w3 = torch.randn(dim, hidden_dim, dtype=dtype_torch, device=torch_device)
        w2 = torch.randn(hidden_dim, dim, dtype=dtype_torch, device=torch_device)
        torch_model.layers[i].feed_forward.w1.weight = torch.nn.Parameter(w1.T)
        torch_model.layers[i].feed_forward.w3.weight = torch.nn.Parameter(w3.T)
        torch_model.layers[i].feed_forward.w2.weight = torch.nn.Parameter(w2.T)

        # Norm weights
        attention_norm = torch.randn(dim, dtype=dtype_torch, device=torch_device)
        ffn_norm = torch.randn(dim, dtype=dtype_torch, device=torch_device)
        torch_model.layers[i].attention_norm.weight = torch.nn.Parameter(attention_norm)
        torch_model.layers[i].ffn_norm.weight = torch.nn.Parameter(ffn_norm)

    # Final norm and output weights
    norm = torch.randn(dim, dtype=dtype_torch, device=torch_device)
    output = torch.randn(dim, vocab_size, dtype=dtype_torch, device=torch_device)
    torch_model.norm.weight = torch.nn.Parameter(norm)
    torch_model.output.weight = torch.nn.Parameter(output.T)

    # 3. Input token generation
    tokens_torch = torch.randint(0, vocab_size, (batch_size, seqlen), device=torch_device)
    
    # 4. Pre-fill KV Caches to be non-empty
    prefill_len = start_pos
    prefill_k = torch.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim, dtype=dtype_torch, device=torch_device)
    prefill_v = torch.randn(n_layers, batch_size, prefill_len, n_kv_heads, head_dim, dtype=dtype_torch, device=torch_device)
    
    for i in range(n_layers):
        layer_cache = KVCache_torch(batch_size, max_seqlen, n_kv_heads, head_dim, dtype=torch.float32, device=torch_device)
        layer_cache.cache_k[:, :prefill_len, :, :] = prefill_k[i]
        layer_cache.cache_v[:, :prefill_len, :, :] = prefill_v[i]
        torch_model.layers[i].attention.cache = layer_cache

    # 5. PyTorch execution
    logits_torch = torch_model.forward_inference(tokens_torch, start_pos)
    
    # 6. Save output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "torch_logits.npy"), logits_torch.cpu().detach().numpy())
    print(f"PyTorch logits saved to {os.path.join(output_dir, 'torch_logits.npy')}")

if __name__ == "__main__":
    generate_torch_output() 