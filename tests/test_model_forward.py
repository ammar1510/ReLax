import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax.core import freeze
import torch_xla

# JAX/Flax model components
from models.llama.model import LLaMa as Llama_jax
from models.llama.config import ModelConfig
from utils.kvcache import KVCache as KVCache_jax

# PyTorch model components
from experiments.torch_llama import (
    Transformer as Llama_torch,
    ModelArgs,
    KVCache as KVCache_torch,
)

jax.config.update("jax_default_matmul_precision", "highest")


def test_model_forward_pass():
    """Test forward pass comparison between PyTorch and JAX models with same weights."""
    # 1. Small Configuration for Verification
    dim = 512
    n_layers = 2
    n_heads = 8
    n_kv_heads = 2
    vocab_size = 1000
    multiple_of = 32
    max_seqlen = 256
    batch_size = 2
    seqlen = 32
    dtype = np.float32


    # Try to import torch_xla for TPU support
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        TORCH_XLA_AVAILABLE = True
    except ImportError:
        TORCH_XLA_AVAILABLE = False

    # Device selection: TPU > CUDA > CPU
    if TORCH_XLA_AVAILABLE:
        device = torch_xla.device()  # TPU device
        print(f"Using PyTorch XLA device: {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device")
    else:
        device = "cpu"
        print("Using CPU device")

    torch_device = device
    torch_dtype = torch.float32
    jax_dtype = jnp.float32

    # PyTorch config
    torch_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seqlen=max_seqlen,
        flash=False,
        max_batch_size=batch_size,
        device=torch_device,
    )

    # JAX config
    ffn_hidden_dim = int(2 * (4 * dim) / 3)
    ffn_hidden_dim = multiple_of * ((ffn_hidden_dim + multiple_of - 1) // multiple_of)

    jax_args = ModelConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        ffn_hidden_dim=ffn_hidden_dim,
        max_seqlen=max_seqlen,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        activation_fn="silu",
        dtype=jax_dtype,
        use_scaled_rope=True,
    )

    head_dim = dim // n_heads

    # 2. Generate Same Weights for Both Models
    np.random.seed(42)  # Fixed seed for reproducibility
    jax_weights = {"params": {}}
    
    # Token embeddings
    tok_embeddings_np = np.random.randn(vocab_size, dim).astype(dtype)
    jax_weights["params"]["tok_embeddings"] = {"embedding": tok_embeddings_np}

    # Layer weights
    for i in range(n_layers):
        layer_key = f"layer_{i}"
        jax_weights["params"][layer_key] = {
            "wq": np.random.randn(dim, n_heads, head_dim).astype(dtype),
            "wk": np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            "wv": np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            "wo": np.random.randn(n_heads * head_dim, dim).astype(dtype),
            "w_gate": np.random.randn(dim, ffn_hidden_dim).astype(dtype),
            "w_up": np.random.randn(dim, ffn_hidden_dim).astype(dtype),
            "w_down": np.random.randn(ffn_hidden_dim, dim).astype(dtype),
            "attention_norm_weight": np.random.randn(dim).astype(dtype),
            "ffn_norm_weight": np.random.randn(dim).astype(dtype),
        }

    # Final norm and output
    jax_weights["params"]["norm_weight"] = np.random.randn(dim).astype(dtype)
    # Output is a param directly, not a dict
    jax_weights["params"]["output"] = np.random.randn(dim, vocab_size).astype(dtype)

    # 3. Initialize JAX Model
    jax_model = Llama_jax(jax_args)
    jax_params = freeze(jax_weights["params"])

    # 4. Initialize PyTorch Model and Load Same Weights
    torch_model = Llama_torch(torch_args).to(torch_device)
    torch_model.eval()
    
    # Load token embeddings
    torch_model.tok_embeddings.weight = torch.nn.Parameter(
        torch.tensor(tok_embeddings_np, device=torch_device, dtype=torch_dtype)
    )

    # Load layer weights
    for i in range(n_layers):
        layer_weights = jax_weights["params"][f"layer_{i}"]
        
        # Attention weights
        wq = layer_weights["wq"].reshape(dim, n_heads * head_dim)
        wk = layer_weights["wk"].reshape(dim, n_kv_heads * head_dim)
        wv = layer_weights["wv"].reshape(dim, n_kv_heads * head_dim)
        wo = layer_weights["wo"]
        
        torch_model.layers[i].attention.wq.weight = torch.nn.Parameter(
            torch.tensor(wq.T, device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].attention.wk.weight = torch.nn.Parameter(
            torch.tensor(wk.T, device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].attention.wv.weight = torch.nn.Parameter(
            torch.tensor(wv.T, device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].attention.wo.weight = torch.nn.Parameter(
            torch.tensor(wo.T, device=torch_device, dtype=torch_dtype)
        )

        # Feed-forward weights (w1=gate, w3=up, w2=down)
        w_gate = layer_weights["w_gate"]
        w_up = layer_weights["w_up"]
        w_down = layer_weights["w_down"]
        
        torch_model.layers[i].feed_forward.w1.weight = torch.nn.Parameter(
            torch.tensor(w_gate.T, device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].feed_forward.w3.weight = torch.nn.Parameter(
            torch.tensor(w_up.T, device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].feed_forward.w2.weight = torch.nn.Parameter(
            torch.tensor(w_down.T, device=torch_device, dtype=torch_dtype)
        )

        # Norm weights
        torch_model.layers[i].attention_norm.weight = torch.nn.Parameter(
            torch.tensor(layer_weights["attention_norm_weight"], device=torch_device, dtype=torch_dtype)
        )
        torch_model.layers[i].ffn_norm.weight = torch.nn.Parameter(
            torch.tensor(layer_weights["ffn_norm_weight"], device=torch_device, dtype=torch_dtype)
        )

    # Final norm and output
    torch_model.norm.weight = torch.nn.Parameter(
        torch.tensor(jax_weights["params"]["norm_weight"], device=torch_device, dtype=torch_dtype)
    )
    torch_model.output.weight = torch.nn.Parameter(
        torch.tensor(jax_weights["params"]["output"].T, device=torch_device, dtype=torch_dtype)
    )

    # 5. Create Input Tokens
    tokens_np = np.random.randint(0, vocab_size, (batch_size, seqlen))
    tokens_jax = jnp.array(tokens_np, dtype=jnp.int32)
    tokens_torch = torch.tensor(tokens_np, device=torch_device, dtype=torch.long)

    # 6. Initialize Empty KV Caches
    kv_cache_jax = KVCache_jax.new(
        n_layers, batch_size, max_seqlen, n_kv_heads, head_dim, dtype=jax_dtype
    )
    
    # PyTorch KV caches
    for i in range(n_layers):
        layer_cache = KVCache_torch(
            batch_size,
            max_seqlen,
            n_kv_heads,
            head_dim,
            dtype=torch_dtype,
            device=torch_device,
        )
        torch_model.layers[i].attention.cache = layer_cache

    # 7. Forward Pass - JAX
    seq_lengths = jnp.full((batch_size,), seqlen, dtype=jnp.int32)
    logits_jax, updated_kv_cache_jax = jax_model.apply(
        {"params": jax_params},
        tokens=tokens_jax,
        seq_lengths=seq_lengths,
        kv_cache=kv_cache_jax,
    )

    # 8. Forward Pass - PyTorch
    start_pos = 0
    logits_torch = torch_model.forward_inference(tokens_torch, start_pos)

    # 9. Compare Outputs
    logits_jax_np = np.array(logits_jax)
    logits_torch_np = logits_torch.cpu().detach().numpy()

    # Basic shape checks
    assert logits_jax_np.shape == logits_torch_np.shape, (
        f"Shape mismatch: JAX {logits_jax_np.shape} vs PyTorch {logits_torch_np.shape}"
    )
    assert logits_jax_np.shape == (batch_size, seqlen, vocab_size), (
        f"Unexpected output shape: {logits_jax_np.shape}"
    )

    # Check for NaN or Inf
    assert not np.any(np.isnan(logits_jax_np)), "JAX output contains NaN"
    assert not np.any(np.isnan(logits_torch_np)), "PyTorch output contains NaN"
    assert not np.any(np.isinf(logits_jax_np)), "JAX output contains Inf"
    assert not np.any(np.isinf(logits_torch_np)), "PyTorch output contains Inf"

    # Compare outputs
    print(f"JAX output shape: {logits_jax_np.shape}")
    print(f"PyTorch output shape: {logits_torch_np.shape}")
    print(f"Max absolute difference: {np.max(np.abs(logits_jax_np - logits_torch_np))}")
    print(f"Mean absolute difference: {np.mean(np.abs(logits_jax_np - logits_torch_np))}")
    print(f"Relative difference (max): {np.max(np.abs(logits_jax_np - logits_torch_np) / (np.abs(logits_torch_np) + 1e-8))}")

    # Assert close with reasonable tolerance
    np.testing.assert_allclose(
        logits_jax_np,
        logits_torch_np,
        rtol=1e-3,
        atol=1e-3,
        err_msg="JAX and PyTorch outputs do not match"
    )

    print("✓ Model forward pass comparison passed!")


if __name__ == "__main__":
    test_model_forward_pass()

