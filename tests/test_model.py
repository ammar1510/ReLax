import numpy as np
import torch
import jax
import jax.numpy as jnp

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


def test_transformer_forward_pass():
    # 1. Shared Configuration
    dim = 3072
    n_layers = 28
    n_heads = 24
    n_kv_heads = 8
    vocab_size = 128256
    multiple_of = 256
    max_seqlen = 8192
    batch_size = 4  # Increased batch size
    seqlen = 64
    start_pos = 10  # This will be the size of our pre-filled cache
    dtype = np.float64

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
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
    )

    head_dim = dim // n_heads

    # 2. Weight Generation
    np.random.seed(0)
    jax_weights = {"params": {}}
    tok_embeddings_np = np.random.randn(vocab_size, dim).astype(dtype)
    jax_weights["params"]["tok_embeddings"] = {"embedding": tok_embeddings_np}

    for i in range(n_layers):
        layer_key = f"layer_{i}"
        # This structure must be flat to match how Flax registers params with self.param(), fixing the ScopeParamNotFoundError
        jax_weights["params"][layer_key] = {
            "wq": np.random.randn(dim, n_heads, head_dim).astype(dtype),
            "wk": np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            "wv": np.random.randn(dim, n_kv_heads, head_dim).astype(dtype),
            "wo": np.random.randn(n_heads * head_dim, dim).astype(dtype),
            "w1_gate": np.random.randn(dim, ffn_hidden_dim).astype(dtype),
            "w2_up": np.random.randn(dim, ffn_hidden_dim).astype(dtype),
            "w3_down": np.random.randn(ffn_hidden_dim, dim).astype(dtype),
            "attention_norm_weight": np.random.randn(dim).astype(dtype),
            "ffn_norm_weight": np.random.randn(dim).astype(dtype),
        }

    jax_weights["params"]["norm_weight"] = np.random.randn(dim).astype(dtype)
    jax_weights["params"]["output"] = {
        "kernel": np.random.randn(dim, vocab_size).astype(dtype)
    }

    # 3. JAX Model Initialization
    jax_model = Llama_jax(jax_args)

    # 4. PyTorch Model Initialization and Weight Loading
    torch_model = Llama_torch(torch_args).to(torch_device)
    torch_model.tok_embeddings.weight = torch.nn.Parameter(
        torch.tensor(tok_embeddings_np, device=torch_device)
    )

    for i in range(n_layers):
        wq = jax_weights["params"][f"layer_{i}"]["wq"].reshape(dim, n_heads * head_dim)
        wk = jax_weights["params"][f"layer_{i}"]["wk"].reshape(
            dim, n_kv_heads * head_dim
        )
        wv = jax_weights["params"][f"layer_{i}"]["wv"].reshape(
            dim, n_kv_heads * head_dim
        )
        wo = jax_weights["params"][f"layer_{i}"]["wo"]
        torch_model.layers[i].attention.wq.weight = torch.nn.Parameter(
            torch.tensor(wq.T, device=torch_device)
        )
        torch_model.layers[i].attention.wk.weight = torch.nn.Parameter(
            torch.tensor(wk.T, device=torch_device)
        )
        torch_model.layers[i].attention.wv.weight = torch.nn.Parameter(
            torch.tensor(wv.T, device=torch_device)
        )
        torch_model.layers[i].attention.wo.weight = torch.nn.Parameter(
            torch.tensor(wo.T, device=torch_device)
        )

        w1, w3, w2 = (
            jax_weights["params"][f"layer_{i}"][k]
            for k in ["w1_gate", "w2_up", "w3_down"]
        )
        torch_model.layers[i].feed_forward.w1.weight = torch.nn.Parameter(
            torch.tensor(w1.T, device=torch_device)
        )
        torch_model.layers[i].feed_forward.w3.weight = torch.nn.Parameter(
            torch.tensor(w3.T, device=torch_device)
        )
        torch_model.layers[i].feed_forward.w2.weight = torch.nn.Parameter(
            torch.tensor(w2.T, device=torch_device)
        )

        torch_model.layers[i].attention_norm.weight = torch.nn.Parameter(
            torch.tensor(
                jax_weights["params"][f"layer_{i}"]["attention_norm_weight"],
                device=torch_device,
            )
        )
        torch_model.layers[i].ffn_norm.weight = torch.nn.Parameter(
            torch.tensor(
                jax_weights["params"][f"layer_{i}"]["ffn_norm_weight"],
                device=torch_device,
            )
        )

    torch_model.norm.weight = torch.nn.Parameter(
        torch.tensor(jax_weights["params"]["norm_weight"], device=torch_device)
    )
    torch_model.output.weight = torch.nn.Parameter(
        torch.tensor(jax_weights["params"]["output"]["kernel"].T, device=torch_device)
    )

    # 5. Execution and Comparison
    tokens_np = np.random.randint(0, vocab_size, (batch_size, seqlen))
    tokens_jax, tokens_torch = jnp.array(tokens_np), torch.tensor(
        tokens_np, device=torch_device
    )

    # Pre-fill KV Caches to be non-empty
    prefill_len = start_pos
    prefill_k = np.random.randn(
        n_layers, batch_size, prefill_len, n_kv_heads, head_dim
    ).astype(dtype)
    prefill_v = np.random.randn(
        n_layers, batch_size, prefill_len, n_kv_heads, head_dim
    ).astype(dtype)

    # JAX KV Cache
    k_init_jax = jnp.zeros(
        (n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float64
    )
    v_init_jax = jnp.zeros(
        (n_layers, batch_size, max_seqlen, n_kv_heads, head_dim), dtype=jnp.float64
    )
    k_updated_jax = k_init_jax.at[:, :, :prefill_len, :, :].set(jnp.array(prefill_k))
    v_updated_jax = v_init_jax.at[:, :, :prefill_len, :, :].set(jnp.array(prefill_v))
    kv_cache_jax = KVCache_jax(k=k_updated_jax, v=v_updated_jax)

    # PyTorch KV Cache
    for i in range(n_layers):
        layer_cache = KVCache_torch(
            batch_size,
            max_seqlen,
            n_kv_heads,
            head_dim,
            dtype=torch.float64,
            device=torch_device,
        )
        layer_cache.cache_k[:, :prefill_len, :, :] = torch.tensor(
            prefill_k[i], device=torch_device
        )
        layer_cache.cache_v[:, :prefill_len, :, :] = torch.tensor(
            prefill_v[i], device=torch_device
        )
        torch_model.layers[i].attention.cache = layer_cache

    # JAX execution
    logits_jax, _ = jax_model.apply(
        {"params": jax_weights["params"]},
        tokens=tokens_jax,
        start_pos=start_pos,
        kv_cache=kv_cache_jax,
    )

    # PyTorch execution
    logits_torch = torch_model.forward_inference(tokens_torch, start_pos)

    # Final Comparison
    np.testing.assert_allclose(
        logits_jax, logits_torch.cpu().detach().numpy(), rtol=1e-2, atol=1e-2
    )
