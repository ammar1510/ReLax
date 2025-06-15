import numpy as np
import torch
import jax.numpy as jnp
from utils.ops import precompute_freqs_cis as precompute_freqs_cis_jax
from tests.torch_ops import precompute_freqs_cis as precompute_freqs_cis_torch
from utils.ops import apply_rotary_emb as apply_rotary_emb_jax
from tests.torch_ops import apply_rotary_emb as apply_rotary_emb_torch
from utils.ops import repeat_kv as repeat_kv_jax
from tests.torch_ops import repeat_kv as repeat_kv_torch
from utils.ops import rms_norm as rms_norm_jax
from tests.torch_ops import RMSNorm as RMSNorm_torch
from tests.torch_ops import ModelArgs, Attention as Attention_torch, KVCache as KVCache_torch, FeedForward as FeedForward_torch
from utils.ops import AttentionParams, grouped_query_attention, FeedForwardParams, feed_forward as feed_forward_jax
from utils.kvcache import KVCache as KVCache_jax

def test_precompute_freqs_cis():
    # Parameters for the test
    dim = 128
    end = 1024
    theta = 10000.0

    # JAX implementation
    freqs_cis_jax = precompute_freqs_cis_jax(dim, end, theta)

    # PyTorch implementation
    freqs_cis_torch_tensor = precompute_freqs_cis_torch(dim, end, theta)
    freqs_cis_torch_np = freqs_cis_torch_tensor.numpy()

    # Compare the results
    np.testing.assert_allclose(np.array(freqs_cis_jax), freqs_cis_torch_np, rtol=1e-5)

def test_apply_rotary_emb():
    # Parameters
    bsz = 2
    seqlen = 128
    n_heads = 4
    head_dim = 64
    theta = 10000.0
    dtype = np.float64

    # Create dummy input tensor
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, n_heads, head_dim).astype(dtype)
    
    # JAX input
    x_jax = jnp.array(x_np)

    # PyTorch input
    x_torch = torch.tensor(x_np)

    # Precompute freqs_cis
    freqs_cis_jax = precompute_freqs_cis_jax(head_dim, seqlen, theta)
    freqs_cis_torch = precompute_freqs_cis_torch(head_dim, seqlen, theta)

    # Apply rotary embeddings
    output_jax = apply_rotary_emb_jax(x_jax, freqs_cis_jax)
    output_torch = apply_rotary_emb_torch(x_torch, freqs_cis_torch)

    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.numpy(), rtol=1e-5, atol=1e-5)

def test_repeat_kv():
    # Parameters
    bs = 2
    slen = 64
    n_kv_heads = 4
    head_dim = 32
    n_rep = 2
    dtype = np.float32

    # Create dummy input tensor
    np.random.seed(0)
    x_np = np.random.randn(bs, slen, n_kv_heads, head_dim).astype(dtype)

    # JAX input
    x_jax = jnp.array(x_np)

    # PyTorch input
    x_torch = torch.tensor(x_np)

    # JAX implementation
    output_jax = repeat_kv_jax(x_jax, n_rep)

    # PyTorch implementation
    output_torch = repeat_kv_torch(x_torch, n_rep)

    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.numpy(), rtol=1e-5)

def test_rms_norm():
    # Parameters
    dim = 128
    eps = 1e-6
    dtype = np.float32

    # Create dummy input tensor and weight
    np.random.seed(0)
    x_np = np.random.randn(2, 16, dim).astype(dtype)
    weight_np = np.random.randn(dim).astype(dtype)

    # JAX implementation
    x_jax = jnp.array(x_np)
    weight_jax = jnp.array(weight_np)
    output_jax = rms_norm_jax(x_jax, weight_jax, eps)

    # PyTorch implementation
    x_torch = torch.tensor(x_np)
    weight_torch = torch.tensor(weight_np)
    
    rms_norm_torch_module = RMSNorm_torch(dim, eps=eps)
    rms_norm_torch_module.weight = torch.nn.Parameter(weight_torch)
    output_torch = rms_norm_torch_module(x_torch)

    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().numpy(), rtol=1e-5, atol=1e-5)

def test_attention():
    # 1. Setup Parameters
    bsz = 2
    seqlen = 64
    dim = 128
    n_heads = 4
    n_kv_heads = 2
    head_dim = dim // n_heads
    start_pos = 10
    dtype = np.float32
    max_seq_len = 256

    model_args = ModelArgs(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, flash=False)

    # 2. Create shared weights and inputs
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, dim).astype(dtype)
    freqs_cis_torch = precompute_freqs_cis_torch(head_dim, max_seq_len)
    freqs_cis_jax = jnp.array(freqs_cis_torch.numpy())

    wq_np = np.random.randn(dim, n_heads * head_dim).astype(dtype)
    wk_np = np.random.randn(dim, n_kv_heads * head_dim).astype(dtype)
    wv_np = np.random.randn(dim, n_kv_heads * head_dim).astype(dtype)
    wo_np = np.random.randn(n_heads * head_dim, dim).astype(dtype)

    # 3. JAX setup
    x_jax = jnp.array(x_np)
    jax_params = AttentionParams(
        wq=jnp.array(wq_np).reshape(dim, n_heads, head_dim),
        wk=jnp.array(wk_np).reshape(dim, n_kv_heads, head_dim),
        wv=jnp.array(wv_np).reshape(dim, n_kv_heads, head_dim),
        wo=jnp.array(wo_np)
    )
    kv_cache_jax_initial = KVCache_jax.new(1, bsz, max_seq_len, n_kv_heads, head_dim, dtype=jnp.float32)
    
    # Prefill cache for JAX
    prefill_len = start_pos
    prefill_k_np = np.random.randn(1, bsz, prefill_len, n_kv_heads, head_dim).astype(dtype)
    prefill_v_np = np.random.randn(1, bsz, prefill_len, n_kv_heads, head_dim).astype(dtype)

    k_updated = kv_cache_jax_initial.k.at[0, :, :prefill_len, :, :].set(jnp.array(prefill_k_np[0]))
    v_updated = kv_cache_jax_initial.v.at[0, :, :prefill_len, :, :].set(jnp.array(prefill_v_np[0]))
    kv_cache_jax_initial = KVCache_jax(k=k_updated, v=v_updated)

    # 4. PyTorch setup
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    torch_attention = Attention_torch(model_args)
    torch_attention.wq.weight = torch.nn.Parameter(torch.tensor(wq_np.T))
    torch_attention.wk.weight = torch.nn.Parameter(torch.tensor(wk_np.T))
    torch_attention.wv.weight = torch.nn.Parameter(torch.tensor(wv_np.T))
    torch_attention.wo.weight = torch.nn.Parameter(torch.tensor(wo_np.T))

    kv_cache_torch = KVCache_torch(bsz, max_seq_len, n_kv_heads, head_dim, dtype=torch.float32, device='cpu')
    kv_cache_torch.cache_k[:, :prefill_len, :, :] = torch.tensor(prefill_k_np[0])
    kv_cache_torch.cache_v[:, :prefill_len, :, :] = torch.tensor(prefill_v_np[0])
    torch_attention.cache = kv_cache_torch

    # Verify input caches are identical before the forward pass
    np.testing.assert_allclose(np.array(kv_cache_jax_initial.k[0]), kv_cache_torch.cache_k.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(kv_cache_jax_initial.v[0]), kv_cache_torch.cache_v.numpy(), rtol=1e-5, atol=1e-5)

    # 5. Execute
    # Create the causal mask for the PyTorch version to match the JAX implementation
    mask = torch.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=torch.float32)
    mask[:, :, :, :start_pos] = 0 # Allow attention to all KV cache entries
    causal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf")), diagonal=1)
    mask[:, :, :, start_pos:] = causal_mask

    # JAX
    # We don't need the updated cache for this test, so we can ignore it.
    output_jax, updated_kv_cache_jax = grouped_query_attention(
        x_jax, freqs_cis_jax, jax_params, kv_cache_jax_initial, 0, start_pos, n_heads, n_kv_heads
    )
    
    # PyTorch
    freqs_cis_torch_sliced = freqs_cis_torch[start_pos : start_pos + seqlen]
    output_torch = torch_attention.forward(x_torch, start_pos, freqs_cis_torch_sliced, mask)

    # 6. Compare output tensors
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().numpy(), rtol=1e-5, atol=1e-4)
    
    # 7. Compare KV caches
    updated_k_jax = updated_kv_cache_jax.k[0]
    updated_v_jax = updated_kv_cache_jax.v[0]

    updated_k_torch = torch_attention.cache.cache_k
    updated_v_torch = torch_attention.cache.cache_v

    np.testing.assert_allclose(np.array(updated_k_jax), updated_k_torch.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(updated_v_jax), updated_v_torch.detach().numpy(), rtol=1e-5, atol=1e-5)

def test_feed_forward():
    # 1. Setup Parameters
    bsz = 2
    seqlen = 64
    dim = 128
    multiple_of = 32
    dtype = np.float32

    # Mimic hidden_dim calculation from PyTorch implementation
    hidden_dim = int(2 * (4 * dim) / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # 2. Create shared weights and inputs
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, dim).astype(dtype)

    # PyTorch names weights w1, w2, w3. JAX uses w1_gate, w2_up, w3_down.
    # Mapping: torch.w1 -> jax.w1_gate, torch.w3 -> jax.w2_up, torch.w2 -> jax.w3_down
    w1_np = np.random.randn(dim, hidden_dim).astype(dtype) # gate_proj
    w3_np = np.random.randn(dim, hidden_dim).astype(dtype) # up_proj
    w2_np = np.random.randn(hidden_dim, dim).astype(dtype) # down_proj

    # 3. JAX setup
    x_jax = jnp.array(x_np)
    jax_params = FeedForwardParams(
        w1_gate=jnp.array(w1_np),
        w2_up=jnp.array(w3_np),
        w3_down=jnp.array(w2_np)
    )
    output_jax = feed_forward_jax(x_jax, jax_params, activation_fn='silu')

    # 4. PyTorch setup
    x_torch = torch.tensor(x_np)
    torch_ff = FeedForward_torch(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=None)
    torch_ff.w1.weight = torch.nn.Parameter(torch.tensor(w1_np.T))
    torch_ff.w3.weight = torch.nn.Parameter(torch.tensor(w3_np.T))
    torch_ff.w2.weight = torch.nn.Parameter(torch.tensor(w2_np.T))
    output_torch = torch_ff(x_torch)

    # 5. Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().numpy(), rtol=1e-4, atol=1e-3) 