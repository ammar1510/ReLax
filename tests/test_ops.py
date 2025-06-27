import numpy as np
import torch
import jax
import jax.numpy as jnp
from utils.ops import precompute_freqs_cis as precompute_freqs_cis_jax
from experiments.torch_llama import precompute_freqs_cis as precompute_freqs_cis_torch
from utils.ops import apply_rotary_emb as apply_rotary_emb_jax
from experiments.torch_llama import apply_rotary_emb as apply_rotary_emb_torch
from utils.ops import repeat_kv as repeat_kv_jax
from experiments.torch_llama import repeat_kv as repeat_kv_torch
from utils.ops import rms_norm as rms_norm_jax
from experiments.torch_llama import RMSNorm as RMSNorm_torch
from experiments.torch_llama import ModelArgs, Attention as Attention_torch, KVCache as KVCache_torch, FeedForward as FeedForward_torch
from utils.ops import AttentionParams, grouped_query_attention, FeedForwardParams, feed_forward as feed_forward_jax
from utils.kvcache import KVCache as KVCache_jax
from utils.ops import apply_scaling as apply_scaling_jax
from experiments.torch_llama import apply_scaling as apply_scaling_torch
import math

jax.config.update("jax_default_matmul_precision", "float32")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_precompute_freqs_cis():
    # Parameters for the test
    dim = 128
    end = 1024
    theta = 500000.0
    dtype = np.float32
    # JAX implementation
    freqs_cis_jax = precompute_freqs_cis_jax(dim, end, theta, dtype=dtype)

    # PyTorch implementation
    freqs_cis_torch_tensor = precompute_freqs_cis_torch(dim, end, theta)
    freqs_cis_torch_np = freqs_cis_torch_tensor.detach().cpu().numpy()

    # Compare the results
    np.testing.assert_allclose(np.array(freqs_cis_jax), freqs_cis_torch_np, rtol=5e-4, atol=1e-4)

def test_apply_rotary_emb():
    # Parameters
    bsz = 2
    seqlen = 128
    n_heads = 4
    head_dim = 64
    theta = 500000.0
    dtype = np.float32

    # Create dummy input tensor
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, n_heads, head_dim).astype(dtype)
    
    # JAX input
    x_jax = jnp.array(x_np)

    # PyTorch input
    x_torch = torch.tensor(x_np, device=device)

    # Precompute freqs_cis
    freqs_cis_jax = precompute_freqs_cis_jax(head_dim, seqlen, theta)
    freqs_cis_torch = precompute_freqs_cis_torch(head_dim, seqlen, theta)

    # Apply rotary embeddings
    output_jax = apply_rotary_emb_jax(x_jax, freqs_cis_jax)
    output_torch = apply_rotary_emb_torch(x_torch, freqs_cis_torch)

    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=5e-3, atol=1e-5)

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
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=1e-5)

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
    x_torch = torch.tensor(x_np, device=device)
    weight_torch = torch.tensor(weight_np, device=device)
    
    rms_norm_torch_module = RMSNorm_torch(dim, eps=eps)
    rms_norm_torch_module.weight = torch.nn.Parameter(weight_torch)
    output_torch = rms_norm_torch_module(x_torch)

    # Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

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
    freqs_cis_jax = jnp.array(freqs_cis_torch.detach().cpu().numpy())

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
    x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
    torch_attention = Attention_torch(model_args)
    torch_attention.wq.weight = torch.nn.Parameter(torch.tensor(wq_np.T, device=device))
    torch_attention.wk.weight = torch.nn.Parameter(torch.tensor(wk_np.T, device=device))
    torch_attention.wv.weight = torch.nn.Parameter(torch.tensor(wv_np.T, device=device))
    torch_attention.wo.weight = torch.nn.Parameter(torch.tensor(wo_np.T, device=device))

    kv_cache_torch = KVCache_torch(bsz, max_seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device)
    kv_cache_torch.cache_k[:, :prefill_len, :, :] = torch.tensor(prefill_k_np[0])
    kv_cache_torch.cache_v[:, :prefill_len, :, :] = torch.tensor(prefill_v_np[0])
    torch_attention.cache = kv_cache_torch

    # Verify input caches are identical before the forward pass
    np.testing.assert_allclose(np.array(kv_cache_jax_initial.k[0]), kv_cache_torch.cache_k.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(kv_cache_jax_initial.v[0]), kv_cache_torch.cache_v.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    # 5. Execute
    # Create the causal mask for the PyTorch version to match the JAX implementation
    mask = torch.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=torch.float32, device=device)
    mask[:, :, :, :start_pos] = 0 # Allow attention to all KV cache entries
    causal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
    mask[:, :, :, start_pos:] = causal_mask

    # JAX
    # We don't need the updated cache for this test, so we can ignore it.
    output_jax, updated_kv_cache_jax = grouped_query_attention(
        x_jax, freqs_cis_jax, jax_params, kv_cache_jax_initial, 0, start_pos
    )
    
    # PyTorch
    freqs_cis_torch_sliced = freqs_cis_torch[start_pos : start_pos + seqlen]
    output_torch = torch_attention.forward(x_torch, start_pos, freqs_cis_torch_sliced, mask)

    # 6. Compare output tensors
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=5e-3, atol=5e-3)
    
    # 7. Compare KV caches
    updated_k_jax = updated_kv_cache_jax.k[0]
    updated_v_jax = updated_kv_cache_jax.v[0]

    updated_k_torch = torch_attention.cache.cache_k
    updated_v_torch = torch_attention.cache.cache_v

    np.testing.assert_allclose(np.array(updated_k_jax), updated_k_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.array(updated_v_jax), updated_v_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-4)

def test_attention_with_padding():
    # 1. Setup Parameters for prefill with padding
    bsz = 2
    seqlen = 64 # Prefill length
    dim = 128
    n_heads = 4
    n_kv_heads = 2
    head_dim = dim // n_heads
    start_pos = 0 # Testing prefill
    dtype = np.float32
    max_seq_len = 256

    model_args = ModelArgs(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, flash=False)

    # 2. Create shared weights and inputs with padding
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, dim).astype(dtype)
    
    # Create a padding mask. Pad the second half of the sequence for the first batch element.
    prefill_mask_np = np.ones((bsz, seqlen), dtype=np.bool_)
    padding_start = seqlen // 2
    prefill_mask_np[0, padding_start:] = 0
    # Apply mask to input to zero out padding, mimicking tokenizer behavior
    x_np[0, padding_start:] = 0

    freqs_cis_torch = precompute_freqs_cis_torch(head_dim, max_seq_len)
    freqs_cis_jax = jnp.array(freqs_cis_torch.detach().cpu().numpy())

    wq_np = np.random.randn(dim, n_heads * head_dim).astype(dtype)
    wk_np = np.random.randn(dim, n_kv_heads * head_dim).astype(dtype)
    wv_np = np.random.randn(dim, n_kv_heads * head_dim).astype(dtype)
    wo_np = np.random.randn(n_heads * head_dim, dim).astype(dtype)

    # 3. JAX setup
    x_jax = jnp.array(x_np)
    prefill_mask_jax = jnp.array(prefill_mask_np)
    jax_params = AttentionParams(
        wq=jnp.array(wq_np).reshape(dim, n_heads, head_dim),
        wk=jnp.array(wk_np).reshape(dim, n_kv_heads, head_dim),
        wv=jnp.array(wv_np).reshape(dim, n_kv_heads, head_dim),
        wo=jnp.array(wo_np)
    )
    kv_cache_jax_initial = KVCache_jax.new(1, bsz, max_seq_len, n_kv_heads, head_dim, dtype=jnp.float32)
    
    # 4. PyTorch setup
    x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
    torch_attention = Attention_torch(model_args)
    torch_attention.wq.weight = torch.nn.Parameter(torch.tensor(wq_np.T, device=device))
    torch_attention.wk.weight = torch.nn.Parameter(torch.tensor(wk_np.T, device=device))
    torch_attention.wv.weight = torch.nn.Parameter(torch.tensor(wv_np.T, device=device))
    torch_attention.wo.weight = torch.nn.Parameter(torch.tensor(wo_np.T, device=device))
    kv_cache_torch = KVCache_torch(bsz, max_seq_len, n_kv_heads, head_dim, dtype=torch.float32, device=device)
    torch_attention.cache = kv_cache_torch
    
    # 5. Execute
    # JAX
    output_jax, updated_kv_cache_jax = grouped_query_attention(
        x_jax, freqs_cis_jax, jax_params, kv_cache_jax_initial, 0, start_pos, prefill_mask=prefill_mask_jax
    )

    # PyTorch
    # Create the equivalent mask for PyTorch to match JAX's internal masking
    torch_prefill_mask = torch.tensor(prefill_mask_np, device=device)
    causal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device), diagonal=1)
    
    # Combine causal mask with the padding mask for queries
    # The mask needs to be (bsz, 1, seqlen, seqlen) to be broadcastable
    # JAX logic: mask = causal_mask & query_mask
    # query_mask comes from prefill_mask and is (bsz, seqlen, 1)
    # This combination correctly masks rows (queries) for padded tokens
    torch_mask = causal_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seqlen, seqlen)
    query_mask_torch = torch_prefill_mask.unsqueeze(1).unsqueeze(-1) # (bsz, 1, seqlen, 1)
    final_torch_mask = torch.where(query_mask_torch, torch_mask, float("-inf"))

    freqs_cis_torch_sliced = freqs_cis_torch[start_pos : start_pos + seqlen]
    output_torch = torch_attention.forward(x_torch, start_pos, freqs_cis_torch_sliced, final_torch_mask)

    # Manually zero out the output for padded tokens in the torch output to match the JAX behavior
    output_torch = torch.where(torch_prefill_mask.unsqueeze(-1), output_torch, 0.0)

    # 6. Compare output tensors
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=1e-5, atol=5e-3)

    # 7. Compare KV caches
    # JAX zeros out KV cache entries for padded tokens before caching
    updated_k_jax = updated_kv_cache_jax.k[0]
    updated_v_jax = updated_kv_cache_jax.v[0]

    # The reference torch implementation doesn't have the pre-caching masking, so we apply it manually for comparison
    updated_k_torch = torch_attention.cache.cache_k
    updated_v_torch = torch_attention.cache.cache_v
    
    # Manually mask the torch cache to match JAX's behavior
    prefill_mask_torch_expanded = torch_prefill_mask[:, :seqlen, None, None]
    updated_k_torch[:, :seqlen] = updated_k_torch[:, :seqlen] * prefill_mask_torch_expanded
    updated_v_torch[:, :seqlen] = updated_v_torch[:, :seqlen] * prefill_mask_torch_expanded

    np.testing.assert_allclose(np.array(updated_k_jax), updated_k_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.array(updated_v_jax), updated_v_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-4)

def test_apply_scaling():
    # Parameters
    head_dim = 128
    theta = 500000.0
    dtype = np.float32

    # Generate frequencies like in precompute_freqs_cis
    freqs_np = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(dtype) / head_dim))
    
    # JAX and PyTorch inputs
    freqs_jax = jnp.array(freqs_np)
    freqs_torch = torch.tensor(freqs_np)

    # Get the reference output from the torch implementation
    output_torch_reference = apply_scaling_torch(freqs_torch)
    torch_result_np = output_torch_reference.detach().cpu().numpy()

    # 1. Test JAX implementation with default parameters
    output_jax_default = apply_scaling_jax(freqs_jax)
    np.testing.assert_allclose(np.array(output_jax_default), torch_result_np, rtol=1e-5, atol=1e-5)

    # 2. Test JAX implementation with explicit parameters that match the torch hardcoded values
    scale_factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192.0

    output_jax_explicit = apply_scaling_jax(
        freqs_jax,
        scale_factor=scale_factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len
    )
    
    np.testing.assert_allclose(np.array(output_jax_explicit), torch_result_np, rtol=1e-6, atol=1e-11)

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
    x_torch = torch.tensor(x_np, device=device)
    torch_ff = FeedForward_torch(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=None)
    torch_ff.w1.weight = torch.nn.Parameter(torch.tensor(w1_np.T, device=device))
    torch_ff.w3.weight = torch.nn.Parameter(torch.tensor(w3_np.T, device=device))
    torch_ff.w2.weight = torch.nn.Parameter(torch.tensor(w2_np.T, device=device))
    output_torch = torch_ff(x_torch)

    # 5. Compare
    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().cpu().numpy(), rtol=1e-4, atol=5e-3) 