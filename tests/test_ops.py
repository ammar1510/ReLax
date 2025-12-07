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
from experiments.torch_llama import (
    ModelArgs,
    Attention as Attention_torch,
    KVCache as KVCache_torch,
    FeedForward as FeedForward_torch,
)
from utils.ops import (
    AttentionParams,
    grouped_query_attention,
    FeedForwardParams,
    feed_forward as feed_forward_jax,
)
from utils.kvcache import KVCache as KVCache_jax
from utils.ops import apply_scaling as apply_scaling_jax
from experiments.torch_llama import apply_scaling as apply_scaling_torch
import math

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

jax_dtype = jnp.bfloat16
torch_dtype = torch.bfloat16


def test_precompute_freqs_cis():
    jax.config.update("jax_enable_x64", True)
    jax_dtype = jnp.float64
    torch_dtype = torch.float64
    #Loading with float64 since only precompute_freqs_cis is one time computation. 

    # Parameters for the test
    dim = 128
    end = 1024
    theta = 500000.0
    # JAX implementation
    freqs_cis_jax = precompute_freqs_cis_jax(dim, end, theta, dtype=jax_dtype)

    # PyTorch implementation
    freqs_cis_torch = precompute_freqs_cis_torch(128, 1024, 500000.0).to(dtype=torch_dtype)

    #limits on precision are much softer for precompute_freqs_cis
    np.testing.assert_allclose(freqs_cis_jax, freqs_cis_torch.numpy(), rtol=8e-4, atol=1e-4)
    jax.config.update("jax_enable_x64", False)


def test_apply_rotary_emb():
    # Parameters
    jax.config.update("jax_enable_x64", True)

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
    x_jax = jnp.array(x_np, dtype=jax_dtype)

    # PyTorch input
    x_torch = torch.tensor(x_np, device=device, dtype=torch_dtype)

    # Precompute freqs_cis
    freqs_cis_jax = precompute_freqs_cis_jax(head_dim, seqlen, theta,jnp.float64)
    freqs_cis_torch = torch.from_numpy(np.array(freqs_cis_jax)).to(device=device, dtype=torch_dtype)
    freqs_cis_jax = freqs_cis_jax.astype(jax_dtype)

    # Apply rotary embeddings
    output_jax = apply_rotary_emb_jax(x_jax, freqs_cis_jax)
    output_torch = apply_rotary_emb_torch(x_torch, freqs_cis_torch)

    # Compare
    np.testing.assert_allclose(
        np.array(output_jax), output_torch.float().detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    jax.config.update("jax_enable_x64", False)


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
    x_jax = jnp.array(x_np, dtype=jax_dtype)

    # PyTorch input
    x_torch = torch.tensor(x_np, device=device, dtype=torch_dtype)

    # JAX implementation
    output_jax = repeat_kv_jax(x_jax, n_rep)

    # PyTorch implementation
    output_torch = repeat_kv_torch(x_torch, n_rep)

    # Compare
    np.testing.assert_allclose(
        np.array(output_jax), output_torch.float().detach().cpu().numpy(), rtol=1e-5
    )


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
    x_jax = jnp.array(x_np, dtype=jax_dtype)
    weight_jax = jnp.array(weight_np, dtype=jax_dtype)
    output_jax = rms_norm_jax(x_jax, weight_jax, eps)

    # PyTorch implementation
    x_torch = torch.tensor(x_np, device=device, dtype=torch_dtype)
    weight_torch = torch.tensor(weight_np, device=device, dtype=torch_dtype)

    rms_norm_torch_module = RMSNorm_torch(dim, eps=eps)
    rms_norm_torch_module.weight = torch.nn.Parameter(weight_torch)
    output_torch = rms_norm_torch_module(x_torch)

    # Compare
    np.testing.assert_allclose(
        np.array(output_jax), output_torch.float().detach().cpu().numpy(), rtol=1e-2, atol=1e-4
    )


def test_attention():
    """Basic test for grouped_query_attention with PyTorch comparison."""
    # 1. Setup Parameters
    bsz = 4
    seqlen = 128  # Small sequence for simplicity
    dim = 2048
    n_heads = 16
    n_kv_heads = 4
    head_dim = dim // n_heads
    cache_seq_len = 64  # Sequence length up to which cache is pre-filled
    start_pos = cache_seq_len  # Start from cache_seq_len to perform attention beyond it
    dtype = np.float32
    max_seq_len = 1024
    n_layers = 1

    model_args = ModelArgs(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, flash=False)

    # 2. Create inputs and parameters
    np.random.seed(42)  # Fixed seed for reproducibility
    x_np = np.random.randn(bsz, seqlen, dim).astype(dtype)
    
    # Create weight matrices
    wq_np = np.random.normal(0, 0.02, (dim, n_heads * head_dim)).astype(dtype)
    wk_np = np.random.normal(0, 0.02, (dim, n_kv_heads * head_dim)).astype(dtype)
    wv_np = np.random.normal(0, 0.02, (dim, n_kv_heads * head_dim)).astype(dtype)
    wo_np = np.random.normal(0, 0.02, (n_heads * head_dim, dim)).astype(dtype)

    # Precompute frequencies
    freqs_cis_torch = precompute_freqs_cis_torch(head_dim, max_seq_len).to(dtype=torch_dtype)
    freqs_cis_jax = jnp.array(freqs_cis_torch.float().detach().cpu().numpy(), dtype=jax_dtype)

    # 3. JAX setup
    x_jax = jnp.array(x_np, dtype=jax_dtype)
    jax_params = AttentionParams(
        wq=jnp.array(wq_np, dtype=jax_dtype).reshape(dim, n_heads, head_dim),
        wk=jnp.array(wk_np, dtype=jax_dtype).reshape(dim, n_kv_heads, head_dim),
        wv=jnp.array(wv_np, dtype=jax_dtype).reshape(dim, n_kv_heads, head_dim),
        wo=jnp.array(wo_np, dtype=jax_dtype),
    )
    
    # Initialize KV cache with randomly initialized values up to cache_seq_len
    kv_cache_jax = KVCache_jax.new(
        n_layers, bsz, max_seq_len, n_kv_heads, head_dim, dtype=jax_dtype
    )
    # Pre-fill cache with random values
    prefill_k = np.random.randn(n_layers, bsz, cache_seq_len, n_kv_heads, head_dim).astype(dtype)
    prefill_v = np.random.randn(n_layers, bsz, cache_seq_len, n_kv_heads, head_dim).astype(dtype)
    k_init_jax = jnp.zeros((n_layers, bsz, max_seq_len, n_kv_heads, head_dim), dtype=jax_dtype)
    v_init_jax = jnp.zeros((n_layers, bsz, max_seq_len, n_kv_heads, head_dim), dtype=jax_dtype)
    k_updated_jax = k_init_jax.at[:, :, :cache_seq_len, :, :].set(jnp.array(prefill_k, dtype=jax_dtype))
    v_updated_jax = v_init_jax.at[:, :, :cache_seq_len, :, :].set(jnp.array(prefill_v, dtype=jax_dtype))
    kv_cache_jax = KVCache_jax(
        k=k_updated_jax, 
        v=v_updated_jax, 
        positions=jnp.full((bsz,), cache_seq_len, dtype=jnp.int32)
    )

    # 4. PyTorch setup
    x_torch = torch.tensor(x_np, device=device, dtype=torch_dtype)
    torch_attention = Attention_torch(model_args)
    torch_attention.wq.weight = torch.nn.Parameter(torch.tensor(wq_np.T, device=device, dtype=torch_dtype))
    torch_attention.wk.weight = torch.nn.Parameter(torch.tensor(wk_np.T, device=device, dtype=torch_dtype))
    torch_attention.wv.weight = torch.nn.Parameter(torch.tensor(wv_np.T, device=device, dtype=torch_dtype))
    torch_attention.wo.weight = torch.nn.Parameter(torch.tensor(wo_np.T, device=device, dtype=torch_dtype))
    kv_cache_torch = KVCache_torch(
        bsz, max_seq_len, n_kv_heads, head_dim, dtype=torch_dtype, device=device
    )
    # Pre-fill PyTorch cache with same random values
    kv_cache_torch.cache_k[:, :cache_seq_len, :, :] = torch.tensor(prefill_k[0], device=device, dtype=torch_dtype)
    kv_cache_torch.cache_v[:, :cache_seq_len, :, :] = torch.tensor(prefill_v[0], device=device, dtype=torch_dtype)
    torch_attention.cache = kv_cache_torch

    # 5. Execute JAX attention
    seq_lengths = jnp.full((bsz,), seqlen, dtype=jnp.int32)  # All sequences have same length (no padding)
    output_jax, updated_kv_cache_jax = grouped_query_attention(
        x_jax, freqs_cis_jax, jax_params, kv_cache_jax, start_pos, seq_lengths
    )

    # 6. Execute PyTorch attention
    freqs_cis_torch_sliced = freqs_cis_torch[start_pos : start_pos + seqlen].to(device=device)
    
    # Create causal mask for PyTorch (JAX handles this internally)
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=device), mask]
        ).type_as(x_torch)
    
    output_torch = torch_attention.forward(
        x_torch, start_pos, freqs_cis_torch_sliced, mask
    )

    # 7. Basic sanity checks
    # Check output shape
    assert output_jax.shape == (bsz, seqlen, dim), f"Expected shape {(bsz, seqlen, dim)}, got {output_jax.shape}"
    assert output_torch.shape == (bsz, seqlen, dim), f"Expected shape {(bsz, seqlen, dim)}, got {output_torch.shape}"
    
    # Check that output is not all zeros or NaN
    assert not jnp.all(output_jax == 0), "JAX output should not be all zeros"
    assert not jnp.any(jnp.isnan(output_jax)), "JAX output should not contain NaN values"
    assert not torch.all(output_torch == 0), "PyTorch output should not be all zeros"
    assert not torch.any(torch.isnan(output_torch)), "PyTorch output should not contain NaN values"
    
    # Check KV cache was updated correctly
    updated_k_jax = updated_kv_cache_jax.k[0]  # Layer 0
    updated_v_jax = updated_kv_cache_jax.v[0]  # Layer 0
    
    # Pre-filled cache should still have values at positions 0:cache_seq_len
    assert not jnp.all(updated_k_jax[:, :cache_seq_len, :, :] == 0), "JAX pre-filled keys should remain in cache"
    assert not jnp.all(updated_v_jax[:, :cache_seq_len, :, :] == 0), "JAX pre-filled values should remain in cache"
    
    # New keys and values should be stored at positions cache_seq_len:cache_seq_len+seqlen
    assert not jnp.all(updated_k_jax[:, cache_seq_len:cache_seq_len+seqlen, :, :] == 0), "JAX new keys should be cached"
    assert not jnp.all(updated_v_jax[:, cache_seq_len:cache_seq_len+seqlen, :, :] == 0), "JAX new values should be cached"
    
    # Positions beyond cache_seq_len+seqlen should still be zero (empty)
    if cache_seq_len + seqlen < max_seq_len:
        assert jnp.all(updated_k_jax[:, cache_seq_len+seqlen:, :, :] == 0), "JAX keys beyond cache_seq_len+seqlen should be zero"
        assert jnp.all(updated_v_jax[:, cache_seq_len+seqlen:, :, :] == 0), "JAX values beyond cache_seq_len+seqlen should be zero"
    
    # 8. Compare outputs between JAX and PyTorch
    np.testing.assert_allclose(
        np.array(output_jax), output_torch.float().detach().cpu().numpy(), rtol=1e-3, atol=1e-3
    )
    
    # 9. Compare KV caches
    updated_k_torch = torch_attention.cache.cache_k
    updated_v_torch = torch_attention.cache.cache_v
    
    np.testing.assert_allclose(
        np.array(updated_k_jax),
        updated_k_torch.float().detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(updated_v_jax),
        updated_v_torch.float().detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-4,
    )
    
    print("✓ Basic attention test with PyTorch comparison passed!")


def test_apply_scaling():
    # Parameters
    head_dim = 128
    theta = 500000.0
    dtype = np.float32

    # Generate frequencies like in precompute_freqs_cis
    freqs_np = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(dtype) / head_dim))

    # JAX and PyTorch inputs
    freqs_jax = jnp.array(freqs_np, dtype=jax_dtype)
    freqs_torch = torch.tensor(freqs_np, dtype=torch_dtype)

    # Get the reference output from the torch implementation
    output_torch_reference = apply_scaling_torch(freqs_torch)
    torch_result_np = output_torch_reference.float().detach().cpu().numpy()  # Convert to float32 first

    # Test JAX implementation with explicit parameters that match the torch hardcoded values
    scale_factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192.0

    output_jax_explicit = apply_scaling_jax(
        freqs_jax,
        scale_factor=scale_factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
    )

    np.testing.assert_allclose(
        np.array(output_jax_explicit), torch_result_np, rtol=1e-6, atol=1e-11
    )


#Only Test with both on TPU or both on GPU. Observe precision differences with mixed devices.
#Need to initialize with weight initialization, otherwise the activations explode and tests fail (diff devices).
#For exact precision, run both on xPU.
def test_feed_forward():
    # 1. Setup Parameters
    bsz = 4
    seqlen = 64
    dim = 1536
    multiple_of = 32
    dtype = np.float32

    # Mimic hidden_dim calculation from PyTorch implementation
    hidden_dim =  (dim//3)*8

    # 2. Create shared weights and inputs
    np.random.seed(0)
    x_np = np.random.randn(bsz, seqlen, dim).astype(dtype)

    # PyTorch names weights w1, w2, w3. JAX uses w_gate, w_up, w_down.
    # Mapping: torch.w1 -> jax.w_gate, torch.w3 -> jax.w_up, torch.w2 -> jax.w_down
    w1_np = np.random.normal(0, 0.02, (dim, hidden_dim)).astype(dtype)  # gate_proj
    w3_np = np.random.normal(0, 0.02, (dim, hidden_dim)).astype(dtype)  # up_proj
    w2_np = np.random.normal(0, 0.02, (hidden_dim, dim)).astype(dtype)  # down_proj

    # 3. JAX setup
    x_jax = jnp.array(x_np, dtype=jax_dtype)
    jax_params = FeedForwardParams(
        w_gate=jnp.array(w1_np, dtype=jax_dtype), w_up=jnp.array(w3_np, dtype=jax_dtype), w_down=jnp.array(w2_np, dtype=jax_dtype)
    )
    output_jax = feed_forward_jax(x_jax, jax_params,"silu")

    # 4. PyTorch setup
    x_torch = torch.tensor(x_np, device=device, dtype=torch_dtype)
    torch_ff = FeedForward_torch(
        dim=dim, hidden_dim=hidden_dim, multiple_of=multiple_of, ffn_dim_multiplier=None
    )
    torch_ff.w1.weight = torch.nn.Parameter(torch.tensor(w1_np.T, device=device, dtype=torch_dtype))
    torch_ff.w3.weight = torch.nn.Parameter(torch.tensor(w3_np.T, device=device, dtype=torch_dtype))
    torch_ff.w2.weight = torch.nn.Parameter(torch.tensor(w2_np.T, device=device, dtype=torch_dtype))
    output_torch = torch_ff(x_torch)

    # 5. Compare
    np.testing.assert_allclose(
        np.array(output_jax), output_torch.float().detach().cpu().numpy(), rtol=1e-2, atol=1e-3
    )
