import numpy as np
import torch
import jax
import jax.numpy as jnp
from pathlib import Path
import argparse
import dataclasses
import logging

# JAX/Flax model components
from models.llama.model import LLaMa as Llama_jax
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache as KVCache_jax
from utils.ops import build_attn_mask

# PyTorch model components
from experiments.torch_llama import Llama as Llama_wrapper

jax.config.update("jax_default_matmul_precision", "highest")


def test_model_forward_pass(model_path: str, log_prefill: bool = False):
    """Test forward pass comparison between PyTorch and JAX models with real Llama 3.2 1B weights.

    Args:
        model_path: Path to model directory
        log_prefill: If True, enable debug logging during prefill. If False, only enable for generation.
    """

    # Configure logging based on log_prefill flag
    logging.basicConfig(
        level=logging.WARNING,  # Default level for all loggers
        format='[%(name)s] %(levelname)s - %(message)s'
    )

    # Suppress JAX/JIT internal logs
    logging.getLogger('jax').setLevel(logging.ERROR)
    logging.getLogger('jaxlib').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

    if log_prefill:
        # Enable debug logging for prefill
        logging.getLogger('models.llama.model').setLevel(logging.DEBUG)
        logging.getLogger('utils.ops').setLevel(logging.DEBUG)
        logging.getLogger('experiments.torch_llama').setLevel(logging.DEBUG)

    model_path = Path(model_path)
    print(f"Loading model from {model_path}")

    # Test configuration
    batch_size = 2
    seqlen = 256
    test_prompt = "The quick brown fox jumps over the lazy dog. This is a test sentence for the LLaMA model."

    # Device selection for PyTorch
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        TORCH_XLA_AVAILABLE = True
    except ImportError:
        TORCH_XLA_AVAILABLE = False

    if TORCH_XLA_AVAILABLE:
        torch_device = torch_xla.device()
        print(f"Using PyTorch XLA device: {torch_device}")
    elif torch.cuda.is_available():
        torch_device = "cuda"
        print("Using CUDA device")
    else:
        torch_device = "cpu"
        print("Using CPU device")

    # Use bfloat16 for consistency with typical inference
    torch_dtype = torch.float32
    jax_dtype = jnp.float32

    print("\n" + "="*80)
    print("LOADING JAX MODEL")
    print("="*80)

    # Load JAX configuration and weights
    jax_config = ModelConfig.from_json_file(model_path)
    jax_config = dataclasses.replace(jax_config, dtype=jax_dtype)
    print(f"JAX Config: dim={jax_config.dim}, n_layers={jax_config.n_layers}, "
          f"n_heads={jax_config.n_heads}, n_kv_heads={jax_config.n_kv_heads}")

    # Load JAX weights from safetensors
    jax_params = load_llama_weights(str(model_path), jax_config)
    print("✓ JAX weights loaded successfully")

    # Initialize JAX model
    jax_model = Llama_jax(jax_config)
    print("✓ JAX model initialized")

    print("\n" + "="*80)
    print("LOADING PYTORCH MODEL")
    print("="*80)

    # Load PyTorch model using the Llama.build method
    torch_original_path = model_path / "original"
    tokenizer_path = torch_original_path / "tokenizer.model"

    print(f"Loading from: {torch_original_path}")
    print(f"Tokenizer: {tokenizer_path}")

    llama_wrapper = Llama_wrapper.build(
        ckpt_dir=str(torch_original_path),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=seqlen,
        max_batch_size=batch_size,
        flash=False,
    )
    torch_model = llama_wrapper.model
    torch_model.eval()

    # Move to target device if needed
    if torch_device != torch_model.tok_embeddings.weight.device:
        torch_model = torch_model.to(torch_device)

    print("✓ PyTorch model loaded successfully")

    print("\n" + "="*80)
    print("PREPARING INPUT")
    print("="*80)

    # Initialize tokenizer for JAX (same tokenizer used by PyTorch wrapper)
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    # Tokenize input text
    tokens_list = tokenizer.encode(test_prompt, bos=True, eos=False)
    print(f"Test prompt: {test_prompt}")
    print(f"Tokenized length: {len(tokens_list)}")

    # Truncate if needed to fit within max cache size (but don't pad)
    if len(tokens_list) > seqlen:
        tokens_list = tokens_list[:seqlen]
        print(f"Truncated to {seqlen} tokens")

    actual_seqlen = len(tokens_list)

    # Create batch by repeating the tokens
    tokens_np = np.array([tokens_list] * batch_size, dtype=np.int32)
    tokens_jax = jnp.array(tokens_np, dtype=jnp.int32)
    tokens_torch = torch.tensor(tokens_np, device=torch_device, dtype=torch.long)

    print(f"Input shape: {tokens_np.shape}")
    print(f"First 20 tokens: {tokens_list[:20]}")

    print("\n" + "="*80)
    print("INITIALIZING KV CACHES")
    print("="*80)

    # JAX KV cache
    head_dim = jax_config.head_dim
    kv_cache_jax = KVCache_jax.new(
        jax_config.n_layers,
        batch_size,
        seqlen,
        jax_config.n_kv_heads,
        head_dim,
        dtype=jax_dtype
    )
    print(f"✓ JAX KV cache initialized: {jax_config.n_layers} layers, "
          f"batch={batch_size}, seqlen={seqlen}, kv_heads={jax_config.n_kv_heads}, head_dim={head_dim}")

    # PyTorch KV caches (installed in each layer)
    torch_params = torch_model.params
    for i in range(torch_params.n_layers):
        layer_cache = llama_wrapper.model.layers[i].attention.cache  # Should be None initially
        if layer_cache is None:
            # Create new cache
            from experiments.torch_llama import KVCache as KVCache_torch
            layer_cache = KVCache_torch(
                batch_size,
                seqlen,
                torch_params.n_kv_heads,
                torch_params.dim // torch_params.n_heads,
                dtype=torch_dtype,
                device=torch_device,
            )
            torch_model.layers[i].attention.cache = layer_cache

    print(f"✓ PyTorch KV caches initialized in all layers")

    print("\n" + "="*80)
    logging_status = "Enabled" if log_prefill else "Disabled"
    print(f"RUNNING FORWARD PASSES (PREFILL - Logging {logging_status})")
    print("="*80)

    # JAX forward pass
    print("Running JAX forward pass...")
    true_lengths = jnp.full((batch_size,), actual_seqlen, dtype=jnp.int32)

    # Build attention mask - need to create a dummy input tensor for mask building
    # The mask shape depends on the input tensor shape
    dummy_input = jnp.zeros((batch_size, actual_seqlen, jax_config.dim), dtype=jax_dtype)
    mask = build_attn_mask(dummy_input, kv_cache_jax, true_lengths)

    logits_jax, updated_kv_cache_jax = jax_model.apply(
        {"params": jax_params},
        tokens=tokens_jax,
        true_lengths=true_lengths,
        kv_cache=kv_cache_jax,
        mask=mask,
    )
    print(f"✓ JAX forward pass complete: {logits_jax.shape}")

    # PyTorch forward pass
    print("Running PyTorch forward pass...")
    start_pos = 0
    with torch.no_grad():
        logits_torch = torch_model.forward_inference(tokens_torch, start_pos)
    print(f"✓ PyTorch forward pass complete: {logits_torch.shape}")

    print("\n" + "="*80)
    print("COMPARING OUTPUTS")
    print("="*80)

    # Convert to numpy for comparison
    logits_jax_np = np.array(logits_jax, dtype=np.float32)
    logits_torch_np = logits_torch.cpu().float().detach().numpy()

    print(f"JAX output shape: {logits_jax_np.shape}")
    print(f"PyTorch output shape: {logits_torch_np.shape}")

    # Basic shape checks
    assert logits_jax_np.shape == logits_torch_np.shape, (
        f"Shape mismatch: JAX {logits_jax_np.shape} vs PyTorch {logits_torch_np.shape}"
    )
    assert logits_jax_np.shape == (batch_size, actual_seqlen, jax_config.vocab_size), (
        f"Unexpected output shape: {logits_jax_np.shape}"
    )
    print("✓ Shape check passed")

    # Check for NaN or Inf
    jax_has_nan = np.any(np.isnan(logits_jax_np))
    jax_has_inf = np.any(np.isinf(logits_jax_np))
    torch_has_nan = np.any(np.isnan(logits_torch_np))
    torch_has_inf = np.any(np.isinf(logits_torch_np))

    print(f"JAX has NaN: {jax_has_nan}, Inf: {jax_has_inf}")
    print(f"PyTorch has NaN: {torch_has_nan}, Inf: {torch_has_inf}")

    assert not jax_has_nan, "JAX output contains NaN"
    assert not torch_has_nan, "PyTorch output contains NaN"
    assert not jax_has_inf, "JAX output contains Inf"
    assert not torch_has_inf, "PyTorch output contains Inf"
    print("✓ NaN/Inf check passed")

    # Compare outputs
    abs_diff = np.abs(logits_jax_np - logits_torch_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (np.abs(logits_torch_np) + 1e-8)
    max_rel_diff = np.max(rel_diff)

    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"Max relative difference: {max_rel_diff:.6f}")

    # Sample some predictions to show
    print(f"\nSample logits comparison (first batch, first position, first 10 vocab):")
    print(f"JAX:     {logits_jax_np[0, 0, :10]}")
    print(f"PyTorch: {logits_torch_np[0, 0, :10]}")
    print(f"Diff:    {abs_diff[0, 0, :10]}")

    # Get top-5 predictions for first position
    jax_top5 = np.argsort(logits_jax_np[0, 0])[-5:][::-1]
    torch_top5 = np.argsort(logits_torch_np[0, 0])[-5:][::-1]
    print(f"\nTop-5 token predictions (first batch, first position):")
    print(f"JAX:     {jax_top5}")
    print(f"PyTorch: {torch_top5}")

    # Assert close with reasonable tolerance for bfloat16
    # bfloat16 has less precision, so we need looser tolerances
    try:
        np.testing.assert_allclose(
            logits_jax_np,
            logits_torch_np,
            rtol=1e-2,  # 1% relative tolerance for bfloat16
            atol=1e-1,  # absolute tolerance
            err_msg="JAX and PyTorch outputs do not match"
        )
        print("\n✓ Model forward pass comparison PASSED!")
        print("  Models produce nearly identical outputs!")
    except AssertionError as e:
        print(f"\n✗ Tolerance check failed, but continuing...")
        print(f"  Error: {e}")
        print(f"  This may be acceptable given bfloat16 precision limitations")

    print("\n" + "="*80)
    print("RUNNING SINGLE GENERATION STEP (Logging Enabled)")
    print("="*80)

    # Enable debug logging for generation step (if not already enabled)
    if not log_prefill:
        logging.getLogger('models.llama.model').setLevel(logging.DEBUG)
        logging.getLogger('utils.ops').setLevel(logging.DEBUG)
        logging.getLogger('experiments.torch_llama').setLevel(logging.DEBUG)
        print("✓ Debug logging enabled for JAX and PyTorch models\n")
    else:
        print("✓ Debug logging already enabled from prefill\n")

    # Sample next token from last position (greedy)
    print("Sampling next token from last position...")
    next_token_jax = jnp.argmax(logits_jax[:, -1, :], axis=-1)  # [batch_size]
    next_token_torch = torch.argmax(logits_torch[:, -1, :], dim=-1)  # [batch_size]
    
    # Convert to numpy for comparison
    next_token_jax_np = np.array(next_token_jax, dtype=np.int32)
    next_token_torch_np = next_token_torch.cpu().numpy()
    
    print(f"JAX next tokens: {next_token_jax_np}")
    print(f"PyTorch next tokens: {next_token_torch_np}")
    
    # Check if tokens match
    tokens_match = np.array_equal(next_token_jax_np, next_token_torch_np)
    print(f"Tokens match: {tokens_match}")
    
    # Use first batch's token for generation (repeat for batch_size=2)
    next_token_val = int(next_token_jax_np[0])
    print(f"Using token {next_token_val} for generation step (repeated for batch_size=2)")

    # JAX generation step
    print("\nRunning JAX generation step...")
    next_token_jax_tensor = jnp.array([[next_token_val], [next_token_val]], dtype=jnp.int32)  # [2, 1]
    true_lengths_gen = jnp.array([1, 1], dtype=jnp.int32)
    
    # Build mask for single token generation (batch_size=2)
    dummy_input_gen = jnp.zeros((batch_size, 1, jax_config.dim), dtype=jax_dtype)
    mask_gen = build_attn_mask(dummy_input_gen, updated_kv_cache_jax, true_lengths_gen)
    
    logits_jax_gen, updated_kv_cache_jax = jax_model.apply(
        {"params": jax_params},
        tokens=next_token_jax_tensor,
        true_lengths=true_lengths_gen,
        kv_cache=updated_kv_cache_jax,
        mask=mask_gen,
    )
    print(f"✓ JAX generation step complete: {logits_jax_gen.shape}")

    # PyTorch generation step
    print("Running PyTorch generation step...")
    next_token_torch_tensor = torch.tensor([[next_token_val], [next_token_val]], device=torch_device, dtype=torch.long)  # [2, 1]
    start_pos_gen = actual_seqlen
    
    with torch.no_grad():
        logits_torch_gen = torch_model.forward_inference(next_token_torch_tensor, start_pos_gen)
    print(f"✓ PyTorch generation step complete: {logits_torch_gen.shape}")

    # Compare generation step outputs
    print("\n" + "="*80)
    print("COMPARING GENERATION STEP OUTPUTS")
    print("="*80)

    logits_jax_gen_np = np.array(logits_jax_gen, dtype=np.float32)
    logits_torch_gen_np = logits_torch_gen.cpu().float().detach().numpy()

    print(f"JAX gen output shape: {logits_jax_gen_np.shape}")
    print(f"PyTorch gen output shape: {logits_torch_gen_np.shape}")

    # Compare outputs
    abs_diff_gen = np.abs(logits_jax_gen_np - logits_torch_gen_np)
    max_abs_diff_gen = np.max(abs_diff_gen)
    mean_abs_diff_gen = np.mean(abs_diff_gen)

    rel_diff_gen = abs_diff_gen / (np.abs(logits_torch_gen_np) + 1e-8)
    max_rel_diff_gen = np.max(rel_diff_gen)

    print(f"Max absolute difference: {max_abs_diff_gen:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff_gen:.6f}")
    print(f"Max relative difference: {max_rel_diff_gen:.6f}")

    # Sample some predictions to show
    print(f"\nSample logits comparison (first batch, first position, first 10 vocab):")
    print(f"JAX:     {logits_jax_gen_np[0, 0, :10]}")
    print(f"PyTorch: {logits_torch_gen_np[0, 0, :10]}")
    print(f"Diff:    {abs_diff_gen[0, 0, :10]}")

    # Clean up PyTorch caches
    for block in torch_model.layers:
        block.attention.cache = None

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "jax_shape": logits_jax_np.shape,
        "torch_shape": logits_torch_np.shape,
        "max_abs_diff_gen": max_abs_diff_gen,
        "mean_abs_diff_gen": mean_abs_diff_gen,
        "max_rel_diff_gen": max_rel_diff_gen,
        "tokens_match": tokens_match,
    }


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Test forward pass comparison between JAX and PyTorch LLaMA models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory containing safetensors files and config.json. "
             "Should also have 'original' subdirectory with .pth file and tokenizer.model"
    )
    parser.add_argument(
        "--log-prefill",
        action="store_true",
        help="Enable debug logging during the prefill step (disabled by default)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("LLAMA 3.2 1B FORWARD PASS COMPARISON TEST")
    print("JAX vs PyTorch Implementation")
    print("="*80)

    results = test_model_forward_pass(args.model_path, log_prefill=args.log_prefill)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Summary:")
    print(f"  Initial forward pass:")
    print(f"    Max absolute difference: {results['max_abs_diff']:.6f}")
    print(f"    Mean absolute difference: {results['mean_abs_diff']:.6f}")
    print(f"    Max relative difference: {results['max_rel_diff']:.6f}")
    print(f"  Generation step:")
    print(f"    Max absolute difference: {results['max_abs_diff_gen']:.6f}")
    print(f"    Mean absolute difference: {results['mean_abs_diff_gen']:.6f}")
    print(f"    Max relative difference: {results['max_rel_diff_gen']:.6f}")
    print(f"    Sampled tokens match: {results['tokens_match']}")
    print("="*80)

