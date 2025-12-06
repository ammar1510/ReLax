import numpy as np
import jax

# jax.config.update("jax_default_matmul_precision", "highest")
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
import argparse
import dataclasses

# JAX/Flax model components
from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache
from utils.ops import build_attn_mask


def test_jax_forward_pass(
    model_path: str, output_file: str = "jax_output.txt", max_gen_len: int = 256
):
    """Test JAX model text generation with greedy sampling and save output to file."""

    model_path = Path(model_path)
    print(f"Loading model from {model_path}")

    # Test configuration
    test_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Use bfloat16 for consistency with typical inference
    jax_dtype = jnp.float32
    use_scaled_rope = False

    print("\n" + "=" * 80)
    print("LOADING JAX MODEL")
    print("=" * 80)

    # Load JAX configuration and weights
    config = ModelConfig.from_json_file(str(model_path))
    config = dataclasses.replace(
        config, dtype=jax_dtype, use_scaled_rope=use_scaled_rope
    )
    print(
        f"JAX Config: dim={config.dim}, n_layers={config.n_layers}, "
        f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}"
    )

    # Load JAX weights from PyTorch .pth files
    params = load_llama_weights(str(model_path), config)
    params = jax.tree.map(lambda x: x.astype(jax_dtype), params)
    print("dtype of params: ", params["tok_embeddings"]["embedding"].dtype)
    print("✓ JAX weights loaded successfully")

    # Initialize JAX model
    model = LLaMa(config)
    print("✓ JAX model initialized")

    print("\n" + "=" * 80)
    print("PREPARING INPUT")
    print("=" * 80)

    # Find tokenizer path (try original subdirectory first, then model_path)
    tokenizer_path = model_path / "original" / "tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = model_path / "tokenizer.model"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    # Initialize tokenizer
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    # Tokenize input text
    prompt_tokens = tokenizer.encode(test_prompt, bos=False, eos=False)
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    print("\n" + "=" * 80)
    print("GENERATING TEXT WITH GREEDY SAMPLING")
    print("=" * 80)

    # Initialize KV cache for generation
    batch_size = 1
    max_seq_len = len(prompt_tokens) + max_gen_len
    head_dim = config.head_dim
    kv_cache = KVCache.new(
        config.n_layers,
        batch_size,
        max_seq_len,
        config.n_kv_heads,
        head_dim,
        dtype=jax_dtype,
    )
    print(
        f"✓ KV cache initialized: {config.n_layers} layers, "
        f"batch={batch_size}, max_seq_len={max_seq_len}"
    )

    # Convert prompt tokens to JAX array
    tokens = jnp.array([prompt_tokens], dtype=jnp.int32)  # [1, prompt_len]
    current_seq_len = len(prompt_tokens)

    # Create JIT-compiled forward function
    @jax.jit
    def forward_fn(variables, tokens, true_lengths, kv_cache, mask):
        return model.apply(
            variables,
            tokens=tokens,
            true_lengths=true_lengths,
            kv_cache=kv_cache,
            mask=mask,
        )

    # Prefill: process the prompt
    print(f"Prefilling with {len(prompt_tokens)} tokens...")
    true_lengths = jnp.array([len(prompt_tokens)], dtype=jnp.int32)

    # Build attention mask for prefill using sequence length
    prefill_seqlen = len(prompt_tokens)
    mask = build_attn_mask(prefill_seqlen, kv_cache, true_lengths)

    logits, kv_cache = forward_fn(
        {"params": params},
        tokens=tokens,
        true_lengths=true_lengths,
        kv_cache=kv_cache,
        mask=mask,
    )
    print(f"✓ Prefill complete")

    # Generate tokens autoregressively with greedy sampling
    generated_tokens = []
    stop_tokens = {tokenizer.eos_id, tokenizer.eot_id}

    print(f"Generating up to {max_gen_len} tokens...")
    for step in range(max_gen_len):
        # Get logits for the last token position
        # logits shape: [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # Greedy sampling: argmax
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token_val = int(next_token.item())

        # Stop if we hit a stop token
        if next_token_val in stop_tokens:
            print(f"  Stopped at step {step} (stop token: {next_token_val})")
            break

        generated_tokens.append(next_token_val)

        # Prepare next token for forward pass
        next_token_tensor = jnp.array([[next_token_val]], dtype=jnp.int32)  # [1, 1]

        # Forward pass with single new token (JIT-compiled)
        true_lengths = jnp.array([1], dtype=jnp.int32)

        # Build attention mask for single-token generation
        gen_seqlen = 1
        mask = build_attn_mask(gen_seqlen, kv_cache, true_lengths)

        logits, kv_cache = forward_fn(
            {"params": params},
            tokens=next_token_tensor,
            true_lengths=true_lengths,
            kv_cache=kv_cache,
            mask=mask,
        )

        if (step + 1) % 10 == 0:
            print(f"  Generated {step + 1} tokens...")

    print(f"✓ Generation complete: {len(generated_tokens)} tokens generated")

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)
    full_text = test_prompt + generated_text

    print("\n" + "=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)

    # Save generated text to file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✓ Saved generated text to {output_path}")
    print(f"\nGenerated text ({len(generated_tokens)} tokens):")
    print(f"  {generated_text}")
    print(f"\nFull text (prompt + generated):")
    print(f"  {full_text}")

    print("\n" + "=" * 80)
    print("JAX GENERATION COMPLETE")
    print("=" * 80)

    return {
        "output_file": str(output_path),
        "prompt": test_prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "num_tokens": len(generated_tokens),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test JAX LLaMA model text generation with greedy sampling"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory containing safetensors files and config.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="jax_output.txt",
        help="Output file path for saving generated text (default: jax_output.txt)",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("JAX TEXT GENERATION TEST")
    print("=" * 80)

    results = test_jax_forward_pass(args.model_path, args.output_file, args.max_gen_len)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {results['output_file']}")
    print(f"Generated {results['num_tokens']} tokens")
    print("=" * 80)
