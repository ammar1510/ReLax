from typing import Any, Optional, Set
from functools import partial
import jax
import jax.numpy as jnp
from jax import random, jit
import fire
import time
import dataclasses

from utils.kvcache import KVCache
from sampling import Sampler, TopPSampler
from utils.memory import estimate_pytree_memory_footprint, format_bytes


@partial(jit, static_argnames=['model'])
def _model_step(model, params, tokens, kv_cache, start_pos):
    """JIT-compiled model forward pass."""
    logits, updated_kv_cache = model.apply(
        {'params': params},
        tokens,
        start_pos=start_pos,
        kv_cache=kv_cache
    )
    return logits[:, -1, :], updated_kv_cache


def generate(
    model: Any,
    params: Any,
    tokenizer: Any,
    config: Any,
    prompts: list[str],
    max_gen_len: int,
    sampler: Sampler,
    rng_key: Optional[jax.Array] = None,
    stop_tokens: Optional[Set[int]] = None,
) -> list[str]:
    """
    Model-agnostic batch text generation function.
    
    Args:
        model: The model object (must have .apply method)
        params: Model parameters
        tokenizer: Tokenizer object (must have .encode, .decode, .stop_tokens)
        config: Model config (must have .n_layers, .n_kv_heads, .head_dim, .max_seqlen, .dtype)
        prompts: List of input texts to generate from
        max_gen_len: Maximum number of tokens to generate
        sampler: Sampler object for token sampling (e.g., TopPSampler, GreedySampler, etc.)
        rng_key: JAX random key
        stop_tokens: Optional override for stop tokens (uses tokenizer.stop_tokens if None)
    
    Returns:
        List of generated text strings
    """
    
    # Use tokenizer's stop tokens if not provided
    if stop_tokens is None:
        stop_tokens = tokenizer.stop_tokens
    
    batch_size = len(prompts)
    
    # Initialize KVCache
    kv_cache = KVCache.new(
        n_layers=config.n_layers,
        bsz=batch_size,
        max_seqlen=config.max_seqlen,
        kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=config.dtype,
    )
    
    # Generate random key if not provided
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    # Encode all prompts and pad to same length
    all_prompt_tokens = []
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
        all_prompt_tokens.append(prompt_tokens)
    
    # Find max length and pad all sequences
    max_prompt_len = max(len(tokens) for tokens in all_prompt_tokens)
    padded_tokens = []
    for tokens in all_prompt_tokens:
        padded = tokens + [tokenizer.pad_id] * (max_prompt_len - len(tokens))
        padded_tokens.append(padded)
    
    tokens = jnp.array(padded_tokens, dtype=jnp.int32)  # Shape: (batch_size, seq_len)
    current_pos = 0
    generated_tokens = [list(prompt_tokens) for prompt_tokens in all_prompt_tokens]
    
    # Track which sequences are still generating
    active_mask = jnp.ones(batch_size, dtype=bool)
    
    # Generation loop
    for _ in range(max_gen_len):
        # Forward pass
        logits, kv_cache = _model_step(model, params, tokens, kv_cache, current_pos)
        current_pos += tokens.shape[1]
        
        # Sample next token for all sequences
        assert rng_key is not None  # Should never be None at this point
        rng_key, sample_key = random.split(rng_key)
        next_tokens = sampler.sample(logits, sample_key)  # Shape: (batch_size,)
        
        # Update generated tokens and check for stop conditions
        new_active_mask = []
        for i in range(batch_size):
            if active_mask[i]:
                next_token_id = next_tokens[i].item()
                generated_tokens[i].append(next_token_id)
                # Stop if we hit a stop token
                new_active_mask.append(next_token_id not in stop_tokens)
            else:
                # Already stopped, don't add more tokens
                new_active_mask.append(False)
        
        active_mask = jnp.array(new_active_mask)
        
        # If all sequences have stopped, break early
        if not jnp.any(active_mask):
            break
        
        # Prepare tokens for next iteration
        tokens = next_tokens[:, None]  # Shape: (batch_size, 1)
    
    # Decode all generated sequences
    results = []
    for tokens in generated_tokens:
        results.append(tokenizer.decode(tokens))
    
    return results


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seqlen: int = 8192,
    seed: int = 1,
):
    """
    Entry point for testing the model-agnostic generate function.
    """
    print("Loading model and tokenizer...")
    start_time = time.time()

    # Import here to avoid circular imports
    from .llama.model import LLaMa
    from .llama.config import ModelConfig
    from .llama.load import load_llama_weights
    from .llama.tokenizer import Tokenizer

    model_config = ModelConfig.from_json_file(ckpt_dir)
    model_config = dataclasses.replace(model_config, max_seqlen=max_seqlen)

    tokenizer = Tokenizer(tokenizer_path)
    print("Model config: ", model_config)

    # Load model weights
    params = load_llama_weights(ckpt_dir+"/model.safetensors", model_config)

    # Initialize the model
    model = LLaMa(model_config)

    # Create a JAX random key
    rng_key = random.PRNGKey(seed)

    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")

    # Estimate and print memory usage
    params_size_bytes = estimate_pytree_memory_footprint(params)
    print(f"Estimated model params size: {format_bytes(params_size_bytes)}")

    # Test prompts - demonstrate batch processing
    test_user_prompts = [
        "Write a simple Python function to calculate factorial:",
        "Explain what machine learning is in one sentence:",
    ]

    system_prompt = "You are a helpful AI assistant. Provide clear and concise answers."
    formatted_prompts = []
    for user_prompt in test_user_prompts:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        formatted_prompts.append(prompt)

    print(f"\n--- Batch Generation Test ({len(test_user_prompts)} prompts) ---")
    for i, user_prompt in enumerate(test_user_prompts, 1):
        print(f"Prompt {i}: {user_prompt}")
    
    # Generate text for all prompts in a single batch
    sampler = TopPSampler(p=0.9, temperature=0.6)
    results = generate(
        model=model,
        params=params,
        tokenizer=tokenizer,
        config=model_config,
        prompts=formatted_prompts,
        max_gen_len=200,
        sampler=sampler,
        rng_key=rng_key,
    )
    
    print("\n--- Results ---")
    for i, (user_prompt, result) in enumerate(zip(test_user_prompts, results), 1):
        print(f"\nPrompt {i}: {user_prompt}")
        print(f"Generated: {result}")
        print("-" * 50)


if __name__ == "__main__":
    fire.Fire(main) 