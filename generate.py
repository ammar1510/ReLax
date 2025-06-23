import fire
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import time
import dataclasses
import numpy as np

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache
from sampling import TopPSampler
from utils.memory import estimate_pytree_memory_footprint, format_bytes


def generate(
    model: LLaMa,
    params,
    tokenizer: Tokenizer,
    prompt: str,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    rng_key: jax.random.PRNGKey,
):
    """
    JAX-based text generation function.
    """
    # 1. Initialize sampler and KVCache
    sampler = TopPSampler(p=top_p, temperature=temperature)
    kv_cache = KVCache.new(
        n_layers=model.args.n_layers,
        bsz=1,
        max_seq_len=model.args.max_seqlen,
        kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        dtype=model.args.dtype,
    )
    print(f"KVCache size: {format_bytes(estimate_pytree_memory_footprint(kv_cache))}")

    # 2. Define and JIT-compile the model step function for performance
    @partial(jax.jit)
    def _model_step(params, tokens, kv_cache, start_pos):
        logits, updated_kv_cache = model.apply(
            {'params': params},
            tokens,
            start_pos=start_pos,
            kv_cache=kv_cache
        )
        return logits[:, -1, :], updated_kv_cache

    # 3. Encode prompt and pre-fill KV cache
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = jnp.array([prompt_tokens], dtype=jnp.int32)

    logits, kv_cache = _model_step(params, tokens, kv_cache, 0)

    # 4. Autoregressive generation loop
    generated_tokens = []

    rng_key, sample_key = random.split(rng_key)
    next_token = sampler.sample(logits, sample_key)
    generated_tokens.append(next_token.item())

    current_pos = tokens.shape[1]
    tokens = jnp.concatenate([tokens, next_token.reshape(1,1)], axis=1)

    for _ in range(max_gen_len):
        generated_tokens.append(next_token.item())

        logits, kv_cache = _model_step(params, next_token.reshape(1,1), kv_cache, current_pos)

        rng_key, sample_key = random.split(rng_key)
        next_token = sampler.sample(logits, sample_key)

    return tokenizer.decode(generated_tokens)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str = "",
    max_seqlen: int = 512,
    seed: int = 1,
):
    """
    Entry point for running the Llama JAX model for text generation.
    """
    print("Loading model and tokenizer...")
    start_time = time.time()

    model_config = ModelConfig.from_json_file(ckpt_dir)
    model_config = dataclasses.replace(model_config, max_seqlen=max_seqlen)

    tokenizer = Tokenizer(tokenizer_path)

    # Load model weights
    params = load_llama_weights(ckpt_dir)

    # Initialize the model
    model = LLaMa(model_config)

    # Create a JAX random key
    rng_key = random.PRNGKey(seed)

    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")

    # Estimate and print memory usage
    params_size_bytes = estimate_pytree_memory_footprint(params)
    print(f"Estimated model params size: {format_bytes(params_size_bytes)}")

    # Generate text
    result = generate(
        model,
        params,
        tokenizer,
        prompt,
        max_gen_len=500,
        temperature=0.6,
        top_p=0.9,
        rng_key=rng_key,
    )

    # Print the result
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 20)
    
if __name__ == "__main__":
    fire.Fire(main) 
