import fire
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import time

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache
from sampling import TopPSampler


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
        max_seq_len=model.args.max_seq_len,
        kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        dtype=jnp.bfloat16,
    )

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
    # Sample the first token from the pre-fill logits
    rng_key, sample_key = random.split(rng_key)
    next_token = sampler.sample(logits, sample_key)

    for _ in range(max_gen_len - 1):
        if next_token.item() in tokenizer.stop_tokens:
            break

        generated_tokens.append(next_token.item())

        current_pos = tokens.shape[1]
        tokens = jnp.concatenate([tokens, next_token.reshape(1,1)], axis=1)

        logits, kv_cache = _model_step(params, next_token.reshape(1,1), kv_cache, current_pos)

        rng_key, sample_key = random.split(rng_key)
        next_token = sampler.sample(logits, sample_key)

    return tokenizer.decode(generated_tokens)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str = "The capital of France is",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    seed: int = 1,
):
    """
    Entry point for running the Llama JAX model for text generation.
    """
    print("Loading model and tokenizer...")
    start_time = time.time()

    model_config = ModelConfig.from_json_file(ckpt_dir)
    model_config.max_seq_len = max_seq_len

    tokenizer = Tokenizer(tokenizer_path)

    # Load model weights
    params = load_llama_weights(ckpt_dir)

    # Initialize the model
    model = LLaMa(model_config)

    # Create a JAX random key
    rng_key = random.PRNGKey(seed)

    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")

    # Generate text
    result = generate(
        model,
        params,
        tokenizer,
        prompt,
        max_gen_len,
        temperature,
        top_p,
        rng_key,
    )

    # Print the result
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 20)


if __name__ == "__main__":
    fire.Fire(main) 