import fire
import jax
import jax.numpy as jnp
from jax import random, jit
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
        max_seqlen=model.args.max_seqlen,
        kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        dtype=model.args.dtype,
    )
    print(f"KVCache size: {format_bytes(estimate_pytree_memory_footprint(kv_cache))}")

    # 2. Define and JIT-compile the model step function for performance
    @partial(jit, static_argnames=['model'])
    def _model_step(model, params, tokens, kv_cache, start_pos):
        logits, updated_kv_cache = model.apply(
            {'params': params},
            tokens,
            start_pos=start_pos,
            kv_cache=kv_cache
        )
        return logits[:, -1, :], updated_kv_cache

    # 3. Encode prompt and pre-fill KV cache
    prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
    tokens = jnp.array([prompt_tokens], dtype=jnp.int32)
    current_pos = 0
    generated_tokens = list(prompt_tokens)

    for _ in range(max_gen_len):

        logits, kv_cache = _model_step(model, params, tokens, kv_cache, current_pos)
        current_pos += tokens.shape[1]

        rng_key, sample_key = random.split(rng_key)
        next_token = sampler.sample(logits, sample_key)
        generated_tokens.append(next_token.item())
        tokens = next_token[:,None]
        if next_token == tokenizer.eot_id:
            break

    return tokenizer.decode(generated_tokens)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seqlen: int = 8192,
    seed: int = 1,
):
    """
    Entry point for running the Llama JAX model for text generation.
    """
    system_prompt = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
    user_prompt ="Write a python code to print all prime numbers between 1 and 100"
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    print("Loading model and tokenizer...")
    start_time = time.time()

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
