"""Inference script for LLaMA models.

Loads a LLaMA model with weights from disk and performs batch inference
using the ServingLoop event loop pattern.

Usage:
    python inference.py --model_path /path/to/model
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt


DEFAULT_PROMPTS = [
    "What is JAX and how does it differ from PyTorch?",
    "Explain the transformer architecture in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "What are the benefits of tensor parallelism for LLM inference?",
]


def load_model(model_path: str, config_path: Optional[str] = None):
    """Load model, config, and tokenizer.

    Args:
        model_path: Path to model directory containing:
            - model.safetensors.index.json (required - weight mapping)
            - model-0000X-of-0000Y.safetensors (sharded weight files)
            - config.json (model configuration)
            - tokenizer.model (tokenizer file)
        config_path: Optional separate path to config.json directory

    Returns:
        Tuple of (model, params, config, tokenizer)
    """
    model_path = Path(model_path)

    # Load configuration
    if config_path:
        config = ModelConfig.from_json_file(config_path)
    else:
        config = ModelConfig.from_json_file(str(model_path))

    print(
        f"Loaded config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, {config.n_kv_heads} kv_heads, {config.max_seqlen} max_seqlen"
    )

    # Initialize model
    model = LLaMa(config)

    print(f"Loading weights from {model_path}...")
    params = load_llama_weights(str(model_path), config)
    print("Weights loaded successfully")

    # Load tokenizer
    tokenizer_path = model_path / "original/tokenizer.model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            f"Expected tokenizer.model in {model_path}"
        )

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(str(tokenizer_path))
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    sys.stdout.flush()

    return model, params, config, tokenizer


def format_prompt(prompt: str, tokenizer: Tokenizer) -> List[int]:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(formatted_prompt, bos=False, eos=False)


def generate_batch(
    serving_loop: ServingLoop,
    tokenizer: Tokenizer,
    prompts: List[str],
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using event loop.

    Args:
        serving_loop: ServingLoop instance
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts
        verbose: Print generation progress

    Returns:
        List of generated texts (one per prompt)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch Generation: {len(prompts)} prompts")
        print(f"{'='*80}\n")

    # Submit all requests
    for i, prompt in enumerate(prompts):
        prompt_tokens = format_prompt(prompt, tokenizer)

        if verbose:
            print(f"[request-{i}] Prompt: {prompt[:60]}...")
            print(f"            Tokens: {len(prompt_tokens)}")

        request = UserRequestPrompt(id=i, text=prompt_tokens)
        serving_loop.add_request(request)

    if verbose:
        print(f"\nProcessing {len(prompts)} requests...\n")
        sys.stdout.flush()

    # Event loop
    completed = 0
    start_time = time.time()
    max_iterations = 10000

    for iteration in range(max_iterations):
        serving_loop.serving_step()

        newly_completed = sum(1 for r in serving_loop.results.values() if r.done) - completed
        completed += newly_completed

        if completed >= len(prompts):
            if verbose:
                elapsed = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"All {len(prompts)} requests completed in {elapsed:.2f}s")
                print(f"{'='*80}\n")
            break

    # Decode results
    decoded_results = []
    for i in range(len(prompts)):
        if i in serving_loop.results and serving_loop.results[i].done:
            result = serving_loop.results[i]
            generated_tokens = result.token_list
            if generated_tokens and hasattr(generated_tokens[0], 'item'):
                generated_tokens = [t.item() if hasattr(t, 'item') else t for t in generated_tokens]
            decoded_text = tokenizer.decode(generated_tokens)
            decoded_results.append(decoded_text)

            if verbose:
                print(f"[request-{i}] Generated {len(generated_tokens)} tokens:")
                print(f"            {decoded_text[:200]}")
                print()
        else:
            decoded_results.append("")

    return decoded_results


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA inference with slot-based engine"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory (containing model.safetensors, config.json, tokenizer.model)",
    )
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per request (default: 512)",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS

    # Load model
    print("Loading model...")
    model, params, config, tokenizer = load_model(args.model_path)

    # Create mesh — all devices along single TP axis
    # On multi-chip TPUs (e.g. v4-8), JAX auto-discovers all chips
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(4, 4), ("dp", "tp"))

    print(f"Created mesh with {len(devices)} device(s): {mesh}")
    print(f"Process {jax.process_index()}: devices {jax.local_devices()}")
    sys.stdout.flush()

    # Create serving configuration
    serve_cfg = ServingConfig(
        decode_steps=10,
        decode_batch_size=16,
        prefill_batch_size=4,
        eos_tokens=(tokenizer.eot_id,),
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=args.max_decode_length,
    )

    # Create serving loop
    print(f"\nInitializing serving loop...")
    sys.stdout.flush()
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=(jax.process_index() == 0),
    )

    # Warmup: compile prefill and decode before real inference
    serving_loop.warmup()

    # Run batch generation
    generate_batch(serving_loop, tokenizer, prompts, verbose=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
