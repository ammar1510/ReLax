"""Inference script for LLaMA models.

Loads a LLaMA model with weights from disk and performs batch inference
using the InferenceEngine event loop pattern.

Usage:
    python inference.py --model_path /path/to/model
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List

import jax
import numpy as np
from jax.sharding import Mesh

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_from_orbax
from models.llama.tokenizer import Tokenizer
from models.engine import InferenceEngine, EngineConfig, UserRequestPrompt
from sampling import greedy

DEFAULT_PROMPTS = [
    "What is JAX and how does it differ from PyTorch?",
    "Explain the transformer architecture in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "What are the benefits of tensor parallelism for LLM inference?",
]


def load_model(model_path: str, checkpoint_path: str, mesh: Mesh):
    """Load model, config, tokenizer, and weights from orbax checkpoint.

    Args:
        model_path: Path to model directory containing config.json and tokenizer.model
        checkpoint_path: Orbax checkpoint path (GCS or local)
        mesh: JAX device mesh for sharded restore

    Returns:
        Tuple of (model, params, config, tokenizer)
    """
    model_path = Path(model_path)

    # Load configuration
    config = ModelConfig.from_json_file(str(model_path))
    print(
        f"Loaded config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, {config.n_kv_heads} kv_heads, {config.max_seqlen} max_seqlen"
    )

    # Initialize model
    model = LLaMa(config)

    # Load weights from orbax checkpoint (sharded onto mesh)
    print(f"Loading weights from {checkpoint_path}...")
    params = load_from_orbax(checkpoint_path, mesh=mesh)
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
    return tokenizer.encode(
        formatted_prompt, bos=False, eos=False, allowed_special="all"
    )


def generate_batch(
    engine: InferenceEngine,
    tokenizer: Tokenizer,
    prompts: List[str],
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using event loop.

    Args:
        engine: InferenceEngine instance
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
        engine.add_request(request)

    if verbose:
        print(f"\nProcessing {len(prompts)} requests...\n")
        sys.stdout.flush()

    # Event loop
    completed = 0
    start_time = time.time()
    max_iterations = 10000

    pid = jax.process_index()
    debug = engine.verbose
    for iteration in range(max_iterations):
        if debug:
            print(
                f"[P{pid}] generate_batch iteration={iteration}, completed={completed}/{len(prompts)}"
            )
            sys.stdout.flush()
        engine.serving_step()

        newly_completed = (
            sum(1 for r in engine.results.values() if r.done) - completed
        )
        completed += newly_completed

        if completed >= len(prompts):
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"\n[P{pid}] All {len(prompts)} requests completed in {elapsed:.2f}s"
                )
                sys.stdout.flush()
            if debug:
                print(
                    f"[P{pid}] BREAKING out of generate_batch loop at iteration={iteration}"
                )
                sys.stdout.flush()
            break

    # Decode results
    decoded_results = []
    for i in range(len(prompts)):
        if i in engine.results and engine.results[i].done:
            result = engine.results[i]
            generated_tokens = result.token_list
            if generated_tokens and hasattr(generated_tokens[0], "item"):
                generated_tokens = [
                    t.item() if hasattr(t, "item") else t for t in generated_tokens
                ]
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
    jax.distributed.initialize()
    parser = argparse.ArgumentParser(
        description="LLaMA inference with slot-based engine"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory (containing config.json, tokenizer.model)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Orbax checkpoint path (GCS or local), e.g. gs://bucket/llama-orbax",
    )
    parser.add_argument("--dp", type=int, default=4, help="Data-parallel dim")
    parser.add_argument("--tp", type=int, default=4, help="Tensor-parallel dim")
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per request",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS

    # Create mesh before loading so weights are sharded during restore
    devices = jax.devices()
    assert (
        len(devices) == args.dp * args.tp
    ), f"Expected {args.dp * args.tp} devices, got {len(devices)}"
    mesh = Mesh(np.array(devices).reshape(args.dp, args.tp), ("dp", "tp"))

    print(f"Created mesh with {len(devices)} device(s): {mesh}")
    print(f"Process {jax.process_index()}: devices {jax.local_devices()}")
    sys.stdout.flush()

    # Load model with sharded weights
    print("Loading model...")
    model, params, config, tokenizer = load_model(
        args.model_path, args.checkpoint_path, mesh
    )

    # Create serving configuration
    engine_cfg = EngineConfig(
        sampler=greedy,
        detokenize_fn=tokenizer.decode,
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
    engine = InferenceEngine(
        engine_cfg=engine_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=(jax.process_index() == 0),
    )

    # Run batch generation
    generate_batch(engine, tokenizer, prompts, verbose=True)

    # Ensure all hosts finish before any process exits
    pid = jax.process_index()
    print(f"[P{pid}] reaching shutdown barrier")
    sys.stdout.flush()
    from models.sync_server import SyncServer

    SyncServer.barrier("shutdown", 0)
    print(f"[P{pid}] passed shutdown barrier")
    sys.stdout.flush()

    print(f"\n[P{pid}] Done!")
    jax.distributed.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pid = jax.process_index()
        print(f"\n[HOST {pid}] FATAL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
