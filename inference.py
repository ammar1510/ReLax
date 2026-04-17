"""Inference script for LLaMA models.

Loads a LLaMA model with weights from disk and performs batch inference
using the ServingLoop event loop pattern.

Usage:
    python inference.py --model_path /path/to/model
"""

import argparse
import sys
import threading
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
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt
from utils.kvcache import KVCache
from sampling import greedy


def log(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


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

    config = ModelConfig.from_json_file(str(model_path))
    log(
        f"Loaded config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, {config.n_kv_heads} kv_heads, {config.max_seqlen} max_seqlen"
    )

    model = LLaMa(config)

    log(f"Loading weights from {checkpoint_path}...")
    params = load_from_orbax(checkpoint_path, mesh=mesh)
    log("Weights loaded successfully")

    tokenizer_path = model_path / "original/tokenizer.model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            f"Expected tokenizer.model in {model_path}"
        )

    log(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(str(tokenizer_path))
    log(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    return model, params, config, tokenizer


def format_prompt(prompt: str, tokenizer: Tokenizer) -> List[int]:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(
        formatted_prompt, bos=False, eos=False, allowed_special="all"
    )


def generate_batch(
    serving_loop: ServingLoop,
    tokenizer: Tokenizer,
    prompts: List[str],
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using the ServingLoop event loop.

    Args:
        serving_loop: ServingLoop instance
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts
        verbose: Print generation progress

    Returns:
        List of generated texts (one per prompt)
    """
    if verbose:
        log(f"\n{'='*80}")
        log(f"Batch Generation: {len(prompts)} prompts")
        log(f"{'='*80}\n")

    for i, prompt in enumerate(prompts):
        prompt_tokens = format_prompt(prompt, tokenizer)

        if verbose:
            log(f"[request-{i}] Prompt: {prompt[:60]}...")
            log(f"            Tokens: {len(prompt_tokens)}")

        request = UserRequestPrompt(id=i, text=prompt_tokens)
        serving_loop.add_request(request)

    if verbose:
        log(f"\nProcessing {len(prompts)} requests...\n")

    shutdown = threading.Event()
    serving_loop.serve_forever(shutdown)

    start_time = time.time()
    while sum(1 for r in serving_loop.results.values() if r.done) < len(prompts):
        time.sleep(0.01)

    shutdown.set()
    if verbose:
        elapsed = time.time() - start_time
        log(f"\nAll {len(prompts)} requests completed in {elapsed:.2f}s")

    decoded_results = []
    for i in range(len(prompts)):
        if i in serving_loop.results and serving_loop.results[i].done:
            result = serving_loop.results[i]
            generated_tokens = result.token_list
            if generated_tokens and hasattr(generated_tokens[0], "item"):
                generated_tokens = [
                    t.item() if hasattr(t, "item") else t for t in generated_tokens
                ]
            decoded_text = tokenizer.decode(generated_tokens)
            decoded_results.append(decoded_text)

            if verbose:
                log(f"[request-{i}] Generated {len(generated_tokens)} tokens:")
                log(f"            {decoded_text}")
                log()
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
        default=1024,
        help="Maximum number of tokens to generate per request",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS

    devices = jax.devices()
    assert (
        len(devices) == args.dp * args.tp
    ), f"Expected {args.dp * args.tp} devices, got {len(devices)}"
    mesh = Mesh(np.array(devices).reshape(args.dp, args.tp), ("dp", "tp"))

    log(f"Created mesh with {len(devices)} device(s): {mesh}")
    log(f"Loading model...")
    model, params, config, tokenizer = load_model(
        args.model_path, args.checkpoint_path, mesh
    )

    serve_cfg = ServingConfig(
        sampler=greedy,
        decode_steps=10,
        decode_batch_size=16,
        prefill_batch_size=4,
        eos_tokens=(tokenizer.eot_id,),
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=args.max_decode_length,
        max_cache_seqlen=2048,
    )

    log(f"\nInitializing serving loop...")
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        cache_cls=KVCache,
        is_server=(jax.process_index() == 0),
    )

    serving_loop.warmup()

    generate_batch(serving_loop, tokenizer, prompts, verbose=True)

    from models.sync_server import SyncServer
    log("Reaching shutdown barrier...")
    SyncServer.barrier("shutdown", 0)
    log("Done!")
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
