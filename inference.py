"""Production inference script for LLaMA models.

This script loads a LLaMA model with weights from disk and performs batch inference
on 16 hardcoded prompts using the slot-based engine with prefill and generate steps.

Usage:
    python inference.py --model_path /path/to/model
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import zmq

# Initialize JAX distributed for multi-TPU inference
import jax

jax.distributed.initialize()
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from models.engine import InferenceEngine, InferenceOrchestrator, InferenceRequest


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


def generic_tokenizer(prompt: str, tokenizer: Tokenizer) -> List[int]:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(formatted_prompt, bos=False, eos=False)


def generate_batch(
    orchestrator: InferenceOrchestrator,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts.

    Args:
        orchestrator: Running InferenceOrchestrator instance
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts (expecting 16)
        max_new_tokens: Maximum tokens to generate per prompt
        verbose: Print generation progress

    Returns:
        List of generated texts (one per prompt)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch Generation: {len(prompts)} prompts")
        print(f"{'='*80}\n")

    # Submit all requests
    requests = []
    response_queues = {}

    for i, prompt in enumerate(prompts):
        prompt_tokens = generic_tokenizer(prompt, tokenizer)
        prompt_array = jnp.array(prompt_tokens, dtype=jnp.int32)

        if verbose:
            print(f"[request-{i}] Prompt: {prompt[:60]}...")
            print(f"            Tokens: {len(prompt_tokens)}")

        # Create request
        request = InferenceRequest(
            request_id=f"request-{i}",
            prompt_tokens=prompt_array,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eot_id,
        )

        response_queue = orchestrator.submit(request)
        requests.append(request)
        response_queues[request.request_id] = response_queue

    if verbose:
        print(f"\nProcessing {len(prompts)} requests...\n")
        sys.stdout.flush()

    # Collect results
    results = {}
    completed = 0
    start_time = time.time()

    while completed < len(requests):
        for request_id, response_queue in response_queues.items():
            if request_id in results:
                continue  # Already completed

            try:
                result = response_queue.get(timeout=0.01)

                if result["status"] == "complete":
                    results[request_id] = result
                    completed += 1
                    elapsed = time.time() - start_time

                    if verbose and jax.process_index() == 3:
                        output_text = tokenizer.decode(result["tokens"])
                        print(f"✓ [{request_id}] completed in {elapsed:.2f}s")
                        print(f"  Output: {output_text[:100]}...")
                        print(
                            f"  Tokens: {len(result['tokens'])}, "
                            f"Reason: {result['finish_reason']}\n"
                        )
                        sys.stdout.flush()

                elif result["status"] == "error":
                    if verbose:
                        print(f"✗ [{request_id}] error: {result['error']}")
                        sys.stdout.flush()
                    results[request_id] = result
                    completed += 1

            except:
                pass  # Queue empty

    # Summary
    if verbose:
        total_time = time.time() - start_time
        total_tokens = sum(len(r["tokens"]) for r in results.values() if "tokens" in r)
        print(f"{'='*80}")
        print("Summary:")
        print(f"  Total requests: {len(requests)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {total_tokens / total_time:.2f} tokens/s")
        print(f"  Avg latency: {total_time / len(requests):.2f}s/request")
        print(f"{'='*80}\n")
        sys.stdout.flush()

    # Return outputs in order
    outputs = []
    for i in range(len(prompts)):
        request_id = f"request-{i}"
        if request_id in results and "tokens" in results[request_id]:
            output_text = tokenizer.decode(results[request_id]["tokens"])
            outputs.append(output_text)
        else:
            outputs.append("")

    return outputs


def receive_prompts_from_zmq(num_prompts: int = 16) -> List[str]:
    """Receive prompts from ZMQ SUB socket (multi-host setup).

    Each JAX worker independently binds to port 5555 on its own host and
    subscribes to prompts from the PUB socket. All workers receive identical
    prompts.

    Args:
        num_prompts: Number of prompts to receive before returning

    Returns:
        List of prompt strings
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind("tcp://*:5555")

    # Subscribe to all messages (empty string = all topics)
    socket.subscribe("")

    prompts = []
    print(
        f"[Worker {jax.process_index()}] Waiting for {num_prompts} prompts over ZMQ SUB..."
    )
    sys.stdout.flush()

    while len(prompts) < num_prompts:
        message = socket.recv_json()
        prompt = message.get("prompt", "")
        prompts.append(prompt)
        print(
            f"[Worker {jax.process_index()}] Received prompt {len(prompts)}: {prompt[:40]}..."
        )
        sys.stdout.flush()

    socket.close()
    context.term()
    print(f"[Worker {jax.process_index()}] Received all {len(prompts)} prompts.")
    return prompts


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
        "--num_prompts",
        type=int,
        default=16,
        help="Number of prompts to receive from ZMQ (default: 16)",
    )
    args = parser.parse_args()

    # Receive prompts from ZMQ
    prompts = receive_prompts_from_zmq(args.num_prompts)

    # Load model
    print("Loading model...")
    model, params, config, tokenizer = load_model(args.model_path)

    # Create unified mesh for interleaved mode - use all devices for both prefill and generate
    all_devices = np.array(jax.devices())
    unified_mesh = Mesh(all_devices, "i")
    prefill_mesh = unified_mesh
    generate_mesh = unified_mesh
    # All processes participate in both prefill and generate
    prefill_procs = list(range(len(all_devices)))
    generate_procs = list(range(len(all_devices)))

    print(f"Created unified mesh with {len(all_devices)} device(s): {unified_mesh}")
    print(f"for process-{jax.process_index()}:\n device: {jax.local_devices()} ")

    # Create engine and orchestrator with 16 slots
    max_concurrent_slots = 8
    print(f"\nInitializing inference engine (max_slots={max_concurrent_slots})...")
    sys.stdout.flush()
    engine = InferenceEngine(
        model=model,
        params=params,
        prefill_procs=prefill_procs,
        generate_procs=generate_procs,
        prefill_mesh=prefill_mesh,
        generate_mesh=generate_mesh,
        max_concurrent_slots=max_concurrent_slots,
        pad_id=tokenizer.pad_id,
    )

    # Create orchestrator in interleaved mode (no max_prefill_batch_size needed)
    orchestrator = InferenceOrchestrator(engine)
    orchestrator.start()

    try:
        # Run batch generation with 16 prompts
        generate_batch(
            orchestrator,
            tokenizer,
            prompts,
            max_new_tokens=512,
            verbose=True,
        )

    finally:
        # Cleanup
        print("\nShutting down...")
        orchestrator.stop()
        print("Done!")


if __name__ == "__main__":
    main()
