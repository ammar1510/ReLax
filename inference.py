"""Production inference script for LLaMA models.

This script loads a LLaMA model with weights from disk and performs batch inference
using the ServingLoop event loop pattern with multi-host support.

Features:
- Event loop-based serving (no threading)
- Multi-host coordination via SyncServer
- Flexible batching (configurable prefill/decode batch sizes)
- Multistep decode for efficiency

Usage:
    # Single process
    python inference.py --model_path /path/to/model

    # Multi-host (run on each machine)
    python -m jax.distributed.initialize \\
      --coordinator_address=192.168.1.100:1234 \\
      --num_processes=4 \\
      --process_id=0
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
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt


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
    serving_loop: ServingLoop,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using event loop.

    Args:
        serving_loop: ServingLoop instance
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts
        max_new_tokens: Maximum tokens to generate per prompt
        verbose: Print generation progress

    Returns:
        List of generated texts (one per prompt)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch Generation: {len(prompts)} prompts")
        print(f"{'='*80}\n")

    # Submit all requests (only on server process)
    is_server = jax.process_index() == 0
    if is_server:
        for i, prompt in enumerate(prompts):
            prompt_tokens = generic_tokenizer(prompt, tokenizer)

            if verbose:
                print(f"[request-{i}] Prompt: {prompt[:60]}...")
                print(f"            Tokens: {len(prompt_tokens)}")

            # Create request with list of tokens (not jax.Array)
            request = UserRequestPrompt(
                id=i,
                text=prompt_tokens,  # list[int], not jax.Array
            )

            serving_loop.add_request(request)

        if verbose:
            print(f"\nProcessing {len(prompts)} requests...\n")
            sys.stdout.flush()

    # Event loop (all processes run this)
    completed = 0
    start_time = time.time()
    max_iterations = 10000  # Safety limit

    for iteration in range(max_iterations):
        serving_loop.serving_step()

        # Check results (only on server process)
        if is_server:
            # Count newly completed requests
            newly_completed = sum(1 for result in serving_loop.results.values() if result.done) - completed
            completed += newly_completed

            if newly_completed > 0 and verbose:
                for request_id, result in serving_loop.results.items():
                    if result.done and len(result.token_list) > 0:  # Just completed
                        generated_tokens = result.token_list
                        if hasattr(generated_tokens[0], 'item'):  # Convert jax arrays if needed
                            generated_tokens = [t.item() if hasattr(t, 'item') else t for t in generated_tokens]
                        # Only print once per completion
                        if newly_completed > 0:
                            pass  # Will print summary at end

            # Exit when all requests are done
            if completed >= len(prompts):
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"\n{'='*80}")
                    print(f"All {len(prompts)} requests completed in {elapsed:.2f}s")
                    print(f"{'='*80}\n")
                break

    # Decode results (only on server)
    if is_server:
        decoded_results = []
        for i in range(len(prompts)):
            if i in serving_loop.results and serving_loop.results[i].done:
                result = serving_loop.results[i]
                generated_tokens = result.token_list
                # Convert to int if needed
                if hasattr(generated_tokens[0], 'item'):
                    generated_tokens = [t.item() if hasattr(t, 'item') else t for t in generated_tokens]
                decoded_text = tokenizer.decode(generated_tokens)
                decoded_results.append(decoded_text)

                if verbose:
                    print(f"[request-{i}] Generated {len(generated_tokens)} tokens:")
                    print(f"            {decoded_text[:100]}...")
                    print()
            else:
                decoded_results.append("")  # Empty if not completed

        return decoded_results
    else:
        return []  # Non-server processes return empty list


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

    # Create unified mesh for both prefill and generate
    all_devices = np.array(jax.devices())
    unified_mesh = Mesh(all_devices, "i")
    prefill_mesh = unified_mesh
    decode_mesh = unified_mesh

    print(f"Created unified mesh with {len(all_devices)} device(s): {unified_mesh}")
    print(f"for process-{jax.process_index()}:\n device: {jax.local_devices()} ")
    sys.stdout.flush()

    # Create serving configuration
    serve_cfg = ServingConfig(
        decode_steps=10,  # Generate 10 tokens per decode step
        decode_batch_size=16,  # Max concurrent decode sequences
        prefill_batch_size=4,  # Max concurrent prefill sequences
        eos_tokens=(tokenizer.eot_id,),  # End-of-turn token
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=512,  # Max tokens to generate
    )

    # Create serving loop
    print(f"\nInitializing serving loop...")
    sys.stdout.flush()
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        prefill_mesh=prefill_mesh,
        decode_mesh=decode_mesh,
        is_server=(jax.process_index() == 0),  # Process 0 is server
    )

    # Run batch generation
    generate_batch(
        serving_loop,
        tokenizer,
        prompts,
        max_new_tokens=512,
        verbose=True,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
