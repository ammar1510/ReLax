"""Example usage of ServingLoop event loop-based inference.

This script demonstrates how to use the ServingLoop for efficient
concurrent text generation with multiple requests using an event loop pattern.

Features:
- Event loop-based serving (no threading complexity)
- Multi-host coordination support
- Flexible batching
- Multistep decode for efficiency

Usage:
    python examples/async_inference.py
"""

import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt


def load_model_and_params():
    """Load model and parameters.

    Note: This is a placeholder. In practice, you would load from checkpoint.
    """
    # Example config for a small LLaMA model
    config = ModelConfig(
        vocab_size=32000,
        dim=512,
        ffn_hidden_dim=1376,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        activation_fn="silu",
        max_seqlen=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        dtype="bfloat16",
    )

    # Initialize model
    model = LLaMa(config)

    # Initialize random parameters (in practice, load from checkpoint)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    from utils.kvcache import KVCache
    dummy_cache = KVCache.new(
        n_layers=config.n_layers,
        bsz=1,
        max_seqlen=config.max_seqlen,
        kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
    )
    dummy_seq_lengths = jnp.array([10])

    params = model.init(rng, dummy_input, dummy_seq_lengths, dummy_cache)

    return model, params['params'], config


def single_request_example():
    """Example 1: Single request inference with event loop."""
    print("=" * 80)
    print("Example 1: Single Request Inference (Event Loop)")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create mesh
    all_devices = np.array(jax.devices())
    mesh = Mesh(all_devices, "i")

    # Create serving config
    serve_cfg = ServingConfig(
        decode_steps=10,  # Generate 10 tokens per decode step
        decode_batch_size=4,  # Max concurrent sequences
        prefill_batch_size=2,  # Max concurrent prefill
        eos_tokens=(2,),  # EOS token ID
        token_pad_idx=0,
        max_decode_length=20,  # Max tokens to generate
    )

    # Create serving loop
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=True,  # Single process, so this is the server
    )

    # Create and submit a request
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    request = UserRequestPrompt(
        id=1,
        text=prompt_tokens,  # list[int], not jax.Array
    )

    print(f"\nSubmitting request: {request.id}")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    serving_loop.add_request(request)

    # Event loop
    start_time = time.time()
    max_iterations = 1000

    for iteration in range(max_iterations):
        serving_loop.serving_step()

        # Check if request is complete
        if request.id in serving_loop.results and serving_loop.results[request.id].done:
            result = serving_loop.results[request.id]
            total_time = time.time() - start_time

            print(f"\nGeneration complete!")
            print(f"  Total tokens: {len(result.token_list)}")
            print(f"  Tokens: {result.token_list}")
            print(f"  Time: {total_time:.3f}s")
            print(f"  Tokens/sec: {len(result.token_list) / total_time:.2f}")
            break

    print()


def concurrent_requests_example():
    """Example 2: Multiple concurrent requests with event loop."""
    print("=" * 80)
    print("Example 2: Concurrent Requests (Event Loop + Slot-Based Batching)")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create mesh
    all_devices = np.array(jax.devices())
    mesh = Mesh(all_devices, "i")

    # Create serving config with more slots
    serve_cfg = ServingConfig(
        decode_steps=10,
        decode_batch_size=8,  # Support up to 8 concurrent sequences
        prefill_batch_size=4,  # Process 4 prefills at once
        eos_tokens=(2,),
        token_pad_idx=0,
        max_decode_length=15,
    )

    # Create serving loop
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=True,
    )

    # Create multiple requests with different prompt lengths
    requests = [
        UserRequestPrompt(
            id=i,
            text=list(range(1, 10 + i * 5)),  # Different lengths
        )
        for i in range(5)  # 5 concurrent users
    ]

    print(f"\nSubmitting {len(requests)} concurrent requests:")
    for req in requests:
        print(f"  Request {req.id}: {len(req.text)} tokens")
        serving_loop.add_request(req)

    # Event loop
    start_time = time.time()
    max_iterations = 1000
    completed = 0

    for iteration in range(max_iterations):
        serving_loop.serving_step()

        # Check for newly completed requests
        new_completed = sum(1 for r in serving_loop.results.values() if r.done) - completed
        if new_completed > 0:
            for req_id, result in serving_loop.results.items():
                if result.done and len(result.token_list) > 0:
                    elapsed = time.time() - start_time
                    print(f"\n✓ Request {req_id} completed:")
                    print(f"    Tokens: {len(result.token_list)}")
                    print(f"    Time: {elapsed:.3f}s")
            completed += new_completed

        # Exit when all requests are done
        if completed >= len(requests):
            break

    # Summary
    total_time = time.time() - start_time
    total_tokens = sum(len(r.token_list) for r in serving_loop.results.values())

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total requests: {len(requests)}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Throughput: {total_tokens / total_time:.2f} tokens/sec")
    print(f"  Avg latency: {total_time / len(requests):.3f}s/request")
    print("=" * 80)
    print()


def event_loop_monitoring_example():
    """Example 3: Event loop with progress monitoring."""
    print("=" * 80)
    print("Example 3: Event Loop with Progress Monitoring")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create mesh
    all_devices = np.array(jax.devices())
    mesh = Mesh(all_devices, "i")

    # Create serving config
    serve_cfg = ServingConfig(
        decode_steps=5,  # Smaller steps for more frequent updates
        decode_batch_size=4,
        prefill_batch_size=2,
        eos_tokens=(2,),
        token_pad_idx=0,
        max_decode_length=30,
    )

    # Create serving loop
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=True,
    )

    # Create request
    prompt_tokens = [1, 2, 3, 4, 5]
    request = UserRequestPrompt(id=1, text=prompt_tokens)

    print(f"\nMonitoring generation progress for request {request.id}")
    print("Progress: ", end="", flush=True)

    serving_loop.add_request(request)

    # Event loop with progress tracking
    start_time = time.time()
    max_iterations = 1000
    last_token_count = 0

    for iteration in range(max_iterations):
        serving_loop.serving_step()

        # Check progress
        if request.id in serving_loop.results:
            result = serving_loop.results[request.id]
            current_count = len(result.token_list)

            # Print new tokens
            if current_count > last_token_count:
                for _ in range(current_count - last_token_count):
                    print(".", end="", flush=True)
                last_token_count = current_count

            # Check if complete
            if result.done:
                print(f"\n\nComplete! Generated {len(result.token_list)} tokens")
                print(f"Tokens: {result.token_list}")
                print(f"Time: {time.time() - start_time:.3f}s")
                break

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ServingLoop Event Loop-Based Inference Examples")
    print("=" * 80 + "\n")

    # Run examples
    single_request_example()
    concurrent_requests_example()
    event_loop_monitoring_example()

    print("All examples completed!\n")


if __name__ == "__main__":
    main()
