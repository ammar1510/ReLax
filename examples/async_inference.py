"""Example usage of async slot-based inference engine.

This script demonstrates how to use the InferenceOrchestrator for efficient
concurrent text generation with multiple requests.

Usage:
    python examples/async_inference.py
"""

import time
from pathlib import Path
import jax
import jax.numpy as jnp

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.engine import InferenceEngine, InferenceOrchestrator, InferenceRequest


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
    """Example 1: Single request inference."""
    print("=" * 80)
    print("Example 1: Single Request Inference")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create engine and orchestrator
    engine = InferenceEngine(
        model=model,
        params=params,
        max_concurrent_slots=4,
        pad_id=0,
    )
    orchestrator = InferenceOrchestrator(engine)

    # Start orchestrator
    orchestrator.start()

    try:
        # Create a request
        prompt_tokens = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        request = InferenceRequest(
            request_id="request-1",
            prompt_tokens=prompt_tokens,
            max_new_tokens=20,
            eos_token_id=2,
        )

        # Submit request
        print(f"\nSubmitting request: {request.request_id}")
        print(f"Prompt length: {len(prompt_tokens)} tokens")
        response_queue = orchestrator.submit(request)

        # Collect results
        generated_tokens = []
        start_time = time.time()

        while True:
            result = response_queue.get()

            if result["status"] == "generating":
                token = result["token"]
                generated_tokens.append(token)
                print(f"  Generated token: {token}")

            elif result["status"] == "complete":
                total_time = time.time() - start_time
                print(f"\nGeneration complete!")
                print(f"  Total tokens: {len(result['tokens'])}")
                print(f"  Tokens: {result['tokens']}")
                print(f"  Time: {total_time:.3f}s")
                print(f"  Tokens/sec: {len(result['tokens']) / total_time:.2f}")
                print(f"  Finish reason: {result['finish_reason']}")
                break

            elif result["status"] == "error":
                print(f"  Error: {result['error']}")
                break

    finally:
        # Stop orchestrator
        orchestrator.stop()

    print()


def concurrent_requests_example():
    """Example 2: Multiple concurrent requests."""
    print("=" * 80)
    print("Example 2: Concurrent Requests (Slot-Based Batching)")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create engine with multiple slots
    engine = InferenceEngine(
        model=model,
        params=params,
        max_concurrent_slots=8,  # Support up to 8 concurrent sequences
        pad_id=0,
    )
    orchestrator = InferenceOrchestrator(engine)

    # Start orchestrator
    orchestrator.start()

    try:
        # Create multiple requests with different prompt lengths
        requests = [
            InferenceRequest(
                request_id=f"user-{i}",
                prompt_tokens=jnp.array(list(range(1, 10 + i * 5))),
                max_new_tokens=15,
                eos_token_id=2,
            )
            for i in range(5)  # 5 concurrent users
        ]

        print(f"\nSubmitting {len(requests)} concurrent requests:")
        for req in requests:
            print(f"  {req.request_id}: {len(req.prompt_tokens)} tokens")

        # Submit all requests
        response_queues = {}
        start_time = time.time()

        for request in requests:
            response_queue = orchestrator.submit(request)
            response_queues[request.request_id] = response_queue

        # Collect results from all requests
        completed = 0
        results = {}

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
                        print(f"\n✓ {request_id} completed:")
                        print(f"    Tokens: {len(result['tokens'])}")
                        print(f"    Time: {elapsed:.3f}s")
                        print(f"    Finish reason: {result['finish_reason']}")

                except:
                    pass  # Queue empty, continue

        # Summary
        total_time = time.time() - start_time
        total_tokens = sum(len(r["tokens"]) for r in results.values())

        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Total requests: {len(requests)}")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {total_tokens / total_time:.2f} tokens/sec")
        print(f"  Avg latency: {total_time / len(requests):.3f}s/request")
        print("=" * 80)

    finally:
        # Stop orchestrator
        orchestrator.stop()

    print()


def streaming_example():
    """Example 3: Streaming token-by-token output."""
    print("=" * 80)
    print("Example 3: Streaming Generation")
    print("=" * 80)

    # Load model
    print("Loading model...")
    model, params, config = load_model_and_params()

    # Create engine and orchestrator
    engine = InferenceEngine(model=model, params=params, max_concurrent_slots=4)
    orchestrator = InferenceOrchestrator(engine)

    # Start orchestrator
    orchestrator.start()

    try:
        # Create request
        prompt_tokens = jnp.array([1, 2, 3, 4, 5])
        request = InferenceRequest(
            request_id="stream-test",
            prompt_tokens=prompt_tokens,
            max_new_tokens=30,
            eos_token_id=2,
        )

        print(f"\nStreaming generation for: {request.request_id}")
        print("Generated tokens: ", end="", flush=True)

        response_queue = orchestrator.submit(request)

        # Stream tokens as they arrive
        while True:
            result = response_queue.get()

            if result["status"] == "generating":
                print(f"{result['token']} ", end="", flush=True)

            elif result["status"] == "complete":
                print(f"\n\nComplete! Total: {len(result['tokens'])} tokens")
                break

            elif result["status"] == "error":
                print(f"\nError: {result['error']}")
                break

    finally:
        orchestrator.stop()

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Async Slot-Based Inference Engine Examples")
    print("=" * 80 + "\n")

    # Run examples
    single_request_example()
    concurrent_requests_example()
    streaming_example()

    print("All examples completed!\n")


if __name__ == "__main__":
    main()
