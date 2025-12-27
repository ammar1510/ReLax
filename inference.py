"""Production inference script for LLaMA models.

This script loads a LLaMA model with weights from disk and performs inference
using the slot-based engine with prefill and generate steps to handle multiple
concurrent requests efficiently.

Usage:
    # Single request
    python inference.py --model_path /path/to/model --prompt "Once upon a time"

    # Multiple concurrent requests
    python inference.py --model_path /path/to/model --concurrent

    # Interactive mode
    python inference.py --model_path /path/to/model --interactive
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional
# Initialize JAX distributed for multi-TPU inference
import jax
# TODO: Uncomment for multi-process testing
# jax.distributed.initialize()
import jax.numpy as jnp

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

    return model, params, config, tokenizer


def generic_tokenizer(prompt: str, tokenizer: Tokenizer) -> List[int]:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return tokenizer.encode(formatted_prompt, bos=False, eos=False)


def generate_single(
    orchestrator: InferenceOrchestrator,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    verbose: bool = True,
) -> str:
    """Generate text for a single prompt.

    Args:
        orchestrator: Running InferenceOrchestrator instance
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        verbose: Print streaming tokens

    Returns:
        Generated text (decoded)
    """
    # Encode prompt
    prompt_tokens = generic_tokenizer(prompt, tokenizer)
    prompt_array = jnp.array(prompt_tokens, dtype=jnp.int32)

    if verbose:
        print(f"\nPrompt: {prompt}")
        print(f"Prompt tokens: {len(prompt_tokens)}")
        print("\nGenerating: ", end="", flush=True)

    # Create request
    request = InferenceRequest(
        request_id="single-request",
        prompt_tokens=prompt_array,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eot_id,
    )

    # Submit and collect results
    response_queue = orchestrator.submit(request)
    generated_tokens = []
    start_time = time.time()

    while True:
        result = response_queue.get()

        if result["status"] == "generating":
            token = result["token"]
            generated_tokens.append(token)
            if verbose:
                # Decode and print token
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)

        elif result["status"] == "complete":
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n\n✓ Complete!")
                print(f"  Tokens: {len(result['tokens'])}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Speed: {len(result['tokens']) / elapsed:.2f} tokens/s")
                print(f"  Finish: {result['finish_reason']}")

            # Decode full output
            output_text = tokenizer.decode(result["tokens"])
            return output_text

        elif result["status"] == "error":
            print(f"\n✗ Error: {result['error']}")
            return ""


def generate_concurrent(
    orchestrator: InferenceOrchestrator,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
) -> List[str]:
    """Generate text for multiple prompts concurrently.

    Args:
        orchestrator: Running InferenceOrchestrator instance
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts
        max_new_tokens: Maximum tokens to generate per prompt

    Returns:
        List of generated texts (one per prompt)
    """
    print(f"\n{'='*80}")
    print(f"Concurrent Generation: {len(prompts)} requests")
    print(f"{'='*80}\n")

    # Create and submit all requests
    requests = []
    response_queues = {}

    for i, prompt in enumerate(prompts):
        # Encode prompt
        prompt_tokens = generic_tokenizer(prompt, tokenizer)
        prompt_array = jnp.array(prompt_tokens, dtype=jnp.int32)

        # Create request
        request = InferenceRequest(
            request_id=f"request-{i}",
            prompt_tokens=prompt_array,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eot_id,
        )

        print(f"[{request.request_id}] Prompt: {prompt[:60]}...")
        print(f"            Tokens: {len(prompt_tokens)}")

        # Submit (single-process mode - all requests submitted)
        response_queue = orchestrator.submit(request)
        response_queues[request.request_id] = response_queue
        requests.append(request)

    print(f"\nProcessing {len(prompts)} requests concurrently...\n")

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

                    output_text = tokenizer.decode(result["tokens"])
                    print(f"✓ [{request_id}] completed in {elapsed:.2f}s")
                    print(f"  Output: {output_text}")
                    print(
                        f"  Tokens: {len(result['tokens'])}, "
                        f"Reason: {result['finish_reason']}\n"
                    )

            except:
                pass  # Queue empty

    # Summary
    total_time = time.time() - start_time
    total_tokens = sum(len(r["tokens"]) for r in results.values())

    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  Total requests: {len(requests)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_tokens / total_time:.2f} tokens/s")
    print(f"  Avg latency: {total_time / len(requests):.2f}s/request")
    print(f"{'='*80}\n")

    # Synchronize all workers before returning to ensure they stay in sync
    # TODO: Uncomment for multi-process testing
    # jax.experimental.multihost_utils.sync_global_devices("generate_concurrent_done")

    # Return outputs in order
    outputs = []
    for i in range(len(prompts)):
        request_id = f"request-{i}"
        if request_id in results:
            output_text = tokenizer.decode(results[request_id]["tokens"])
            outputs.append(output_text)
        else:
            outputs.append("")

    return outputs


def interactive_mode(
    orchestrator: InferenceOrchestrator,
    tokenizer: Tokenizer,
    max_new_tokens: int = 200,
):
    """Run interactive chat mode.

    Args:
        orchestrator: Running InferenceOrchestrator instance
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Maximum tokens to generate per turn
    """
    print(f"\n{'='*80}")
    print("Interactive Mode")
    print("Type your prompts (Ctrl+C or 'quit' to exit)")
    print(f"{'='*80}\n")

    try:
        while True:
            # Get user input
            prompt = input("\n> ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not prompt:
                continue

            # Generate response
            generate_single(
                orchestrator,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                verbose=True,
            )

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


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
        "--config_path",
        type=str,
        default=None,
        help="Optional separate path to config.json directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_concurrent_slots",
        type=int,
        default=8,
        help="Maximum number of concurrent sequences in generate batch",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run concurrent generation demo with multiple prompts",
    )

    args = parser.parse_args()


    # Load model
    print("Loading model...")
    model, params, config, tokenizer = load_model(
        args.model_path,
        args.config_path,
    )

    # Create engine and orchestrator
    print(f"\nInitializing inference engine (max_slots={args.max_concurrent_slots})...")
    engine = InferenceEngine(
        model=model,
        params=params,
        max_concurrent_slots=args.max_concurrent_slots,
        pad_id=tokenizer.pad_id,
    )

    orchestrator = InferenceOrchestrator(engine)
    orchestrator.start()

    try:
        if args.interactive:
            # Interactive mode
            interactive_mode(orchestrator, tokenizer, args.max_new_tokens)

        elif args.concurrent:
            # Concurrent demo
            demo_prompts = [
                "The capital of France is",
                "In a galaxy far, far away",
                "The recipe for chocolate cake requires",
                "Python is a programming language that",
                "The meaning of life is",
            ]
            generate_concurrent(
                orchestrator,
                tokenizer,
                demo_prompts,
                max_new_tokens=512,
            )

        elif args.prompt:
            # Single prompt
            generate_single(
                orchestrator,
                tokenizer,
                args.prompt,
                verbose=True,
            )

        else:
            print("\nNo mode specified. Use --prompt, --interactive, or --concurrent")
            print("Examples:")
            print(
                '  python inference.py --model_path /path/to/model --prompt "Once upon a time"'
            )
            print("  python inference.py --model_path /path/to/model --interactive")
            print("  python inference.py --model_path /path/to/model --concurrent")

    finally:
        # Cleanup
        print("\nShutting down...")
        orchestrator.stop()

        # Synchronize all workers before exit to prevent barrier timeout
        # This ensures all processes reach the shutdown point together
        # TODO: Uncomment for multi-process testing
        # jax.experimental.multihost_utils.sync_global_devices("shutdown_sync")
        print("Done!")


if __name__ == "__main__":
    main()
