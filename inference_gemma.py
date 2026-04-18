"""Inference script for Gemma 4 models.

Loads a Gemma model from an orbax checkpoint and performs batch inference
using the ServingLoop event loop pattern.

Usage:
    python inference_gemma.py --model_path /path/to/gemma --checkpoint_path gs://bucket/gemma-orbax

Requires tokenizers:
    pip install tokenizers
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

from models.gemma.model import Gemma
from models.gemma.config import GemmaConfig
from models.gemma.load import load_from_orbax
from models.gemma.tokenizer import GemmaTokenizer
from utils.gemma_cache import GemmaCache
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt

DEFAULT_PROMPTS = [
    "What is JAX and how does it differ from PyTorch?",
    "Explain the transformer architecture in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "What are the benefits of tensor parallelism for LLM inference?",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: str, checkpoint_path: str, mesh: Mesh):
    """Load Gemma model, config, tokenizer, and weights from orbax checkpoint.

    Args:
        model_path: Path to model directory containing config.json and tokenizer.json
        checkpoint_path: Orbax checkpoint path (GCS or local)
        mesh: JAX device mesh for sharded restore

    Returns:
        Tuple of (model, params, config, tokenizer)
    """
    model_path = Path(model_path)

    # Load configuration
    config = GemmaConfig.from_json_file(str(model_path))
    print(
        f"Loaded config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, sliding_kv={config.n_kv_heads}, "
        f"global_kv={config.n_global_kv_heads}, max_seqlen={config.max_seqlen}"
    )
    print(
        f"  Sliding layers: {config.n_sliding_layers}, "
        f"Global layers: {config.n_global_layers}"
    )

    # Initialize model
    model = Gemma(config)

    # Load weights from orbax checkpoint (sharded onto mesh)
    print(f"Loading weights from {checkpoint_path}...")
    params = load_from_orbax(checkpoint_path, mesh=mesh)
    print("Weights loaded successfully")

    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            f"Expected tokenizer.json in {model_path}"
        )

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = GemmaTokenizer(str(tokenizer_path))
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    sys.stdout.flush()

    return model, params, config, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(
    prompt: str, tokenizer: GemmaTokenizer, enable_thinking: bool = False
) -> List[int]:
    """Format a user prompt using the Gemma 4 chat template.

    See gemma4_chat_template.md §2/§3/§8.

    enable_thinking=False (default):
        <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
        The trailing empty thought block suppresses reasoning — required,
        otherwise the model produces runaway <|channel>thought\n... output.

    enable_thinking=True:
        <bos><|turn>system\n<|think|><turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n
    """
    parts = []
    if enable_thinking:
        parts.append("<|turn>system\n<|think|><turn|>\n")
    parts.append(f"<|turn>user\n{prompt}<turn|>\n")
    parts.append("<|turn>model\n")
    if not enable_thinking:
        parts.append("<|channel>thought\n<channel|>")
    return tokenizer.encode("".join(parts), bos=True, eos=False)


def split_thinking(tokens: List[int], tokenizer: GemmaTokenizer):
    """Split a generated token list into (thinking_text, response_text).

    Gemma 4 wraps thinking in <|channel>thought\\n...\\n<channel|>.
    Returns (None, full_text) if channel tokens are absent or the
    pattern is not found.
    """
    start_id = tokenizer.start_channel_id
    end_id = tokenizer.end_channel_id
    if start_id is None or end_id is None:
        return None, tokenizer.decode(tokens)

    # Find the first <|channel> ... <channel|> span
    try:
        start = tokens.index(start_id)
    except ValueError:
        return None, tokenizer.decode(tokens)

    try:
        end = tokens.index(end_id, start + 1)
    except ValueError:
        return None, tokenizer.decode(tokens)

    # The tokens between the channel markers include "thought\n...\n";
    # decode and strip the leading "thought\n" prefix if present.
    thinking_text = tokenizer.decode(tokens[start + 1 : end])
    if thinking_text.startswith("thought\n"):
        thinking_text = thinking_text[len("thought\n"):]

    response_text = tokenizer.decode(tokens[end + 1 :])
    return thinking_text, response_text


# ---------------------------------------------------------------------------
# Batch generation (model-agnostic event loop)
# ---------------------------------------------------------------------------


def generate_batch(
    serving_loop: ServingLoop,
    tokenizer: GemmaTokenizer,
    prompts: List[str],
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using the event loop.

    Args:
        serving_loop: ServingLoop instance
        tokenizer: GemmaTokenizer for encoding/decoding
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

    # Run serving loop in background thread, poll for completion
    pid = jax.process_index()
    shutdown = threading.Event()
    serving_loop.serve_forever(shutdown)

    start_time = time.time()
    while sum(1 for r in serving_loop.results.values() if r.done) < len(prompts):
        time.sleep(0.01)

    shutdown.set()
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n[P{pid}] All {len(prompts)} requests completed in {elapsed:.2f}s")
        sys.stdout.flush()

    # Decode results
    decoded_results = []
    for i in range(len(prompts)):
        if i in serving_loop.results and serving_loop.results[i].done:
            result = serving_loop.results[i]
            generated_tokens = result.token_list
            if generated_tokens and hasattr(generated_tokens[0], "item"):
                generated_tokens = [
                    t.item() if hasattr(t, "item") else t for t in generated_tokens
                ]

            thinking_text, response_text = split_thinking(generated_tokens, tokenizer)
            decoded_results.append(response_text)

            if verbose:
                print(f"[request-{i}] Generated {len(generated_tokens)} tokens:")
                if thinking_text is not None:
                    print(f"  --- Thinking ---")
                    for line in thinking_text.strip().splitlines():
                        print(f"  {line}")
                    print(f"  --- Response ---")
                for line in response_text.strip().splitlines():
                    print(f"  {line}")
                print()
        else:
            decoded_results.append("")

    return decoded_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    jax.distributed.initialize()
    parser = argparse.ArgumentParser(
        description="Gemma 4 inference with slot-based engine"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory (containing config.json and tokenizer.json)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Orbax checkpoint path (GCS or local), e.g. gs://bucket/gemma-orbax",
    )
    parser.add_argument("--dp", type=int, default=2, help="Data-parallel dim")
    parser.add_argument("--tp", type=int, default=8, help="Tensor-parallel dim")
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per request",
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=8,
        help="Maximum number of concurrent decode sequences",
    )
    parser.add_argument(
        "--max_cache_seqlen",
        type=int,
        default=2048,
        help="Maximum cached sequence length (prefill + decode)",
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

    # Load model with weights
    print("Loading model...")
    model, params, config, tokenizer = load_model(
        args.model_path, args.checkpoint_path, mesh
    )

    # Create serving configuration
    serve_cfg = ServingConfig(
        decode_steps=10,
        decode_batch_size=args.decode_batch_size,
        prefill_batch_size=4,
        eos_tokens=tokenizer.stop_ids,
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=args.max_decode_length,
        max_cache_seqlen=args.max_cache_seqlen,
    )

    # Create serving loop — pass GemmaCache so the engine allocates the right cache
    print("\nInitializing serving loop...")
    sys.stdout.flush()
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        cache_cls=GemmaCache,
        is_server=(jax.process_index() == 0),
    )

    # Run batch generation
    generate_batch(serving_loop, tokenizer, prompts, verbose=True)

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
