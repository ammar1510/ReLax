"""Inference script for Qwen3.5 MoE models.

Loads a Qwen3.5 model with weights from disk and performs batch inference
using the ServingLoop event loop pattern.

Usage:
    python qwen_inference.py --model_path /path/to/Qwen3.5-122B-A10B
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from models.qwen.model import Qwen
from models.qwen.config import QwenConfig
from models.qwen.load import load_qwen_weights, load_from_orbax
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt
from utils.hybrid_cache import HybridCache, DeltaNetState
from utils.kvcache import KVCache
from utils.mesh_helpers import MeshHelper


DEFAULT_PROMPTS = [
    "What is JAX and how does it differ from PyTorch?",
    "Explain the transformer architecture in simple terms.",
    "Write a Python function that checks if a number is prime.",
    "What are the benefits of tensor parallelism for LLM inference?",
]


# ---------------------------------------------------------------------------
# HybridCache callbacks for engine integration
# ---------------------------------------------------------------------------


def make_cache_factory(config: QwenConfig, max_cache_seqlen: int, dtype):
    """Create a cache factory closure for the engine."""
    def factory(bsz):
        return HybridCache.new(config, bsz, max_cache_seqlen, dtype)
    return factory


def qwen_cache_slicer(cache: HybridCache, idx: int) -> HybridCache:
    """Slice a single sequence from a batched HybridCache."""
    return HybridCache(
        kv_cache=KVCache(
            k=cache.kv_cache.k[:, idx : idx + 1, :, :, :],
            v=cache.kv_cache.v[:, idx : idx + 1, :, :, :],
            seq_positions=cache.kv_cache.seq_positions[idx : idx + 1],
        ),
        deltanet_state=DeltaNetState(
            state=cache.deltanet_state.state[:, idx : idx + 1, :, :, :],
            conv_state=cache.deltanet_state.conv_state[:, idx : idx + 1, :, :],
        ),
    )


def qwen_cache_updater(decode_cache, entries, slot_idxs, lens, next_tokens, curr_tokens):
    """Batch update HybridCache for multiple slot insertions."""
    idx = jnp.array(slot_idxs)

    # KV cache update
    stacked_k = jnp.concatenate([e.kv_cache.k for e in entries], axis=1)
    stacked_v = jnp.concatenate([e.kv_cache.v for e in entries], axis=1)
    new_k = decode_cache.kv_cache.k.at[:, idx, :, :, :].set(stacked_k)
    new_v = decode_cache.kv_cache.v.at[:, idx, :, :, :].set(stacked_v)
    new_positions = decode_cache.kv_cache.seq_positions.at[idx].set(jnp.array(lens))

    # DeltaNet state update
    stacked_state = jnp.concatenate([e.deltanet_state.state for e in entries], axis=1)
    stacked_conv = jnp.concatenate([e.deltanet_state.conv_state for e in entries], axis=1)
    new_state = decode_cache.deltanet_state.state.at[:, idx, :, :, :].set(stacked_state)
    new_conv = decode_cache.deltanet_state.conv_state.at[:, idx, :, :].set(stacked_conv)

    new_cache = HybridCache(
        kv_cache=KVCache(k=new_k, v=new_v, seq_positions=new_positions),
        deltanet_state=DeltaNetState(state=new_state, conv_state=new_conv),
    )
    new_tokens = curr_tokens.at[idx, 0].set(jnp.array(next_tokens))
    return new_cache, new_tokens


def qwen_mask_cache_extractor(cache: HybridCache) -> KVCache:
    """Extract KV cache from HybridCache for build_attn_mask."""
    return cache.kv_cache


def qwen_place_cache(cache: HybridCache, mesh) -> HybridCache:
    """Place HybridCache on mesh."""
    return MeshHelper.place_hybrid_cache(cache, mesh)


# ---------------------------------------------------------------------------
# Model loading and tokenization
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    mesh=None,
):
    """Load Qwen model, config, and tokenizer.

    Args:
        model_path: Path to model directory containing config.json and tokenizer.
            Also used for safetensors loading when checkpoint_path is not set.
        config_path: Optional separate path to config.json directory.
        checkpoint_path: Optional orbax checkpoint path (local or gs://...).
            When provided, weights are loaded from orbax instead of safetensors.
        mesh: JAX mesh for sharded orbax restore. Required when using
            checkpoint_path so each host loads only its shards.

    Returns:
        Tuple of (model, params, config, tokenizer)
    """
    model_path = Path(model_path)

    # Load configuration
    cfg_path = config_path or str(model_path)
    config = QwenConfig.from_json_file(cfg_path)

    print(
        f"Loaded config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, {config.n_kv_heads} kv_heads, "
        f"{config.num_experts} experts (top-{config.num_experts_per_tok}), "
        f"full_attn={config.n_full_attn_layers}, linear_attn={config.n_linear_attn_layers}"
    )

    # Initialize model
    model = Qwen(config)

    if checkpoint_path is not None:
        print(f"Loading weights from orbax checkpoint: {checkpoint_path}...")
        params = load_from_orbax(checkpoint_path, mesh=mesh)
    else:
        print(f"Loading weights from safetensors: {model_path}...")
        params = load_qwen_weights(str(model_path), config)
    print("Weights loaded successfully")

    # Load tokenizer (HuggingFace AutoTokenizer for Qwen)
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    sys.stdout.flush()

    return model, params, config, tokenizer


def format_prompt(prompt: str, tokenizer) -> List[int]:
    """Format a prompt using Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text)


def generate_batch(
    serving_loop: ServingLoop,
    tokenizer,
    prompts: List[str],
    verbose: bool = True,
) -> List[str]:
    """Generate text for a batch of prompts using event loop."""
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

    pid = jax.process_index()
    debug = serving_loop.verbose
    for iteration in range(max_iterations):
        if debug:
            print(f"[P{pid}] generate_batch iteration={iteration}, completed={completed}/{len(prompts)}")
            sys.stdout.flush()
        serving_loop.serving_step()

        newly_completed = sum(1 for r in serving_loop.results.values() if r.done) - completed
        completed += newly_completed

        if completed >= len(prompts):
            if verbose:
                elapsed = time.time() - start_time
                print(f"\n[P{pid}] All {len(prompts)} requests completed in {elapsed:.2f}s")
                sys.stdout.flush()
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
    jax.distributed.initialize()
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MoE inference with slot-based engine"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Qwen3.5 model directory (config.json + tokenizer)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Orbax checkpoint path (local or gs://...). "
        "When set, weights are loaded from orbax instead of safetensors.",
    )
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per request",
    )
    parser.add_argument(
        "--max_cache_seqlen",
        type=int,
        default=4096,
        help="Maximum KV cache sequence length",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS

    # Create mesh first — needed for sharded orbax restore
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(4, 4), ("dp", "tp"))

    # Load model (from orbax checkpoint if provided, else safetensors)
    print("Loading Qwen3.5 model...")
    model, params, config, tokenizer = load_model(
        args.model_path,
        checkpoint_path=args.checkpoint_path,
        mesh=mesh if args.checkpoint_path else None,
    )

    print(f"Created mesh with {len(devices)} device(s): {mesh}")
    print(f"Process {jax.process_index()}: devices {jax.local_devices()}")
    sys.stdout.flush()

    # Determine EOS tokens
    eos_token_id = tokenizer.eos_token_id
    eos_tokens = (eos_token_id,) if isinstance(eos_token_id, int) else tuple(eos_token_id or [])

    # Cache factory for this config
    dtype = jnp.dtype(config.dtype)
    cache_factory = make_cache_factory(config, args.max_cache_seqlen, dtype)

    # Create serving configuration
    serve_cfg = ServingConfig(
        decode_steps=10,
        decode_batch_size=16,
        prefill_batch_size=4,
        eos_tokens=eos_tokens,
        token_pad_idx=tokenizer.pad_token_id or 0,
        max_decode_length=args.max_decode_length,
        max_cache_seqlen=args.max_cache_seqlen,
    )

    # Create serving loop with Qwen cache callbacks
    print(f"\nInitializing serving loop...")
    sys.stdout.flush()
    serving_loop = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        is_server=(jax.process_index() == 0),
        cache_factory=cache_factory,
        cache_slicer=qwen_cache_slicer,
        cache_updater=qwen_cache_updater,
        mask_cache_extractor=qwen_mask_cache_extractor,
        place_cache=qwen_place_cache,
    )

    # Warmup
    serving_loop.warmup()

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
