"""Baseline reward-signal check for CodeContests RL training.

Before committing to a 500-iteration GRPO run, sanity-check that the base model
produces enough reward variance for GRPO to learn from. Mirrors the rollout
phase of train_codecontests.py exactly (same temperature, group_size, prompt
format, executor settings) but skips training.

What it reports:
    - compile_rate: fraction of completions that produce valid C++.
    - any_pass_rate: fraction passing >=1 test.
    - all_pass_rate: fraction fully solving.
    - frac_groups_with_variance: fraction of problems where the group of
        `group_size` completions did NOT all score the same reward. THIS is the
        GRPO-criticality metric — zero-variance groups give zero gradient.

Usage:
    python scripts/baseline_check_codecontests.py \
        --model_path /path/to/model \
        --checkpoint_path gs://... \
        --num_problems 30
"""

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import jax
import numpy as np
from datasets import load_dataset
from jax.sharding import Mesh

from inference import load_model
from models.engine import ServingConfig, ServingLoop, UserRequestPrompt
from models.llama.tokenizer import Tokenizer
from models.sync_server import SyncServer
from sampling import categorical
from scripts.code_executor import (
    ExecutionResult,
    execute_cpp_batch,
    extract_cpp,
)
from utils.kvcache import KVCache

# ── Hyperparameters (must match train_codecontests.py for the check to mean anything) ──

GROUP_SIZE = 16
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 2048
MAX_CACHE_SEQLEN = 4096
DECODE_STEPS = 10
ROLLOUT_BATCH_SIZE = 32  # also caps decode_batch_size below

# Executor settings — generous time limit because we want to measure model
# capability, not infrastructure noise.
TIME_LIMIT_S = 5.0
MEM_LIMIT_MB = 512
COMPILE_TIMEOUT_S = 10.0


SYSTEM_PROMPT = (
    "You are a competitive programming assistant. Read the problem carefully, "
    "then write a complete C++ solution that reads from standard input and "
    "writes to standard output. Wrap your final code in a ```cpp ... ``` "
    "fenced block. Keep any reasoning brief."
)


def build_prompt(problem_text: str, tokenizer: Tokenizer) -> List[int]:
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{problem_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return tokenizer.encode(formatted, bos=False, eos=False, allowed_special="all")


def collect_tests(example: dict) -> List[Tuple[str, str]]:
    """Pull (input, output) pairs from a CodeContests example.

    CodeContests stores tests as dicts with 'input' and 'output' parallel lists,
    under three keys: public_tests, private_tests, generated_tests. We use only
    public + private (generated tests are model-synthesized and noisier).
    """
    tests: List[Tuple[str, str]] = []
    for key in ("public_tests", "private_tests"):
        block = example.get(key)
        if not block:
            continue
        for stdin, expected in zip(block.get("input", []), block.get("output", [])):
            tests.append((stdin, expected))
    return tests


def load_problems(
    tokenizer: Tokenizer,
    num_problems: int,
    seed: int,
    max_prompt_tokens: int,
) -> List[Tuple[List[int], List[Tuple[str, str]], str]]:
    """Sample N CodeContests problems with non-empty test suites and prompts
    that fit within `max_prompt_tokens`.

    Returns list of (prompt_tokens, tests, problem_id).
    """
    ds = load_dataset("deepmind/code_contests", split="train", streaming=False)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))

    out: List[Tuple[List[int], List[Tuple[str, str]], str]] = []
    skipped_no_tests = 0
    skipped_too_long = 0
    for idx in indices:
        if len(out) >= num_problems:
            break
        ex = ds[int(idx)]
        tests = collect_tests(ex)
        if not tests:
            skipped_no_tests += 1
            continue
        prompt_tokens = build_prompt(ex["description"], tokenizer)
        if len(prompt_tokens) > max_prompt_tokens:
            skipped_too_long += 1
            continue
        out.append((prompt_tokens, tests, ex.get("name", f"idx-{int(idx)}")))

    print(
        f"Loaded {len(out)} problems "
        f"(skipped {skipped_no_tests} with no tests, "
        f"{skipped_too_long} with prompts > {max_prompt_tokens} tokens)"
    )
    return out


def generate_completions(
    engine: ServingLoop,
    prompts: List[List[int]],
    is_main: bool,
) -> List[List[int]]:
    """Enqueue group_size completions per prompt and drain the serving loop."""
    total = len(prompts) * GROUP_SIZE
    engine.results = {}
    engine.done_count = 0

    if is_main:
        for pi, prompt_tokens in enumerate(prompts):
            for g in range(GROUP_SIZE):
                req_id = pi * GROUP_SIZE + g
                engine.add_request(UserRequestPrompt(id=req_id, text=list(prompt_tokens)))

    while not engine.serving_step(should_stop=engine.done_count >= total):
        pass

    completions: List[List[int]] = []
    for pi in range(len(prompts)):
        for g in range(GROUP_SIZE):
            req_id = pi * GROUP_SIZE + g
            r = engine.results.get(req_id)
            completions.append(r.token_list if r is not None else [])
    return completions


def score_group(
    completions_text: List[str],
    tests: List[Tuple[str, str]],
) -> List[ExecutionResult]:
    items = [(extract_cpp(t), tests) for t in completions_text]
    return execute_cpp_batch(
        items,
        time_limit_s=TIME_LIMIT_S,
        mem_limit_mb=MEM_LIMIT_MB,
        compile_timeout_s=COMPILE_TIMEOUT_S,
    )


def summarize_group(results: List[ExecutionResult]) -> dict:
    rewards = [r.pass_fraction for r in results]
    return {
        "n": len(results),
        "compile_rate": sum(1 for r in results if r.compiled) / len(results),
        "any_pass_rate": sum(1 for r in results if r.pass_fraction > 0) / len(results),
        "all_pass_rate": sum(1 for r in results if r.pass_fraction >= 1.0) / len(results),
        "mean_reward": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_path",
        type=str,
        default="./baseline_codecontests.json",
    )
    args = parser.parse_args()

    is_main = jax.process_index() == 0
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(2, 8), ("dp", "tp"))
    if is_main:
        print(f"Devices: {len(jax.local_devices())} local / {len(devices)} total")

    model, params, model_config, tokenizer = load_model(
        args.model_path, args.checkpoint_path, mesh
    )
    if is_main:
        print(
            f"Model: {model_config.n_layers}L {model_config.dim}D "
            f"{model_config.n_heads}H {model_config.n_kv_heads}KVH"
        )

    max_prompt_tokens = MAX_CACHE_SEQLEN - MAX_NEW_TOKENS
    problems = load_problems(
        tokenizer, args.num_problems, args.seed, max_prompt_tokens
    )
    if not problems:
        if is_main:
            print("No usable problems loaded — abort.")
        sys.exit(1)

    eos_tokens = (tokenizer.eot_id, tokenizer.eos_id, tokenizer.eom_id)
    serve_cfg = ServingConfig(
        sampler=partial(categorical, temperature=TEMPERATURE),
        decode_steps=DECODE_STEPS,
        decode_batch_size=min(ROLLOUT_BATCH_SIZE * GROUP_SIZE, 256),
        prefill_batch_size=GROUP_SIZE,
        eos_tokens=eos_tokens,
        token_pad_idx=tokenizer.pad_id,
        max_decode_length=MAX_NEW_TOKENS,
        max_cache_seqlen=MAX_CACHE_SEQLEN,
        rng_seed=args.seed,
    )
    engine = ServingLoop(
        serve_cfg=serve_cfg,
        model=model,
        params=params,
        mesh=mesh,
        cache_cls=KVCache,
        is_server=is_main,
    )

    # Generate everything in chunks of ROLLOUT_BATCH_SIZE prompts to keep the
    # decode batch within decode_batch_size.
    per_problem_results: List[List[ExecutionResult]] = []
    t_total = time.time()
    for chunk_start in range(0, len(problems), ROLLOUT_BATCH_SIZE):
        chunk = problems[chunk_start : chunk_start + ROLLOUT_BATCH_SIZE]
        chunk_prompts = [p for p, _, _ in chunk]
        chunk_tests = [t for _, t, _ in chunk]

        if is_main:
            print(
                f"\n[{chunk_start + 1}-{chunk_start + len(chunk)}/{len(problems)}] "
                f"generating {len(chunk) * GROUP_SIZE} completions..."
            )
        t0 = time.time()
        flat_completions = generate_completions(engine, chunk_prompts, is_main)
        if is_main:
            print(f"  generation: {time.time() - t0:.1f}s")

        if is_main:
            t1 = time.time()
            for pi, tests in enumerate(chunk_tests):
                start = pi * GROUP_SIZE
                end = start + GROUP_SIZE
                texts = [tokenizer.decode(toks) for toks in flat_completions[start:end]]
                per_problem_results.append(score_group(texts, tests))
            print(f"  execution:  {time.time() - t1:.1f}s")

    SyncServer.barrier("baseline_done", 0)

    if not is_main:
        jax.distributed.shutdown()
        return

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")

    per_problem_stats = []
    for (_, _, name), results in zip(problems, per_problem_results):
        s = summarize_group(results)
        s["problem_id"] = name
        per_problem_stats.append(s)

    overall = {
        "num_problems": len(per_problem_stats),
        "group_size": GROUP_SIZE,
        "compile_rate": statistics.fmean(s["compile_rate"] for s in per_problem_stats),
        "any_pass_rate": statistics.fmean(s["any_pass_rate"] for s in per_problem_stats),
        "all_pass_rate": statistics.fmean(s["all_pass_rate"] for s in per_problem_stats),
        "mean_reward": statistics.fmean(s["mean_reward"] for s in per_problem_stats),
        "median_reward": statistics.median(s["mean_reward"] for s in per_problem_stats),
        "frac_groups_with_variance": (
            sum(1 for s in per_problem_stats if s["reward_std"] > 1e-6)
            / len(per_problem_stats)
        ),
    }

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")

    fgv = overall["frac_groups_with_variance"]
    print("\nVERDICT:")
    if fgv > 0.4:
        print(f"  ✅ {fgv:.1%} groups have reward variance — strong signal, train.")
    elif fgv > 0.2:
        print(
            f"  ⚠️  {fgv:.1%} groups have reward variance — trainable but slow.\n"
            "     Consider SFT warmup on solutions or curriculum (easier subset)."
        )
    else:
        print(
            f"  ❌ Only {fgv:.1%} groups have reward variance — model can't engage.\n"
            "     Drop to Python, do SFT first, or use a stronger base model."
        )

    # Pick one passing and one failing example per problem (when available) for inspection.
    examples = []
    for (_, tests, name), results in zip(problems, per_problem_results):
        passing = next((r for r in results if r.pass_fraction >= 1.0), None)
        failing = next((r for r in results if r.pass_fraction < 1.0), None)
        examples.append({
            "problem_id": name,
            "n_tests": len(tests),
            "passing_example": _result_summary(passing),
            "failing_example": _result_summary(failing),
        })

    out = {
        "overall": overall,
        "per_problem": per_problem_stats,
        "examples": examples,
    }
    Path(args.output_path).write_text(json.dumps(out, indent=2))
    print(f"\nFull results written to {args.output_path}")

    SyncServer.barrier("shutdown", 0)
    jax.distributed.shutdown()


def _result_summary(r: Optional[ExecutionResult]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "compiled": r.compiled,
        "compile_error": r.compile_error,
        "pass_fraction": r.pass_fraction,
        "statuses": [t.status for t in r.test_results],
    }


if __name__ == "__main__":
    main()
