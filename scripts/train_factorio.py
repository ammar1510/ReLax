"""GRPO training on Factorio Learning Environment (single-turn).

The model generates one Python program per task, which is executed in FLE.
Rewards are based on task completion, code correctness, and production score.

Usage:
    python scripts/train_factorio.py --model_path /path/to/model

Prerequisites:
    pip install factorio-learning-environment
"""

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from trainers.grpo_trainer import GRPOTrainer, GRPOConfig
from models.sync_server import SyncServer

from scripts.factorio_env import FactorioRewardEvaluator
from scripts.factorio_tasks import build_prompt_dataset, get_all_tasks

# ── Hardcoded training config ──────────────────────────────────────────────────

ROLLOUT_BATCH_SIZE = 8       # Fewer prompts (env evaluation is slow)
GROUP_SIZE = 4               # Fewer completions per prompt
MAX_NEW_TOKENS = 1024        # Code completions can be longer
MAX_CACHE_SEQLEN = 2048      # Longer context for API prompt + code
TEMPERATURE = 0.8
NUM_ITERATIONS = 200
MINIBATCH_SIZE = 8
KL_COEF = 0.1
LEARNING_RATE = 1e-5
REFERENCE_MODE = "static"
OUTPUT_DIR = "./factorio_output"
CHECKPOINT_FREQ = 50

# ── Code extraction ───────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)


def extract_code(text: str) -> str:
    """Extract Python code from model output.

    Handles both raw code and markdown-fenced code blocks.
    """
    # Try to extract from markdown code blocks first
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Otherwise treat the entire output as code
    return text.strip()


# ── Reward function ───────────────────────────────────────────────────────────

def make_reward_fn(tokenizer: Tokenizer, evaluator: FactorioRewardEvaluator):
    """Return a reward function that executes code in FLE.

    Rewards:
        1.0  — task objective achieved
        0.5  — valid code, partial progress
        0.25 — valid code, no progress
        0.1  — code errors but some state change
        0.0  — complete failure
    """
    def reward_fn(
        completions: List[List[int]], ground_truths: List[Any]
    ) -> jax.Array:
        rewards = []
        for tokens, task_config in zip(completions, ground_truths):
            text = tokenizer.decode(tokens)
            code = extract_code(text)

            if not code:
                rewards.append(0.0)
                continue

            r = evaluator.evaluate(code, task_config)
            rewards.append(r)

        return jnp.array(rewards, dtype=jnp.float32)

    return reward_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser(description="GRPO training on Factorio")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--wandb_project", type=str, default="relax-grpo-factorio", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no_labplay", action="store_true", help="Skip Lab-Play benchmark tasks")
    parser.add_argument("--manage_cluster", action="store_true", default=True, help="Auto-manage FLE cluster")
    parser.add_argument("--no_manage_cluster", action="store_true", help="Don't auto-manage FLE cluster")
    args = parser.parse_args()

    manage_cluster = args.manage_cluster and not args.no_manage_cluster

    is_main = jax.process_index() == 0
    devices = jax.devices()
    num_devices = len(devices)

    # Build device mesh — adapt to available device count
    if num_devices >= 16:
        mesh = Mesh(np.array(devices).reshape(4, 4), ("dp", "tp"))
    elif num_devices >= 8:
        mesh = Mesh(np.array(devices).reshape(2, 4), ("dp", "tp"))
    elif num_devices >= 4:
        mesh = Mesh(np.array(devices).reshape(2, 2), ("dp", "tp"))
    elif num_devices >= 2:
        mesh = Mesh(np.array(devices).reshape(1, 2), ("dp", "tp"))
    else:
        mesh = Mesh(np.array(devices).reshape(1, 1), ("dp", "tp"))

    print(f"Process {jax.process_index()}: {len(jax.local_devices())} local devices, {num_devices} total devices")

    output_dir = Path(OUTPUT_DIR)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb and is_main
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "task": "factorio",
                "rollout_batch_size": ROLLOUT_BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_cache_seqlen": MAX_CACHE_SEQLEN,
                "temperature": TEMPERATURE,
                "num_iterations": NUM_ITERATIONS,
                "minibatch_size": MINIBATCH_SIZE,
                "kl_coef": KL_COEF,
                "learning_rate": LEARNING_RATE,
                "reference_mode": REFERENCE_MODE,
                "model_path": args.model_path,
            },
        )
    elif not _WANDB_AVAILABLE and not args.no_wandb and is_main:
        print("wandb not installed; skipping W&B logging. Install with: pip install wandb")

    # Load model
    print(f"Loading model from {args.model_path}...")
    config = dataclasses.replace(ModelConfig.from_json_file(args.model_path), max_seqlen=8192)
    model = LLaMa(config)
    params = load_llama_weights(args.model_path, config)
    print(f"Model loaded: {config.n_layers}L {config.dim}D {config.n_heads}H {config.n_kv_heads}KVH")

    # Load tokenizer
    tokenizer_path = Path(args.model_path) / "original/tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path(args.model_path) / "tokenizer.model"
    tokenizer = Tokenizer(str(tokenizer_path))

    # Initialize FLE evaluator (only on main process — reward_fn runs on main)
    evaluator = FactorioRewardEvaluator(manage_cluster=manage_cluster)
    if is_main:
        evaluator.start()

    # Build task dataset
    print("Loading Factorio tasks...")
    tasks = get_all_tasks(include_labplay=not args.no_labplay)
    prompt_dataset = build_prompt_dataset(tokenizer, tasks)
    print(f"Loaded {len(prompt_dataset)} tasks")

    # GRPO config
    grpo_config = GRPOConfig(
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        group_size=GROUP_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_cache_seqlen=MAX_CACHE_SEQLEN,
        temperature=TEMPERATURE,
        num_iterations=NUM_ITERATIONS,
        minibatch_size=MINIBATCH_SIZE,
        kl_coef=KL_COEF,
        learning_rate=LEARNING_RATE,
        reference_mode=REFERENCE_MODE,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eot_id,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        params=params,
        grpo_config=grpo_config,
        reward_fn=make_reward_fn(tokenizer, evaluator),
        mesh=mesh,
        seed=42,
        detokenize_fn=tokenizer.decode,
    )

    # W&B logging callback
    def wandb_log(iteration_metrics: dict):
        if use_wandb:
            wandb.log(
                {
                    "reward/mean": iteration_metrics["mean_reward"],
                    "loss/total": iteration_metrics["mean_loss"],
                    "loss/pg": iteration_metrics["mean_pg_loss"],
                    "loss/kl_div": iteration_metrics["mean_kl_div"],
                },
                step=iteration_metrics["iteration"],
            )

    # Train
    print(f"Starting GRPO training for {NUM_ITERATIONS} iterations...")
    metrics = trainer.train(
        prompt_dataset=prompt_dataset,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_freq=CHECKPOINT_FREQ,
        step_callback=wandb_log,
    )

    # Save results
    if is_main:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    trainer.save_checkpoint(str(output_dir / "final_checkpoint"))
    if use_wandb:
        wandb.finish()

    # Cleanup
    if is_main:
        evaluator.stop()
        print("Training complete.")
        last = metrics[-1]
        print(f"  Mean reward:  {last['mean_reward']:.4f}")
        print(f"  Mean loss:    {last['mean_loss']:.4f}")
        print(f"  Mean KL div:  {last['mean_kl_div']:.4f}")

    SyncServer.barrier("shutdown", 0)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
