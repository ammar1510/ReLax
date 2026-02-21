"""GRPO training on GSM8K math reasoning dataset.

Usage:
    python examples/train_gsm8k.py --model_path /path/to/model
"""

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
from datasets import load_dataset

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from trainers.grpo_trainer import GRPOTrainer, GRPOConfig

# ── Hardcoded training config ──────────────────────────────────────────────────

ROLLOUT_BATCH_SIZE = 64
GROUP_SIZE = 8
MAX_NEW_TOKENS = 512
MAX_CACHE_SEQLEN = 1024
TEMPERATURE = 0.8
NUM_ITERATIONS = 500
MINIBATCH_SIZE = 32
KL_COEF = 0.1
LEARNING_RATE = 1e-5
REFERENCE_MODE = "static"
OUTPUT_DIR = "./gsm8k_output"
CHECKPOINT_FREQ = 100

# ── GSM8K answer extraction ────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r'####\s*(.+)')


def extract_answer(text: str) -> str:
    """Extract the answer after #### in a GSM8K-formatted string."""
    match = _ANSWER_RE.search(text)
    if match is None:
        return ""
    # Normalise: strip whitespace and commas (e.g. "1,000" → "1000")
    return match.group(1).strip().replace(",", "")


# ── Reward function ────────────────────────────────────────────────────────────

def make_reward_fn(tokenizer: Tokenizer):
    """Return a reward function closed over the tokenizer.

    Rewards:
        1.0  — correct format AND correct answer
        0.5  — correct format but wrong answer
        0.0  — no #### marker found
    """
    def reward_fn(completions: List[List[int]], ground_truths: List[str]) -> jax.Array:
        rewards = []
        for tokens, expected in zip(completions, ground_truths):
            text = tokenizer.decode(tokens)
            extracted = extract_answer(text)
            if extracted == "":
                rewards.append(0.0)
            elif extracted == expected:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        return jnp.array(rewards, dtype=jnp.float32)

    return reward_fn


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_gsm8k(tokenizer: Tokenizer) -> List[Tuple[List[int], str]]:
    """Load GSM8K train split and return (prompt_tokens, expected_answer) pairs."""
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    pairs = []
    for example in dataset:
        prompt = f"Question: {example['question']}\nAnswer:"
        tokens = tokenizer.encode(prompt, bos=True, eos=False)
        expected = extract_answer(example["answer"])
        if expected:  # skip any malformed entries
            pairs.append((tokens, expected))

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training on GSM8K")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    args = parser.parse_args()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    config = dataclasses.replace(ModelConfig.from_json_file(args.model_path), max_seqlen=8192)
    model = LLaMa(config)
    params = load_llama_weights(args.model_path, config)
    print(f"Model loaded: {config.n_layers}L {config.dim}D {config.n_heads}H {config.n_kv_heads}KVH")

    tokenizer_path = Path(args.model_path) / "original/tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path(args.model_path) / "tokenizer.model"
    tokenizer = Tokenizer(str(tokenizer_path))

    print("Loading GSM8K dataset...")
    prompt_dataset = load_gsm8k(tokenizer)
    print(f"Loaded {len(prompt_dataset)} examples")

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
        reward_fn=make_reward_fn(tokenizer),
        seed=42,
    )

    print(f"Starting GRPO training for {NUM_ITERATIONS} iterations...")
    metrics = trainer.train(
        prompt_dataset=prompt_dataset,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_freq=CHECKPOINT_FREQ,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    trainer.save_checkpoint(str(output_dir / "final_checkpoint"))
    print("Training complete.")
    last = metrics[-1]
    print(f"  Mean reward:  {last['mean_reward']:.4f}")
    print(f"  Mean loss:    {last['mean_loss']:.4f}")
    print(f"  Mean KL div:  {last['mean_kl_div']:.4f}")


if __name__ == "__main__":
    main()
