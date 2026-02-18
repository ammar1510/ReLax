"""Example GRPO training script.

This script demonstrates how to set up and run GRPO training on a LLaMA model.

Usage:
    python examples/train_grpo.py --model_path /path/to/model --dataset_path /path/to/prompts.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from trainers.grpo_trainer import GRPOTrainer, GRPOConfig


def example_reward_function(completions: List[List[int]]) -> jax.Array:
    """Example reward function based on sequence length.

    In practice, this should be replaced with a task-specific reward:
    - Code execution results for code generation
    - Format validation for structured output
    - External reward model scores
    - Human feedback

    Args:
        completions: List of token sequences.

    Returns:
        Rewards array [num_sequences].
    """
    # Simple example: reward based on length (prefer moderate length)
    rewards = []
    for completion in completions:
        length = len(completion)
        # Penalize very short or very long completions
        if length < 10:
            reward = -1.0
        elif length > 500:
            reward = -0.5
        else:
            # Reward moderate lengths
            reward = 1.0 - abs(length - 100) / 100.0

        rewards.append(reward)

    return jnp.array(rewards, dtype=jnp.float32)


def load_prompt_dataset(dataset_path: str, tokenizer: Tokenizer) -> List[List[int]]:
    """Load prompts from a JSON file and tokenize them.

    Expected format:
    {
        "prompts": [
            "Prompt 1 text here",
            "Prompt 2 text here",
            ...
        ]
    }

    Args:
        dataset_path: Path to JSON file with prompts.
        tokenizer: Tokenizer instance.

    Returns:
        List of tokenized prompts.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    prompts = data["prompts"]
    tokenized_prompts = []

    for prompt_text in prompts:
        # Encode prompt (adjust formatting based on your model)
        tokens = tokenizer.encode(prompt_text, bos=True, eos=False)
        tokenized_prompts.append(tokens)

    return tokenized_prompts


def main():
    parser = argparse.ArgumentParser(description="GRPO training for LLaMA models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to prompt dataset (JSON file)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=32,
        help="Number of prompts per rollout",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Number of completions per prompt",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.1,
        help="KL divergence coefficient",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./grpo_output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=100,
        help="Save checkpoint every N iterations (0 to disable)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("GRPO Training Setup")
    print(f"{'='*80}\n")

    # Load model configuration
    print(f"Loading model from {args.model_path}...")
    config = ModelConfig.from_json_file(args.model_path)
    print(
        f"Config: {config.n_layers} layers, {config.dim} dim, "
        f"{config.n_heads} heads, {config.n_kv_heads} kv_heads"
    )

    # Initialize model
    model = LLaMa(config)

    # Load weights
    print("Loading model weights...")
    params = load_llama_weights(args.model_path, config)
    print("Weights loaded successfully")

    # Load tokenizer
    tokenizer_path = Path(args.model_path) / "original/tokenizer.model"
    if not tokenizer_path.exists():
        # Try alternative path
        tokenizer_path = Path(args.model_path) / "tokenizer.model"

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(str(tokenizer_path))
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Load prompt dataset
    print(f"\nLoading prompts from {args.dataset_path}...")
    prompt_dataset = load_prompt_dataset(args.dataset_path, tokenizer)
    print(f"Loaded {len(prompt_dataset)} prompts")

    # Create GRPO configuration
    grpo_config = GRPOConfig(
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        max_new_tokens=256,
        temperature=0.8,
        num_iterations=args.num_iterations,
        ppo_epochs=2,
        minibatch_size=32,
        kl_coef=args.kl_coef,
        learning_rate=args.learning_rate,
        reference_mode="static",  # Use static reference model
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eot_id,
    )

    print(f"\nGRPO Configuration:")
    print(f"  Rollout batch size: {grpo_config.rollout_batch_size}")
    print(f"  Group size: {grpo_config.group_size}")
    print(
        f"  Completions per iteration: {grpo_config.rollout_batch_size * grpo_config.group_size}"
    )
    print(f"  PPO epochs: {grpo_config.ppo_epochs}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  KL coefficient: {grpo_config.kl_coef}")

    # Create trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        config=config,
        params=params,
        grpo_config=grpo_config,
        reward_fn=example_reward_function,  # Replace with your reward function
        seed=42,
    )

    print("Trainer initialized successfully")
    print(f"\nStarting training for {args.num_iterations} iterations...")
    print(f"{'='*80}\n")

    # Train
    checkpoint_dir = (
        str(output_dir / "checkpoints") if args.checkpoint_freq > 0 else None
    )
    metrics = trainer.train(
        prompt_dataset=prompt_dataset,
        num_iterations=args.num_iterations,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
    )

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save final checkpoint
    trainer.save_checkpoint(str(output_dir / "final_checkpoint"))
    print("\nTraining complete!")
    print(f"\nFinal metrics:")
    print(f"  Mean reward: {metrics[-1]['mean_reward']:.4f}")
    print(f"  Mean loss: {metrics[-1]['mean_loss']:.4f}")
    print(f"  Mean KL div: {metrics[-1]['mean_kl_div']:.4f}")


if __name__ == "__main__":
    main()
