"""Group Relative Policy Optimization (GRPO) Trainer.

GRPO is an RL algorithm for fine-tuning language models using relative advantages
within groups of completions for the same prompt. It doesn't require a separate
reward model during training (rewards are computed once during rollout).

Key Features:
- Static reference model for KL penalty
- Group-relative advantage calculation (compares within prompt groups)
- Memory-efficient: caches reference logprobs during rollout
- On-policy: generates new rollouts each iteration
- Batched generation via ServingLoop/InferenceEngine

Training Loop:
1. Rollout: Generate multiple completions per prompt via ServingLoop
2. Reward: Score completions using task-specific reward function
3. Advantages: Normalize rewards relative to group mean/std
4. Optimize: Update policy using advantages and KL penalty
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from jax.sharding import Mesh

from models.llama.model import LLaMa
from models.llama.config import ModelConfig
from models.engine import ServingLoop, ServingConfig, UserRequestPrompt
from utils.kvcache import KVCache
from utils.ops import build_attn_mask
from utils.mesh_helpers import MeshHelper
from functools import partial
from sampling import categorical
from trainers.trainer import Trainer, TrainState


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Rollout settings
    rollout_batch_size: int = 64 # Number of prompts per rollout
    group_size: int = 16  # Completions per prompt
    max_new_tokens: int = 512  # Max tokens to generate per completion
    temperature: float = 0.7  # Sampling temperature for diversity

    # Training settings
    num_iterations: int = 1000  # Number of rollout-train cycles
    minibatch_size: int = 64  # Gradient update batch size

    # Loss coefficients
    kl_coef: float = 0.1  # KL divergence penalty coefficient

    # Reference model settings
    reference_mode: str = "static"  # "static", "ema", or "periodic"
    ema_alpha: float = 0.99  # EMA decay (if using EMA mode)
    reference_update_freq: int = 500  # Update frequency (if using periodic mode)

    # Optimization
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0  # Gradient clipping

    # Generation
    pad_token_id: int = 0
    eos_token_ids: Tuple[int, ...] = (2,)
    decode_steps: int = 10  # Tokens per multistep decode call
    max_cache_seqlen: Optional[int] = None  # KV cache size; defaults to model's max_seqlen if None

    def __post_init__(self):
        """Validate configuration."""
        if self.reference_mode not in ["static", "ema", "periodic"]:
            raise ValueError(
                f"reference_mode must be 'static', 'ema', or 'periodic', got {self.reference_mode}"
            )
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for advantage calculation, got {self.group_size}")


@dataclass
class RolloutBatch:
    """Container for a batch of rollout data."""

    # Tokens: [num_prompts * group_size, max_seq_len]
    tokens: jax.Array

    # Logprobs from reference model (cached): [num_prompts * group_size, max_seq_len]
    reference_logprobs: jax.Array

    # Advantages (group-normalized): [num_prompts * group_size]
    advantages: jax.Array

    # Mask indicating valid (non-padding) tokens: [num_prompts * group_size, max_seq_len]
    mask: jax.Array

    # Actual sequence lengths: [num_prompts * group_size]
    seq_lengths: jax.Array


class GRPOTrainer(Trainer):
    """GRPO trainer using ServingLoop for batched generation."""

    def __init__(
        self,
        model: LLaMa,
        config: ModelConfig,
        params: FrozenDict,
        grpo_config: GRPOConfig,
        reward_fn: Callable[[List[List[int]], List[Any]], jax.Array],
        detokenize_fn: Callable[[List[int]], str],
        mesh: Optional[Mesh] = None,
        seed: int = 42,
    ):
        """Initialize GRPO trainer.

        Args:
            model: LLaMa model instance.
            config: Model configuration.
            params: Model parameters (will be used for both policy and reference).
            grpo_config: GRPO-specific configuration.
            reward_fn: Function that takes a list of token sequences and returns rewards.
                       Should return array of shape [batch_size].
            mesh: JAX mesh for sharded computation. If None, creates a default mesh.
            seed: Random seed.
            detokenize_fn: Function to decode token IDs to text for logging.
        """
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(grpo_config.max_grad_norm),
            optax.adam(grpo_config.learning_rate),
        )

        # Setup mesh before sharding
        if mesh is None:
            mesh = Mesh(np.array(jax.devices()).reshape(4, 4), ("dp", "tp"))
        self.mesh = mesh

        # Shard params once; TrainState and ServingLoop share the same sharded copy
        sharded_params = MeshHelper.shard_params(params, mesh)

        # Initialize base trainer with sharded params
        super().__init__(model, sharded_params, optimizer, seed)

        self.config = config
        self.grpo_config = grpo_config
        self.reward_fn = reward_fn
        self.detokenize_fn = detokenize_fn

        # Create reference model parameters (frozen copy, also sharded)
        self.reference_params = jax.tree.map(lambda x: x.copy(), sharded_params)

        # Create ServingLoop for batched rollout generation
        serve_cfg = ServingConfig(
            decode_steps=grpo_config.decode_steps,
            decode_batch_size=grpo_config.rollout_batch_size,
            prefill_batch_size=grpo_config.group_size,
            eos_tokens=grpo_config.eos_token_ids,
            token_pad_idx=grpo_config.pad_token_id,
            max_decode_length=grpo_config.max_new_tokens,
            max_cache_seqlen=grpo_config.max_cache_seqlen or config.max_seqlen,
            sampler=partial(categorical, temperature=grpo_config.temperature),
            rng_seed=seed,
        )
        self.is_main = jax.process_index() == 0
        self.serving_loop = ServingLoop(
            serve_cfg=serve_cfg,
            model=model,
            params=sharded_params,
            mesh=mesh,
            is_server=self.is_main,
        )

        # Compile JIT functions
        self._compile_functions()

        self._log(f"GRPOTrainer initialized: {grpo_config.rollout_batch_size} prompts x "
                  f"{grpo_config.group_size} completions, is_server={self.is_main}")

    def _log(self, msg: str):
        """Log with process index prefix and flush."""
        print(f"[GRPOTrainer P{jax.process_index()}] {msg}")
        sys.stdout.flush()

    def _shard_batch(self, data: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        """Shard a batch dict along dp on the batch (first) axis."""
        def _shard(x):
            spec = MeshHelper.batch_axis_spec(self.mesh, rank=x.ndim, batch_axis=0)
            return MeshHelper.put_on_mesh(x, self.mesh, spec)
        return {k: _shard(v) for k, v in data.items()}

    def _shard_array(self, x: jax.Array) -> jax.Array:
        """Shard a single array along dp on the batch (first) axis."""
        spec = MeshHelper.batch_axis_spec(self.mesh, rank=x.ndim, batch_axis=0)
        return MeshHelper.put_on_mesh(x, self.mesh, spec)

    def _compile_functions(self):
        """Pre-compile frequently used functions."""
        self.train_step_jit = jax.jit(self._train_step_fn)

    # ==================== PHASE 1: ROLLOUT ====================

    def _reset_serving_loop(self):
        """Reset serving loop state for a new rollout batch.

        Note: _it is NOT reset — it must increase monotonically across prompts
        because SyncServer uses _it to namespace KV store keys, and keys are
        global for the lifetime of the process.
        """
        kv_cache, tokens = self.serving_loop.engine.init_decode_state()
        self.serving_loop.decode_work.curr_tokens = tokens
        self.serving_loop.decode_work.cache = kv_cache
        self.serving_loop.decode_work.active_results = [
            None for _ in range(self.grpo_config.rollout_batch_size)
        ]
        self.serving_loop.prefill_work.requests = []
        self.serving_loop.prefill_work.to_prefill = []
        self.serving_loop.prefill_work.to_decode = []
        self.serving_loop.results = {}
        self.serving_loop.decode_output = (None, None)

    def generate_rollouts(
        self,
        prompts: List[List[int]],
    ) -> RolloutBatch:
        """Generate rollouts for a batch of prompts using ServingLoop.

        All prompts are enqueued at once. The serving loop runs continuously
        until every completion is done — no per-prompt reset or iteration cap.

        Args:
            prompts: List of token sequences (prompts). Length = rollout_batch_size.

        Returns:
            RolloutBatch containing generated tokens and logprobs.
        """
        total_sequences = len(prompts) * self.grpo_config.group_size

        # Point engine to current policy params (already sharded)
        self.serving_loop.engine.params = self.state.params

        # Single reset before the whole rollout batch
        self._reset_serving_loop()

        t0 = time.time()
        self._log(f"Rollout: generating {total_sequences} completions "
                  f"({len(prompts)} prompts x {self.grpo_config.group_size} each)")

        # Enqueue all requests at once; unique id = pi * group_size + g
        if self.is_main:
            for pi, prompt_tokens in enumerate(prompts):
                for g in range(self.grpo_config.group_size):
                    req_id = pi * self.grpo_config.group_size + g
                    self.serving_loop.add_request(
                        UserRequestPrompt(id=req_id, text=list(prompt_tokens))
                    )

        # Drive the serving loop until every completion is done
        step = 0
        while True:
            self.serving_loop.serving_step()
            step += 1
            results = self.serving_loop.results
            if len(results) == total_sequences and all(r.done for r in results.values()):
                break

        self._log(f"Rollout: all {total_sequences} completions done in {step} serving steps, "
                  f"{time.time() - t0:.1f}s")

        # Collect results in prompt order
        all_prompt_tokens = []
        all_generated_tokens = []
        for pi, prompt_tokens in enumerate(prompts):
            for g in range(self.grpo_config.group_size):
                req_id = pi * self.grpo_config.group_size + g
                result = self.serving_loop.results.get(req_id)
                generated = result.token_list if result is not None else []
                all_prompt_tokens.append(list(prompt_tokens))
                all_generated_tokens.append(generated)

        if self.is_main:
            for pi in range(len(prompts)):
                if pi % 50 == 0:
                    idx = pi * self.grpo_config.group_size
                    sample_tokens = all_prompt_tokens[idx] + all_generated_tokens[idx]
                    self._log(f"Sample (prompt {pi}):\n{self.detokenize_fn(sample_tokens)}\n{'─'*60}")

        # Assemble full sequences and pad to common length
        max_seq_len = max(
            len(p) + len(g)
            for p, g in zip(all_prompt_tokens, all_generated_tokens)
        )
        max_seq_len = max(max_seq_len, 2)

        all_tokens = []
        all_masks = []
        all_seq_lengths = []

        for prompt_toks, gen_toks in zip(all_prompt_tokens, all_generated_tokens):
            full_seq = prompt_toks + gen_toks
            seq_len = len(full_seq)
            pad_len = max_seq_len - seq_len

            padded = full_seq + [self.grpo_config.pad_token_id] * pad_len
            mask = [1.0] * seq_len + [0.0] * pad_len

            all_tokens.append(padded)
            all_masks.append(mask)
            all_seq_lengths.append(seq_len)

        tokens_arr = self._shard_array(jnp.array(all_tokens, dtype=jnp.int32))
        mask_arr = self._shard_array(jnp.array(all_masks, dtype=jnp.float32))
        seq_lengths_arr = self._shard_array(jnp.array(all_seq_lengths, dtype=jnp.int32))

        self._log(f"Rollout: generation done in {time.time() - t0:.1f}s, "
                  f"shape={tokens_arr.shape}, computing reference logprobs...")
        t1 = time.time()

        # Compute reference logprobs (cached for KL penalty during training)
        reference_logprobs = self._compute_logprobs_batch(
            tokens_arr, mask_arr, self.reference_params
        )

        self._log(f"Rollout: reference logprobs computed in {time.time() - t1:.1f}s")

        return RolloutBatch(
            tokens=tokens_arr,
            reference_logprobs=reference_logprobs,
            advantages=self._shard_array(jnp.zeros(total_sequences)),
            mask=mask_arr,
            seq_lengths=seq_lengths_arr,
        )

    def _compute_logprobs_batch(
        self,
        tokens: jax.Array,
        mask: jax.Array,
        params: FrozenDict,
    ) -> jax.Array:
        """Compute log probabilities for a batch of sequences.

        Processes in chunks to avoid OOM on large batches. Chunk size matches
        the serving loop's group_size (decode_batch_size), which is already
        sized to fit in memory during rollout generation.

        Args:
            tokens: Token sequences [bsz, seq_len].
            mask: Validity mask [bsz, seq_len].
            params: Model parameters.

        Returns:
            Log probabilities for each token [bsz, seq_len].
        """
        bsz, seq_len = tokens.shape
        # Use group_size as chunk size — already proven to fit in memory during rollout
        dp_size = self.mesh.shape["dp"] if "dp" in self.mesh.shape else 1
        chunk_size = self.grpo_config.group_size * dp_size

        all_logprobs = []
        for start in range(0, bsz, chunk_size):
            end = min(start + chunk_size, bsz)
            chunk_tokens = self._shard_array(tokens[start:end])
            chunk_mask = self._shard_array(mask[start:end])

            chunk_logprobs = self._compute_logprobs_chunk(
                chunk_tokens, chunk_mask, params
            )
            all_logprobs.append(chunk_logprobs)

        return self._shard_array(jnp.concatenate(all_logprobs, axis=0))

    def _compute_logprobs_chunk(
        self,
        tokens: jax.Array,
        mask: jax.Array,
        params: FrozenDict,
    ) -> jax.Array:
        """Compute log probabilities for a single chunk of sequences."""
        bsz, seq_len = tokens.shape
        true_lengths = jnp.sum(mask, axis=-1).astype(jnp.int32)

        kv_cache = KVCache.new(
            n_layers=self.config.n_layers,
            bsz=bsz,
            max_seqlen=seq_len,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.bfloat16,
        )
        kv_cache = MeshHelper.init_kv_cache_on_mesh(kv_cache, self.mesh)

        attn_mask = build_attn_mask(seq_len, kv_cache, true_lengths)

        logits, _ = self.model.apply(
            {"params": params},
            tokens,
            true_lengths=true_lengths,
            kv_cache=kv_cache,
            mask=attn_mask,
        )

        log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
        target_tokens = tokens[:, 1:]
        token_logprobs = jnp.take_along_axis(
            log_probs, target_tokens[:, :, None], axis=-1
        ).squeeze(-1)

        token_logprobs = jnp.pad(token_logprobs, ((0, 0), (1, 0)))
        token_logprobs = token_logprobs * mask

        return token_logprobs

    # ==================== PHASE 2: REWARD ====================

    def compute_rewards(self, rollout: RolloutBatch, ground_truths: List[Any]) -> jax.Array:
        """Compute rewards for generated completions.

        Args:
            rollout: Rollout batch containing generated sequences.
            ground_truths: Expected answers for each prompt [num_prompts]. Each entry
                           is repeated group_size times to align with completions.

        Returns:
            Rewards array [num_sequences].
        """
        # Expand: each prompt has group_size completions sharing the same ground truth
        expanded_truths = [gt for gt in ground_truths for _ in range(self.grpo_config.group_size)]

        token_lists = []
        for i in range(rollout.tokens.shape[0]):
            seq_len = int(rollout.seq_lengths[i])
            token_list = rollout.tokens[i, :seq_len].tolist()
            token_lists.append(token_list)

        rewards = self.reward_fn(token_lists, expanded_truths)
        return rewards

    # ==================== PHASE 3: ADVANTAGES ====================

    def compute_advantages(self, rewards: jax.Array) -> jax.Array:
        """Compute group-relative advantages.

        For each group (same prompt), normalize rewards relative to group mean/std.

        Args:
            rewards: Rewards for all sequences [num_prompts * group_size].

        Returns:
            Advantages array [num_prompts * group_size].
        """
        num_groups = len(rewards) // self.grpo_config.group_size
        advantages = jnp.zeros_like(rewards)

        for group_idx in range(num_groups):
            start_idx = group_idx * self.grpo_config.group_size
            end_idx = start_idx + self.grpo_config.group_size

            group_rewards = rewards[start_idx:end_idx]

            mean_reward = jnp.mean(group_rewards)
            std_reward = jnp.std(group_rewards) + 1e-8

            group_advantages = (group_rewards - mean_reward) / std_reward
            advantages = advantages.at[start_idx:end_idx].set(group_advantages)

        return advantages

    # ==================== PHASE 4: POLICY OPTIMIZATION ====================

    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single training step on a minibatch.

        Args:
            state: Current training state.
            batch: Minibatch from rollout buffer.
            rng: Random key (unused, kept for interface compatibility).

        Returns:
            Tuple of (updated_state, metrics).
        """
        return self.train_step_jit(state, batch, self.grpo_config.kl_coef)

    def _train_step_fn(
        self,
        state: TrainState,
        batch: Dict[str, jax.Array],
        kl_coef: float,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Training step with logprob recomputation from current policy.

        Runs a full forward pass inside the loss function so gradients
        flow through the current policy params, making multiple PPO epochs
        meaningful.

        Args:
            state: Training state.
            batch: Minibatch with keys: tokens, mask, advantages, policy_logprobs, reference_logprobs.
            kl_coef: KL coefficient.

        Returns:
            Updated state and metrics.
        """
        model = self.model
        config = self.config

        def loss_fn(params):
            tokens = batch["tokens"]
            mask = batch["mask"]
            advantages = batch["advantages"]
            ref_logprobs = batch["reference_logprobs"]

            bsz, seq_len = tokens.shape
            true_lengths = jnp.sum(mask, axis=-1).astype(jnp.int32)

            # Scratch KV cache for forward pass
            kv_cache = KVCache.new(
                n_layers=config.n_layers,
                bsz=bsz,
                max_seqlen=seq_len,
                kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                dtype=jnp.bfloat16,
            )

            attn_mask = build_attn_mask(seq_len, kv_cache, true_lengths)

            # Forward pass with current params (differentiable)
            logits, _ = model.apply(
                {"params": params},
                tokens,
                true_lengths=true_lengths,
                kv_cache=kv_cache,
                mask=attn_mask,
            )

            # Compute per-token logprobs from current policy
            log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
            target_tokens = tokens[:, 1:]
            policy_logprobs = jnp.take_along_axis(
                log_probs, target_tokens[:, :, None], axis=-1
            ).squeeze(-1)
            policy_logprobs = jnp.pad(policy_logprobs, ((0, 0), (1, 0)))
            policy_logprobs = policy_logprobs * mask

            # KL divergence: D_KL(policy || reference)
            kl_div = (policy_logprobs - ref_logprobs) * mask
            kl_per_seq = jnp.sum(kl_div, axis=-1)

            # Policy gradient loss weighted by advantages
            per_seq_logprobs = jnp.sum(policy_logprobs * mask, axis=-1)
            pg_loss = -jnp.mean(advantages * per_seq_logprobs)

            total_loss = pg_loss + kl_coef * jnp.mean(kl_per_seq)

            return total_loss, {
                "loss": total_loss,
                "pg_loss": pg_loss,
                "kl_div": jnp.mean(kl_per_seq),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads, tx=self.optimizer)

        return new_state, metrics

    def train_on_rollout(self, rollout: RolloutBatch, rewards: jax.Array) -> Dict[str, List[float]]:
        """Train policy on a rollout buffer for multiple epochs.

        Args:
            rollout: Rollout batch.
            rewards: Computed rewards.

        Returns:
            Training metrics accumulated over all minibatches.
        """
        advantages = self._shard_array(self.compute_advantages(rewards))

        rollout = RolloutBatch(
            tokens=rollout.tokens,
            reference_logprobs=rollout.reference_logprobs,
            advantages=advantages,
            mask=rollout.mask,
            seq_lengths=rollout.seq_lengths,
        )

        all_metrics = {"loss": [], "pg_loss": [], "kl_div": []}

        num_sequences = rollout.tokens.shape[0]
        num_minibatches = (num_sequences + self.grpo_config.minibatch_size - 1) // self.grpo_config.minibatch_size
        self._log(f"Training: {num_sequences} sequences in {num_minibatches} minibatches")
        t0 = time.time()
        perm = jax.random.permutation(self.get_rng(), num_sequences)

        for i in range(0, num_sequences, self.grpo_config.minibatch_size):
            end_idx = min(i + self.grpo_config.minibatch_size, num_sequences)
            batch_indices = perm[i:end_idx]

            batch = self._shard_batch({
                "tokens": rollout.tokens[batch_indices],
                "reference_logprobs": rollout.reference_logprobs[batch_indices],
                "advantages": rollout.advantages[batch_indices],
                "mask": rollout.mask[batch_indices],
            })

            self.state, metrics = self.train_step(
                self.state,
                batch,
                self.get_rng(),
            )

            for key, value in metrics.items():
                all_metrics[key].append(float(value))

        self._log(f"Training: {num_minibatches} minibatches done in {time.time() - t0:.1f}s")
        return all_metrics

    # ==================== MAIN TRAINING LOOP ====================

    def train(
        self,
        prompt_dataset: List[Tuple[List[int], Any]],
        num_iterations: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 100,
        step_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """Main GRPO training loop.

        Args:
            prompt_dataset: Dataset of (token_sequence, ground_truth) pairs.
            num_iterations: Number of rollout-train iterations (uses config if None).
            checkpoint_dir: Directory to save checkpoints. If None, no checkpointing.
            checkpoint_freq: Save checkpoint every N iterations.

        Returns:
            List of metrics per iteration.
        """
        if num_iterations is None:
            num_iterations = self.grpo_config.num_iterations

        all_iteration_metrics = []

        for iteration in range(num_iterations):
            iter_t0 = time.time()
            self._log(f"{'='*60}")
            self._log(f"Iteration {iteration + 1}/{num_iterations}")

            # Sample prompts for this iteration
            num_prompts = self.grpo_config.rollout_batch_size
            prompt_indices = jax.random.choice(
                self.get_rng(),
                len(prompt_dataset),
                shape=(num_prompts,),
                replace=False,
            )
            batch = [prompt_dataset[int(idx)] for idx in prompt_indices]
            prompts = [b[0] for b in batch]
            ground_truths = [b[1] for b in batch]

            # Phase 1: Generate rollouts
            self._log("Phase 1/4: Generating rollouts...")
            rollout = self.generate_rollouts(prompts)

            # Phase 2: Compute rewards
            self._log("Phase 2/4: Computing rewards...")
            rewards = self.compute_rewards(rollout, ground_truths)
            mean_reward = float(jnp.mean(rewards))
            self._log(f"Phase 2/4: Mean reward = {mean_reward:.4f}")

            # Phase 3 & 4: Compute advantages and train
            self._log("Phase 3/4: Computing advantages...")
            self._log("Phase 4/4: Training on rollout buffer...")
            train_metrics = self.train_on_rollout(rollout, rewards)

            # Aggregate metrics for this iteration
            iteration_metrics = {
                "iteration": iteration,
                "mean_reward": mean_reward,
                "mean_loss": float(jnp.mean(jnp.array(train_metrics["loss"]))),
                "mean_pg_loss": float(jnp.mean(jnp.array(train_metrics["pg_loss"]))),
                "mean_kl_div": float(jnp.mean(jnp.array(train_metrics["kl_div"]))),
            }
            all_iteration_metrics.append(iteration_metrics)

            self._log(f"Iteration {iteration + 1} complete in {time.time() - iter_t0:.1f}s: "
                      f"loss={iteration_metrics['mean_loss']:.4f} "
                      f"pg_loss={iteration_metrics['mean_pg_loss']:.4f} "
                      f"kl_div={iteration_metrics['mean_kl_div']:.4f} "
                      f"reward={mean_reward:.4f}")

            if step_callback is not None:
                step_callback(iteration_metrics)

            # Update reference model (if using periodic or EMA mode)
            self._update_reference_model(iteration)

            # Save checkpoint
            if checkpoint_dir is not None and (iteration + 1) % checkpoint_freq == 0:
                self._log(f"Saving checkpoint at step {iteration + 1}...")
                ckpt_path = os.path.join(checkpoint_dir, f"step_{iteration + 1}")
                self.save_checkpoint(ckpt_path)

        return all_iteration_metrics

    def _update_reference_model(self, iteration: int):
        """Update reference model based on configured mode.

        Args:
            iteration: Current training iteration.
        """
        if self.grpo_config.reference_mode == "static":
            pass
        elif self.grpo_config.reference_mode == "ema":
            alpha = self.grpo_config.ema_alpha
            self.reference_params = jax.tree.map(
                lambda ref, policy: alpha * ref + (1 - alpha) * policy,
                self.reference_params,
                self.state.params,
            )
        elif self.grpo_config.reference_mode == "periodic":
            if (iteration + 1) % self.grpo_config.reference_update_freq == 0:
                self._log(f"Updating reference model at iteration {iteration + 1}")
                self.reference_params = jax.tree.map(
                    lambda x: x.copy(),
                    self.state.params,
                )

    # ==================== CHECKPOINTING ====================

    def save_checkpoint(self, path: str):
        """Save training checkpoint using Orbax.

        Uses tensorstore-backed array serialization for efficient, sharding-aware
        saves. Metadata (step, config, RNG) is saved as a separate JSON file.

        Args:
            path: Directory path to save checkpoint files.
        """
        ckpt_dir = os.path.abspath(path)
        checkpointer = ocp.StandardCheckpointer()

        # Save JAX pytrees via Orbax (params, opt_state, ref_params)
        ckpt_state = {
            "params": self.state.params,
            "opt_state": self.state.opt_state,
            "ref_params": self.reference_params,
        }
        checkpointer.save(os.path.join(ckpt_dir, "state"), ckpt_state)
        checkpointer.wait_until_finished()

        # Save lightweight metadata as JSON (only rank 0 writes to avoid races)
        if self.is_main:
            metadata = {
                "step": int(self.state.step),
                "seed": self.seed,
                "grpo_config": {
                    field: getattr(self.grpo_config, field)
                    for field in self.grpo_config.__dataclass_fields__
                },
                "rng_state": self.rng.tolist(),
            }
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Checkpoint saved to {ckpt_dir} at step {int(self.state.step)}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint using Orbax.

        Restores model params, optimizer state, reference params, and metadata.

        Args:
            path: Directory path to load checkpoint from.
        """
        ckpt_dir = os.path.abspath(path)
        checkpointer = ocp.StandardCheckpointer()

        # Build abstract target structure for restoration
        target = {
            "params": self.state.params,
            "opt_state": self.state.opt_state,
            "ref_params": self.reference_params,
        }
        ckpt_state = checkpointer.restore(
            os.path.join(ckpt_dir, "state"),
            args=ocp.args.StandardRestore(target),
        )

        self.reference_params = ckpt_state["ref_params"]

        # Load metadata
        with open(os.path.join(ckpt_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.state = TrainState(
            step=metadata["step"],
            params=ckpt_state["params"],
            opt_state=ckpt_state["opt_state"],
        )

        if "rng_state" in metadata:
            self.rng = jnp.array(metadata["rng_state"], dtype=jnp.uint32)

        print(f"Checkpoint loaded from {ckpt_dir} at step {metadata['step']}")
