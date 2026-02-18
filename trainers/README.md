# GRPO Training for ReLax

This directory contains the implementation of Group Relative Policy Optimization (GRPO) for training LLaMA models in ReLax.

## Overview

GRPO is a reinforcement learning algorithm designed for fine-tuning language models. It's more efficient than traditional RLHF because:

1. **No separate reward model needed** - Rewards are computed once during rollout, then cached
2. **Group-relative advantages** - Compares completions within groups (same prompt), reducing reward model bias
3. **Memory efficient** - Reference model logprobs are cached, so reference model only needed during rollout

## Architecture

### Key Components

1. **`trainer.py`** - Base trainer class with `TrainState` for JAX optimization
2. **`grpo_trainer.py`** - Full GRPO implementation with all 4 training phases
3. **`examples/train_grpo.py`** - Example training script

### Training Phases

#### Phase 1: Rollout Generation
```python
# Generate multiple completions per prompt
rollout = trainer.generate_rollouts(prompts)
```
- Samples `group_size` completions for each prompt (e.g., 4 completions per prompt)
- Uses current policy model with temperature sampling for diversity
- Computes and caches logprobs from both policy and reference models

#### Phase 2: Reward Computation
```python
# Score each completion
rewards = trainer.compute_rewards(rollout)
```
- Apply task-specific reward function (code execution, format validation, etc.)
- Rewards are scalar values per completion

#### Phase 3: Advantage Calculation
```python
# Normalize within groups
advantages = trainer.compute_advantages(rewards)
```
- **Key innovation**: Advantages are relative to group mean/std
- Formula: `advantage = (reward - group_mean) / group_std`
- This removes reward bias and makes training more stable

#### Phase 4: Policy Optimization
```python
# Train on rollout buffer
metrics = trainer.train_on_rollout(rollout, rewards)
```
- Update policy using policy gradient weighted by advantages
- Add KL penalty to prevent drift from reference model
- Loss: `L = -advantages * log_probs + β * KL(policy || reference)`
- Train for multiple epochs on the rollout buffer

## Memory Management

### Static Reference Model (Recommended)

GRPO uses a **static reference model** approach by default:

```
During Rollout:
  - Policy model (active): N parameters
  - Reference model (frozen): N parameters
  - Total: 2N parameters

During Training:
  - Policy model (active): N parameters
  - Policy optimizer state (Adam): 2N parameters
  - Policy gradients: N parameters
  - Reference logprobs (cached): Small array
  - Total: 4N parameters

Peak Memory: max(2N, 4N) = 4N parameters
```

The reference model can be offloaded to CPU between rollout phases if needed.

## Configuration

### GRPOConfig

```python
@dataclass
class GRPOConfig:
    # Rollout settings
    rollout_batch_size: int = 256  # Prompts per rollout
    group_size: int = 4  # Completions per prompt
    max_new_tokens: int = 512  # Max generation length
    temperature: float = 0.8  # Sampling temperature

    # Training settings
    num_iterations: int = 1000  # Rollout-train cycles
    ppo_epochs: int = 2  # Epochs per rollout
    minibatch_size: int = 64  # Batch size for gradients

    # Loss coefficients
    kl_coef: float = 0.1  # KL penalty weight

    # Optimization
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
```

### Key Hyperparameters

- **`rollout_batch_size`**: Number of unique prompts per iteration
  - Typical: 64-512
  - More prompts = better diversity, more computation

- **`group_size`**: Completions per prompt
  - Typical: 4-8
  - More completions = better advantage estimates, more computation

- **`kl_coef`**: KL divergence penalty
  - Typical: 0.05-0.2
  - Higher = stays closer to reference, less exploration
  - Lower = more exploration, risk of forgetting

- **`ppo_epochs`**: Training epochs per rollout
  - Typical: 1-4
  - More epochs = better optimization, risk of overfitting to rollout

- **`learning_rate`**: Policy learning rate
  - Typical: 1e-6 to 1e-5
  - Start small, especially if reference is a good SFT model

## Usage Example

### 1. Prepare Prompt Dataset

Create a JSON file with prompts:

```json
{
  "prompts": [
    "Write a function to calculate fibonacci numbers:",
    "Explain how binary search works:",
    ...
  ]
}
```

### 2. Define Reward Function

```python
def my_reward_function(completions: List[List[int]]) -> jax.Array:
    """Compute rewards for completions.

    Args:
        completions: List of token sequences

    Returns:
        Array of rewards [num_completions]
    """
    rewards = []
    for tokens in completions:
        # Example: reward based on length
        reward = compute_task_specific_reward(tokens)
        rewards.append(reward)

    return jnp.array(rewards)
```

Common reward functions:
- **Code generation**: Execute code, return 1.0 if passes tests, 0.0 otherwise
- **Format validation**: Check JSON/XML structure, return 1.0 if valid
- **External model**: Use a reward model to score completion quality
- **Rule-based**: Check for specific patterns, keywords, constraints

### 3. Run Training

```bash
python examples/train_grpo.py \
    --model_path /path/to/model \
    --dataset_path examples/example_prompts.json \
    --num_iterations 100 \
    --rollout_batch_size 32 \
    --group_size 4 \
    --learning_rate 1e-5 \
    --kl_coef 0.1
```

### 4. Monitor Training

The training script will output:
```
================================================================================
Iteration 1/100
================================================================================

Generating 128 completions...
Computing rewards...
Mean reward: 0.7234
Training on rollout buffer...
Iteration 1 complete:
  Loss: 2.3456
  PG Loss: 1.8234
  KL Div: 0.0523
```

Metrics are saved to `grpo_output/training_metrics.json`.

## Workflow Comparison

### Traditional Supervised Fine-Tuning (SFT)
```
for batch in dataset:
    loss = cross_entropy(model(batch.input), batch.target)
    update_model(loss)
```

### GRPO Training
```
for iteration in range(num_iterations):
    # Generate new data with current policy
    prompts = sample_prompts(dataset)
    completions = generate_multiple_per_prompt(prompts)

    # Score completions
    rewards = reward_function(completions)

    # Compute advantages (group-relative)
    advantages = normalize_by_group(rewards)

    # Train on rollout buffer
    for epoch in range(ppo_epochs):
        for batch in shuffle(completions):
            loss = -advantages * log_probs + kl_penalty
            update_model(loss)
```

## Advanced Features

### Reference Model Modes

The trainer supports three reference model update strategies:

#### 1. Static (Recommended)
```python
GRPOConfig(reference_mode="static")
```
- Reference never changes
- Maximum stability
- Preserves original model capabilities

#### 2. Exponential Moving Average (EMA)
```python
GRPOConfig(reference_mode="ema", ema_alpha=0.99)
```
- Reference smoothly tracks policy
- More exploration allowed
- May drift from original capabilities

#### 3. Periodic Updates
```python
GRPOConfig(reference_mode="periodic", reference_update_freq=500)
```
- Update reference every N iterations
- Middle ground between static and EMA

## Implementation Notes

### Current Limitations

1. **Generation is not batched** - Each completion is generated separately
   - TODO: Implement batched generation for efficiency
   - Can use your existing `ServingLoop` infrastructure

2. **Logprob recomputation simplified** - Training currently uses cached logprobs
   - TODO: Recompute logprobs in training step for proper PPO
   - Needed for multiple PPO epochs

3. **No checkpointing** - Model checkpoints not yet implemented
   - TODO: Add checkpoint saving/loading

4. **Single-device only** - No multi-GPU/TPU sharding yet
   - TODO: Add model parallelism using JAX sharding

### Recommended Extensions

1. **Better Generation**
   - Use batched generation from `models/engine.py`
   - Add KV cache reuse across group completions

2. **Efficient Logprob Computation**
   - Batch logprob computation in training step
   - Recompute with current policy (not cached) for proper PPO

3. **Advanced Rewards**
   - Combine multiple reward signals
   - Use learned reward models
   - Add diversity bonuses

4. **Distributed Training**
   - Shard model across devices
   - Distribute rollout generation
   - Aggregate gradients across workers

## References

- **GRPO Paper**: Group Relative Policy Optimization for language model alignment
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **RLHF**: Training language models to follow instructions with human feedback (Ouyang et al., 2022)

## Troubleshooting

### Common Issues

**Q: Training is very slow**
- A: Reduce `rollout_batch_size` or `group_size`
- A: Implement batched generation
- A: Use smaller model for debugging

**Q: Rewards are not improving**
- A: Check reward function is working correctly
- A: Try lower `kl_coef` for more exploration
- A: Increase `learning_rate`
- A: Verify prompts are diverse enough

**Q: Model forgets original capabilities**
- A: Increase `kl_coef` to stay closer to reference
- A: Use `reference_mode="static"`
- A: Lower `learning_rate`
- A: Reduce number of `ppo_epochs`

**Q: Out of memory errors**
- A: Reduce `rollout_batch_size`
- A: Reduce `max_new_tokens`
- A: Offload reference model to CPU during training phase
- A: Use gradient checkpointing (not yet implemented)

## Next Steps

1. Run example training script with your model
2. Implement task-specific reward function
3. Tune hyperparameters on small dataset
4. Scale up to full training
5. Evaluate on held-out prompts
