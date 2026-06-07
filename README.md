# ReLax

**RL post-training for LLMs on a JAX/Flax stack — built to teach language models to play [Factorio](https://factorio.com).**

ReLax is a from-scratch JAX/Flax implementation of the LLaMA (and Gemma/Qwen) transformer family, paired with a production-grade inference engine and a Group Relative Policy Optimization (GRPO) trainer. The headline use case is **agentic reinforcement learning in the Factorio Learning Environment (FLE)**: the model writes Python programs that are executed inside live Factorio game instances, and the resulting game state is turned into a reward signal that drives on-policy GRPO updates.

The same machinery also trains on math reasoning (GSM8K) and serves plain inference, but the project is organized around the RL loop.

---

## Table of Contents

- [Why Factorio](#why-factorio)
- [How the RL loop works](#how-the-rl-loop-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Factorio RL training](#factorio-rl-training)
- [GSM8K RL training](#gsm8k-rl-training)
- [Inference](#inference)
- [Multi-host / TPU notes](#multi-host--tpu-notes)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Security & operational notes](#security--operational-notes)

---

## Why Factorio

Factorio is a factory-automation game where success requires long-horizon planning, spatial reasoning, and composing primitive actions (mining, smelting, belts, inserters, power) into working production chains. The [Factorio Learning Environment (FLE)](https://github.com/JackHopkins/factorio-learning-environment) exposes the game through a Python API, which makes it a rich, *verifiable* RL benchmark: a completion either builds a working factory or it doesn't, and the environment can score it directly.

ReLax uses FLE in a **single-turn** setting: the policy model is given a task description plus an API reference, it emits one Python program, and that program is executed in a fresh Factorio instance. The outcome is mapped to a scalar reward. Because the reward is computed by the game rather than a learned reward model, GRPO can optimize against it cleanly.

The graded reward (`scripts/factorio_env.py`):

| Reward | Meaning |
|-------:|---------|
| `1.0`  | Task objective verified as achieved |
| `0.5`  | Code ran cleanly and made positive progress |
| `0.25` | Code ran cleanly but made no measurable progress |
| `0.1`  | Code errored but caused some state change |
| `0.0`  | Failed to parse/execute, or empty output |

Tasks live in `scripts/factorio_tasks.py`: a handful of hand-written tasks (mine iron, smelt iron, craft gears, green circuits, power setup) plus an optional loader for FLE's **Lab-Play** benchmark suite. Each task ships with a system prompt embedding a compact [Factorio API summary](scripts/factorio_tasks.py) so the model knows the available `move_to`, `place_entity`, `connect_entities`, `craft_item`, etc.

---

## How the RL loop works

GRPO (Group Relative Policy Optimization, DeepSeek-style) is an on-policy RL algorithm that drops the separate value/reward model used in classic PPO-RLHF. Each iteration runs four phases:

1. **Rollout** — For each prompt, sample `group_size` completions with temperature sampling, using the batched `ServingLoop` inference engine. Reference-model and old-policy logprobs are cached during rollout.
2. **Reward** — Execute each completion in its environment (Factorio code execution, or GSM8K answer matching) to produce a scalar reward.
3. **Advantage** — Normalize rewards *within each prompt group*: `advantage = (reward - group_mean) / group_std`. This removes per-prompt baseline bias without a critic.
4. **Optimize** — Policy-gradient update weighted by advantage, with a PPO-style importance-ratio clip (`clip_epsilon`) and a KL penalty toward the reference model (`kl_coef`).

See [`trainers/README.md`](trainers/README.md) for a deep dive on the algorithm, memory profile, and reference-model modes (`static` / `ema` / `periodic`).

---

## Repository layout

```
models/
  engine.py            # InferenceEngine + ServingLoop: slot-based batched serving
  sync_server.py       # Multi-host coordination (JAX distributed)
  llama/               # LLaMA model.py, config.py, load.py, tokenizer.py
  gemma/  qwen/         # Gemma & Qwen variants

trainers/
  trainer.py           # Base Trainer + TrainState (optax)
  grpo_trainer.py      # GRPO implementation (rollout → reward → advantage → optimize)
  README.md            # GRPO algorithm & tuning guide

scripts/
  train_factorio.py    # ⭐ GRPO on the Factorio Learning Environment
  factorio_env.py      #     FLE cluster lifecycle + reward evaluator
  factorio_tasks.py    #     Task definitions + prompt construction
  train_gsm8k.py       # GRPO on GSM8K math reasoning
  convert_*_to_orbax.py# HF checkpoint → Orbax (sharded) conversion
  code_executor.py     # Sandboxed code execution helper

utils/
  ops.py               # GQA, RoPE, RMSNorm, feed-forward
  kvcache.py           # Per-sequence-position KV cache
  mesh_helpers.py      # Device-mesh helpers for DP/TP sharding
  padding.py memory.py # Bucketing/padding + memory estimation

inference.py           # Single/multi-host LLaMA inference entrypoint
sampling.py            # greedy / top-k / top-p / categorical samplers
```

---

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/). **Always run scripts via `uv run python ...`.**

```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -e .[dev]
```

On TPU, add the TPU JAX wheel:

```bash
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

The Factorio path additionally requires `factorio-learning-environment>=0.3.0` (already a declared dependency) plus a working Factorio install/cluster that the `fle` CLI can manage. See the [FLE docs](https://github.com/JackHopkins/factorio-learning-environment) for game-server setup.

> ⚠️ **Dependency security note:** `factorio-learning-environment` and `gymnasium` are third-party packages and a core part of this stack. Before installing in any shared/CI/production environment, verify them against your organization's internal-approved tools list and raise an IT/security ticket if they are not yet approved. Prefer pinned, audited versions.

---

## Factorio RL training

```bash
uv run python scripts/train_factorio.py \
    --model_path "$HOME/Llama-3.1-8B-Instruct"
```

What happens:
1. `jax.distributed.initialize()` runs and a DP×TP device mesh is built automatically from the available device count (1 → 4×4).
2. The model + Orbax weights + tokenizer are loaded.
3. The FLE cluster is started automatically (`fle cluster start`) on the main process and torn down at exit.
4. Tasks are loaded (simple + optional Lab-Play), prompts are tokenized, and GRPO training runs for `NUM_ITERATIONS`.
5. Metrics stream to Weights & Biases; checkpoints write to `./factorio_output/checkpoints`.

Useful flags:

| Flag | Effect |
|------|--------|
| `--no_wandb` | Disable W&B logging |
| `--no_labplay` | Skip the FLE Lab-Play benchmark tasks |
| `--no_manage_cluster` | Don't auto start/stop the FLE cluster (manage it yourself) |
| `--wandb_project` / `--wandb_run_name` | Override W&B project/run names |

Training hyperparameters are currently set at the top of `scripts/train_factorio.py` (rollout batch 8, group size 4, `max_new_tokens` 1024, `max_cache_seqlen` 2048 — env evaluation is slow, so batches are small). Edit them there.

The model output is parsed for a ```` ```python ```` code block (falling back to the whole completion); that code is the program executed in Factorio.

---

## GSM8K RL training

A faster RL loop with a deterministic reward (exact answer match), useful for validating the trainer before committing to slow Factorio rollouts:

```bash
uv run python scripts/train_gsm8k.py \
    --model_path "$HOME/Llama-3.1-8B-Instruct" \
    --checkpoint_path gs://<your-bucket>/llama-3.1-8b-instruct
```

Defaults (top of `scripts/train_gsm8k.py`): rollout batch 64, group size 16, `lr` 3e-6, 500 iterations, checkpoints to GCS.

---

## Inference

Plain batched generation through the slot-based serving engine:

```bash
uv run python inference.py \
    --model_path "$HOME/Llama-3.1-8B-Instruct" \
    --checkpoint_path gs://<your-bucket>/llama-3.1-8b-instruct \
    --dp 4 --tp 4 \
    --max_decode_length 1024
```

| Arg | Meaning |
|-----|---------|
| `--model_path` | Directory with `config.json` and `tokenizer.model` |
| `--checkpoint_path` | Orbax checkpoint (local or `gs://`) |
| `--dp` / `--tp` | Data-parallel / tensor-parallel mesh dims (product must equal device count) |
| `--max_decode_length` | Max tokens generated per request |

The engine (`models/engine.py`) separates **prefill** (prompts processed individually) from **decode** (fixed-size slots batched together), with per-sequence KV-cache positions so sequences of different lengths share a batch without uniform padding. Gemma uses `inference_gemma.py`; Qwen uses `qwen_inference.py`.

### Getting weights

HF checkpoints are converted to sharded Orbax format before serving/training:

```bash
uv run python scripts/convert_llama_to_orbax.py --repo meta-llama/Llama-3.1-8B-Instruct \
    --gcs_path gs://<your-bucket>/llama-3.1-8b-instruct --tp 16
```

(Analogous `convert_gemma_to_orbax.py` / `convert_qwen_to_orbax.py` exist.)

---

## Multi-host / TPU notes

- Entry scripts call `jax.distributed.initialize()` at startup; you cannot import `inference.py` without a distributed context.
- `models/sync_server.py` (`SyncServer`) coordinates prefill/decode and barriers across hosts.
- For manual multi-host launch, run on each machine with matching `--coordinator_address` / `--num_processes` / `--process_id` (see [`CLAUDE.md`](CLAUDE.md)).
- Mesh axes are `("dp", "tp")`; `utils/mesh_helpers.py` builds them.

---

## Configuration reference

`GRPOConfig` (`trainers/grpo_trainer.py`) — key fields:

| Field | Default | Notes |
|-------|--------:|-------|
| `rollout_batch_size` | 64 | Unique prompts per iteration |
| `group_size` | 16 | Completions per prompt (≥2 required for advantages) |
| `max_new_tokens` | 512 | Generation length |
| `max_cache_seqlen` | 1024 | KV cache length = prompt + generated |
| `temperature` | 0.7 | Rollout sampling temperature |
| `kl_coef` (β) | 0.1 | KL penalty toward reference |
| `clip_epsilon` (ε) | 0.2 | PPO importance-ratio clip |
| `reference_mode` | `static` | `static` / `ema` / `periodic` |
| `learning_rate` | 1e-5 | Policy LR |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `decode_steps` | 10 | Tokens per multistep decode call |

Model config (`ModelConfig`, `models/llama/config.py`) loads HuggingFace `config.json` via `ModelConfig.from_json_file()` and validates GQA constraints (`n_heads % n_kv_heads == 0`).

---

## Testing

```bash
uv run pytest                                   # all tests
uv run pytest tests/test_kvcache.py             # a specific file
```

Tests compare the JAX implementation against a PyTorch reference (`experiments/torch_llama.py` / `hf_reference.py`) for numerical equivalence.

> Note: `tests/test_ops.py` currently has a pre-existing import error (`repeat_kv` was removed from `utils.ops`).

---

## Security & operational notes

- **Do not commit secrets.** The local shell scripts (`run_gsm8k.sh`, `setup.sh`, `llama_setup.sh`) now read credentials from the environment rather than hardcoding them. Provide them at runtime from your secrets manager:
  ```bash
  export WANDB_API_KEY=...   # from your secrets store, not source control
  export HF_TOKEN=...
  ```
  These scripts previously contained live W&B / HuggingFace / GitHub tokens. Even though they were never committed to git, **rotate any token that was ever written to disk** — treat it as compromised.
- **Generated code is executed.** The Factorio reward path runs model-produced Python inside FLE game instances. Keep that execution sandboxed/isolated and never point it at production infrastructure.
- Route all training/inference deployments through your approved deployment pipeline; don't install or run directly on production systems.

---

## License

See repository for license details.
