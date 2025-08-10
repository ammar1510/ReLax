# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReLax is a JAX-based implementation of the LLaMA transformer architecture optimized for efficient inference and training. The codebase focuses on variable-length sequence processing, grouped query attention (GQA), and efficient KV caching.

## Development Commands

**Testing:**
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_ops.py
pytest tests/test_kvcache.py
pytest tests/test_gqa_benchmark.py

# Run with specific Python path (already configured in pyproject.toml)
python -m pytest tests/
```

**Code Formatting:**
```bash
# Format code (black is in dev dependencies)
python -m black .
```

**Model Generation:**
```bash
# Generate text using the model
python generate.py --model_path <path_to_model> --prompt "Your prompt here"

# Interactive chat interface
python chat.py
```

## Architecture Overview

### Core Components

**Models (`models/`):**
- `models/llama/model.py`: Core LLaMA transformer implementation using Flax
- `models/llama/config.py`: Model configuration with support for loading HuggingFace configs
- `models/llama/load.py`: Model weight loading utilities
- `models/llama/tokenizer.py`: Tokenizer implementation

**Optimized Operations (`utils/`):**
- `utils/ops.py`: High-performance JAX operations including:
  - `grouped_query_attention()`: Variable-length GQA with KV caching
  - `apply_rotary_emb()` and `apply_rotary_emb_batch()`: RoPE implementations
  - `feed_forward()`: Configurable activation functions (SwiGLU, etc.)
  - `rms_norm()`: RMS normalization
- `utils/kvcache.py`: Efficient KV cache with per-sequence position tracking
- `utils/sharding.py`: Multi-device computation utilities
- `utils/memory.py`: Memory footprint estimation tools

**Trainers (`trainers/`):**
- `trainers/trainer.py`: Abstract base trainer with custom TrainState
- `trainers/sft_trainer.py`: Supervised fine-tuning implementation
- `trainers/grpo_trainer.py`: Group Relative Policy Optimization trainer

### Key Design Patterns

**Variable-Length Sequence Processing:**
The codebase implements sophisticated variable-length sequence handling:
- `grouped_query_attention()` takes `seq_lengths` array instead of uniform `start_pos`
- `KVCache.update_batch()` uses `jax.vmap` for per-sequence cache updates
- Attention masks combine causal, validity, and sequence-length constraints
- JAX's `nn.dot_product_attention` handles GQA automatically

**JAX Optimization:**
- Extensive use of `@jit` decorators with static arguments
- `donate_argnums` for memory efficiency
- Structured arrays using `flax.struct.dataclass`
- Functional programming patterns for JAX transformations

**Configuration Management:**
- `ModelConfig` dataclass with HuggingFace compatibility
- Support for loading models via `ModelConfig.from_json_file()`
- Validation of GQA constraints (n_heads % n_kv_heads == 0)

## Important Implementation Details

**Grouped Query Attention:**
- Uses `n_kv_heads` < `n_heads` for memory efficiency
- JAX's dot_product_attention automatically handles head repetition
- No manual `repeat_kv` calls needed when using `nn.dot_product_attention`

**KV Caching:**
- Per-sequence position tracking via `positions` array in `KVCache`
- `update_batch()` method handles variable-length updates using vmap
- Cache positions determine valid attention regions

**RoPE (Rotary Positional Embeddings):**
- `precompute_freqs_cis()` with optional scaling support  
- `apply_rotary_emb_batch()` for per-sequence position application
- Complex number representation using stacked cos/sin components

**Testing Strategy:**
Tests compare JAX implementations against PyTorch reference implementations to ensure correctness. The `experiments/` directory contains PyTorch comparison code.

## Model Configuration

Models expect HuggingFace-compatible config.json files with these key parameters:
- `num_attention_heads`, `num_key_value_heads`: For GQA configuration
- `rope_scaling.original_max_position_embeddings`: Maximum sequence length
- `hidden_act`: Activation function (silu, relu, gelu)
- `rope_theta`: RoPE frequency base parameter

## Performance Considerations

- Use `seq_lengths` arrays for variable-length processing rather than uniform padding
- KV cache updates are optimized with vmap for batch processing
- Attention masking is handled efficiently by JAX's native implementation
- Memory estimation tools available in `utils.memory` for monitoring usage