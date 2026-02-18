# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReLax is a JAX/Flax implementation of the LLaMA transformer architecture optimized for efficient inference and training. The codebase focuses on variable-length sequence processing, grouped query attention (GQA), efficient KV caching, and production-ready serving infrastructure.

## Development Commands

### Dependencies and Setup
- Install dependencies: `pip install -e .[dev]`
- Project uses Python with JAX/Flax for model implementation
- Core dependencies: tiktoken, jax, flax, numpy, safetensors
- Dev dependencies: pytest, torch, chex, huggingface_hub, fire, black

### Testing
- Run all tests: `pytest`
- Run specific test files: `pytest tests/test_ops.py tests/test_kvcache.py`
- Key test files:
  - `test_model.py` - Model architecture tests (compares against PyTorch reference)
  - `test_ops.py` - JAX operations tests
  - `test_kvcache.py` - KV cache tests
  - `test_llama_tokenizer.py` - Tokenizer tests
  - `test_gqa_benchmark.py` - GQA performance benchmarks

### Code Formatting
- Format code: `black .`

### Running Inference
- Single-host inference: `python inference.py --model_path <path_to_model>`
- Multi-host inference (run on each machine):
  ```bash
  python -m jax.distributed.initialize \
    --coordinator_address=192.168.1.100:1234 \
    --num_processes=4 \
    --process_id=0
  python inference.py --model_path <path_to_model>
  ```
- Async inference example: `python examples/async_inference.py`

## Project Architecture

### Core Components

**Models (`models/`)**
- `models/llama/model.py`: Core LLaMA transformer implementation using Flax
  - `TransformerBlock`: Single transformer layer with attention + FFN
  - `LLaMa`: Full model with token embeddings, layers, and output projection
- `models/llama/config.py`: Model configuration with HuggingFace compatibility
  - `ModelConfig.from_json_file()` loads HuggingFace config.json files
  - Validates GQA constraints (n_heads % n_kv_heads == 0)
- `models/llama/load.py`: Model weight loading utilities for safetensors format
- `models/llama/tokenizer.py`: Tokenizer implementation
- `models/engine.py`: Production inference engine with slot-based batching
  - `ServingLoop`: Event loop-based serving with prefill/decode separation
  - `DecodeState`: Fixed-size slot system for efficient batched generation
- `models/sync_server.py`: Multi-host coordination via SyncServer

**Optimized Operations (`utils/`)**
- `utils/ops.py`: High-performance JAX operations
  - `grouped_query_attention()`: Variable-length GQA with KV caching and per-sequence positions
  - `apply_rotary_emb()` and `apply_rotary_emb_batch()`: RoPE implementations
  - `precompute_freqs_cis()`: Precomputes rotary frequency embeddings with optional scaling
  - `feed_forward()`: Configurable activation functions (SwiGLU, etc.)
  - `rms_norm()`: RMS normalization
  - `build_attn_mask()`: Builds attention masks for variable-length sequences
- `utils/kvcache.py`: Efficient KV cache with per-sequence position tracking
  - `KVCache.update()`: Updates cache for all sequences at their own positions
  - `KVCache.update_positions()`: Increments positions after processing
- `utils/memory.py`: Memory footprint estimation tools
- `utils/mesh_helpers.py`: Multi-device computation utilities
- `utils/padding.py`: Bucketing and padding utilities for efficient batching

**Sampling (`sampling.py`)**
- `temperature_scale()`: Temperature scaling for logits
- `sample_top_k()`: Top-k sampling
- `sample_top_p()`: Nucleus (top-p) sampling

### Key Design Patterns

**Variable-Length Sequence Processing**

The codebase implements sophisticated variable-length sequence handling without requiring uniform padding:

- `grouped_query_attention()` takes `mask` array (shape `[bsz, seqlen, max_seqlen]`) instead of assuming uniform positions
- `KVCache.update()` uses per-sequence positions tracked in `seq_positions` array
  - Each sequence in a batch can be at a different cache position
  - Loop-based updates (unrolled at compile time) with `lax.dynamic_update_slice`
- Attention masks combine:
  - Causal masking: query at position i can only attend to keys at positions ≤ i
  - Valid query masking: only non-padded queries are real
  - Per-sequence position tracking via `seq_positions`

**JAX/Flax Functional Design**

- Uses Flax modules (`nn.Module`) with explicit parameter management
- Extensive use of `@jit` decorators with `static_argnames` for compile-time constants
- `donate_argnums` for memory efficiency (donates arrays to avoid copies)
- Functional programming style with immutable state (uses `flax.struct.dataclass`)
- Custom parameter dataclasses (`AttentionParams`, `FeedForwardParams`)

**Configuration Management**

- `ModelConfig` frozen dataclass with HuggingFace compatibility
- Supports loading models via `ModelConfig.from_json_file()`
- Key parameters:
  - Architecture: `vocab_size`, `dim`, `ffn_hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`
  - Positional: `max_seqlen`, `rope_theta`, `use_scaled_rope`
  - Normalization: `rms_norm_eps`
  - Activation: `activation_fn` (silu, relu, gelu)

**Production Serving Architecture**

The inference engine uses a slot-based batching system:

- **Prefill**: Processes prompts one at a time (sequential)
- **Decode**: Batches multiple sequences using fixed-size slots
  - `DecodeState` maintains fixed batch dimension with "slots"
  - Each slot can be occupied by a generating sequence
  - Enables efficient batching without dynamic reshaping
- **Multi-host support**: Via `SyncServer` for TPU pod coordination
- **Streaming**: Response queues for token-by-token streaming

## Important Implementation Details

**Grouped Query Attention (GQA)**

- Uses `n_kv_heads` < `n_heads` for memory efficiency
- Key/value heads are repeated to match query heads: `n_rep = n_heads // n_kv_heads`
- Manual repetition: `jnp.repeat(keys, n_rep, axis=1)` before attention computation
- Cache stores only `n_kv_heads` (not full `n_heads`)

**KV Caching**

- Shape: `[n_layers, bsz, n_kv_heads, max_seqlen, head_dim]`
- Per-sequence position tracking via `seq_positions` array `[bsz]`
- `update()` method handles variable-length updates:
  - Loops over batch dimension (unrolled at compile time)
  - Uses `lax.dynamic_update_slice` for each sequence at its own position
- Cache positions determine valid attention regions in mask construction

**RoPE (Rotary Positional Embeddings)**

- `precompute_freqs_cis()` with optional scaling support (Llama 3 style)
  - Returns shape `[max_seqlen, head_dim//2, 2]` with cos/sin components
- `apply_rotary_emb_batch()` for per-sequence position application:
  - Takes `freqs_cis` of shape `[bsz, seqlen, head_dim//2, 2]`
  - Allows each sequence to use different position indices
- Complex number representation using stacked cos/sin instead of complex64

**Attention Masking**

`build_attn_mask()` constructs boolean masks `[bsz, seqlen, max_seqlen]`:
- Causal mask: `query_positions[:, None] >= key_positions[None, :]`
- Valid query mask: filters padding in queries via `query_offsets < true_length`
- Combined with `jax.vmap` for batched per-sequence mask building
- Used with `nn.softmax(..., where=mask)` for masked softmax

**Testing Strategy**

Tests compare JAX implementations against PyTorch reference implementations:
- `experiments/torch_llama.py`: PyTorch reference model
- Tests verify numerical equivalence between JAX and PyTorch
- Use `jax.config.update("jax_default_matmul_precision", "highest")` for accuracy

## Model Configuration

Models expect HuggingFace-compatible `config.json` files:
- `num_attention_heads`, `num_key_value_heads`: For GQA configuration
- `rope_scaling.original_max_position_embeddings`: Maximum sequence length
- `rope_scaling.rope_type`: Use `"llama3"` for scaled RoPE
- `hidden_act`: Activation function (silu, relu, gelu)
- `rope_theta`: RoPE frequency base parameter (default 500000.0)
- `torch_dtype`: Model dtype (bfloat16, float32, etc.)

## Performance Considerations

- Use per-sequence position tracking (`seq_positions`) rather than uniform padding
- KV cache updates optimized with compile-time unrolled loops
- Attention masking handled efficiently with boolean masks and `where` parameter
- Memory estimation tools available in `utils.memory.estimate_pytree_memory_footprint()`
- Slot-based batching in serving engine avoids dynamic reshaping
- Multi-step decode (`decode_steps` parameter) amortizes overhead
- Bucketing system in `utils.padding` groups similar-length sequences

## Multi-Host Inference

For TPU pods or multi-GPU setups:
1. Initialize JAX distributed before importing model code:
   ```python
   import jax
   jax.distributed.initialize()
   ```
2. Use `utils.mesh_helpers.MeshHelper` for device mesh creation
3. `SyncServer` coordinates prefill/decode across hosts
4. See `inference.py` for complete multi-host example
