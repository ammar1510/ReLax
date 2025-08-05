# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies and Setup
- Install dependencies: `pip install -e .[dev]`
- Project uses Python with JAX/Flax for model implementation
- Dependencies include: tiktoken, jax, flax, numpy, safetensors
- Dev dependencies: pytest, torch, chex, huggingface_hub, fire, black

### Testing
- Run tests: `pytest`
- Test files are in `tests/` directory
- Key test files:
  - `test_model.py` - Model architecture tests
  - `test_ops.py` - JAX operations tests  
  - `test_llama_tokenizer.py` - Tokenizer tests
  - `test_kvcache.py` - KV cache tests

### Code Formatting
- Format code: `black .`

### Running the Model
- Generate text: `python generate.py --ckpt_dir <checkpoint_dir> --tokenizer_path <tokenizer_path>`
- Chat interface: `python chat.py`

## Project Architecture

### Core Components

**Models (`models/`)**
- `models/llama/` - LLaMA model implementation in JAX/Flax
  - `model.py` - Main model classes (LLaMa, TransformerBlock)
  - `config.py` - Model configuration dataclasses
  - `load.py` - Weight loading utilities
  - `tokenizer.py` - Tokenizer implementation

**Training Framework (`trainers/`)**
- Modular training architecture following abstract base class pattern
- `trainer.py` - Abstract Trainer base class with TrainState dataclass
- `sft_trainer.py` - Supervised Fine-Tuning implementation
- `grpo_trainer.py` - Group Relative Policy Optimization trainer
- Training follows functional programming with JAX transformations

**Utilities (`utils/`)**
- `ops.py` - Core JAX operations (attention, RMSNorm, RoPE, feed-forward)
- `kvcache.py` - KV cache implementation for efficient inference
- `sharding.py` - Model sharding utilities
- `memory.py` - Memory estimation utilities

### Key Architecture Patterns

**JAX/Flax Design**
- Uses Flax modules with explicit parameter management
- JAX transformations (jit, grad) for performance
- Functional programming style with immutable state
- Custom TrainState dataclass for training state management

**Attention Mechanism**
- Grouped Query Attention (GQA) implementation
- RoPE (Rotary Position Embeddings) for positional encoding
- Flash attention optimization for TPU
- KV caching for efficient autoregressive generation

**Training Framework**
- Abstract Trainer base class for modularity
- Separate trainer implementations (SFT, GRPO)
- Configuration-driven approach with dataclasses
- JAX-native training loops with proper state management

### Model Configuration
- Uses dataclasses for type-safe configuration
- ModelConfig handles all model hyperparameters
- Supports various activation functions (SiLU, ReLU, GELU)
- Configurable attention patterns and FFN dimensions

### Generation Pipeline
- `generate.py` - Main generation script with configurable sampling
- `sampling.py` - Top-p sampling implementation
- `chat.py` - Interactive chat interface
- Support for temperature and top-p sampling strategies