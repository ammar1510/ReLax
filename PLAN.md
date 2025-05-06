# ReLax: LLaMA Inference Plan

This plan outlines the steps to build a simple, JAX-efficient chat interface for the LLaMA model.

## Core Components

1.  **Tokenizer Integration (`tokenizer_utils.py`):**
    *   Interface with a SentencePiece tokenizer (using `sentencepiece` library or `transformers`).
    *   Provide simple `encode` (text -> token IDs) and `decode` (token IDs -> text) functions.
    *   Needs the specific tokenizer model file (`tokenizer.model`).

2.  **Sampling Logic (`sampling.py`):**
    *   Implement core sampling functions operating on logits (`jax.Array`):
        *   `temperature_scale(logits, temperature)`
        *   `sample_top_p(logits, p, key)`
        *   `sample_top_k(logits, k, key)` (Potentially combine with top-p or use separately)
    *   These functions will use `jax.random` for pseudo-random number generation (requires passing PRNG keys).
    *   Aim for simplicity and efficiency suitable for JAX compilation.

3.  **Model Loading (`load_utils.py`):**
    *   Function(s) to load pre-trained LLaMA weights (e.g., from `.safetensors` or PyTorch `.bin`).
    *   Map loaded weights to the parameter structure defined in `models/llama.py`.
    *   Leverage `flax.serialization` if possible, otherwise manual mapping.

4.  **Generation Loop (`inference.py`):**
    *   The core autoregressive generation function.
    *   Input: Model parameters, tokenizer, initial prompt (token IDs), `ModelArgs`, sampling parameters (temp, top_p/top_k), max length, PRNG key.
    *   Initialization:
        *   Initialize `KVCache`.
        *   Process the initial prompt through the model to populate the `KVCache`.
    *   Generation Step (inside a JAX-compilable loop, potentially using `jax.lax.scan` for efficiency):
        *   Get logits for the *last* token.
        *   Apply sampling logic (`sampling.py`) to get the next token ID.
        *   Check for EOS token.
        *   Run the model on *only the new token*, passing the updated `KVCache` and correct `start_pos`.
        *   Update `KVCache` and the sequence of generated tokens.
    *   Output: Completed sequence of token IDs.
    *   **Crucially, this function should be decorated with `@jax.jit` for maximum performance.**

5.  **Chat Interface (`chat.py`):**
    *   Main executable script.
    *   Argument parsing (`argparse`) for model path, tokenizer path, sampling params, etc.
    *   Load model (`load_utils.py`) and tokenizer (`tokenizer_utils.py`).
    *   Initialize model parameters on the target device (CPU/GPU/TPU).
    *   Simple terminal loop:
        *   Get user input.
        *   Format input (add chat templates if needed, e.g., `[INST] ... [/INST]`).
        *   Encode prompt using tokenizer.
        *   Call the JIT-compiled generation function (`inference.py`).
        *   Decode the output sequence using the tokenizer.
        *   Print the response streamingly (token-by-token) or all at once.
        *   Handle basic commands (e.g., `/exit`, `/clear`).

## Proposed Workflow

1.  Implement `tokenizer_utils.py`.
2.  Implement `sampling.py` functions using JAX.
3.  Implement `load_utils.py` (this might depend heavily on the specific checkpoint format).
4.  Develop and `@jax.jit` the `inference.py` generation loop.
5.  Build the user-facing `chat.py` script, integrating all components.

## Design Principles

*   **Minimalism:** Only include essential features for a functional chat demo.
*   **JAX Efficiency:** Leverage `jax.jit`, functional updates, and efficient JAX operations (`lax.dynamic_slice`, `lax.scan` if applicable) where possible, especially in the core generation loop.
*   **Clarity:** Keep the code clean and understandable.
*   **Shardability:** Structure the codebase to facilitate potential future integration with `jax.sharding` for distributed inference across multiple devices. 