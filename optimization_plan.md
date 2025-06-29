# TPU Performance Optimization Plan

This document outlines a plan to optimize the LLaMa model implementation for better performance and utilization on TPUs. The current TPU utilization is around 0.2%, indicating significant room for improvement.

The following are key areas for optimization, ordered by potential impact:

### 1. Replace Manual Attention with Pallas Flash Attention

-   **File to Modify:** `utils/ops.py`
-   **Function:** `grouped_query_attention`
-   **Current Implementation:** Uses `nn.dot_product_attention`, a general-purpose JAX implementation.
-   **Proposed Change:** Replace `nn.dot_product_attention` with the highly optimized `jax.experimental.pallas.ops.tpu.flash_attention`.
-   **Justification:** Flash Attention is a memory-aware attention algorithm that is significantly faster on TPUs, especially for long sequences during the prefill stage. It avoids materializing the large `(seqlen, seqlen)` attention matrix, which is a major bottleneck for memory I/O. This is likely the single most impactful change for prefill performance.

### 2. Unroll Transformer Loop with `jax.lax.scan`

-   **File to Modify:** `models/llama/model.py`
-   **Function:** `LLaMa.__call__`
-   **Current Implementation:** Uses a standard Python `for` loop to iterate through the `TransformerBlock` layers.
-   **Proposed Change:** Refactor the loop over the transformer layers to use `jax.lax.scan`.
-   **Justification:** A Python `for` loop forces JAX to compile and execute each layer's computation graph one by one. By using `jax.lax.scan`, the entire sequence of layers is represented as a single, unrolled computation graph. This allows the JAX compiler to perform more aggressive, cross-layer optimizations like instruction fusion and scheduling, leading to better TPU utilization.

### 3. Implement Custom Fused Kernels with Pallas

-   **Justification:** Many operations in the current code are memory-bound (i.e., the time is spent reading/writing to memory, not on computation). Pallas allows for the creation of custom, fused kernels that combine multiple operations into a single TPU kernel launch, drastically reducing memory bandwidth requirements.
-   **Target Functions for Fusion:**
    -   **`apply_rotary_emb` in `utils/ops.py`**: The current implementation involves several separate operations (reshaping, arithmetic, stacking, reshaping again). These can be fused into a single Pallas kernel to perform the rotary embedding transformation in one pass without creating intermediate tensors.
    -   **`feed_forward` in `utils/ops.py`**: The SwiGLU MLP consists of three `einsum` operations and element-wise activations. This is a prime candidate for a fused kernel that combines the matrix multiplications and the non-linear activation function into a single, efficient operation.
    -   **`rms_norm` in `utils/ops.py`**: RMS Normalization is a memory-bound operation. It can be fused with the subsequent operation (e.g., the QKV projection `einsum`s inside `grouped_query_attention`) to eliminate a separate memory round-trip.

### 4. Optimize KV Cache Implementation

-   **File to Modify:** `utils/kvcache.py` (and its usage in `utils/ops.py`)
-   **Justification:** The efficiency of the Key-Value cache is critical for fast decoding performance.
-   **Proposed Changes:**
    -   Ensure the cache is pre-allocated to the maximum sequence length to avoid repeated re-allocations.
    -   Use `jax.lax.dynamic_update_slice` for in-place updates to avoid creating new copies of the cache at each step.
    -   Verify that the data layout of the cache is optimized for TPU memory access patterns (e.g., matching the tiling of the attention computation).

### 5. Review and Standardize Data Types

-   **Files to Modify:** `models/llama/model.py`, `utils/ops.py`
-   **Justification:** The use of `jnp.float64` is highly inefficient on TPUs. The most performant data type for training and inference on modern TPUs is `jnp.bfloat16`.
-   **Proposed Changes:**
    -   Remove the global `jax_enable_x64` flag.
    -   Conduct a thorough review to ensure all computations (weights, activations, intermediate values) are performed using `jnp.bfloat16` or `jnp.float32` where precision is strictly necessary. The `freqs_cis` calculation, for instance, should be done in `float32` and then cast, but the main model parameters and activations should be `bfloat16`. 

### 6. Aggressive JIT Compilation with XLA-Specific Optimizations

-   **Files to Modify:** `models/llama/model.py`, `utils/ops.py`, `generate.py`
-   **Current Gap:** Basic JIT usage is missing several XLA-specific optimizations that can dramatically improve TPU utilization.
-   **Proposed Optimizations:**
    -   Add `donate_argnums` to JIT functions to tell XLA it can reuse input memory for outputs, reducing memory allocations. Critical for KV cache updates.
    -   Use `jax.checkpoint` (remat) for gradient computation to reduce memory usage during training, allowing larger batch sizes.
    -   Configure XLA compiler flags: Set `jax.config.update('jax_default_matmul_precision', 'tensorfloat32')` for TPU-optimized matmul precision.
    -   Add `device_put` with explicit sharding to ensure data is placed optimally on TPU cores.
-   **Example Implementation:** `@partial(jax.jit, static_argnames=['model'], donate_argnums=(2,))` for model forward pass to donate KV cache memory.

### 7. Memory Layout and Tensor Sharding Optimizations

-   **Files to Modify:** `utils/sharding.py`, `models/llama/model.py`
-   **Current Implementation:** Basic sharding utilities exist but aren't used effectively.
-   **Proposed Changes:**
    -   Implement parameter sharding to distribute model weights across TPU cores using `jax.sharding.NamedSharding`.
    -   Optimize tensor layouts for TPU tiling - TPUs perform best with specific tensor shapes that align with their 128x128 matrix units.
    -   Add sequence parallelism for long sequences, partitioning the sequence dimension across TPU cores.
    -   Implement activation sharding to shard intermediate activations and reduce memory per core.
-   **Justification:** Proper sharding can fix the 0.2% utilization issue by ensuring all TPU cores are actively used.

### 8. Advanced Einsum Optimizations

-   **Files to Modify:** `utils/ops.py`
-   **Current Issue:** 7 separate einsum operations that could be optimized.
-   **Specific Optimizations:**
    -   Replace einsum with `jax.lax.dot_general` for more efficient specific contractions.
    -   Batch einsum operations where possible to combine multiple einsums.
    -   Use `jax.lax.conv_general_dilated` for certain attention patterns that can be implemented as convolutions.
    -   Optimize einsum equation formats by reordering indices for better memory access patterns.
-   **Expected Impact:** Quick wins with significant performance improvements for matrix operations.

### 9. Custom Attention Mask Optimization

-   **Files to Modify:** `utils/ops.py` (in `grouped_query_attention`)
-   **Current Issue:** Mask construction is inefficient - creating masks dynamically each time.
-   **Proposed Changes:**
    -   Pre-compute causal masks and cache triangular masks for common sequence lengths.
    -   Use `jax.lax.select` instead of `jnp.where` for more efficient conditional operations.
    -   Implement block-sparse attention patterns for very long sequences.
-   **Justification:** Mask operations are currently memory-bound and recomputed unnecessarily.

### 10. Numerical Stability and Precision Optimizations

-   **Files to Modify:** `utils/ops.py`, `models/llama/model.py`
-   **Current Issues:** Inconsistent precision usage and potential numerical instabilities.
-   **Proposed Changes:**
    -   Use `jnp.bfloat16` consistently except for specific operations requiring higher precision.
    -   Add mixed-precision training support using `jax.lax.Precision.HIGH` for critical operations.
    -   Implement numerically stable softmax using `jax.nn.log_softmax` followed by `jnp.exp`.
    -   Add gradient clipping using `jax.clip_by_global_norm` to prevent gradient explosion.
-   **Justification:** Proper precision management enables higher throughput while maintaining numerical stability.

### 11. Kernel Fusion Beyond Pallas

-   **Files to Modify:** `utils/ops.py`
-   **Additional Fusion Opportunities:**
    -   QKV projection fusion: Combine all three attention projections into a single kernel.
    -   Attention + residual fusion: Combine attention output with residual connection.
    -   Embedding + position encoding fusion: Fuse token embeddings with positional encoding.
    -   Softmax + masking fusion: Combine attention masking with softmax computation.
-   **Justification:** Additional kernel fusion opportunities beyond the basic Pallas implementations for long-term performance gains.

### 12. Compilation and Caching Optimizations

-   **Files to Modify:** `generate.py`, `chat.py`
-   **Current Issue:** Functions are recompiled unnecessarily, causing startup delays.
-   **Proposed Changes:**
    -   Implement persistent compilation cache using `jax.config.update('jax_compilation_cache_dir', '/path/to/cache')`.
    -   Add function versioning to prevent recompilation when function signatures haven't changed.
    -   Pre-compile critical paths by compiling all generation functions at startup.
    -   Use `jax.jit` with `static_argnames` more aggressively, marking more arguments as static where possible.
-   **Expected Impact:** Dramatically reduces startup time and eliminates runtime recompilation overhead.

### 13. TPU-Specific Memory Management

-   **Files to Modify:** `utils/kvcache.py`, `generate.py`
-   **Current Gap:** No TPU-specific memory optimizations implemented.
-   **Proposed Changes:**
    -   Implement memory pooling to reuse memory allocations for repeated operations.
    -   Use `jax.lax.with_sharding_constraint` to ensure optimal memory placement across TPU cores.
    -   Add memory-mapped I/O for large models using memory mapping for model weights that don't fit in HBM.
    -   Implement gradient accumulation to allow training with larger effective batch sizes.
-   **Justification:** TPU memory management differs significantly from GPU and requires specialized approaches.

### 14. Quantization and Compression

-   **Files to Modify:** `models/llama/model.py`, `utils/ops.py`
-   **New Optimization:** Quantization techniques for reduced memory usage and increased throughput.
-   **Proposed Changes:**
    -   Implement INT8 quantization using `jax.experimental.aqt` for weights and activations.
    -   Add dynamic quantization to quantize activations on-the-fly during inference.
    -   Implement key-value cache quantization to reduce memory usage for long sequences.
    -   Add support for mixed bit-width quantization (e.g., 4-bit weights with 8-bit activations).
-   **Expected Impact:** Significant memory reduction allowing larger batch sizes and longer sequences.

## Priority Ranking

**Immediate Impact (Implement First):**
1. JIT + XLA optimizations (#6) - 2-3x improvement likely
2. Memory layout optimization (#7) - Can fix 0.2% utilization issue
3. Einsum optimizations (#8) - Quick wins with significant impact

**Medium Term:**
4. Compilation caching (#12) - Reduces startup time dramatically
5. Numerical precision (#10) - Enables higher throughput
6. Advanced kernel fusion (#11) - Long-term performance gains

**Advanced Optimizations:**
7. Attention mask optimization (#9)
8. TPU memory management (#13)
9. Quantization (#14)

The combination of proper sharding, XLA optimizations, and kernel fusion should achieve 60-80% TPU utilization for compute-bound operations. 