# JIT Compatibility Test Plan for `utils/ops.py`

## 1. Goal

The primary objective is to ensure that all public functions within `utils/ops.py` are compatible with JAX's Just-In-Time (JIT) compilation. This involves verifying two key aspects:
1.  The functions can be successfully decorated with `@jax.jit`.
2.  The output of the jitted functions remains numerically identical to their non-jitted (eager) counterparts.

## 2. General Test Strategy

For each function under test, the following steps will be taken:
1.  Define a jitted version of the function using `jitted_function = jax.jit(original_function)`.
2.  Generate a set of realistic, shared input tensors (and parameters).
3.  Execute both the original and the jitted versions of the function with these inputs.
4.  Compare the outputs from both versions using `np.testing.assert_allclose` to ensure numerical consistency.

## 3. Test Cases by Function

### 3.1 `rms_norm`

-   **Test:** `test_jit_rms_norm`
-   **Description:** A straightforward test to verify that the basic `rms_norm` function can be jitted.
-   **Inputs:** A random input tensor `x` and a weight tensor.
-   **Verification:** Compare the output of `rms_norm(x, weight)` with `jitted_rms_norm(x, weight)`.

### 3.2 `apply_rotary_emb`

-   **Test:** `test_jit_apply_rotary_emb`
-   **Description:** Ensure that the rotary embedding logic is JIT-compatible.
-   **Inputs:** A random input tensor `x` and a precomputed `freqs_cis` tensor.
-   **Verification:** Compare `apply_rotary_emb(x, freqs_cis)` with `jitted_apply_rotary_emb(x, freqs_cis)`.

### 3.3 `repeat_kv`

-   **Test:** `test_jit_repeat_kv`
-   **Description:** Verify JIT compilation for the KV head repetition logic. The `n_rep` argument must be treated as a static argument for the jitted function.
-   **Setup:** `jitted_repeat_kv = jax.jit(repeat_kv, static_argnums=(1,))`
-   **Inputs:** A random tensor `x` for keys/values and an integer `n_rep`.
-   **Verification:** Compare `repeat_kv(x, n_rep)` with `jitted_repeat_kv(x, n_rep)`.

### 3.4 `feed_forward`

-   **Test:** `test_jit_feed_forward`
-   **Description:** Verify JIT for the feed-forward network. The `activation_fn` string argument must be static.
-   **Setup:** `jitted_feed_forward = jax.jit(feed_forward, static_argnames=('activation_fn',))`
-   **Inputs:** A random input tensor `x`, a `FeedForwardParams` dataclass, and an activation function name (e.g., `'silu'`).
-   **Verification:** Compare `feed_forward(x, params, 'silu')` with `jitted_feed_forward(x, params, 'silu')`.

### 3.5 `grouped_query_attention`

This is the most critical function and requires testing in its two primary operational modes.

**Note on `layer_idx`:** Contrary to common JAX patterns where indices used in control flow must be static, `layer_idx` does not need to be a static argument. This is because it is used as a data-dependent index into `jax.lax.dynamic_update_slice`, which is designed to handle dynamic indices. As it doesn't change the computational graph's structure (i.e., the `jaxpr`), JAX can compile a single, general version of the function that does not recompile for different values of `layer_idx`.

#### Scenario A: Prefill Mode

-   **Test:** `test_jit_attention_prefill`
-   **Description:** Test the JIT-compiled attention function during a prefill step, which involves a multi-token sequence and a padding mask.
-   **Setup:**
    -   `jitted_gqa = jax.jit(grouped_query_attention)`
    -   `start_pos = 0`
    -   `seqlen > 1` (e.g., 64)
    -   A `prefill_mask` is created and passed to the function.
-   **Inputs:** Input tensor `x`, `freqs_cis`, `AttentionParams`, an initial `KVCache`, `layer_idx=0`, `start_pos=0`, and the `prefill_mask`.
-   **Verification:**
    -   Compare the output tensor from the jitted and non-jitted functions.
    -   Compare the updated `k` and `v` tensors from the `KVCache` returned by both functions.

#### Scenario B: Decode Mode

-   **Test:** `test_jit_attention_decode`
-   **Description:** Test the JIT-compiled attention function during a single-token decoding step.
-   **Setup:**
    -   `jitted_gqa = jax.jit(grouped_query_attention)`
    -   `start_pos > 0` (e.g., 64, simulating that tokens are already in the cache)
    -   `seqlen = 1`
    -   `prefill_mask = None`
-   **Inputs:** A single-token input tensor `x`, `freqs_cis`, `AttentionParams`, a `KVCache` pre-filled with some data, `layer_idx=0`, and `start_pos=64`.
-   **Verification:**
    -   Compare the output tensor from the jitted and non-jitted functions.
    -   Compare the updated `k` and `v` tensors from the `KVCache` returned by both functions. 