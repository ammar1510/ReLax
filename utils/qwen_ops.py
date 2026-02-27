"""
Qwen3.5-specific JAX operations.

Contains ops that differ from the LLaMA implementations in utils/ops.py:
- qwen_rms_norm: (1 + weight) scaling variant
- apply_partial_rotary_emb_batch: partial RoPE on first rotary_dim dimensions
- gated_deltanet_step / gated_deltanet_prefill: Gated DeltaNet linear attention
- moe_forward: Mixture-of-Experts with shared expert
"""

import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.lax as lax
from jax import jit
from flax import struct
from functools import partial

from .ops import FeedForwardParams, feed_forward


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@struct.dataclass
class DeltaNetParams:
    in_proj_qkv: jax.Array  # [dim, key_dim*2 + value_dim]
    in_proj_z: jax.Array  # [dim, value_dim]
    in_proj_a: jax.Array  # [dim, n_v_heads]
    in_proj_b: jax.Array  # [dim, n_v_heads]
    conv1d_weight: jax.Array  # [conv_dim, kernel_size]
    dt_bias: jax.Array  # [n_v_heads]
    A_log: jax.Array  # [n_v_heads]
    norm_weight: jax.Array  # [value_dim]
    out_proj: jax.Array  # [value_dim, dim]


@struct.dataclass
class MoEParams:
    router_weight: jax.Array  # [num_experts, dim]
    expert_gate_up: jax.Array  # [num_experts, 2 * moe_intermediate, dim]
    expert_down: jax.Array  # [num_experts, dim, moe_intermediate]
    shared_gate: jax.Array  # [dim, shared_intermediate]
    shared_up: jax.Array  # [dim, shared_intermediate]
    shared_down: jax.Array  # [shared_intermediate, dim]
    shared_expert_gate: jax.Array  # [dim, 1]


# ---------------------------------------------------------------------------
# RMS Norm (Qwen variant)
# ---------------------------------------------------------------------------


@jit
def qwen_rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMS normalization with (1 + weight) scaling used by Qwen models.

    Qwen initializes norm weights to zero, so scaling by (1 + weight) gives
    identity at initialization (unlike LLaMA which initializes to ones and
    scales by weight directly).
    """
    output = x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return output * (1.0 + weight)


@jit
def qwen_rms_norm_gated(
    x: jax.Array, gate: jax.Array, weight: jax.Array, eps: float = 1e-6
) -> jax.Array:
    """Gated RMS norm: normalize x, scale by weight, then multiply by silu(gate).

    Used in DeltaNet output normalization.
    """
    normed = x * lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return weight * normed * nn.silu(gate)


# ---------------------------------------------------------------------------
# Partial Rotary Embeddings
# ---------------------------------------------------------------------------


@partial(jit, static_argnames=["rotary_dim"])
def apply_partial_rotary_emb_batch(
    x: jax.Array, freqs_cis: jax.Array, rotary_dim: int
) -> jax.Array:
    """Apply RoPE to only the first rotary_dim dimensions of the head.

    For Qwen3.5: head_dim=256, partial_rotary_factor=0.25, so rotary_dim=64.
    The first 64 dims get rotary embeddings, the remaining 192 pass through unchanged.

    Args:
        x: [bsz, seqlen, n_heads, head_dim]
        freqs_cis: [bsz, seqlen, rotary_dim // 2, 2] (cos/sin for the rotary portion)
        rotary_dim: Number of head dimensions to rotate.

    Returns:
        x with RoPE applied to first rotary_dim dimensions.
    """
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Apply RoPE to the rotary portion
    # x_rot: [bsz, seqlen, n_heads, rotary_dim]
    x_shaped = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
    x_r, x_i = x_shaped[..., 0], x_shaped[..., 1]

    # freqs_cis: [bsz, seqlen, rotary_dim//2, 2] -> [bsz, seqlen, 1, rotary_dim//2, 2]
    freqs_cis = freqs_cis[:, :, None, :, :]
    freqs_cos, freqs_sin = freqs_cis[..., 0], freqs_cis[..., 1]

    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_rotated = jnp.stack([x_out_r, x_out_i], axis=-1).reshape(x_rot.shape)

    return jnp.concatenate([x_rotated, x_pass], axis=-1)


# ---------------------------------------------------------------------------
# Gated DeltaNet (Linear Attention)
# ---------------------------------------------------------------------------


@partial(
    jit,
    static_argnames=["n_k_heads", "k_head_dim", "n_v_heads", "v_head_dim"],
)
def gated_deltanet_step(
    x: jax.Array,
    state: jax.Array,
    conv_state: jax.Array,
    params: DeltaNetParams,
    n_k_heads: int,
    k_head_dim: int,
    n_v_heads: int,
    v_head_dim: int,
) -> tuple:
    """Single-step recurrent inference for Gated DeltaNet.

    Args:
        x: Input [bsz, dim] (single token per sequence).
        state: Recurrent state [bsz, n_v_heads, k_head_dim, v_head_dim].
        conv_state: Conv buffer [bsz, conv_dim, kernel_size - 1].
        params: DeltaNetParams.
        n_k_heads, k_head_dim, n_v_heads, v_head_dim: Static shape info.

    Returns:
        (output [bsz, dim], new_state, new_conv_state)
    """
    bsz = x.shape[0]
    key_dim = n_k_heads * k_head_dim
    value_dim = n_v_heads * v_head_dim
    n_rep = n_v_heads // n_k_heads

    # 1. Input projections
    qkv = jnp.einsum("bd,do->bo", x, params.in_proj_qkv)  # [bsz, key_dim*2 + value_dim]
    z = jnp.einsum("bd,do->bo", x, params.in_proj_z)  # [bsz, value_dim]
    alpha_in = jnp.einsum("bd,do->bo", x, params.in_proj_a)  # [bsz, n_v_heads]
    beta_in = jnp.einsum("bd,do->bo", x, params.in_proj_b)  # [bsz, n_v_heads]

    # 2. Causal conv1d: update rolling buffer and apply depthwise conv
    # conv_state: [bsz, conv_dim, kernel_size - 1]
    # qkv[:, :key_dim*2] goes through conv (q and k portion)
    # Actually the full qkv goes through conv based on HF impl
    # But only q,k portion based on the conv_dim = key_dim*2
    # Let's check: conv_dim in HF = key_dim * 2 (q and k dims, not v)
    # Actually from config: conv_dim = linear_key_dim * 2 + linear_value_dim
    # So ALL of qkv goes through conv
    conv_input = qkv  # [bsz, conv_dim]

    # Shift conv_state left and append new input
    new_conv_state = jnp.concatenate(
        [conv_state[:, :, 1:], conv_input[:, :, None]], axis=-1
    )  # [bsz, conv_dim, kernel_size - 1]

    # Depthwise conv: sum over the kernel window including current input
    # conv1d_weight: [conv_dim, kernel_size]
    # Full window: [conv_state, current_input] = [bsz, conv_dim, kernel_size]
    full_window = jnp.concatenate(
        [new_conv_state, conv_input[:, :, None]], axis=-1
    )  # [bsz, conv_dim, kernel_size]
    conv_out = jnp.sum(full_window * params.conv1d_weight[None, :, :], axis=-1)  # [bsz, conv_dim]

    # Split back into q, k, v after conv
    q_conv = conv_out[:, :key_dim]
    k_conv = conv_out[:, key_dim : key_dim * 2]
    v = qkv[:, key_dim * 2 :]  # v doesn't go through conv based on typical DeltaNet

    # Wait - if conv_dim includes v, then v also went through conv
    # Let me re-check: conv_dim = key_dim*2 + value_dim
    # So actually all of qkv goes through conv, and we split after
    v = conv_out[:, key_dim * 2 :]

    # 3. Activation on q
    q = nn.silu(q_conv)  # [bsz, key_dim]
    k = k_conv  # [bsz, key_dim]

    # 4. Reshape to multi-head
    q = q.reshape(bsz, n_k_heads, k_head_dim)
    k = k.reshape(bsz, n_k_heads, k_head_dim)
    v = v.reshape(bsz, n_v_heads, v_head_dim)

    # 5. Compute gates
    # alpha (forget gate): sigmoid(-exp(A_log)) * sigmoid(alpha_in)
    A = jnp.exp(params.A_log)  # [n_v_heads]
    alpha = nn.sigmoid(-A)[None, :] * nn.sigmoid(alpha_in)  # [bsz, n_v_heads]

    # beta (update gate): sigmoid(beta_in + dt_bias)
    beta = nn.sigmoid(beta_in + params.dt_bias[None, :])  # [bsz, n_v_heads]

    # 6. Delta rule recurrence
    # Expand k from n_k_heads to n_v_heads
    k_expanded = jnp.repeat(k, n_rep, axis=1)  # [bsz, n_v_heads, k_head_dim]

    # Outer product: kv = k^T @ v -> [bsz, n_v_heads, k_head_dim, v_head_dim]
    kv = jnp.einsum("bvk,bvd->bvkd", k_expanded, v)

    # State update: S = alpha * S + beta * kv
    new_state = (
        alpha[:, :, None, None] * state + beta[:, :, None, None] * kv
    )  # [bsz, n_v_heads, k_head_dim, v_head_dim]

    # 7. Read from state: output = q @ S
    q_expanded = jnp.repeat(q, n_rep, axis=1)  # [bsz, n_v_heads, k_head_dim]
    output = jnp.einsum(
        "bvk,bvkd->bvd", q_expanded, new_state
    )  # [bsz, n_v_heads, v_head_dim]

    # 8. Gated RMS norm + output gate
    output_flat = output.reshape(bsz, value_dim)
    output_normed = qwen_rms_norm_gated(
        output_flat, z, params.norm_weight
    )  # [bsz, value_dim]

    # 9. Output projection
    result = jnp.einsum("bd,do->bo", output_normed, params.out_proj)  # [bsz, dim]

    return result, new_state, new_conv_state


def gated_deltanet_prefill(
    x_seq: jax.Array,
    state: jax.Array,
    conv_state: jax.Array,
    params: DeltaNetParams,
    n_k_heads: int,
    k_head_dim: int,
    n_v_heads: int,
    v_head_dim: int,
) -> tuple:
    """Process a sequence through DeltaNet using lax.scan (for prefill).

    Args:
        x_seq: Input [bsz, seqlen, dim].
        state: Initial recurrent state [bsz, n_v_heads, k_head_dim, v_head_dim].
        conv_state: Initial conv buffer [bsz, conv_dim, kernel_size - 1].
        params: DeltaNetParams.
        n_k_heads, k_head_dim, n_v_heads, v_head_dim: Static shape info.

    Returns:
        (output_seq [bsz, seqlen, dim], final_state, final_conv_state)
    """
    # Transpose to [seqlen, bsz, dim] for scanning over time
    x_time = jnp.transpose(x_seq, (1, 0, 2))

    def scan_fn(carry, x_t):
        s, cs = carry
        out, s_new, cs_new = gated_deltanet_step(
            x_t, s, cs, params, n_k_heads, k_head_dim, n_v_heads, v_head_dim
        )
        return (s_new, cs_new), out

    (final_state, final_conv_state), outputs = lax.scan(
        scan_fn, (state, conv_state), x_time
    )

    # outputs: [seqlen, bsz, dim] -> [bsz, seqlen, dim]
    output_seq = jnp.transpose(outputs, (1, 0, 2))

    return output_seq, final_state, final_conv_state


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------


@partial(jit, static_argnames=["num_experts_per_tok", "activation_fn"])
def moe_forward(
    x: jax.Array,
    params: MoEParams,
    num_experts_per_tok: int,
    activation_fn: str = "silu",
) -> jax.Array:
    """Mixture-of-Experts forward pass with shared expert.

    Args:
        x: Input [bsz, seqlen, dim].
        params: MoEParams containing router, expert, and shared expert weights.
        num_experts_per_tok: Number of routed experts per token (top-k).
        activation_fn: Activation function name.

    Returns:
        Output [bsz, seqlen, dim].
    """
    bsz, seqlen, dim = x.shape

    # --- Routed experts ---

    # Router: compute expert probabilities
    router_logits = jnp.einsum(
        "bsd,ed->bse", x, params.router_weight
    )  # [bsz, seqlen, num_experts]
    router_probs = nn.softmax(router_logits.astype(jnp.float32), axis=-1)

    # Top-k selection
    top_k_probs, top_k_indices = lax.top_k(
        router_probs, num_experts_per_tok
    )  # [bsz, seqlen, k]

    # Renormalize selected expert weights
    top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
    top_k_probs = top_k_probs.astype(x.dtype)

    # Gather selected expert weights
    # expert_gate_up: [num_experts, 2*intermediate, dim]
    # expert_down: [num_experts, dim, intermediate]
    selected_gate_up = params.expert_gate_up[
        top_k_indices
    ]  # [bsz, seqlen, k, 2*inter, dim]
    selected_down = params.expert_down[
        top_k_indices
    ]  # [bsz, seqlen, k, dim, inter]

    # SwiGLU per expert: x @ gate_up^T -> split -> silu(gate) * up -> @ down^T
    # x: [bsz, seqlen, dim] -> [bsz, seqlen, 1, dim]
    x_expanded = x[:, :, None, :]

    # gate_up: [bsz, seqlen, k, 2*inter]
    gate_up = jnp.einsum("bskd,bskid->bski", x_expanded, selected_gate_up)
    intermediate = gate_up.shape[-1] // 2
    gate = gate_up[..., :intermediate]
    up = gate_up[..., intermediate:]

    if activation_fn == "silu":
        activated = nn.silu(gate)
    elif activation_fn == "gelu":
        activated = nn.gelu(gate, approximate=False)
    else:
        activated = nn.relu(gate)

    expert_hidden = activated * up  # [bsz, seqlen, k, inter]

    # Down projection: [bsz, seqlen, k, dim]
    expert_out = jnp.einsum("bski,bskdi->bskd", expert_hidden, selected_down)

    # Weighted sum over experts
    routed_output = jnp.einsum(
        "bskd,bsk->bsd", expert_out, top_k_probs
    )  # [bsz, seqlen, dim]

    # --- Shared expert ---
    shared_params = FeedForwardParams(
        w_gate=params.shared_gate,
        w_up=params.shared_up,
        w_down=params.shared_down,
    )
    shared_output = feed_forward(x, shared_params, activation_fn)

    # Shared expert gate: sigmoid(x @ shared_expert_gate)
    shared_gate = nn.sigmoid(
        jnp.einsum("bsd,do->bso", x, params.shared_expert_gate)
    )  # [bsz, seqlen, 1]

    # Combine
    output = routed_output + shared_gate * shared_output

    return output
