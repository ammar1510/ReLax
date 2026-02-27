"""Tests for Qwen3.5-specific JAX operations."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from utils.qwen_ops import (
    qwen_rms_norm,
    qwen_rms_norm_gated,
    apply_partial_rotary_emb_batch,
    gated_deltanet_step,
    gated_deltanet_prefill,
    moe_forward,
    DeltaNetParams,
    MoEParams,
)
from utils.ops import precompute_freqs_cis, FeedForwardParams


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# qwen_rms_norm
# ---------------------------------------------------------------------------


class TestQwenRMSNorm:
    def test_basic_shape(self, rng):
        x = jax.random.normal(rng, (2, 4, 64))
        weight = jnp.zeros(64)
        out = qwen_rms_norm(x, weight)
        assert out.shape == x.shape

    def test_zero_weight_is_identity_norm(self, rng):
        """With weight=0, (1+weight)=1, so output is just the normalized x."""
        x = jax.random.normal(rng, (2, 4, 64))
        weight = jnp.zeros(64)
        out = qwen_rms_norm(x, weight)

        # Manual RMS norm
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        expected = x / rms
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_differs_from_standard_rms_norm(self, rng):
        """Qwen uses (1+weight) while standard uses weight. They should differ."""
        from utils.ops import rms_norm

        x = jax.random.normal(rng, (2, 4, 64))
        weight = jax.random.normal(jax.random.PRNGKey(1), (64,)) * 0.1
        qwen_out = qwen_rms_norm(x, weight)
        std_out = rms_norm(x, weight)
        assert not jnp.allclose(qwen_out, std_out, atol=1e-6)


class TestQwenRMSNormGated:
    def test_basic(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        x = jax.random.normal(k1, (2, 64))
        gate = jax.random.normal(k2, (2, 64))
        weight = jnp.ones(64)
        out = qwen_rms_norm_gated(x, gate, weight)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# apply_partial_rotary_emb_batch
# ---------------------------------------------------------------------------


class TestPartialRoPE:
    def test_shape_preserved(self, rng):
        bsz, seqlen, n_heads, head_dim = 2, 4, 8, 256
        rotary_dim = 64
        x = jax.random.normal(rng, (bsz, seqlen, n_heads, head_dim))
        freqs = precompute_freqs_cis(rotary_dim, seqlen * 2, 10000000.0)
        # Build per-batch frequencies
        positions = jnp.arange(seqlen)[None, :].repeat(bsz, axis=0)
        batch_freqs = freqs[positions]  # [bsz, seqlen, rotary_dim//2, 2]
        out = apply_partial_rotary_emb_batch(x, batch_freqs, rotary_dim)
        assert out.shape == x.shape

    def test_non_rotary_dims_unchanged(self, rng):
        bsz, seqlen, n_heads, head_dim = 2, 4, 8, 256
        rotary_dim = 64
        x = jax.random.normal(rng, (bsz, seqlen, n_heads, head_dim))
        freqs = precompute_freqs_cis(rotary_dim, seqlen * 2, 10000000.0)
        positions = jnp.arange(seqlen)[None, :].repeat(bsz, axis=0)
        batch_freqs = freqs[positions]
        out = apply_partial_rotary_emb_batch(x, batch_freqs, rotary_dim)
        # Last (head_dim - rotary_dim) dims should be unchanged
        np.testing.assert_array_equal(
            np.array(out[..., rotary_dim:]),
            np.array(x[..., rotary_dim:]),
        )

    def test_rotary_dims_changed(self, rng):
        bsz, seqlen, n_heads, head_dim = 2, 4, 8, 256
        rotary_dim = 64
        x = jax.random.normal(rng, (bsz, seqlen, n_heads, head_dim))
        freqs = precompute_freqs_cis(rotary_dim, seqlen * 2, 10000000.0)
        positions = jnp.arange(seqlen)[None, :].repeat(bsz, axis=0)
        batch_freqs = freqs[positions]
        out = apply_partial_rotary_emb_batch(x, batch_freqs, rotary_dim)
        # First rotary_dim dims should be different (except at position 0 where cos=1, sin=0)
        assert not jnp.allclose(out[:, 1:, :, :rotary_dim], x[:, 1:, :, :rotary_dim])


# ---------------------------------------------------------------------------
# gated_deltanet_step
# ---------------------------------------------------------------------------


class TestGatedDeltaNetStep:
    @pytest.fixture
    def deltanet_params(self, rng):
        n_k_heads, k_head_dim = 4, 32
        n_v_heads, v_head_dim = 8, 32
        dim = 64
        key_dim = n_k_heads * k_head_dim
        value_dim = n_v_heads * v_head_dim
        conv_dim = key_dim * 2 + value_dim
        kernel_size = 4

        keys = jax.random.split(rng, 10)
        return DeltaNetParams(
            in_proj_qkv=jax.random.normal(keys[0], (dim, key_dim * 2 + value_dim)) * 0.02,
            in_proj_z=jax.random.normal(keys[1], (dim, value_dim)) * 0.02,
            in_proj_a=jax.random.normal(keys[2], (dim, n_v_heads)) * 0.02,
            in_proj_b=jax.random.normal(keys[3], (dim, n_v_heads)) * 0.02,
            conv1d_weight=jax.random.normal(keys[4], (conv_dim, kernel_size)) * 0.02,
            dt_bias=jnp.ones(n_v_heads),
            A_log=jnp.zeros(n_v_heads),
            norm_weight=jnp.zeros(value_dim),
            out_proj=jax.random.normal(keys[5], (value_dim, dim)) * 0.02,
        ), n_k_heads, k_head_dim, n_v_heads, v_head_dim

    def test_output_shape(self, rng, deltanet_params):
        params, n_k_heads, k_head_dim, n_v_heads, v_head_dim = deltanet_params
        bsz, dim = 2, 64
        key_dim = n_k_heads * k_head_dim
        value_dim = n_v_heads * v_head_dim
        conv_dim = key_dim * 2 + value_dim

        x = jax.random.normal(rng, (bsz, dim))
        state = jnp.zeros((bsz, n_v_heads, k_head_dim, v_head_dim))
        conv_state = jnp.zeros((bsz, conv_dim, 3))  # kernel_size - 1

        out, new_state, new_conv_state = gated_deltanet_step(
            x, state, conv_state, params,
            n_k_heads, k_head_dim, n_v_heads, v_head_dim,
        )

        assert out.shape == (bsz, dim)
        assert new_state.shape == state.shape
        assert new_conv_state.shape == conv_state.shape

    def test_state_updates(self, rng, deltanet_params):
        """State should change after processing a token."""
        params, n_k_heads, k_head_dim, n_v_heads, v_head_dim = deltanet_params
        bsz, dim = 2, 64
        key_dim = n_k_heads * k_head_dim
        value_dim = n_v_heads * v_head_dim
        conv_dim = key_dim * 2 + value_dim

        x = jax.random.normal(rng, (bsz, dim))
        state = jnp.zeros((bsz, n_v_heads, k_head_dim, v_head_dim))
        conv_state = jnp.zeros((bsz, conv_dim, 3))

        _, new_state, new_conv_state = gated_deltanet_step(
            x, state, conv_state, params,
            n_k_heads, k_head_dim, n_v_heads, v_head_dim,
        )

        assert not jnp.allclose(new_state, state)
        assert not jnp.allclose(new_conv_state, conv_state)


# ---------------------------------------------------------------------------
# gated_deltanet_prefill
# ---------------------------------------------------------------------------


class TestGatedDeltaNetPrefill:
    def test_matches_sequential_steps(self, rng):
        """Prefill via scan should match sequential single steps."""
        n_k_heads, k_head_dim = 2, 16
        n_v_heads, v_head_dim = 4, 16
        dim = 32
        key_dim = n_k_heads * k_head_dim
        value_dim = n_v_heads * v_head_dim
        conv_dim = key_dim * 2 + value_dim
        kernel_size = 4
        bsz, seqlen = 1, 3

        keys = jax.random.split(rng, 10)
        params = DeltaNetParams(
            in_proj_qkv=jax.random.normal(keys[0], (dim, key_dim * 2 + value_dim)) * 0.02,
            in_proj_z=jax.random.normal(keys[1], (dim, value_dim)) * 0.02,
            in_proj_a=jax.random.normal(keys[2], (dim, n_v_heads)) * 0.02,
            in_proj_b=jax.random.normal(keys[3], (dim, n_v_heads)) * 0.02,
            conv1d_weight=jax.random.normal(keys[4], (conv_dim, kernel_size)) * 0.02,
            dt_bias=jnp.ones(n_v_heads),
            A_log=jnp.zeros(n_v_heads),
            norm_weight=jnp.zeros(value_dim),
            out_proj=jax.random.normal(keys[5], (value_dim, dim)) * 0.02,
        )

        x_seq = jax.random.normal(keys[6], (bsz, seqlen, dim))
        state0 = jnp.zeros((bsz, n_v_heads, k_head_dim, v_head_dim))
        conv0 = jnp.zeros((bsz, conv_dim, kernel_size - 1))

        # Prefill (scan)
        out_prefill, state_pf, conv_pf = gated_deltanet_prefill(
            x_seq, state0, conv0, params,
            n_k_heads, k_head_dim, n_v_heads, v_head_dim,
        )

        # Sequential steps
        outputs = []
        s, cs = state0, conv0
        for t in range(seqlen):
            o, s, cs = gated_deltanet_step(
                x_seq[:, t, :], s, cs, params,
                n_k_heads, k_head_dim, n_v_heads, v_head_dim,
            )
            outputs.append(o)
        out_seq = jnp.stack(outputs, axis=1)

        np.testing.assert_allclose(
            np.array(out_prefill), np.array(out_seq), atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(state_pf), np.array(s), atol=1e-5
        )


# ---------------------------------------------------------------------------
# moe_forward
# ---------------------------------------------------------------------------


class TestMoEForward:
    def test_output_shape(self, rng):
        bsz, seqlen, dim = 2, 4, 32
        num_experts = 8
        num_experts_per_tok = 2
        intermediate = 16
        shared_intermediate = 16

        keys = jax.random.split(rng, 8)
        params = MoEParams(
            router_weight=jax.random.normal(keys[0], (num_experts, dim)) * 0.02,
            expert_gate_up=jax.random.normal(keys[1], (num_experts, 2 * intermediate, dim)) * 0.02,
            expert_down=jax.random.normal(keys[2], (num_experts, dim, intermediate)) * 0.02,
            shared_gate=jax.random.normal(keys[3], (dim, shared_intermediate)) * 0.02,
            shared_up=jax.random.normal(keys[4], (dim, shared_intermediate)) * 0.02,
            shared_down=jax.random.normal(keys[5], (shared_intermediate, dim)) * 0.02,
            shared_expert_gate=jax.random.normal(keys[6], (dim, 1)) * 0.02,
        )

        x = jax.random.normal(keys[7], (bsz, seqlen, dim))
        out = moe_forward(x, params, num_experts_per_tok, "silu")

        assert out.shape == (bsz, seqlen, dim)

    def test_shared_expert_always_active(self, rng):
        """Even with zero router weights, shared expert should contribute."""
        bsz, seqlen, dim = 1, 1, 16
        num_experts = 4
        num_experts_per_tok = 2
        intermediate = 8

        keys = jax.random.split(rng, 8)
        params = MoEParams(
            router_weight=jnp.zeros((num_experts, dim)),
            expert_gate_up=jnp.zeros((num_experts, 2 * intermediate, dim)),
            expert_down=jnp.zeros((num_experts, dim, intermediate)),
            shared_gate=jax.random.normal(keys[0], (dim, intermediate)) * 0.1,
            shared_up=jax.random.normal(keys[1], (dim, intermediate)) * 0.1,
            shared_down=jax.random.normal(keys[2], (intermediate, dim)) * 0.1,
            shared_expert_gate=jnp.ones((dim, 1)),  # sigmoid(1) ≈ 0.73
        )

        x = jax.random.normal(keys[3], (bsz, seqlen, dim))
        out = moe_forward(x, params, num_experts_per_tok, "silu")

        # Output should be non-zero from shared expert
        assert jnp.any(out != 0)
