"""Tests for Qwen3.5 MoE model architecture."""

import pytest
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from models.qwen.config import QwenConfig
from models.qwen.model import Qwen, FullAttentionBlock, LinearAttentionBlock
from utils.hybrid_cache import HybridCache
from utils.ops import build_attn_mask


# ---------------------------------------------------------------------------
# Tiny config for testing (small enough to run on CPU)
# ---------------------------------------------------------------------------

TINY_CONFIG = QwenConfig(
    vocab_size=256,
    dim=64,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    head_dim=32,
    rms_norm_eps=1e-6,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_value_head_dim=16,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"),
    attn_output_gate=True,
    rope_theta=10000000.0,
    partial_rotary_factor=0.25,
    max_seqlen=128,
    activation_fn="silu",
    dtype="float32",
)


@pytest.fixture
def tiny_config():
    return TINY_CONFIG


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestQwenConfig:
    def test_derived_properties(self, tiny_config):
        assert tiny_config.rotary_dim == 8  # 32 * 0.25
        assert tiny_config.n_full_attn_layers == 1
        assert tiny_config.n_linear_attn_layers == 3
        assert tiny_config.linear_key_dim == 32  # 2 * 16
        assert tiny_config.linear_value_dim == 64  # 4 * 16
        assert tiny_config.linear_conv_dim == 128  # 32*2 + 64

    def test_gqa_validation(self):
        with pytest.raises(ValueError, match="divisible"):
            QwenConfig(
                vocab_size=256, dim=64, n_layers=1,
                n_heads=4, n_kv_heads=3, head_dim=32,  # 4 % 3 != 0
                rms_norm_eps=1e-6,
                linear_conv_kernel_dim=4, linear_key_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_value_head_dim=16,
                num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=32, shared_expert_intermediate_size=32,
                layer_types=("full_attention",),
                attn_output_gate=True,
                rope_theta=10000000.0, partial_rotary_factor=0.25,
                max_seqlen=128, activation_fn="silu",
            )

    def test_from_json_file(self, tmp_path):
        config_data = {
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 3072,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 64,
                "linear_value_head_dim": 128,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "full_attention_interval": 4,
                "attn_output_gate": True,
                "max_position_embeddings": 262144,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                },
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        cfg = QwenConfig.from_json_file(str(tmp_path))
        assert cfg.vocab_size == 248320
        assert cfg.dim == 3072
        assert cfg.n_layers == 4
        assert cfg.layer_types == (
            "linear_attention", "linear_attention", "linear_attention", "full_attention"
        )
        assert cfg.rope_theta == 10000000
        assert cfg.partial_rotary_factor == 0.25


# ---------------------------------------------------------------------------
# Model forward pass tests
# ---------------------------------------------------------------------------


class TestQwenForwardPass:
    def test_full_model_shapes(self, tiny_config, rng):
        """Test that the full model produces correct output shapes."""
        bsz, seqlen = 2, 8
        max_cache_seqlen = 32

        model = Qwen(tiny_config)
        params = model.init(
            rng,
            tokens=jnp.zeros((bsz, seqlen), dtype=jnp.int32),
            true_lengths=jnp.ones(bsz, dtype=jnp.int32) * seqlen,
            hybrid_cache=HybridCache.new(
                tiny_config, bsz, max_cache_seqlen, dtype=jnp.float32
            ),
            mask=jnp.ones((bsz, seqlen, max_cache_seqlen), dtype=jnp.bool_),
        )

        # Forward pass with real data
        tokens = jax.random.randint(rng, (bsz, seqlen), 0, tiny_config.vocab_size)
        true_lengths = jnp.array([seqlen, seqlen - 2])
        cache = HybridCache.new(tiny_config, bsz, max_cache_seqlen, dtype=jnp.float32)
        mask = build_attn_mask(seqlen, cache.kv_cache, true_lengths)

        logits, updated_cache = model.apply(
            params, tokens, true_lengths, cache, mask
        )

        assert logits.shape == (bsz, seqlen, tiny_config.vocab_size)
        # KV cache positions should be updated
        np.testing.assert_array_equal(
            np.array(updated_cache.kv_cache.seq_positions),
            np.array(true_lengths),
        )

    def test_decode_step(self, tiny_config, rng):
        """Test single-token decode after prefill."""
        bsz = 1
        prefill_len = 4
        max_cache_seqlen = 32

        model = Qwen(tiny_config)

        # Init
        tokens_init = jnp.zeros((bsz, prefill_len), dtype=jnp.int32)
        cache_init = HybridCache.new(
            tiny_config, bsz, max_cache_seqlen, dtype=jnp.float32
        )
        mask_init = jnp.ones(
            (bsz, prefill_len, max_cache_seqlen), dtype=jnp.bool_
        )
        params = model.init(
            rng, tokens_init,
            jnp.ones(bsz, dtype=jnp.int32) * prefill_len,
            cache_init, mask_init,
        )

        # Prefill
        tokens = jax.random.randint(rng, (bsz, prefill_len), 0, tiny_config.vocab_size)
        true_lengths = jnp.array([prefill_len])
        cache = HybridCache.new(tiny_config, bsz, max_cache_seqlen, dtype=jnp.float32)
        mask = build_attn_mask(prefill_len, cache.kv_cache, true_lengths)

        _, cache_after_prefill = model.apply(
            params, tokens, true_lengths, cache, mask
        )

        # Decode: single token
        decode_token = jnp.array([[42]])
        decode_len = jnp.array([1])
        decode_mask = build_attn_mask(1, cache_after_prefill.kv_cache, decode_len)

        logits, cache_after_decode = model.apply(
            params, decode_token, decode_len, cache_after_prefill, decode_mask
        )

        assert logits.shape == (bsz, 1, tiny_config.vocab_size)
        # Positions should have advanced by 1
        expected_pos = prefill_len + 1
        assert int(cache_after_decode.kv_cache.seq_positions[0]) == expected_pos


# ---------------------------------------------------------------------------
# HybridCache tests
# ---------------------------------------------------------------------------


class TestHybridCache:
    def test_creation(self, tiny_config):
        bsz = 2
        cache = HybridCache.new(tiny_config, bsz, max_cache_seqlen=64, dtype=jnp.float32)

        # KV cache: 1 full-attn layer
        assert cache.kv_cache.k.shape == (1, 2, 2, 64, 32)  # [layers, bsz, kv_heads, seqlen, head_dim]

        # DeltaNet state: 3 linear-attn layers
        assert cache.deltanet_state.state.shape == (3, 2, 4, 16, 16)  # [layers, bsz, v_heads, k_dim, v_dim]
        assert cache.deltanet_state.conv_state.shape == (3, 2, 128, 3)  # [layers, bsz, conv_dim, kernel-1]

    def test_position_update(self, tiny_config):
        bsz = 2
        cache = HybridCache.new(tiny_config, bsz, max_cache_seqlen=64, dtype=jnp.float32)

        updated = cache.update_positions(jnp.array([5, 3]))
        np.testing.assert_array_equal(
            np.array(updated.kv_cache.seq_positions), [5, 3]
        )
        # DeltaNet state should be unchanged
        np.testing.assert_array_equal(
            np.array(updated.deltanet_state.state),
            np.array(cache.deltanet_state.state),
        )
