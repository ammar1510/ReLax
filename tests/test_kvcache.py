import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.kvcache import KVCache

class TestKVCache:
    def test_new_cache(self):
        n_layers = 2
        bsz = 4
        max_seq_len = 128
        kv_heads = 8
        head_dim = 64
        dtype = jnp.float32

        cache = KVCache.new(n_layers, bsz, max_seq_len, kv_heads, head_dim, dtype=dtype)

        assert cache.k.shape == (n_layers, bsz, max_seq_len, kv_heads, head_dim)
        assert cache.v.shape == (n_layers, bsz, max_seq_len, kv_heads, head_dim)
        assert cache.k.dtype == dtype
        assert cache.v.dtype == dtype
        assert jnp.all(cache.k == 0)
        assert jnp.all(cache.v == 0)

    def test_update_cache(self):
        n_layers = 2
        bsz = 1
        max_seq_len = 10
        kv_heads = 1
        head_dim = 2
        dtype = jnp.float32

        cache = KVCache.new(n_layers, bsz, max_seq_len, kv_heads, head_dim, dtype=dtype)

        layer_idx = 0
        start_pos = 2
        seqlen_update = 3

        # Create dummy xk and xv tensors
        # xk, xv shapes: [bsz, seqlen_update, kv_heads, head_dim]
        xk_update_val = jnp.arange(bsz * seqlen_update * kv_heads * head_dim, dtype=dtype).reshape(
            bsz, seqlen_update, kv_heads, head_dim
        ) + 1.0 # Add 1 to distinguish from initial zeros
        xv_update_val = jnp.arange(bsz * seqlen_update * kv_heads * head_dim, dtype=dtype).reshape(
            bsz, seqlen_update, kv_heads, head_dim
        ) + 10.0 # Add 10 to distinguish from xk and initial zeros

        updated_cache = cache.update(xk_update_val, xv_update_val, layer_idx, start_pos)

        # Check shapes of updated cache (should be unchanged)
        assert updated_cache.k.shape == (n_layers, bsz, max_seq_len, kv_heads, head_dim)
        assert updated_cache.v.shape == (n_layers, bsz, max_seq_len, kv_heads, head_dim)

        # Verify the updated slice in k
        expected_k_slice = xk_update_val
        actual_k_slice = updated_cache.k[layer_idx, :, start_pos:start_pos + seqlen_update, :, :]
        np.testing.assert_array_equal(actual_k_slice, expected_k_slice)

        # Verify the updated slice in v
        expected_v_slice = xv_update_val
        actual_v_slice = updated_cache.v[layer_idx, :, start_pos:start_pos + seqlen_update, :, :]
        np.testing.assert_array_equal(actual_v_slice, expected_v_slice)

        # Verify that other parts of the cache remain zero (or original value)
        # Check layer before update (if layer_idx > 0)
        if layer_idx > 0:
            assert jnp.all(updated_cache.k[layer_idx -1, ...] == 0)
            assert jnp.all(updated_cache.v[layer_idx -1, ...] == 0)
        # Check layer after update (if layer_idx < n_layers - 1)
        if layer_idx < n_layers - 1:
            assert jnp.all(updated_cache.k[layer_idx + 1, ...] == 0)
            assert jnp.all(updated_cache.v[layer_idx + 1, ...] == 0)
        
        # Check sequence positions before the update
        if start_pos > 0:
            assert jnp.all(updated_cache.k[layer_idx, :, :start_pos, :, :] == 0)
            assert jnp.all(updated_cache.v[layer_idx, :, :start_pos, :, :] == 0)
        
        # Check sequence positions after the update
        if start_pos + seqlen_update < max_seq_len:
            assert jnp.all(updated_cache.k[layer_idx, :, start_pos + seqlen_update:, :, :] == 0)
            assert jnp.all(updated_cache.v[layer_idx, :, start_pos + seqlen_update:, :, :] == 0)

    def test_get_layer(self):
        n_layers = 2
        bsz = 1
        max_seq_len = 10
        kv_heads = 1
        head_dim = 2
        dtype = jnp.float32

        cache = KVCache.new(n_layers, bsz, max_seq_len, kv_heads, head_dim, dtype=dtype)

        layer_idx_to_update = 0
        start_pos_update = 0
        seqlen_update = 5 

        # Create dummy xk and xv tensors for update
        xk_update_val = jnp.arange(bsz * seqlen_update * kv_heads * head_dim, dtype=dtype).reshape(
            bsz, seqlen_update, kv_heads, head_dim
        ) + 1.0
        xv_update_val = jnp.arange(bsz * seqlen_update * kv_heads * head_dim, dtype=dtype).reshape(
            bsz, seqlen_update, kv_heads, head_dim
        ) + 10.0

        cache = cache.update(xk_update_val, xv_update_val, layer_idx_to_update, start_pos_update)

        # Parameters for get_layer
        layer_idx_to_get = 0

        retrieved_k, retrieved_v = cache.get_layer(layer_idx_to_get)

        # Expected shapes: [bsz, max_seq_len, kv_heads, head_dim]
        assert retrieved_k.shape == (bsz, max_seq_len, kv_heads, head_dim)
        assert retrieved_v.shape == (bsz, max_seq_len, kv_heads, head_dim)

        # Verify the content of the retrieved k
        # The first part should contain the updated values
        np.testing.assert_array_equal(retrieved_k[:, :seqlen_update, :, :], xk_update_val)
        # The rest should be zeros
        assert jnp.all(retrieved_k[:, seqlen_update:, :, :] == 0)

        # Verify the content of the retrieved v
        np.testing.assert_array_equal(retrieved_v[:, :seqlen_update, :, :], xv_update_val)
        assert jnp.all(retrieved_v[:, seqlen_update:, :, :] == 0)
        
        # Test getting from a different layer (should be all zeros)
        if n_layers > 1:
            layer_idx_other = 1
            retrieved_k_other, retrieved_v_other = cache.get_layer(layer_idx_other)
            assert jnp.all(retrieved_k_other == 0)
            assert jnp.all(retrieved_v_other == 0) 