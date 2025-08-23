"""
Benchmark tests for grouped_query_attention function with different shapes and dtypes.

This module provides comprehensive timing benchmarks for the grouped_query_attention
function across various configuration scenarios including different model sizes,
sequence lengths, batch sizes, and data types.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial
from typing import Dict, List, Tuple, Optional

from utils.ops import (
    grouped_query_attention,
    precompute_freqs_cis,
    AttentionParams,
)
from utils.kvcache import KVCache
from utils.memory import estimate_pytree_memory_footprint, format_bytes


# Test configurations for different model sizes
TEST_CONFIGS = [
    # Small model configs
    {
        "name": "small-1B",
        "dim": 2048,
        "n_heads": 8,
        "n_kv_heads": 2,
        "batch_sizes": [32, 64, 128, 256],
        "seq_lengths": [256, 512, 1024],
        "dtypes": [jnp.float32, jnp.bfloat16],
    },
    # Medium model configs  
    {
        "name": "medium-3B",
        "dim": 3072,
        "n_heads": 24,
        "n_kv_heads": 8,
        "batch_sizes": [64, 128, 256, 512],
        "seq_lengths": [512, 1024, 2048],
        "dtypes": [jnp.float32, jnp.bfloat16],
    },
    # Large model configs
    {
        "name": "large-8B",
        "dim": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "batch_sizes": [128, 256],
        "seq_lengths": [512, 1024],
        "dtypes": [jnp.float32, jnp.bfloat16],
    },
]

# Decode vs Prefill scenarios
SCENARIOS = [
    {
        "name": "prefill",
        "description": "Initial prompt processing",
        "start_pos": 0,
        "use_prefill_mask": True,
    },
    {
        "name": "decode",
        "description": "Token-by-token generation",
        "start_pos": lambda seq_len: seq_len - 1,  # Decode next token
        "use_prefill_mask": False,
        "decode_seq_len": 1,
    },
]


class AttentionBenchmarkSuite:
    """Comprehensive benchmark suite for grouped_query_attention."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.max_seq_len = 2048  # Maximum sequence length for cache allocation
        
    def create_test_inputs(
        self,
        batch_size: int,
        seq_len: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        dtype: jnp.dtype,
        scenario: Dict,
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, AttentionParams, KVCache, Optional[jax.Array]]:
        """Create test inputs for the attention function."""
        head_dim = dim // n_heads
        
        # Split random keys
        key_x, key_wq, key_wk, key_wv, key_wo, key_cache = jax.random.split(key, 6)
        
        # Adjust sequence length for decode scenario
        actual_seq_len = scenario.get("decode_seq_len", seq_len)
        
        # Create input tensor
        x = jax.random.normal(key_x, (batch_size, actual_seq_len, dim), dtype=dtype)
        
        # Create frequency embeddings
        freqs_cis = precompute_freqs_cis(head_dim, self.max_seq_len, dtype=dtype)
        
        # Create attention parameters
        params = AttentionParams(
            wq=jax.random.normal(key_wq, (dim, n_heads, head_dim), dtype=dtype),
            wk=jax.random.normal(key_wk, (dim, n_kv_heads, head_dim), dtype=dtype),
            wv=jax.random.normal(key_wv, (dim, n_kv_heads, head_dim), dtype=dtype),
            wo=jax.random.normal(key_wo, (n_heads * head_dim, dim), dtype=dtype),
        )
        
        # Create KV cache
        kv_cache = KVCache.new(
            n_layers=1,
            bsz=batch_size,
            max_seqlen=self.max_seq_len,
            kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        
        # For decode scenario, pre-fill cache with some data
        start_pos_value = scenario["start_pos"]
        if callable(start_pos_value):
            start_pos_value = start_pos_value(seq_len)
        else:
            start_pos_value = start_pos_value
            
        if scenario["name"] == "decode" and start_pos_value > 0:
            # Pre-fill cache with random data
            prefill_k = jax.random.normal(
                key_cache, (1, batch_size, start_pos_value, n_kv_heads, head_dim), dtype=dtype
            )
            prefill_v = jax.random.normal(
                key_cache, (1, batch_size, start_pos_value, n_kv_heads, head_dim), dtype=dtype
            )
            k_updated = kv_cache.k.at[0, :, :start_pos_value, :, :].set(prefill_k[0])
            v_updated = kv_cache.v.at[0, :, :start_pos_value, :, :].set(prefill_v[0])
            kv_cache = KVCache(k=k_updated, v=v_updated)
        
        # Create prefill mask if needed
        prefill_mask = None
        if scenario["use_prefill_mask"]:
            prefill_mask = jnp.ones((batch_size, actual_seq_len), dtype=jnp.bool_)
            # Add some padding for realistic scenarios
            if batch_size > 1:
                # Add padding to the second batch element
                pad_start = actual_seq_len // 2
                prefill_mask = prefill_mask.at[1, pad_start:].set(False)
        
        return x, freqs_cis, params, kv_cache, prefill_mask, start_pos_value
    
    def benchmark_attention(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        params: AttentionParams,
        kv_cache: KVCache,
        start_pos: int,
        prefill_mask: Optional[jax.Array],
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ) -> Dict:
        """Benchmark the attention function with timing and memory stats."""
        layer_idx = 0
        
        # Create JIT-compiled function
        jitted_attention = jax.jit(grouped_query_attention)
        
        # Warmup runs
        for _ in range(warmup_iterations):
            output, updated_cache = jitted_attention(
                x, freqs_cis, params, kv_cache, layer_idx, start_pos, prefill_mask
            )
            output.block_until_ready()
        
        # Benchmark runs
        times = []
        for _ in range(benchmark_iterations):
            start_time = time.perf_counter()
            output, updated_cache = jitted_attention(
                x, freqs_cis, params, kv_cache, layer_idx, start_pos, prefill_mask
            )
            output.block_until_ready()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Memory footprint estimation
        input_memory = (
            estimate_pytree_memory_footprint(x) +
            estimate_pytree_memory_footprint(params) +
            estimate_pytree_memory_footprint(kv_cache)
        )
        output_memory = estimate_pytree_memory_footprint(output)
        
        return {
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "input_memory_bytes": input_memory,
            "output_memory_bytes": output_memory,
            "total_memory_bytes": input_memory + output_memory,
            "output_shape": output.shape,
            "cache_shape": updated_cache.k.shape,
        }
    
    def run_single_benchmark(
        self,
        config: Dict,
        scenario: Dict,
        batch_size: int,
        seq_len: int,
        dtype: jnp.dtype,
        key: jax.random.PRNGKey,
    ) -> Dict:
        """Run a single benchmark configuration."""
        # Create test inputs
        x, freqs_cis, params, kv_cache, prefill_mask, start_pos = self.create_test_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            dim=config["dim"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            dtype=dtype,
            scenario=scenario,
            key=key,
        )
        
        # Run benchmark
        benchmark_results = self.benchmark_attention(
            x, freqs_cis, params, kv_cache, start_pos, prefill_mask
        )
        
        # Combine configuration and results
        result = {
            "config_name": config["name"],
            "scenario": scenario["name"],
            "batch_size": batch_size,
            "seq_length": seq_len,
            "dim": config["dim"],
            "n_heads": config["n_heads"],
            "n_kv_heads": config["n_kv_heads"],
            "dtype": str(dtype),
            "head_dim": config["dim"] // config["n_heads"],
            **benchmark_results,
        }
        
        return result
    
    def run_full_benchmark_suite(self, verbose: bool = True) -> List[Dict]:
        """Run the complete benchmark suite."""
        print("Starting grouped_query_attention benchmark suite...")
        print("=" * 80)
        
        key = jax.random.PRNGKey(42)
        
        for config in TEST_CONFIGS:
            if verbose:
                print(f"\nBenchmarking {config['name']} configuration:")
                print(f"  dim={config['dim']}, n_heads={config['n_heads']}, n_kv_heads={config['n_kv_heads']}")
            
            for scenario in SCENARIOS:
                if verbose:
                    print(f"\n  Scenario: {scenario['name']} ({scenario['description']})")
                
                for batch_size in config["batch_sizes"]:
                    for seq_len in config["seq_lengths"]:
                        for dtype in config["dtypes"]:
                            key, subkey = jax.random.split(key)
                            
                            try:
                                result = self.run_single_benchmark(
                                    config, scenario, batch_size, seq_len, dtype, subkey
                                )
                                self.results.append(result)
                                
                                if verbose:
                                    print(f"    BS={batch_size:2d}, SeqLen={seq_len:4d}, "
                                          f"dtype={str(dtype):>11s}: "
                                          f"{result['mean_time_ms']:6.2f}ms ± {result['std_time_ms']:5.2f}ms, "
                                          f"Memory: {format_bytes(result['total_memory_bytes'])}")
                                    
                            except Exception as e:
                                if verbose:
                                    print(f"    BS={batch_size:2d}, SeqLen={seq_len:4d}, "
                                          f"dtype={str(dtype):>11s}: FAILED - {str(e)}")
        
        return self.results
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Group by scenario
        for scenario_name in ["prefill", "decode"]:
            scenario_results = [r for r in self.results if r["scenario"] == scenario_name]
            if not scenario_results:
                continue
                
            print(f"\n{scenario_name.upper()} SCENARIO:")
            print("-" * 40)
            
            # Find fastest and slowest
            fastest = min(scenario_results, key=lambda x: x["mean_time_ms"])
            slowest = max(scenario_results, key=lambda x: x["mean_time_ms"])
            
            print(f"Fastest: {fastest['config_name']} (BS={fastest['batch_size']}, "
                  f"SeqLen={fastest['seq_length']}, {fastest['dtype']}) - "
                  f"{fastest['mean_time_ms']:.2f}ms")
            print(f"Slowest: {slowest['config_name']} (BS={slowest['batch_size']}, "
                  f"SeqLen={slowest['seq_length']}, {slowest['dtype']}) - "
                  f"{slowest['mean_time_ms']:.2f}ms")
            
            # Memory usage summary
            max_memory = max(scenario_results, key=lambda x: x["total_memory_bytes"])
            print(f"Max Memory: {format_bytes(max_memory['total_memory_bytes'])} "
                  f"({max_memory['config_name']})")


def test_attention_benchmark_small():
    """Test attention benchmarking with small configurations."""
    suite = AttentionBenchmarkSuite()
    
    # Run just the small config for testing
    small_config = TEST_CONFIGS[0]  # small-1B
    key = jax.random.PRNGKey(42)
    
    # Test one configuration per scenario
    for scenario in SCENARIOS:
        key, subkey = jax.random.split(key)
        result = suite.run_single_benchmark(
            config=small_config,
            scenario=scenario,
            batch_size=1,
            seq_len=64,
            dtype=jnp.float32,
            key=subkey,
        )
        
        # Verify result structure
        assert "mean_time_ms" in result
        assert "output_shape" in result
        assert result["mean_time_ms"] > 0
        assert result["config_name"] == "small-1B"
        assert result["scenario"] == scenario["name"]


def test_attention_benchmark_dtype_comparison():
    """Test attention performance comparison between dtypes."""
    suite = AttentionBenchmarkSuite()
    
    config = TEST_CONFIGS[0]  # small-1B
    scenario = SCENARIOS[0]   # prefill
    key = jax.random.PRNGKey(42)
    
    results = {}
    for dtype in [jnp.float32, jnp.bfloat16]:
        key, subkey = jax.random.split(key)
        result = suite.run_single_benchmark(
            config=config,
            scenario=scenario,
            batch_size=2,
            seq_len=128,
            dtype=dtype,
            key=subkey,
        )
        results[str(dtype)] = result
    
    # Verify both dtypes were tested
    assert len(results) == 2
    assert "<class 'jax._src.dtypes.bfloat16'>" in results or "bfloat16" in results
    assert "<class 'jax._src.dtypes.float32'>" in results or "float32" in results


if __name__ == "__main__":
    # Run the full benchmark suite when executed directly
    suite = AttentionBenchmarkSuite()
    results = suite.run_full_benchmark_suite(verbose=True)
    suite.print_summary()
    
    # Save results to file
    import json
    
    # Convert jax arrays to lists for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, tuple):
                json_result[key] = list(value)
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open("attention_benchmark_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to attention_benchmark_results.json")
    print(f"Total configurations tested: {len(results)}") 