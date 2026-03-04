"""Convert Qwen safetensors weights to orbax checkpoint on GCS.

One-time script: loads weights from local safetensors files, shards them
onto the device mesh, and saves as an orbax checkpoint to a GCS bucket
(or local path). Subsequent inference runs can then load directly from
the orbax checkpoint, avoiding repeated safetensors parsing.

This script must be run on all hosts (each host saves its own shards).

Usage (run on each host):
    python scripts/convert_qwen_to_orbax.py \
        --model_path /path/to/Qwen3.5-MoE \
        --output_path gs://your-bucket/qwen-orbax-ckpt

    # Or to a local path:
    python scripts/convert_qwen_to_orbax.py \
        --model_path /path/to/Qwen3.5-MoE \
        --output_path /local/orbax_ckpt
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from models.qwen.config import QwenConfig
from models.qwen.load import load_qwen_weights, save_orbax_weights


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser(
        description="Convert Qwen safetensors to orbax checkpoint (supports GCS)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Qwen model directory with *.safetensors and config.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for orbax checkpoint (local or gs://bucket/path)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional separate path to config.json directory",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=4,
        help="Data-parallel mesh dimension (default: 4)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor-parallel mesh dimension (default: 4)",
    )
    args = parser.parse_args()

    is_main = jax.process_index() == 0

    # Load config
    cfg_path = args.config_path or args.model_path
    config = QwenConfig.from_json_file(cfg_path)
    if is_main:
        print(
            f"Config: {config.n_layers} layers, {config.dim} dim, "
            f"{config.n_heads} heads, {config.n_kv_heads} kv_heads, "
            f"{config.num_experts} experts (top-{config.num_experts_per_tok}), "
            f"full_attn={config.n_full_attn_layers}, "
            f"linear_attn={config.n_linear_attn_layers}"
        )
        sys.stdout.flush()

    # Load weights from safetensors (every host loads independently)
    if is_main:
        print(f"Loading safetensors weights from {args.model_path}...")
        sys.stdout.flush()
    t0 = time.time()
    params = load_qwen_weights(args.model_path, config)
    if is_main:
        print(f"Weights loaded in {time.time() - t0:.1f}s")
        sys.stdout.flush()

    # Build mesh
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(args.dp, args.tp), ("dp", "tp"))
    if is_main:
        print(f"Mesh: {mesh.shape} axes={mesh.axis_names} ({len(devices)} devices)")
        sys.stdout.flush()

    # Save as orbax checkpoint (sharded onto mesh)
    if is_main:
        print(f"Saving orbax checkpoint to {args.output_path}...")
        sys.stdout.flush()
    t0 = time.time()
    save_orbax_weights(params, args.output_path, mesh=mesh)
    if is_main:
        print(f"Checkpoint saved in {time.time() - t0:.1f}s")
        print("Done!")
        sys.stdout.flush()

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
