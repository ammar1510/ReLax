"""Convert LLaMA .pth weights to orbax checkpoint format.

Loads weights from the original PyTorch .pth files, converts them to
ReLax format, and saves as an orbax checkpoint. This script must be
run on all hosts so that each host writes its local shards.

After conversion, use load_from_orbax() for fast sharded loading.

Usage (run on each host):
    python scripts/convert_to_orbax.py \
        --model_path /path/to/llama \
        --output_path /path/to/orbax_ckpt
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from models.llama.config import ModelConfig
from models.llama.load import load_llama_weights, save_orbax_weights


def main():
    jax.distributed.initialize()

    parser = argparse.ArgumentParser(
        description="Convert LLaMA .pth weights to orbax checkpoint format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LLaMA model directory containing original/*.pth files and config.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the orbax checkpoint will be written",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    is_main = jax.process_index() == 0

    if is_main and output_path.exists():
        print(f"Output path {output_path} already exists, overwriting...")

    # Load config
    config = ModelConfig.from_json_file(str(model_path))
    if is_main:
        print(
            f"Config: {config.n_layers} layers, {config.dim} dim, "
            f"{config.n_heads} heads, {config.n_kv_heads} kv_heads"
        )

    # Load weights from .pth (every host loads independently)
    if is_main:
        print(f"Loading weights from {model_path}...")
    params = load_llama_weights(str(model_path), config)
    if is_main:
        print("Weights loaded successfully")

    # Build mesh
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(4, 4), ("dp", "tp"))
    if is_main:
        print(f"Mesh: {mesh.shape} axes={mesh.axis_names} ({len(devices)} devices)")

    # Save as orbax — params are sharded onto mesh, each host writes its shards
    if is_main:
        print(f"Saving orbax checkpoint to {output_path}...")
    save_orbax_weights(params, str(output_path), mesh=mesh)
    if is_main:
        print("Done!")


if __name__ == "__main__":
    main()
