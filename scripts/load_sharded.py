"""
Minimal script to load LLaMA weights from a GCS Orbax checkpoint
onto a sharded mesh of TPU workers.

Usage (run on each TPU worker):
    python scripts/load_sharded.py \
        --model_path /path/to/model \
        --gcs_path gs://bucket/llama-orbax \
        --dp 1 --tp 4
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.llama.config import ModelConfig
from models.llama.model import LLaMa
from utils.mesh_helpers import MeshHelper


def main():
    parser = argparse.ArgumentParser(description="Load sharded LLaMA from GCS")
    parser.add_argument("--model_path", required=True,
                        help="Directory containing config.json")
    parser.add_argument("--gcs_path",
                        default="gs://model-weights-1510/llama-3.1-8b-instruct/0",
                        help="GCS Orbax checkpoint path")
    parser.add_argument("--dp", type=int, default=1, help="Data-parallel dim")
    parser.add_argument("--tp", type=int, default=4, help="Tensor-parallel dim")
    args = parser.parse_args()

    # 1. Initialize distributed JAX (must be first)
    jax.distributed.initialize()

    # 2. Build mesh
    devices = jax.devices()
    assert len(devices) == args.dp * args.tp, (
        f"Expected {args.dp * args.tp} devices, got {len(devices)}"
    )
    mesh = Mesh(np.array(devices).reshape(args.dp, args.tp), ("dp", "tp"))
    if jax.process_index() == 0:
        print(f"Mesh: {mesh.shape} on {len(devices)} devices")

    # 3. Load config
    config = ModelConfig.from_json_file(args.model_path)
    if jax.process_index() == 0:
        print(f"Config: {config.n_layers}L, dim={config.dim}, "
              f"heads={config.n_heads}, kv_heads={config.n_kv_heads}")

    # 4. Get checkpoint metadata (shapes/dtypes, no data loaded)
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(args.gcs_path)

    # 5. Build per-parameter restore args with sharding specs
    def _get_key_name(key) -> str:
        if hasattr(key, "key"):
            return str(key.key)
        return str(key)

    def make_restore_args(path, meta):
        name = "/".join(_get_key_name(k) for k in path)
        spec = MeshHelper.param_sharding(meta, name, mesh)
        sharding = NamedSharding(mesh, spec)
        return ocp.ArrayRestoreArgs(sharding=sharding)

    restore_args = jax.tree.map_with_path(make_restore_args, metadata)

    # 6. Restore from GCS — each worker reads only its shards
    if jax.process_index() == 0:
        print(f"\nRestoring from {args.gcs_path} ...")

    with mesh:
        params = checkpointer.restore(
            args.gcs_path,
            restore_args=restore_args,
        )

    if jax.process_index() == 0:
        print("Loaded successfully!\n")
        print("Parameter sharding:")
        for key, val in jax.tree.leaves_with_path(params):
            name = "/".join(_get_key_name(k) for k in key)
            sharding = val.sharding if hasattr(val, "sharding") else "N/A"
            print(f"  {name}: {val.shape} {val.dtype} | {sharding}")


if __name__ == "__main__":
    main()
