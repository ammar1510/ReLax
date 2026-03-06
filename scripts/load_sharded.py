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
import numpy as np
from jax.sharding import Mesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.llama.config import ModelConfig
from models.llama.load import load_from_orbax
from models.llama.model import LLaMa
from utils.ops import precompute_freqs_cis


def main():
    parser = argparse.ArgumentParser(description="Load sharded LLaMA from GCS")
    parser.add_argument("--model_path", required=True,
                        help="Directory containing config.json")
    parser.add_argument("--gcs_path", required=True,
                        help="GCS Orbax checkpoint path, e.g. gs://bucket/llama-orbax")
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

    # 4. Load weights sharded onto mesh
    params = load_from_orbax(args.gcs_path, mesh=mesh)

    # 5. Verify sharding
    if jax.process_index() == 0:
        print("\nParameter sharding:")
        for key, val in jax.tree.leaves_with_path(params):
            name = "/".join(str(k.key) for k in key)
            sharding = val.sharding if hasattr(val, "sharding") else "N/A"
            print(f"  {name}: {val.shape} {val.dtype} | {sharding}")

    # 6. Quick forward pass sanity check
    model = LLaMa(config)
    freqs_cis = precompute_freqs_cis(
        config.head_dim, config.max_seqlen,
        config.rope_theta, config.use_scaled_rope,
    )

    # Dummy single-token input
    tokens = jax.numpy.ones((1, 1), dtype=jax.numpy.int32)
    logits = model.apply({"params": params}, tokens, 0, freqs_cis)
    jax.block_until_ready(logits)

    if jax.process_index() == 0:
        print(f"\nForward pass OK — logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
