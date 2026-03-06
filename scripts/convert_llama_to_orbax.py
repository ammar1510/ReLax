"""
Memory-efficient script to convert LLaMA safetensors weights from a local
model directory to an Orbax checkpoint on GCS (or local disk).

Two-pass approach:
  Pass 1 — Load each .pth shard, apply the HF→ReLax transformation
            (transpose, reshape), save each tensor as a .npy file, then
            delete the shard from memory.
  Pass 2 — Memory-map every .npy file, assemble the ReLax pytree, and let
            Orbax stream the checkpoint to GCS.

Usage:
    # Dry run — print HF→ReLax key mappings and shapes, no writes:
    python scripts/convert_llama_to_orbax.py --model_path /path/to/model --dry_run

    # Full conversion:
    python scripts/convert_llama_to_orbax.py \
        --model_path /path/to/model \
        --gcs_path gs://my-bucket/llama-orbax
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

import ml_dtypes  # noqa: F401 – registers bfloat16 with numpy
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llama.config import ModelConfig


_TEMP_DIR = Path("./temp_npy_weights")


# ---- helpers ---------------------------------------------------------------

def _transform(tensor: np.ndarray, *, transpose: bool = False,
               reshape: tuple | None = None) -> np.ndarray:
    if transpose:
        tensor = tensor.T
    if reshape is not None:
        tensor = tensor.reshape(reshape)
    return tensor


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # numpy doesn't support bfloat16 natively; go through float32 first
    return tensor.detach().cpu().float().numpy().astype(ml_dtypes.bfloat16)


# ---- per-key transformation table ------------------------------------------

def _build_key_map(config: ModelConfig) -> dict[str, tuple[str, dict]]:
    """Map every HF .pth key to (relax_key, transform_kwargs)."""
    key_map: dict[str, tuple[str, dict]] = {}

    key_map["tok_embeddings.weight"] = ("tok_embeddings.embedding", {})
    key_map["output.weight"] = ("output", {"transpose": True})
    key_map["norm.weight"] = ("norm_weight", {})

    for i in range(config.n_layers):
        pfx = f"layers.{i}"
        lk = f"layer_{i}"

        key_map[f"{pfx}.attention.wq.weight"] = (
            f"{lk}.wq",
            {"transpose": True, "reshape": (config.dim, config.n_heads, config.head_dim)},
        )
        key_map[f"{pfx}.attention.wk.weight"] = (
            f"{lk}.wk",
            {"transpose": True, "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
        )
        key_map[f"{pfx}.attention.wv.weight"] = (
            f"{lk}.wv",
            {"transpose": True, "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
        )
        key_map[f"{pfx}.attention.wo.weight"] = (f"{lk}.wo", {"transpose": True})

        key_map[f"{pfx}.feed_forward.w1.weight"] = (f"{lk}.w_gate", {"transpose": True})
        key_map[f"{pfx}.feed_forward.w3.weight"] = (f"{lk}.w_up",   {"transpose": True})
        key_map[f"{pfx}.feed_forward.w2.weight"] = (f"{lk}.w_down", {"transpose": True})

        key_map[f"{pfx}.attention_norm.weight"] = (f"{lk}.attention_norm_weight", {})
        key_map[f"{pfx}.ffn_norm.weight"]       = (f"{lk}.ffn_norm_weight", {})

    return key_map


# ---- pytree builder --------------------------------------------------------

def _insert_into_pytree(tree: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        tree = tree.setdefault(part, {})
    tree[parts[-1]] = value


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaMA .pth weights to Orbax checkpoint"
    )
    parser.add_argument("--model_path", required=True,
                        help="Directory containing config.json and original/*.pth")
    parser.add_argument("--gcs_path",
                        help="Destination path, e.g. gs://bucket/checkpoint or a local dir")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print HF→ReLax mappings and shapes only, no writes")
    args = parser.parse_args()

    if not args.dry_run and not args.gcs_path:
        parser.error("--gcs_path is required unless --dry_run is set")

    model_path = Path(args.model_path)
    pth_dir = model_path / "original"

    config = ModelConfig.from_json_file(str(model_path))
    print(f"Config: {config.n_layers} layers, dim={config.dim}, "
          f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, "
          f"head_dim={config.head_dim}")

    key_map = _build_key_map(config)
    print(f"Key map has {len(key_map)} entries")

    shard_files = sorted(pth_dir.glob("*.pth"))
    if not shard_files:
        raise FileNotFoundError(f"No .pth files found in {pth_dir}")
    print(f"Found {len(shard_files)} shard(s) in {pth_dir}")

    # ------------------------------------------------------------------
    # PASS 1: load each shard, transform, save to .npy, free shard
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n--- DRY RUN: HF key → ReLax key + shape ---")
    else:
        print("\n--- PASS 1: Extract + transform → local .npy ---")
        os.makedirs(_TEMP_DIR, exist_ok=True)

    seen_relax_keys: set[str] = set()

    for shard in shard_files:
        print(f"\n  Loading {shard.name} ...")
        checkpoint = torch.load(str(shard), map_location="cpu", weights_only=True)

        for hf_key, value in checkpoint.items():
            if hf_key not in key_map:
                print(f"  [SKIP] Unknown key: {hf_key}")
                continue

            relax_key, xform_kw = key_map[hf_key]
            tensor = _to_numpy(value)
            tensor = _transform(tensor, **xform_kw)
            tensor = tensor.astype(ml_dtypes.bfloat16)

            if args.dry_run:
                print(f"  {hf_key}  →  {relax_key}  {tuple(tensor.shape)}  {tensor.dtype}")
            else:
                npy_path = _TEMP_DIR / (relax_key.replace(".", "__") + ".npy")
                np.save(str(npy_path), tensor)

            seen_relax_keys.add(relax_key)
            del tensor, value

        del checkpoint
        gc.collect()
        if not args.dry_run:
            print(f"  Done with {shard.name}")

    # Sanity check
    missing = set(v[0] for v in key_map.values()) - seen_relax_keys
    if missing:
        print(f"\n  WARNING: {len(missing)} ReLax keys not found in shards:")
        for m in sorted(missing)[:20]:
            print(f"    {m}")

    if args.dry_run:
        print(f"\nTotal: {len(seen_relax_keys)} tensors mapped")
        return

    # ------------------------------------------------------------------
    # PASS 2: mmap .npy files, build pytree, stream to destination
    # ------------------------------------------------------------------
    import orbax.checkpoint as ocp

    print("\n--- PASS 2: Build lazy pytree → Orbax ---")
    jax_pytree: dict = {}

    for relax_key in sorted(seen_relax_keys):
        npy_path = _TEMP_DIR / (relax_key.replace(".", "__") + ".npy")
        lazy = np.load(str(npy_path), mmap_mode="r").view(ml_dtypes.bfloat16)
        _insert_into_pytree(jax_pytree, relax_key, lazy)

    print(f"Lazy pytree assembled ({len(seen_relax_keys)} arrays, ~0 RAM)")

    is_gcs = args.gcs_path.startswith("gs://")
    if is_gcs:
        from orbax.checkpoint.options import MultiprocessingOptions
        options = ocp.CheckpointManagerOptions(
            multiprocessing_options=MultiprocessingOptions(primary_host=0),
        )
    else:
        options = ocp.CheckpointManagerOptions()

    mngr = ocp.CheckpointManager(args.gcs_path, options=options)
    print(f"Streaming to {args.gcs_path} ...")
    mngr.save(step=0, args=ocp.args.StandardSave(jax_pytree))
    mngr.wait_until_finished()
    print("Upload complete!")

    shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
