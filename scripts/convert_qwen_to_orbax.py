"""
Memory-efficient script to convert Qwen3.5 MoE safetensors weights to an
Orbax checkpoint on GCS.

Downloads safetensors shards from HuggingFace one at a time, extracts and
transforms each tensor into a local .npy file, then deletes the shard.
After all shards are processed, memory-maps the .npy files and streams
the assembled pytree to GCS via Orbax.

Usage:
    # Dry run — print HF→ReLax key mappings and shapes (downloads 1 tensor
    # per shard at most, lightweight):
    python scripts/orbax_load.py --repo Qwen/Qwen3.5-MoE --dry_run

    # Full conversion:
    python scripts/orbax_load.py \
        --repo Qwen/Qwen3.5-MoE \
        --gcs_path gs://my-bucket/qwen3.5-orbax
"""

import argparse
import gc
import os
import shutil

# Limit tensorstore parallelism to avoid OOM during Orbax upload.
# Must be set before importing orbax/tensorstore.
os.environ.setdefault("TENSORSTORE_NUM_THREADS", "2")
import time
from pathlib import Path

import ml_dtypes  # noqa: F401 – registers bfloat16 with numpy
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

# ---------------------------------------------------------------------------
# We import QwenConfig so we know the exact shapes and layer types.
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.qwen.config import QwenConfig


# ---- HF key prefix for Qwen3.5 MoE text backbone -------------------------
_HF_PREFIX = "model.language_model.layers"


# ---- helpers ---------------------------------------------------------------


def _transform(tensor: np.ndarray, *, transpose: bool = False,
               reshape: tuple | None = None,
               squeeze_axis: int | None = None) -> np.ndarray:
    if squeeze_axis is not None:
        tensor = tensor.squeeze(axis=squeeze_axis)
    if transpose:
        tensor = tensor.T
    if reshape is not None:
        tensor = tensor.reshape(reshape)
    return tensor


# ---- per-key transformation table ------------------------------------------

def _build_key_map(config: QwenConfig):
    """Return a dict mapping every HF key to (relax_key, transform_kwargs).

    This mirrors the logic in models/qwen/load.py but produces flat
    (dot-separated) ReLax keys so we can save each tensor independently.
    """
    key_map: dict[str, tuple[str, dict]] = {}

    # --- embeddings & head ---
    key_map["model.language_model.embed_tokens.weight"] = (
        "tok_embeddings.embedding", {}
    )
    key_map["lm_head.weight"] = (
        "output", {"transpose": True}
    )
    key_map["model.language_model.norm.weight"] = (
        "norm_weight", {}
    )

    # --- per-layer ---
    for i in range(config.n_layers):
        pfx = f"{_HF_PREFIX}.{i}"
        lt = config.layer_types[i]
        lk = f"layer_{i}"

        # ---- attention ----
        if lt == "full_attention":
            attn = f"{pfx}.self_attn"
            key_map[f"{attn}.q_proj.weight"] = (
                f"{lk}.wq",
                {"transpose": True, "reshape": (config.dim, config.n_heads, config.head_dim * 2)},
            )
            key_map[f"{attn}.k_proj.weight"] = (
                f"{lk}.wk",
                {"transpose": True, "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
            )
            key_map[f"{attn}.v_proj.weight"] = (
                f"{lk}.wv",
                {"transpose": True, "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
            )
            key_map[f"{attn}.o_proj.weight"] = (
                f"{lk}.wo", {"transpose": True},
            )
            key_map[f"{attn}.q_norm.weight"] = (f"{lk}.q_norm", {})
            key_map[f"{attn}.k_norm.weight"] = (f"{lk}.k_norm", {})
        else:
            # linear attention (Gated DeltaNet)
            attn = f"{pfx}.linear_attn"
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                key_map[f"{attn}.{proj}.weight"] = (
                    f"{lk}.{proj}", {"transpose": True},
                )
            key_map[f"{attn}.conv1d.weight"] = (
                f"{lk}.conv1d_weight", {"squeeze_axis": 1},
            )
            key_map[f"{attn}.dt_bias"] = (f"{lk}.dt_bias", {})
            key_map[f"{attn}.A_log"] = (f"{lk}.A_log", {})
            key_map[f"{attn}.norm.weight"] = (f"{lk}.norm_weight", {})

        # ---- MoE (every layer) ----
        mlp = f"{pfx}.mlp"
        key_map[f"{mlp}.gate.weight"] = (f"{lk}.router_weight", {})
        key_map[f"{mlp}.experts.gate_up_proj"] = (f"{lk}.expert_gate_up", {})
        key_map[f"{mlp}.experts.down_proj"] = (f"{lk}.expert_down", {})
        key_map[f"{mlp}.shared_expert.gate_proj.weight"] = (
            f"{lk}.shared_gate", {"transpose": True},
        )
        key_map[f"{mlp}.shared_expert.up_proj.weight"] = (
            f"{lk}.shared_up", {"transpose": True},
        )
        key_map[f"{mlp}.shared_expert.down_proj.weight"] = (
            f"{lk}.shared_down", {"transpose": True},
        )
        key_map[f"{mlp}.shared_expert_gate.weight"] = (
            f"{lk}.shared_expert_gate", {"transpose": True},
        )

        # ---- layer norms ----
        key_map[f"{pfx}.input_layernorm.weight"] = (
            f"{lk}.attention_norm_weight", {},
        )
        key_map[f"{pfx}.post_attention_layernorm.weight"] = (
            f"{lk}.ffn_norm_weight", {},
        )

    return key_map


# ---- pytree builder --------------------------------------------------------

def _insert_into_pytree(tree: dict, dotted_key: str, value):
    """Insert *value* into a nested dict following a dot-separated key."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        tree = tree.setdefault(part, {})
    tree[parts[-1]] = value


# ---- HuggingFace helpers ---------------------------------------------------

def _list_safetensor_files(repo_id: str) -> list[str]:
    """List all .safetensors filenames in a HF repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id)
    return sorted(f for f in files if f.endswith(".safetensors"))


def _download_config(repo_id: str, cache_dir: Path) -> Path:
    """Download config.json from a HF repo, return local path."""
    return Path(hf_hub_download(
        repo_id, "config.json", local_dir=cache_dir,
    ))


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 MoE safetensors to Orbax on GCS"
    )
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. Qwen/Qwen3.5-MoE")
    parser.add_argument("--gcs_path",
                        help="GCS destination, e.g. gs://bucket/checkpoint")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print HF→ReLax key mappings and shapes, skip writing")
    parser.add_argument("--temp_dir", default="./temp_npy_weights",
                        help="Directory for intermediate .npy files. Point to a "
                             "GCS FUSE mount (e.g. /mnt/gcs/temp_weights) to avoid "
                             "filling local disk. Defaults to ./temp_npy_weights.")
    parser.add_argument("--no_mmap", action="store_true",
                        help="Load .npy files into RAM instead of memory-mapping. "
                             "Requires ~244GB RAM but avoids potential dtype issues.")
    args = parser.parse_args()

    if not args.dry_run and not args.gcs_path:
        parser.error("--gcs_path is required unless --dry_run is set")

    temp_dir = Path(args.temp_dir)
    download_dir = Path("./temp_hf_download")
    target_dtype = ml_dtypes.bfloat16

    # Download config.json
    os.makedirs(download_dir, exist_ok=True)
    config_path = _download_config(args.repo, download_dir)
    config = QwenConfig.from_json_file(str(config_path.parent))
    print(f"Config: {config.n_layers} layers, {config.num_experts} experts, "
          f"dim={config.dim}, head_dim={config.head_dim}")

    # Build the mapping from HF keys → (relax_key, transform)
    key_map = _build_key_map(config)
    print(f"Key map has {len(key_map)} entries")

    shard_filenames = _list_safetensor_files(args.repo)
    if not shard_filenames:
        raise FileNotFoundError(f"No safetensors files in repo {args.repo}")
    print(f"Found {len(shard_filenames)} safetensors file(s) in {args.repo}")

    # ------------------------------------------------------------------
    # PASS 1: download each shard, extract + transform, delete shard
    # (skipped if temp_dir already contains .npy files)
    # ------------------------------------------------------------------
    existing_npy = list(temp_dir.glob("*.npy")) if temp_dir.exists() else []

    seen_relax_keys: set[str] = set()

    if existing_npy and not args.dry_run:
        print(f"\n--- PASS 1: SKIPPED ({len(existing_npy)} .npy files already in {temp_dir}) ---")
        for npy_path in existing_npy:
            relax_key = npy_path.stem.replace("__", ".")
            seen_relax_keys.add(relax_key)
        missing = set(v[0] for v in key_map.values()) - seen_relax_keys
        if missing:
            print(f"  WARNING: {len(missing)} ReLax keys not found in temp_dir:")
            for m in sorted(missing)[:20]:
                print(f"    {m}")
    else:
        if args.dry_run:
            print("\n--- DRY RUN: HF key → ReLax key + shape ---")
        else:
            print("\n--- PASS 1: Download + extract + transform → local .npy ---")
            os.makedirs(temp_dir, exist_ok=True)

        for shard_name in shard_filenames:
            print(f"\n  Downloading {shard_name} ...")
            local_shard = Path(hf_hub_download(
                args.repo, shard_name, local_dir=download_dir,
            ))

            with safe_open(str(local_shard), framework="np", device="cpu") as f:
                for hf_key in f.keys():
                    if hf_key not in key_map:
                        print(f"  [SKIP] Unknown HF key: {hf_key}")
                        continue

                    relax_key, xform_kw = key_map[hf_key]

                    tensor = f.get_tensor(hf_key)
                    tensor = _transform(tensor, **xform_kw)
                    tensor = tensor.astype(target_dtype)

                    if args.dry_run:
                        print(f"  {hf_key}  →  {relax_key}  {tuple(tensor.shape)}  {tensor.dtype}")
                    else:
                        npy_path = temp_dir / (relax_key.replace(".", "__") + ".npy")
                        nbytes = tensor.nbytes
                        t0 = time.time()
                        np.save(str(npy_path), tensor)
                        elapsed = time.time() - t0
                        print(f"    saved {relax_key}  {tuple(tensor.shape)}  "
                              f"{nbytes / 1e6:.1f} MB  ({elapsed:.2f}s)")

                    seen_relax_keys.add(relax_key)
                    del tensor

            # Delete the shard to reclaim disk space
            local_shard.unlink(missing_ok=True)
            gc.collect()
            if not args.dry_run:
                print(f"  Processed and deleted {shard_name}")

        # Sanity check
        missing = set(v[0] for v in key_map.values()) - seen_relax_keys
        if missing:
            print(f"\n  WARNING: {len(missing)} ReLax keys not found in safetensors:")
            for m in sorted(missing)[:20]:
                print(f"    {m}")

        # Clean up download directory
        shutil.rmtree(download_dir, ignore_errors=True)

        if args.dry_run:
            print(f"\nTotal: {len(seen_relax_keys)} tensors mapped")
            return

    # ------------------------------------------------------------------
    # PASS 2: mmap .npy files, build pytree, stream to GCS via Orbax
    # ------------------------------------------------------------------
    print("\n--- PASS 2: Build lazy pytree → Orbax to GCS ---")
    jax_pytree: dict = {}
    total_bytes = 0

    sorted_keys = sorted(seen_relax_keys)
    for idx, relax_key in enumerate(sorted_keys, 1):
        npy_path = temp_dir / (relax_key.replace(".", "__") + ".npy")
        if args.no_mmap:
            arr = np.load(str(npy_path))
        else:
            arr = np.load(str(npy_path), mmap_mode="r")
        # np.load mmap may return void16 instead of bfloat16; fix the dtype view
        if arr.dtype == np.dtype("V2"):
            arr = arr.view(ml_dtypes.bfloat16)
        nbytes = arr.nbytes
        total_bytes += nbytes
        mode = "load" if args.no_mmap else "mmap"
        print(f"  [{idx}/{len(sorted_keys)}] {mode} {relax_key}  {tuple(arr.shape)}  {nbytes / 1e6:.1f} MB")
        _insert_into_pytree(jax_pytree, relax_key, arr)

    print(f"\nLazy pytree assembled: {len(seen_relax_keys)} arrays, "
          f"total logical size {total_bytes / 1e9:.2f} GB")

    shared = args.gcs_path.startswith("gs://")
    if shared:
        from orbax.checkpoint.options import MultiprocessingOptions
        options = ocp.CheckpointManagerOptions(
            multiprocessing_options=MultiprocessingOptions(primary_host=0),
        )
    else:
        options = ocp.CheckpointManagerOptions()

    mngr = ocp.CheckpointManager(args.gcs_path, options=options)

    print(f"\nStreaming to {args.gcs_path} ...")
    t_save_start = time.time()
    mngr.save(step=0, args=ocp.args.StandardSave(jax_pytree))
    print("  save() returned, waiting for upload to finish ...")
    mngr.wait_until_finished()
    elapsed_save = time.time() - t_save_start
    throughput = total_bytes / elapsed_save / 1e6
    print(f"Upload complete! ({elapsed_save:.1f}s, ~{throughput:.0f} MB/s)")

    # Clean up .npy temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
