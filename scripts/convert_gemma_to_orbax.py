"""
Convert Gemma 4 safetensors weights from HuggingFace to a sharded Orbax
checkpoint on GCS.

Three stages:
  1. Download safetensors shards from HF, transform each tensor to .npy,
     delete the shard to reclaim disk.  (CPU only)
  2. Load .npy files, build the ReLax param pytree, create a TPU mesh,
     and shard params onto TPU via auto-sharding.
  3. Save the sharded pytree to GCS via Orbax.

Gemma-specific notes:
  - Sliding attention layers have Q/K/V/O projections + QK norms.
  - Global attention layers have Q/K/O only (K=V, no v_proj).
  - Tied word embeddings: no separate output projection.
  - Sandwich norms: 4 layer norms per layer.

Usage:
    # Dry run -- print HF->ReLax key mappings and shapes, no writes:
    python scripts/convert_gemma_to_orbax.py \
        --repo google/gemma-4-27b-it --dry_run

    # Full conversion (on a v4-16 TPU):
    python scripts/convert_gemma_to_orbax.py \
        --repo google/gemma-4-27b-it \
        --gcs_path gs://my-bucket/gemma4-orbax \
        --tp 16
"""

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

# Limit tensorstore parallelism to avoid OOM during Orbax upload.
os.environ.setdefault("TENSORSTORE_NUM_THREADS", "2")

import ml_dtypes  # noqa: F401 -- registers bfloat16 with numpy
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.gemma.config import GemmaConfig


# ---- helpers ---------------------------------------------------------------

def _transform(tensor: np.ndarray, *, transpose: bool = False,
               reshape: tuple | None = None) -> np.ndarray:
    if transpose:
        tensor = tensor.T
    if reshape is not None:
        tensor = tensor.reshape(reshape)
    return tensor


def _detect_prefix(keys: list[str]) -> str:
    """Detect whether weights use 'model.layers' or 'model.language_model.layers'."""
    for key in keys:
        if key.startswith("model.language_model.layers."):
            return "model.language_model"
        if key.startswith("model.layers."):
            return "model"
    raise KeyError("Could not detect weight prefix: no layer weights found")


# ---- per-key transformation table ------------------------------------------

def _build_key_map(config: GemmaConfig, prefix: str) -> dict[str, tuple[str, dict]]:
    """Map every HF safetensors key to (relax_key, transform_kwargs)."""
    key_map: dict[str, tuple[str, dict]] = {}

    # --- embeddings (tied -- no separate output projection) ---
    key_map[f"{prefix}.embed_tokens.weight"] = (
        "tok_embeddings.embedding", {}
    )

    # --- final norm ---
    key_map[f"{prefix}.norm.weight"] = ("norm_weight", {})

    # --- per-layer ---
    for i in range(config.n_layers):
        lp = f"{prefix}.layers.{i}"
        lk = f"layer_{i}"
        attn = f"{lp}.self_attn"
        layer_type = config.layer_types[i]

        # ---- attention projections ----
        if layer_type == "sliding_attention":
            hd = config.head_dim
            key_map[f"{attn}.q_proj.weight"] = (
                f"{lk}.wq",
                {"transpose": True,
                 "reshape": (config.dim, config.n_heads, hd)},
            )
            key_map[f"{attn}.k_proj.weight"] = (
                f"{lk}.wk",
                {"transpose": True,
                 "reshape": (config.dim, config.n_kv_heads, hd)},
            )
            key_map[f"{attn}.v_proj.weight"] = (
                f"{lk}.wv",
                {"transpose": True,
                 "reshape": (config.dim, config.n_kv_heads, hd)},
            )
        else:
            # Global attention: K=V, no v_proj
            ghd = config.global_head_dim
            key_map[f"{attn}.q_proj.weight"] = (
                f"{lk}.wq",
                {"transpose": True,
                 "reshape": (config.dim, config.n_heads, ghd)},
            )
            key_map[f"{attn}.k_proj.weight"] = (
                f"{lk}.wk",
                {"transpose": True,
                 "reshape": (config.dim, config.n_global_kv_heads, ghd)},
            )

        # O proj and QK norms (both layer types)
        key_map[f"{attn}.o_proj.weight"] = (f"{lk}.wo", {"transpose": True})
        key_map[f"{attn}.q_norm.weight"] = (f"{lk}.q_norm", {})
        key_map[f"{attn}.k_norm.weight"] = (f"{lk}.k_norm", {})

        # ---- FFN ----
        mlp = f"{lp}.mlp"
        key_map[f"{mlp}.gate_proj.weight"] = (f"{lk}.w_gate", {"transpose": True})
        key_map[f"{mlp}.up_proj.weight"] = (f"{lk}.w_up", {"transpose": True})
        key_map[f"{mlp}.down_proj.weight"] = (f"{lk}.w_down", {"transpose": True})

        # ---- 4 sandwich layer norms ----
        key_map[f"{lp}.input_layernorm.weight"] = (
            f"{lk}.input_layernorm", {}
        )
        key_map[f"{lp}.post_attention_layernorm.weight"] = (
            f"{lk}.post_attention_layernorm", {}
        )
        key_map[f"{lp}.pre_feedforward_layernorm.weight"] = (
            f"{lk}.pre_feedforward_layernorm", {}
        )
        key_map[f"{lp}.post_feedforward_layernorm.weight"] = (
            f"{lk}.post_feedforward_layernorm", {}
        )

        # Per-layer residual scalar
        key_map[f"{lp}.layer_scalar"] = (f"{lk}.layer_scalar", {})

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
    api = HfApi()
    files = api.list_repo_files(repo_id)
    return sorted(f for f in files if f.endswith(".safetensors"))


def _download_config(repo_id: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(repo_id, "config.json", local_dir=cache_dir))


def _peek_hf_keys(repo_id: str, shard_filenames: list[str],
                   download_dir: Path) -> list[str]:
    """Return weight key names from the safetensors index (no shard download needed)."""
    import json
    # Try the sharded index file first -- it lists all keys without downloading weights.
    for index_name in ("model.safetensors.index.json",
                       "model.safetensors.index.json"):
        try:
            index_path = Path(hf_hub_download(
                repo_id, index_name, local_dir=download_dir,
            ))
            with open(index_path) as fh:
                data = json.load(fh)
            return list(data["weight_map"].keys())
        except Exception:
            pass
    # Fallback: download only the first shard (may be large).
    first = Path(hf_hub_download(
        repo_id, shard_filenames[0], local_dir=download_dir,
    ))
    with safe_open(str(first), framework="np", device="cpu") as f:
        keys = list(f.keys())
    return keys


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma 4 safetensors -> sharded Orbax on GCS"
    )
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. google/gemma-4-27b-it")
    parser.add_argument("--gcs_path",
                        help="GCS destination, e.g. gs://bucket/checkpoint")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print HF->ReLax key mappings and shapes only")
    parser.add_argument("--temp_dir", default="./temp_gemma_npy",
                        help="Directory for intermediate .npy files")
    parser.add_argument("--tp", type=int, default=16,
                        help="Tensor-parallel mesh dimension (default: 16)")
    args = parser.parse_args()

    if not args.dry_run and not args.gcs_path:
        parser.error("--gcs_path is required unless --dry_run is set")

    temp_dir = Path(args.temp_dir)
    download_dir = Path("./temp_hf_download")
    target_dtype = ml_dtypes.bfloat16

    # ------------------------------------------------------------------
    # Download config and build key map
    # ------------------------------------------------------------------
    os.makedirs(download_dir, exist_ok=True)
    config_path = _download_config(args.repo, download_dir)
    config = GemmaConfig.from_json_file(str(config_path.parent))
    print(f"Config: {config.n_layers} layers, dim={config.dim}, "
          f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, "
          f"head_dim={config.head_dim}, global_head_dim={config.global_head_dim}, "
          f"n_global_kv_heads={config.n_global_kv_heads}")

    shard_filenames = _list_safetensor_files(args.repo)
    if not shard_filenames:
        raise FileNotFoundError(f"No safetensors files in repo {args.repo}")
    print(f"Found {len(shard_filenames)} safetensors file(s) in {args.repo}")

    # Detect prefix from the first shard's keys
    first_keys = _peek_hf_keys(args.repo, shard_filenames, download_dir)
    prefix = _detect_prefix(first_keys)
    print(f"Detected HF prefix: '{prefix}'")

    key_map = _build_key_map(config, prefix)
    print(f"Key map has {len(key_map)} entries")

    # ------------------------------------------------------------------
    # STAGE 1: Download shards, transform, save as .npy  (CPU only)
    # ------------------------------------------------------------------
    existing_npy = list(temp_dir.glob("*.npy")) if temp_dir.exists() else []
    seen_relax_keys: set[str] = set()

    if existing_npy and not args.dry_run:
        print(f"\n--- STAGE 1: SKIPPED ({len(existing_npy)} .npy files "
              f"already in {temp_dir}) ---")
        for npy_path in existing_npy:
            relax_key = npy_path.stem.replace("__", ".")
            seen_relax_keys.add(relax_key)
    else:
        if args.dry_run:
            print("\n--- DRY RUN: HF key -> ReLax key + shape ---")
        else:
            print("\n--- STAGE 1: Download + transform -> .npy ---")
            os.makedirs(temp_dir, exist_ok=True)

        for shard_name in shard_filenames:
            print(f"\n  Downloading {shard_name} ...")
            local_shard = Path(hf_hub_download(
                args.repo, shard_name, local_dir=download_dir,
            ))

            with safe_open(str(local_shard), framework="np",
                           device="cpu") as f:
                for hf_key in f.keys():
                    if hf_key not in key_map:
                        print(f"  [SKIP] Unknown HF key: {hf_key}")
                        continue

                    relax_key, xform_kw = key_map[hf_key]
                    tensor = f.get_tensor(hf_key)
                    tensor = _transform(tensor, **xform_kw)
                    tensor = tensor.astype(target_dtype)

                    if args.dry_run:
                        print(f"  {hf_key}  ->  {relax_key}  "
                              f"{tuple(tensor.shape)}  {tensor.dtype}")
                    else:
                        npy_path = temp_dir / (
                            relax_key.replace(".", "__") + ".npy"
                        )
                        t0 = time.time()
                        np.save(str(npy_path), tensor)
                        elapsed = time.time() - t0
                        print(f"    saved {relax_key}  {tuple(tensor.shape)}  "
                              f"{tensor.nbytes / 1e6:.1f} MB  ({elapsed:.2f}s)")

                    seen_relax_keys.add(relax_key)
                    del tensor

            local_shard.unlink(missing_ok=True)
            gc.collect()
            if not args.dry_run:
                print(f"  Processed and deleted {shard_name}")

        shutil.rmtree(download_dir, ignore_errors=True)

    # Sanity check
    missing = set(v[0] for v in key_map.values()) - seen_relax_keys
    if missing:
        print(f"\n  WARNING: {len(missing)} ReLax keys missing:")
        for m in sorted(missing)[:20]:
            print(f"    {m}")

    if args.dry_run:
        print(f"\nTotal: {len(seen_relax_keys)} tensors mapped")
        return

    # ------------------------------------------------------------------
    # STAGE 2: Load .npy -> shard onto TPU (one tensor at a time)
    # ------------------------------------------------------------------
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
    from models.sync_server import SyncServer

    print(f"\n--- STAGE 2: Load and shard onto TPU tensor-by-tensor "
          f"(tp={args.tp}) ---")

    # Create TP-only mesh
    devices = jax.devices()
    assert len(devices) == args.tp, (
        f"Expected {args.tp} devices (tp={args.tp}), got {len(devices)}"
    )
    mesh = Mesh(np.array(devices), ("tp",))
    print(f"Mesh: {mesh.shape} on {len(devices)} device(s)")

    def _auto_shard_spec(shape: tuple, tp: int) -> PS:
        """Find the first axis divisible by tp and shard on it; else replicate."""
        for axis in range(len(shape)):
            if shape[axis] % tp == 0:
                spec = [None] * len(shape)
                spec[axis] = "tp"
                return PS(*spec)
        return PS()

    # Load each .npy, shard onto TPU, discard numpy array immediately.
    params: dict = {}
    total_bytes = 0
    t_shard = time.time()

    sorted_keys = sorted(seen_relax_keys)
    for idx, relax_key in enumerate(sorted_keys, 1):
        npy_path = temp_dir / (relax_key.replace(".", "__") + ".npy")
        arr = np.load(str(npy_path))
        if arr.dtype == np.uint16:
            arr = arr.view(ml_dtypes.bfloat16)
        total_bytes += arr.nbytes

        # Convert to bfloat16 if needed
        if np.issubdtype(arr.dtype, np.floating) and arr.dtype != ml_dtypes.bfloat16:
            arr = arr.astype(ml_dtypes.bfloat16)

        # Sync all hosts before placing on device
        SyncServer.barrier("shard_param", idx)

        # Shard on first divisible axis, else replicate
        spec = _auto_shard_spec(arr.shape, args.tp)
        t0 = time.time()
        jax_arr = jax.device_put(arr, NamedSharding(mesh, spec))
        dt = time.time() - t0
        del arr  # free host RAM immediately

        _insert_into_pytree(params, relax_key, jax_arr)

        print(f"  [{idx}/{len(sorted_keys)}] {relax_key}  "
              f"{tuple(jax_arr.shape)}  {spec}  "
              f"{jax_arr.nbytes / 1e6:.1f} MB  ({dt:.2f}s)")

    jax.block_until_ready(params)
    elapsed_shard = time.time() - t_shard
    print(f"\nSharded pytree on TPU: {len(seen_relax_keys)} arrays, "
          f"{total_bytes / 1e9:.2f} GB ({elapsed_shard:.1f}s)")
    gc.collect()

    # ------------------------------------------------------------------
    # STAGE 3: Save sharded params to GCS via Orbax
    # ------------------------------------------------------------------
    import orbax.checkpoint as ocp
    from orbax.checkpoint.options import MultiprocessingOptions

    print(f"\n--- STAGE 3: Orbax save -> {args.gcs_path} ---")

    shared = args.gcs_path.startswith("gs://")
    primary = 0 if shared else None
    options = ocp.CheckpointManagerOptions(
        multiprocessing_options=MultiprocessingOptions(primary_host=primary),
    )
    mngr = ocp.CheckpointManager(args.gcs_path, options=options)

    print("Saving...")
    t_save = time.time()
    mngr.save(step=0, args=ocp.args.StandardSave(params))
    mngr.wait_until_finished()
    elapsed_save = time.time() - t_save
    throughput = total_bytes / elapsed_save / 1e6
    print(f"Upload complete! ({elapsed_save:.1f}s, ~{throughput:.0f} MB/s)")

    if jax.process_index() == 0:
        print(f"Checkpoint saved to {args.gcs_path}")

    # Clean up temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
