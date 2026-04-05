"""
Convert Qwen3.5-122B-A10B safetensors weights from HuggingFace to a sharded
Orbax checkpoint on GCS.

Three stages:
  1. Download safetensors shards from HF, transform each tensor to .npy,
     delete the shard to reclaim disk.  (CPU only)
  2. Load .npy files, build the ReLax param pytree, create a TPU mesh,
     and shard params onto TPU via MeshHelper.shard_params.
  3. Save the sharded pytree to GCS via Orbax.

Usage:
    # Dry run — print HF→ReLax key mappings and shapes, no writes:
    python scripts/convert_qwen_to_orbax.py \
        --repo Qwen/Qwen3.5-122B-A10B --dry_run

    # Full conversion (on a v4-16 TPU, 4×4 mesh):
    python scripts/convert_qwen_to_orbax.py \
        --repo Qwen/Qwen3.5-122B-A10B \
        --gcs_path gs://my-bucket/qwen3.5-orbax \
        --dp 4 --tp 4
"""

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

# Limit tensorstore parallelism to avoid OOM during Orbax upload.
os.environ.setdefault("TENSORSTORE_NUM_THREADS", "2")

import ml_dtypes  # noqa: F401 – registers bfloat16 with numpy
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.qwen.config import QwenConfig


# ---- HF key prefix --------------------------------------------------------

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
    """Map every HF safetensors key to (relax_key, transform_kwargs)."""
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
                {"transpose": True,
                 "reshape": (config.dim, config.n_heads, config.head_dim * 2)},
            )
            key_map[f"{attn}.k_proj.weight"] = (
                f"{lk}.wk",
                {"transpose": True,
                 "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
            )
            key_map[f"{attn}.v_proj.weight"] = (
                f"{lk}.wv",
                {"transpose": True,
                 "reshape": (config.dim, config.n_kv_heads, config.head_dim)},
            )
            key_map[f"{attn}.o_proj.weight"] = (
                f"{lk}.wo", {"transpose": True},
            )
            key_map[f"{attn}.q_norm.weight"] = (f"{lk}.q_norm", {})
            key_map[f"{attn}.k_norm.weight"] = (f"{lk}.k_norm", {})
        else:
            # linear attention (Gated DeltaNet)
            attn = f"{pfx}.linear_attn"
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a",
                         "in_proj_b", "out_proj"):
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
    api = HfApi()
    files = api.list_repo_files(repo_id)
    return sorted(f for f in files if f.endswith(".safetensors"))


def _download_config(repo_id: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(repo_id, "config.json", local_dir=cache_dir))


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 MoE safetensors → sharded Orbax on GCS"
    )
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--gcs_path", required=True,
                        help="GCS destination, e.g. gs://bucket/checkpoint")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print HF→ReLax key mappings and shapes only")
    parser.add_argument("--temp_dir", required=True,
                        help="Directory for intermediate .npy files")
    parser.add_argument("--dp", type=int, default=2,
                        help="Data-parallel mesh dimension (default: 2)")
    parser.add_argument("--tp", type=int, default=8,
                        help="Tensor-parallel mesh dimension (default: 8)")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    download_dir = Path("./temp_hf_download")
    target_dtype = ml_dtypes.bfloat16

    # ------------------------------------------------------------------
    # Download config and build key map
    # ------------------------------------------------------------------
    os.makedirs(download_dir, exist_ok=True)
    config_path = _download_config(args.repo, download_dir)
    config = QwenConfig.from_json_file(str(config_path.parent))
    print(f"Config: {config.n_layers} layers, {config.num_experts} experts, "
          f"dim={config.dim}, head_dim={config.head_dim}")

    key_map = _build_key_map(config)
    print(f"Key map has {len(key_map)} entries")

    shard_filenames = _list_safetensor_files(args.repo)
    if not shard_filenames:
        raise FileNotFoundError(f"No safetensors files in repo {args.repo}")
    print(f"Found {len(shard_filenames)} safetensors file(s) in {args.repo}")

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
            print("\n--- DRY RUN: HF key → ReLax key + shape ---")
        else:
            print("\n--- STAGE 1: Download + transform → .npy ---")
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
                        print(f"  {hf_key}  →  {relax_key}  "
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
    # STAGE 2: Load .npy → shard onto TPU (one tensor at a time)
    # ------------------------------------------------------------------
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from utils.mesh_helpers import MeshHelper
    from models.sync_server import SyncServer

    print(f"\n--- STAGE 2: Load and shard onto TPU tensor-by-tensor "
          f"(dp={args.dp}, tp={args.tp}) ---")

    # Create TPU mesh
    devices = jax.devices()
    expected = args.dp * args.tp
    assert len(devices) == expected, (
        f"Expected {expected} devices (dp={args.dp} × tp={args.tp}), "
        f"got {len(devices)}"
    )
    mesh = Mesh(np.array(devices).reshape(args.dp, args.tp), ("dp", "tp"))
    print(f"Mesh: {mesh.shape} on {len(devices)} device(s)")

    # Load each .npy, shard onto TPU, discard numpy array immediately.
    # Barrier before each device_put ensures all hosts are ready.
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

        # Determine sharding and place directly on TPU
        spec = MeshHelper.param_sharding(arr, relax_key, mesh)
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

    print(f"\n--- STAGE 3: Orbax save → {args.gcs_path} ---")

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
