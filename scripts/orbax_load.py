"""
Memory-efficient script to convert Qwen3.5 MoE safetensors weights to an
Orbax checkpoint on GCS.

Two-pass approach:
  Pass 1 — Stream each tensor from safetensors, apply the HF→ReLax
           transformation (transpose, reshape, etc.), and save as a .npy
           file on local disk.  Only one tensor is in RAM at a time.
  Pass 2 — Memory-map every .npy file, assemble the ReLax pytree, and let
           Orbax stream chunks to GCS.

Usage:
    python scripts/orbax_load.py \
        --model_path /path/to/downloaded/model \
        --gcs_path gs://my-bucket/qwen3.5-orbax \
        [--temp_dir ./temp_npy_weights]
"""

import argparse
import gc
import os
from pathlib import Path

import ml_dtypes  # noqa: F401 – registers bfloat16 with numpy
import numpy as np
import orbax.checkpoint as ocp
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


def _relax_key(parts: list[str]) -> str:
    """Join parts into a dot-separated ReLax pytree key."""
    return ".".join(parts)


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


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 MoE safetensors to Orbax on GCS"
    )
    parser.add_argument("--model_path", required=True,
                        help="Directory with *.safetensors and config.json")
    parser.add_argument("--gcs_path", required=True,
                        help="GCS destination, e.g. gs://bucket/checkpoint")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    temp_dir = Path("./temp_npy_weights")
    target_dtype = ml_dtypes.bfloat16

    # Load config
    config = QwenConfig.from_json_file(str(model_path))
    print(f"Config: {config.n_layers} layers, {config.num_experts} experts, "
          f"dim={config.dim}, head_dim={config.head_dim}")

    # Build the mapping from HF keys → (relax_key, transform)
    key_map = _build_key_map(config)
    print(f"Key map has {len(key_map)} entries")

    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")
    print(f"Found {len(safetensor_files)} safetensors file(s)")

    os.makedirs(temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PASS 1: extract, transform, save to local .npy (one tensor at a time)
    # ------------------------------------------------------------------
    print("\n--- PASS 1: Extract + transform → local .npy ---")
    seen_relax_keys: set[str] = set()

    for sf in safetensor_files:
        with safe_open(str(sf), framework="np", device="cpu") as f:
            for hf_key in f.keys():
                if hf_key not in key_map:
                    print(f"  [SKIP] Unknown HF key: {hf_key}")
                    continue

                relax_key, xform_kw = key_map[hf_key]

                tensor = f.get_tensor(hf_key)
                tensor = _transform(tensor, **xform_kw)
                tensor = tensor.astype(target_dtype)

                npy_path = temp_dir / (relax_key.replace(".", "__") + ".npy")
                np.save(str(npy_path), tensor)
                seen_relax_keys.add(relax_key)

                del tensor

        gc.collect()
        print(f"  Processed {sf.name}")

    # Sanity check
    expected = set(key_map.values().__iter__().__class__.__name__)  # just count
    missing = set(v[0] for v in key_map.values()) - seen_relax_keys
    if missing:
        print(f"\n  WARNING: {len(missing)} ReLax keys not found in safetensors:")
        for m in sorted(missing)[:20]:
            print(f"    {m}")

    # ------------------------------------------------------------------
    # PASS 2: mmap .npy files, build pytree, stream to GCS via Orbax
    # ------------------------------------------------------------------
    print("\n--- PASS 2: Build lazy pytree → Orbax to GCS ---")
    jax_pytree: dict = {}

    for relax_key in sorted(seen_relax_keys):
        npy_path = temp_dir / (relax_key.replace(".", "__") + ".npy")
        lazy = np.load(str(npy_path), mmap_mode="r")
        _insert_into_pytree(jax_pytree, relax_key, lazy)

    print(f"Lazy pytree assembled ({len(seen_relax_keys)} arrays, ~0 RAM)")

    shared = args.gcs_path.startswith("gs://")
    if shared:
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


if __name__ == "__main__":
    main()
