"""
Functions to load Qwen3.5 MoE weights from HuggingFace safetensors format.

Handles the conversion from HF weight naming conventions to the ReLax
parameter structure expected by the Qwen model.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np

from .config import QwenConfig


def load_qwen_weights(model_path: str, config: QwenConfig) -> Dict[str, Any]:
    """
    Load Qwen3.5 MoE weights from safetensors files.

    Args:
        model_path: Path to directory containing *.safetensors files.
        config: QwenConfig instance.

    Returns:
        ReLax parameter dictionary matching the Qwen model structure.
    """
    from safetensors import safe_open

    model_path = Path(model_path)
    shard_files = sorted(model_path.glob("*.safetensors"))

    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}"
        )

    print(f"Loading model from {model_path}")
    print(f"Found {len(shard_files)} safetensors file(s)")

    all_weights = {}
    for shard_path in shard_files:
        print(f"  Loading {shard_path.name}...")
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    print(f"Loaded {len(all_weights)} tensors total")

    params = _convert_hf_to_relax(all_weights, config)
    return params


def _get(weights: dict, key: str) -> np.ndarray:
    """Get a weight tensor, raising a clear error if missing."""
    if key not in weights:
        raise KeyError(f"Weight tensor not found: {key}")
    return weights[key]


def _to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 for parameter storage."""
    return arr.astype(np.float32)


def _convert_hf_to_relax(
    hf_weights: Dict[str, np.ndarray],
    config: QwenConfig,
) -> Dict[str, Any]:
    """Convert HuggingFace Qwen3.5 MoE weights to ReLax format."""
    print("Converting HuggingFace weights to ReLax format...")

    params = {}

    # --- Embeddings ---
    embed_key = "model.language_model.embed_tokens.weight"
    embed_weight = _to_f32(_get(hf_weights, embed_key))
    params["tok_embeddings"] = {"embedding": embed_weight}
    print(f"  Embeddings: {embed_weight.shape}")

    # --- Output projection ---
    lm_head = _to_f32(_get(hf_weights, "lm_head.weight"))
    params["output"] = lm_head.T  # [vocab, dim] -> [dim, vocab]

    # --- Final norm ---
    norm_key = "model.language_model.norm.weight"
    if norm_key in hf_weights:
        params["norm_weight"] = _to_f32(_get(hf_weights, norm_key))

    # --- Transformer layers ---
    print(f"  Converting {config.n_layers} transformer layers...")

    for i in range(config.n_layers):
        layer_type = config.layer_types[i]
        layer_params = _convert_layer(hf_weights, i, layer_type, config)
        params[f"layer_{i}"] = layer_params

        if i == 0:
            print(f"    Layer 0 ({layer_type}) shapes:")
            for k, v in layer_params.items():
                if isinstance(v, np.ndarray):
                    print(f"      {k}: {v.shape}")

    print(f"  Converted all {config.n_layers} layers")
    return params


def _convert_layer(
    hf_weights: Dict[str, np.ndarray],
    layer_idx: int,
    layer_type: str,
    config: QwenConfig,
) -> Dict[str, np.ndarray]:
    """Convert a single transformer layer to ReLax format."""
    prefix = f"model.language_model.layers.{layer_idx}"
    layer = {}

    # --- Attention weights ---
    if layer_type == "full_attention":
        _convert_full_attention(hf_weights, prefix, config, layer)
    else:
        _convert_linear_attention(hf_weights, prefix, config, layer)

    # --- MoE weights (same for all layers) ---
    _convert_moe(hf_weights, prefix, config, layer)

    # --- Layer norms ---
    layer["attention_norm_weight"] = _to_f32(
        _get(hf_weights, f"{prefix}.input_layernorm.weight")
    )
    layer["ffn_norm_weight"] = _to_f32(
        _get(hf_weights, f"{prefix}.post_attention_layernorm.weight")
    )

    return layer


def _convert_full_attention(
    hf_weights: dict, prefix: str, config: QwenConfig, layer: dict
):
    """Convert full attention (GQA) weights."""
    attn = f"{prefix}.self_attn"

    # Q: [n_heads * head_dim * 2, dim] -> [dim, n_heads, head_dim * 2]
    q_proj = _to_f32(_get(hf_weights, f"{attn}.q_proj.weight"))
    layer["wq"] = q_proj.T.reshape(config.dim, config.n_heads, config.head_dim * 2)

    # K: [n_kv_heads * head_dim, dim] -> [dim, n_kv_heads, head_dim]
    k_proj = _to_f32(_get(hf_weights, f"{attn}.k_proj.weight"))
    layer["wk"] = k_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)

    # V: [n_kv_heads * head_dim, dim] -> [dim, n_kv_heads, head_dim]
    v_proj = _to_f32(_get(hf_weights, f"{attn}.v_proj.weight"))
    layer["wv"] = v_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)

    # O: [dim, n_heads * head_dim] -> [n_heads * head_dim, dim]
    o_proj = _to_f32(_get(hf_weights, f"{attn}.o_proj.weight"))
    layer["wo"] = o_proj.T

    # QK norms
    layer["q_norm"] = _to_f32(_get(hf_weights, f"{attn}.q_norm.weight"))
    layer["k_norm"] = _to_f32(_get(hf_weights, f"{attn}.k_norm.weight"))


def _convert_linear_attention(
    hf_weights: dict, prefix: str, config: QwenConfig, layer: dict
):
    """Convert linear attention (Gated DeltaNet) weights."""
    attn = f"{prefix}.linear_attn"

    # Projection weights: [out, in] -> [in, out]
    layer["in_proj_qkv"] = _to_f32(
        _get(hf_weights, f"{attn}.in_proj_qkv.weight")
    ).T
    layer["in_proj_z"] = _to_f32(
        _get(hf_weights, f"{attn}.in_proj_z.weight")
    ).T
    layer["in_proj_a"] = _to_f32(
        _get(hf_weights, f"{attn}.in_proj_a.weight")
    ).T
    layer["in_proj_b"] = _to_f32(
        _get(hf_weights, f"{attn}.in_proj_b.weight")
    ).T
    layer["out_proj"] = _to_f32(
        _get(hf_weights, f"{attn}.out_proj.weight")
    ).T

    # Conv1d: [conv_dim, 1, kernel_size] -> [conv_dim, kernel_size]
    conv_w = _to_f32(_get(hf_weights, f"{attn}.conv1d.weight"))
    layer["conv1d_weight"] = conv_w.squeeze(1)

    # Scalar params (no transpose needed)
    layer["dt_bias"] = _to_f32(_get(hf_weights, f"{attn}.dt_bias"))
    layer["A_log"] = _to_f32(_get(hf_weights, f"{attn}.A_log"))

    # Gated norm weight
    layer["norm_weight"] = _to_f32(
        _get(hf_weights, f"{attn}.norm.weight")
    )


def _convert_moe(
    hf_weights: dict, prefix: str, config: QwenConfig, layer: dict
):
    """Convert MoE block weights."""
    mlp = f"{prefix}.mlp"

    # Router: [num_experts, dim] — keep as-is
    layer["router_weight"] = _to_f32(
        _get(hf_weights, f"{mlp}.gate.weight")
    )

    # Fused expert weights: these are already in the right shape for gather
    # gate_up_proj: [num_experts, 2*intermediate, dim]
    layer["expert_gate_up"] = _to_f32(
        _get(hf_weights, f"{mlp}.experts.gate_up_proj")
    )
    # down_proj: [num_experts, dim, intermediate]
    layer["expert_down"] = _to_f32(
        _get(hf_weights, f"{mlp}.experts.down_proj")
    )

    # Shared expert: [out, in] -> [in, out]
    layer["shared_gate"] = _to_f32(
        _get(hf_weights, f"{mlp}.shared_expert.gate_proj.weight")
    ).T
    layer["shared_up"] = _to_f32(
        _get(hf_weights, f"{mlp}.shared_expert.up_proj.weight")
    ).T
    layer["shared_down"] = _to_f32(
        _get(hf_weights, f"{mlp}.shared_expert.down_proj.weight")
    ).T

    # Shared expert gate: [1, dim] -> [dim, 1]
    layer["shared_expert_gate"] = _to_f32(
        _get(hf_weights, f"{mlp}.shared_expert_gate.weight")
    ).T


# ---------------------------------------------------------------------------
# Orbax checkpoint save/load (reuses pattern from llama/load.py)
# ---------------------------------------------------------------------------


def _make_checkpoint_manager(checkpoint_path, shared_storage: bool = False):
    """Create a CheckpointManager.

    Args:
        checkpoint_path: Local or GCS (gs://...) path.
        shared_storage: If True (e.g. GCS), use primary_host=0 so only one
            host writes metadata. If False (local disk per host), use
            primary_host=None so each host writes independently.
    """
    import orbax.checkpoint as ocp
    from orbax.checkpoint.options import MultiprocessingOptions

    primary = 0 if shared_storage else None
    options = ocp.CheckpointManagerOptions(
        multiprocessing_options=MultiprocessingOptions(primary_host=primary),
    )
    return ocp.CheckpointManager(checkpoint_path, options=options)


def save_orbax_weights(
    params: Dict[str, Any], checkpoint_path: str, mesh=None
) -> None:
    """Save ReLax params to an orbax checkpoint directory.

    Automatically detects GCS paths (gs://...) and uses shared-storage mode
    so only host 0 writes metadata.
    """
    import orbax.checkpoint as ocp
    import jax

    shared = str(checkpoint_path).startswith("gs://")

    if mesh is not None:
        from utils.mesh_helpers import MeshHelper

        params = MeshHelper.shard_params(params, mesh)

    mngr = _make_checkpoint_manager(checkpoint_path, shared_storage=shared)
    mngr.save(0, args=ocp.args.StandardSave(params))
    mngr.wait_until_finished()
    if jax.process_index() == 0:
        print(f"Saved orbax checkpoint to {checkpoint_path}")


def load_from_orbax(
    checkpoint_path: str,
    mesh=None,
) -> Dict[str, Any]:
    """Load ReLax params from an orbax checkpoint, optionally sharding onto a mesh.

    Automatically detects GCS paths (gs://...) and uses shared-storage mode.
    """
    import orbax.checkpoint as ocp
    import jax

    shared = str(checkpoint_path).startswith("gs://")
    mngr = _make_checkpoint_manager(checkpoint_path, shared_storage=shared)
    step = mngr.latest_step()

    if mesh is None:
        params = mngr.restore(step, args=ocp.args.StandardRestore(None))
        if jax.process_index() == 0:
            print(f"Loaded orbax checkpoint from {checkpoint_path}")
        return params

    from jax.sharding import NamedSharding
    from utils.mesh_helpers import MeshHelper

    item_meta = mngr.item_metadata(step)

    def _get_key_name(key) -> str:
        if hasattr(key, "key"):
            return str(key.key)
        elif hasattr(key, "idx"):
            return str(key.idx)
        elif hasattr(key, "name"):
            return str(key.name)
        return str(key)

    def _build_target(path, meta):
        name = "/".join(_get_key_name(k) for k in path)
        spec = MeshHelper.param_sharding(meta, name, mesh)
        sharding = NamedSharding(mesh, spec)
        return jax.ShapeDtypeStruct(meta.shape, meta.dtype, sharding=sharding)

    target = jax.tree.map_with_path(_build_target, item_meta)
    params = mngr.restore(step, args=ocp.args.StandardRestore(target))

    if jax.process_index() == 0:
        print(
            f"Loaded orbax checkpoint from {checkpoint_path} "
            f"(sharded onto mesh {mesh.axis_names})"
        )
    return params
