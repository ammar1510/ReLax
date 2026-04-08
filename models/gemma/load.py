"""
Functions to load Gemma 4 weights from HuggingFace safetensors format.

Handles the conversion from HF weight naming conventions to the ReLax
parameter structure expected by the Gemma model.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np

from .config import GemmaConfig


def load_gemma_weights(model_path: str, config: GemmaConfig) -> Dict[str, Any]:
    """
    Load Gemma 4 weights from safetensors files.

    Args:
        model_path: Path to directory containing *.safetensors files.
        config: GemmaConfig instance.

    Returns:
        ReLax parameter dictionary matching the Gemma model structure.
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


def _detect_prefix(hf_weights: Dict[str, np.ndarray]) -> str:
    """Detect whether weights use 'model.layers' or 'model.language_model.layers'."""
    for key in hf_weights:
        if key.startswith("model.language_model.layers."):
            return "model.language_model"
        if key.startswith("model.layers."):
            return "model"
    raise KeyError("Could not detect weight prefix: no layer weights found")


def _convert_hf_to_relax(
    hf_weights: Dict[str, np.ndarray],
    config: GemmaConfig,
) -> Dict[str, Any]:
    """Convert HuggingFace Gemma 4 weights to ReLax format."""
    print("Converting HuggingFace weights to ReLax format...")

    prefix = _detect_prefix(hf_weights)
    params = {}

    # --- Embeddings ---
    embed_key = f"{prefix}.embed_tokens.weight"
    embed_weight = _to_f32(_get(hf_weights, embed_key))
    params["tok_embeddings"] = {"embedding": embed_weight}
    print(f"  Embeddings: {embed_weight.shape}")

    # --- Final norm ---
    norm_key = f"{prefix}.norm.weight"
    params["norm_weight"] = _to_f32(_get(hf_weights, norm_key))

    # --- Transformer layers ---
    print(f"  Converting {config.n_layers} transformer layers...")

    for i in range(config.n_layers):
        layer_type = config.layer_types[i]
        layer_params = _convert_layer(hf_weights, prefix, i, layer_type, config)
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
    prefix: str,
    layer_idx: int,
    layer_type: str,
    config: GemmaConfig,
) -> Dict[str, np.ndarray]:
    """Convert a single transformer layer to ReLax format."""
    lp = f"{prefix}.layers.{layer_idx}"
    layer = {}

    # --- Attention weights ---
    if layer_type == "sliding_attention":
        _convert_sliding_attention(hf_weights, lp, config, layer)
    else:
        _convert_global_attention(hf_weights, lp, config, layer)

    # --- FFN weights ---
    _convert_ffn(hf_weights, lp, layer)

    # --- 4 layer norms ---
    layer["input_layernorm"] = _to_f32(
        _get(hf_weights, f"{lp}.input_layernorm.weight")
    )
    layer["post_attention_layernorm"] = _to_f32(
        _get(hf_weights, f"{lp}.post_attention_layernorm.weight")
    )
    layer["pre_feedforward_layernorm"] = _to_f32(
        _get(hf_weights, f"{lp}.pre_feedforward_layernorm.weight")
    )
    layer["post_feedforward_layernorm"] = _to_f32(
        _get(hf_weights, f"{lp}.post_feedforward_layernorm.weight")
    )

    return layer


def _convert_sliding_attention(
    hf_weights: dict, prefix: str, config: GemmaConfig, layer: dict
):
    """Convert sliding attention layer weights (has V proj)."""
    attn = f"{prefix}.self_attn"
    hd = config.head_dim

    # Q: [n_heads * head_dim, dim] -> [dim, n_heads, head_dim]
    q_proj = _to_f32(_get(hf_weights, f"{attn}.q_proj.weight"))
    layer["wq"] = q_proj.T.reshape(config.dim, config.n_heads, hd)

    # K: [n_kv_heads * head_dim, dim] -> [dim, n_kv_heads, head_dim]
    k_proj = _to_f32(_get(hf_weights, f"{attn}.k_proj.weight"))
    layer["wk"] = k_proj.T.reshape(config.dim, config.n_kv_heads, hd)

    # V: [n_kv_heads * head_dim, dim] -> [dim, n_kv_heads, head_dim]
    v_proj = _to_f32(_get(hf_weights, f"{attn}.v_proj.weight"))
    layer["wv"] = v_proj.T.reshape(config.dim, config.n_kv_heads, hd)

    # O: [dim, n_heads * head_dim] -> [n_heads * head_dim, dim]
    o_proj = _to_f32(_get(hf_weights, f"{attn}.o_proj.weight"))
    layer["wo"] = o_proj.T

    # QK norms
    layer["q_norm"] = _to_f32(_get(hf_weights, f"{attn}.q_norm.weight"))
    layer["k_norm"] = _to_f32(_get(hf_weights, f"{attn}.k_norm.weight"))


def _convert_global_attention(
    hf_weights: dict, prefix: str, config: GemmaConfig, layer: dict
):
    """Convert global attention layer weights (no V proj — K=V)."""
    attn = f"{prefix}.self_attn"
    ghd = config.global_head_dim

    # Q: [n_heads * global_head_dim, dim] -> [dim, n_heads, global_head_dim]
    q_proj = _to_f32(_get(hf_weights, f"{attn}.q_proj.weight"))
    layer["wq"] = q_proj.T.reshape(config.dim, config.n_heads, ghd)

    # K: [n_global_kv_heads * global_head_dim, dim] -> [dim, n_global_kv_heads, global_head_dim]
    k_proj = _to_f32(_get(hf_weights, f"{attn}.k_proj.weight"))
    layer["wk"] = k_proj.T.reshape(config.dim, config.n_global_kv_heads, ghd)

    # O: [dim, n_heads * global_head_dim] -> [n_heads * global_head_dim, dim]
    o_proj = _to_f32(_get(hf_weights, f"{attn}.o_proj.weight"))
    layer["wo"] = o_proj.T

    # QK norms
    layer["q_norm"] = _to_f32(_get(hf_weights, f"{attn}.q_norm.weight"))
    layer["k_norm"] = _to_f32(_get(hf_weights, f"{attn}.k_norm.weight"))


def _convert_ffn(hf_weights: dict, prefix: str, layer: dict):
    """Convert FFN (gate/up/down projections)."""
    mlp = f"{prefix}.mlp"

    # gate_proj: [intermediate, dim] -> [dim, intermediate]
    layer["w_gate"] = _to_f32(_get(hf_weights, f"{mlp}.gate_proj.weight")).T
    # up_proj: [intermediate, dim] -> [dim, intermediate]
    layer["w_up"] = _to_f32(_get(hf_weights, f"{mlp}.up_proj.weight")).T
    # down_proj: [dim, intermediate] -> [intermediate, dim]
    layer["w_down"] = _to_f32(_get(hf_weights, f"{mlp}.down_proj.weight")).T


# ---------------------------------------------------------------------------
# Orbax checkpoint save/load
# ---------------------------------------------------------------------------


def _make_checkpoint_manager(checkpoint_path, shared_storage: bool = False):
    """Create a CheckpointManager."""
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
    """Save ReLax params to an orbax checkpoint directory."""
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
    """Load ReLax params from an orbax checkpoint, optionally sharding onto a mesh."""
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
