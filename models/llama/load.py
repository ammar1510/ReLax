"""
Functions to load model weights from PyTorch checkpoint format.

This module handles loading LLaMA weights from sharded .pth files
in the PyTorch format, converting them to the ReLax model structure.
It also provides orbax-based save/load for fast subsequent loads with
optional mesh sharding.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .config import ModelConfig


def load_llama_weights(model_path: str, config: ModelConfig) -> Dict[str, Any]:
    """
    Loads LLaMA model weights from PyTorch sharded .pth format.

    Args:
        model_path: Path to directory containing *.pth files
        config: ModelConfig instance for the LLaMa model.

    Returns:
        Flax parameter dictionary matching the LLaMa model structure.
    """
    model_path = Path(model_path) / "original"

    # Find all .pth checkpoint files
    shard_files = sorted(model_path.glob("*.pth"))

    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"No .pth checkpoint files found in {model_path}\n" f"Expected *.pth files"
        )

    print(f"Loading model from {model_path}")
    print(f"Found {len(shard_files)} checkpoint file(s)")

    # Load all weights from all shards
    all_weights = {}
    for shard_path in shard_files:
        print(f"  Loading {shard_path.name}...")
        checkpoint = torch.load(shard_path, map_location="cpu", weights_only=True)
        # Convert PyTorch tensors to numpy arrays
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                # Convert BFloat16 to float32 (numpy doesn't support bfloat16 natively)
                if value.dtype == torch.bfloat16:
                    value = value.float()
                all_weights[key] = value.detach().cpu().numpy()
            else:
                all_weights[key] = value

    print(f"Loaded {len(all_weights)} tensors total")

    # Convert to ReLax format (keeping as numpy arrays on host)
    params = _convert_hf_to_relax(all_weights, config)

    return params


def _convert_hf_to_relax(
    hf_weights: Dict[str, np.ndarray],
    config: ModelConfig,
) -> Dict[str, Any]:
    """
    Convert PyTorch/HuggingFace weight format to ReLax model structure.
    Returns weights as numpy arrays to avoid early accelerator placement.
    """
    print("Converting PyTorch weights to ReLax format...")

    params = {}

    # Token embeddings
    embed_weight = hf_weights.get("tok_embeddings.weight")
    if embed_weight is None:
        raise ValueError("tok_embeddings.weight not found in checkpoint")

    params["tok_embeddings"] = {
        "embedding": embed_weight.astype(np.float32)
    }
    print(f"  ✓ Embeddings: {embed_weight.shape}")

    # Output layer
    output_weight = hf_weights.get("output.weight")
    params["output"] = output_weight.T.astype(np.float32)

    # Final norm
    norm = hf_weights.get("norm.weight")
    params["norm_weight"] = norm.astype(np.float32)

    # Transformer layers
    print(f"  Converting {config.n_layers} transformer layers...")
    for i in range(config.n_layers):
        layer_params = _convert_layer(hf_weights, i, config)
        params[f"layer_{i}"] = layer_params

        if i == 0:
            # Print shapes for first layer as reference
            print(f"    Layer 0 shapes:")
            for k, v in layer_params.items():
                print(f"      {k}: {v.shape}")

    print(f"  ✓ Converted all {config.n_layers} layers")

    return params


def _convert_layer(
    hf_weights: Dict[str, np.ndarray],
    layer_idx: int,
    config: ModelConfig,
) -> Dict[str, np.ndarray]:
    """Convert a single transformer layer to ReLax format (as numpy arrays)."""
    prefix = f"layers.{layer_idx}"

    # Attention weights
    q_proj = hf_weights[f"{prefix}.attention.wq.weight"]
    k_proj = hf_weights[f"{prefix}.attention.wk.weight"]
    v_proj = hf_weights[f"{prefix}.attention.wv.weight"]
    o_proj = hf_weights[f"{prefix}.attention.wo.weight"]

    # Transpose and reshape
    wq = q_proj.T.astype(np.float32).reshape(
        config.dim, config.n_heads, config.head_dim
    )
    wk = k_proj.T.astype(np.float32).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )
    wv = v_proj.T.astype(np.float32).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )
    wo = o_proj.T.astype(np.float32)

    # MLP/Feed-forward weights
    gate_proj = hf_weights[f"{prefix}.feed_forward.w1.weight"]
    up_proj = hf_weights[f"{prefix}.feed_forward.w3.weight"]
    down_proj = hf_weights[f"{prefix}.feed_forward.w2.weight"]

    w_gate = gate_proj.T.astype(np.float32)
    w_up = up_proj.T.astype(np.float32)
    w_down = down_proj.T.astype(np.float32)

    # Normalization weights
    attention_norm = hf_weights[f"{prefix}.attention_norm.weight"].astype(np.float32)
    ffn_norm = hf_weights[f"{prefix}.ffn_norm.weight"].astype(np.float32)

    return {
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "attention_norm_weight": attention_norm,
        "ffn_norm_weight": ffn_norm,
    }


def save_orbax_weights(
    params: Dict[str, Any], checkpoint_path: str, mesh=None
) -> None:
    """
    Save ReLax params to an orbax checkpoint directory.
    """
    import orbax.checkpoint as ocp
    import jax

    checkpoint_path = Path(checkpoint_path)

    if mesh is not None:
        from utils.mesh_helpers import MeshHelper
        # Pass model_config indirectly via dtype from first param or default
        params = MeshHelper.shard_params(params, mesh)
        jax.block_until_ready(params)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, params, force=True)
    checkpointer.wait_until_finished()
    if jax.process_index() == 0:
        print(f"Saved orbax checkpoint to {checkpoint_path}")


def load_from_orbax(
    checkpoint_path: str,
    mesh=None,
) -> Dict[str, Any]:
    """
    Load ReLax params from an orbax checkpoint, optionally sharding onto a mesh.
    """
    import orbax.checkpoint as ocp
    import jax
    from jax.sharding import NamedSharding

    checkpoint_path = Path(checkpoint_path)
    checkpointer = ocp.StandardCheckpointer()

    if mesh is None:
        params = checkpointer.restore(checkpoint_path)
        if jax.process_index() == 0:
            print(f"Loaded orbax checkpoint from {checkpoint_path}")
        return params

    from utils.mesh_helpers import MeshHelper

    # Read metadata to get shapes/dtypes without loading data
    step_metadata = checkpointer.metadata(checkpoint_path)
    item_metadata = step_metadata.item_metadata

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

    target = jax.tree.map_with_path(_build_target, item_metadata)

    params = checkpointer.restore(checkpoint_path, target=target)
    if jax.process_index() == 0:
        print(f"Loaded orbax checkpoint from {checkpoint_path} (sharded onto mesh {mesh.axis_names})")

    return params
