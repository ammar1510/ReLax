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

    PyTorch weight naming (input):
        - model.embed_tokens.weight
        - model.layers.{i}.self_attn.q_proj.weight
        - model.layers.{i}.self_attn.k_proj.weight
        - model.layers.{i}.self_attn.v_proj.weight
        - model.layers.{i}.self_attn.o_proj.weight
        - model.layers.{i}.mlp.gate_proj.weight
        - model.layers.{i}.mlp.up_proj.weight
        - model.layers.{i}.mlp.down_proj.weight
        - model.layers.{i}.input_layernorm.weight
        - model.layers.{i}.post_attention_layernorm.weight
        - model.norm.weight
        - lm_head.weight

    ReLax model structure (output):
        - tok_embeddings.embedding
        - layer_{i}.wq, wk, wv, wo
        - layer_{i}.w_gate, w_up, w_down
        - layer_{i}.attention_norm_weight
        - layer_{i}.ffn_norm_weight
        - norm_weight
        - output
    """
    model_path = Path(model_path) / "original"

    # Find all .pth checkpoint files
    shard_files = sorted(model_path.glob("*.pth"))
    print(model_path)

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
                # Convert BFloat16 to float32 (numpy doesn't support bfloat16)
                if value.dtype == torch.bfloat16:
                    value = value.float()
                all_weights[key] = value.detach().cpu().numpy()
            else:
                all_weights[key] = value

    print(f"Loaded {len(all_weights)} tensors total")

    # Convert to ReLax format
    params = _convert_hf_to_relax(all_weights, config)

    return params


def _convert_hf_to_relax(
    hf_weights: Dict[str, np.ndarray],
    config: ModelConfig,
) -> Dict[str, Any]:
    """
    Convert PyTorch/HuggingFace weight format to ReLax model structure.

    Args:
        hf_weights: Dictionary of PyTorch weights (numpy arrays)
        config: Model configuration

    Returns:
        Flax parameter dictionary for ReLax LLaMa model
    """
    print("Converting PyTorch weights to ReLax format...")

    params = {}

    # Token embeddings
    # HF: model.embed_tokens.weight [vocab_size, dim]
    # ReLax: tok_embeddings.embedding [vocab_size, dim]
    embed_weight = hf_weights.get("tok_embeddings.weight")
    if embed_weight is None:
        raise ValueError("model.embed_tokens.weight not found in checkpoint")

    import jax.numpy as jnp

    jax_dtype = getattr(jnp, config.dtype)

    params["tok_embeddings"] = {
        "embedding": jnp.array(embed_weight, dtype=jax_dtype)
    }
    print(f"  ✓ Embeddings: {embed_weight.shape}")

    # Output layer (language model head)
    # PyTorch: lm_head.weight [vocab_size, dim] or output.weight [dim, vocab_size] (if exists)
    # Otherwise: use tied embeddings (reuse embed_tokens.weight)
    # ReLax: output [dim, vocab_size]
    output_weight = hf_weights.get("output.weight")

    params["output"] = jnp.array(output_weight.T, dtype=jax_dtype)

    # Final norm
    # HF: model.norm.weight [dim]
    # ReLax: norm_weight [dim]
    norm = hf_weights.get("norm.weight")
    params["norm_weight"] = jnp.array(norm, dtype=jax_dtype)

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
    """
    Convert a single transformer layer from HF to ReLax format.

    HF format:
        - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        - model.layers.{i}.mlp.{gate,up,down}_proj.weight
        - model.layers.{i}.input_layernorm.weight
        - model.layers.{i}.post_attention_layernorm.weight

    ReLax format:
        - wq: [dim, n_heads, head_dim]
        - wk: [dim, n_kv_heads, head_dim]
        - wv: [dim, n_kv_heads, head_dim]
        - wo: [n_heads * head_dim, dim]
        - w_gate: [dim, ffn_hidden_dim]
        - w_up: [dim, ffn_hidden_dim]
        - w_down: [ffn_hidden_dim, dim]
        - attention_norm_weight: [dim]
        - ffn_norm_weight: [dim]
    """
    import jax.numpy as jnp

    prefix = f"layers.{layer_idx}"
    jax_dtype = getattr(jnp, config.dtype)

    # Attention weights
    # HF: [n_heads * head_dim, dim] or [n_kv_heads * head_dim, dim]
    # ReLax: need to transpose and reshape
    q_proj = hf_weights[f"{prefix}.attention.wq.weight"]  # [n_heads * head_dim, dim]
    k_proj = hf_weights[f"{prefix}.attention.wk.weight"]  # [n_kv_heads * head_dim, dim]
    v_proj = hf_weights[f"{prefix}.attention.wv.weight"]  # [n_kv_heads * head_dim, dim]
    o_proj = hf_weights[f"{prefix}.attention.wo.weight"]  # [dim, n_heads * head_dim]

    # Transpose and reshape for ReLax format
    # q_proj: [n_heads * head_dim, dim] -> [dim, n_heads * head_dim] -> [dim, n_heads, head_dim]
    wq = jnp.array(q_proj.T, dtype=jax_dtype).reshape(
        config.dim, config.n_heads, config.head_dim
    )

    wk = jnp.array(k_proj.T, dtype=jax_dtype).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )

    wv = jnp.array(v_proj.T, dtype=jax_dtype).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )

    # o_proj: [dim, n_heads * head_dim] -> [n_heads * head_dim, dim]
    wo = jnp.array(o_proj.T, dtype=jax_dtype)

    # MLP/Feed-forward weights
    # HF: [ffn_hidden_dim, dim]
    # ReLax: transpose to [dim, ffn_hidden_dim] or [ffn_hidden_dim, dim]
    gate_proj = hf_weights[f"{prefix}.feed_forward.w1.weight"]  # [ffn_hidden_dim, dim]
    up_proj = hf_weights[f"{prefix}.feed_forward.w3.weight"]  # [ffn_hidden_dim, dim]
    down_proj = hf_weights[f"{prefix}.feed_forward.w2.weight"]  # [dim, ffn_hidden_dim]

    w_gate = jnp.array(gate_proj.T, dtype=jax_dtype)  # [dim, ffn_hidden_dim]
    w_up = jnp.array(up_proj.T, dtype=jax_dtype)  # [dim, ffn_hidden_dim]
    w_down = jnp.array(down_proj.T, dtype=jax_dtype)  # [ffn_hidden_dim, dim]

    # Normalization weights
    attention_norm = jnp.array(
        hf_weights[f"{prefix}.attention_norm.weight"], dtype=jax_dtype
    )

    ffn_norm = jnp.array(
        hf_weights[f"{prefix}.ffn_norm.weight"], dtype=jax_dtype
    )

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

    This is used once after converting from .pth format. Subsequent loads
    can use load_from_orbax() which skips PyTorch entirely.

    When a mesh is provided, params are sharded across devices before saving
    so that each host writes only its local shards. This is required for
    multi-host setups.

    Args:
        params: Nested params dict as returned by load_llama_weights()
        checkpoint_path: Directory where the checkpoint will be written
        mesh: Optional JAX Mesh. When provided, params are sharded before
              saving for proper multi-host coordination.
    """
    import orbax.checkpoint as ocp
    import jax

    checkpoint_path = Path(checkpoint_path)

    if mesh is not None:
        from utils.mesh_helpers import MeshHelper

        params = MeshHelper.shard_params(params, mesh)
        jax.block_until_ready(params)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path, params, force=True)
    if jax.process_index() == 0:
        print(f"Saved orbax checkpoint to {checkpoint_path}")


def load_from_orbax(
    checkpoint_path: str,
    mesh=None,
) -> Dict[str, Any]:
    """
    Load ReLax params from an orbax checkpoint, optionally sharding onto a mesh.

    When a mesh is provided, each host reads only its own shards directly from
    disk (no full-load-then-reshard). This is done by passing a target pytree
    of jax.ShapeDtypeStruct with sharding specs to orbax's restore.

    Args:
        checkpoint_path: Directory written by save_orbax_weights()
        mesh: Optional JAX Mesh. When provided, params are restored directly
              onto the mesh using per-array sharding specs.

    Returns:
        Nested params dict ready for model.apply() or InferenceEngine.
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
