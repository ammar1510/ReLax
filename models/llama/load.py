"""
Functions to load model weights from HuggingFace format.

This module handles loading LLaMA weights from sharded safetensors files
in the HuggingFace format, converting them to the ReLax model structure.
"""

import json
import jax.numpy as jnp
from safetensors import safe_open
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .config import ModelConfig


def load_llama_weights(model_path: str, config: ModelConfig) -> Dict[str, Any]:
    """
    Loads LLaMA model weights from HuggingFace sharded safetensors format.

    Args:
        model_path: Path to directory containing:
            - model.safetensors.index.json (weight mapping)
            - model-0000X-of-0000Y.safetensors (sharded weight files)
        config: ModelConfig instance for the LLaMa model.

    Returns:
        Flax parameter dictionary matching the LLaMa model structure.

    HuggingFace weight naming (input):
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
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Model index file not found: {index_path}\n"
            f"Expected model.safetensors.index.json in {model_path}"
        )

    print(f"Loading sharded model from {model_path}")

    # Load the index file
    with open(index_path, 'r') as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Group weights by shard file
    shard_to_weights = {}
    for weight_name, shard_file in weight_map.items():
        if shard_file not in shard_to_weights:
            shard_to_weights[shard_file] = []
        shard_to_weights[shard_file].append(weight_name)

    print(f"Loading from {len(shard_to_weights)} shard files...")

    # Load all weights from all shards
    all_weights = {}
    for shard_file, weight_names in shard_to_weights.items():
        shard_path = model_path / shard_file
        print(f"  Loading {shard_file}... ({len(weight_names)} tensors)")

        with safe_open(shard_path, framework="numpy") as f:
            for weight_name in weight_names:
                all_weights[weight_name] = f.get_tensor(weight_name)

    print(f"Loaded {len(all_weights)} tensors total")

    # Convert to ReLax format
    return _convert_hf_to_relax(all_weights, config)


def _convert_hf_to_relax(
    hf_weights: Dict[str, np.ndarray],
    config: ModelConfig,
) -> Dict[str, Any]:
    """
    Convert HuggingFace weight format to ReLax model structure.

    Args:
        hf_weights: Dictionary of HuggingFace weights (numpy arrays)
        config: Model configuration

    Returns:
        Flax parameter dictionary for ReLax LLaMa model
    """
    print("Converting HuggingFace weights to ReLax format...")

    params = {}

    # Token embeddings
    # HF: model.embed_tokens.weight [vocab_size, dim]
    # ReLax: tok_embeddings.embedding [vocab_size, dim]
    embed_weight = hf_weights.get("model.embed_tokens.weight")
    if embed_weight is not None:
        params["tok_embeddings"] = {
            "embedding": jnp.asarray(embed_weight, dtype=config.dtype)
        }
        print(f"  ✓ Embeddings: {embed_weight.shape}")

    # Output layer (language model head)
    # HF: lm_head.weight [vocab_size, dim]
    # ReLax: output [dim, vocab_size] (transposed)
    lm_head = hf_weights.get("lm_head.weight")
    if lm_head is not None:
        params["output"] = jnp.asarray(lm_head.T, dtype=config.dtype)
        print(f"  ✓ LM head: {lm_head.shape} -> {params['output'].shape}")

    # Final norm
    # HF: model.norm.weight [dim]
    # ReLax: norm_weight [dim]
    norm = hf_weights.get("model.norm.weight")
    if norm is not None:
        params["norm_weight"] = jnp.asarray(norm, dtype=config.dtype)
        print(f"  ✓ Final norm: {norm.shape}")

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
) -> Dict[str, jnp.ndarray]:
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
    prefix = f"model.layers.{layer_idx}"

    # Attention weights
    # HF: [n_heads * head_dim, dim] or [n_kv_heads * head_dim, dim]
    # ReLax: need to transpose and reshape
    q_proj = hf_weights[f"{prefix}.self_attn.q_proj.weight"]  # [n_heads * head_dim, dim]
    k_proj = hf_weights[f"{prefix}.self_attn.k_proj.weight"]  # [n_kv_heads * head_dim, dim]
    v_proj = hf_weights[f"{prefix}.self_attn.v_proj.weight"]  # [n_kv_heads * head_dim, dim]
    o_proj = hf_weights[f"{prefix}.self_attn.o_proj.weight"]  # [dim, n_heads * head_dim]

    # Transpose and reshape for ReLax format
    # q_proj: [n_heads * head_dim, dim] -> [dim, n_heads * head_dim] -> [dim, n_heads, head_dim]
    wq = jnp.asarray(q_proj.T, dtype=config.dtype).reshape(
        config.dim, config.n_heads, config.head_dim
    )
    wk = jnp.asarray(k_proj.T, dtype=config.dtype).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )
    wv = jnp.asarray(v_proj.T, dtype=config.dtype).reshape(
        config.dim, config.n_kv_heads, config.head_dim
    )
    # o_proj: [dim, n_heads * head_dim] -> [n_heads * head_dim, dim]
    wo = jnp.asarray(o_proj.T, dtype=config.dtype)

    # MLP/Feed-forward weights
    # HF: [ffn_hidden_dim, dim]
    # ReLax: transpose to [dim, ffn_hidden_dim] or [ffn_hidden_dim, dim]
    gate_proj = hf_weights[f"{prefix}.mlp.gate_proj.weight"]  # [ffn_hidden_dim, dim]
    up_proj = hf_weights[f"{prefix}.mlp.up_proj.weight"]      # [ffn_hidden_dim, dim]
    down_proj = hf_weights[f"{prefix}.mlp.down_proj.weight"]  # [dim, ffn_hidden_dim]

    w_gate = jnp.asarray(gate_proj.T, dtype=config.dtype)  # [dim, ffn_hidden_dim]
    w_up = jnp.asarray(up_proj.T, dtype=config.dtype)      # [dim, ffn_hidden_dim]
    w_down = jnp.asarray(down_proj.T, dtype=config.dtype)  # [ffn_hidden_dim, dim]

    # Normalization weights
    attention_norm = hf_weights[f"{prefix}.input_layernorm.weight"]
    ffn_norm = hf_weights[f"{prefix}.post_attention_layernorm.weight"]

    return {
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "attention_norm_weight": jnp.asarray(attention_norm, dtype=config.dtype),
        "ffn_norm_weight": jnp.asarray(ffn_norm, dtype=config.dtype),
    }
