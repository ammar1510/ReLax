"""
Functions to load model weights.
"""

import jax
import jax.numpy as jnp
from safetensors.torch import safe_open
from safetensors.flax import save_file, load_file
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
from pathlib import Path
import numpy as np
import torch

from .config import ModelConfig
from .model import LLaMa


def load_llama_weights(model_path: str, config: ModelConfig) -> LLaMa:
    """
    Loads LLaMa model weights from a .safetensors file into a LLaMa model instance.
    Args:
        model_path: Path to the .safetensors file.
        config: ModelConfig instance for the LLaMa model.
    Returns:
        Parameters for the LLaMa model.
    """
    tensors = load_file(model_path)
    params = unflatten_dict(tensors, sep=".")

    return params


def pth_to_safetensors(pth_model_path: str, config_dir: str, output_dir: str):
    """
    Reads model weights from a .pth file, converts them to JAX arrays
    compatible with the LLaMa model, and saves them to a .safetensors file.
    Args:
        pth_model_path: Path to the .pth file containing model weights.
        config_dir: Path to the directory containing config.json.
        output_dir: Directory where the .safetensors file will be saved.
    """
    # Load tensors from .pth file
    tensors = torch.load(pth_model_path, map_location="cpu")

    config = ModelConfig.from_json_file(config_dir)

    # This will be the final Flax parameter dictionary, structured to match model.py
    params = {}
    torch_dtypes = {}
    torch_dtypes["float32"] = torch.float32
    torch_dtypes["float64"] = torch.float64
    torch_dtypes["float16"] = torch.float16
    torch_dtypes["bfloat16"] = torch.bfloat16
    torch_dtypes["int8"] = torch.int8

    # Token embeddings
    params["tok_embeddings"] = {
        "embedding": jnp.asarray(
            tensors["tok_embeddings.weight"].to(torch_dtypes[config.dtype]),
            dtype=config.dtype,
        )
    }

    # Final normalization
    params["norm_weight"] = jnp.asarray(
        tensors["norm.weight"].to(torch_dtypes[config.dtype]), dtype=config.dtype
    )
    params["output"] = jnp.asarray(
        tensors["output.weight"].to(torch_dtypes[config.dtype]), dtype=config.dtype
    ).T

    # Transformer layers
    for i in range(config.n_layers):
        layer_prefix = f"layers.{i}."

        # Get all weights from the tensor dict
        q_proj = jnp.asarray(
            tensors[layer_prefix + "attention.wq.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        )
        k_proj = jnp.asarray(
            tensors[layer_prefix + "attention.wk.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        )
        v_proj = jnp.asarray(
            tensors[layer_prefix + "attention.wv.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        )
        o_proj = jnp.asarray(
            tensors[layer_prefix + "attention.wo.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        )

        # Reshape attention weights to match the model's expected format (dim, n_heads, head_dim)
        wq = q_proj.T.reshape(config.dim, config.n_heads, config.head_dim)
        wk = k_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wv = v_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wo = o_proj.T

        # Get feed-forward weights
        w_gate = jnp.asarray(
            tensors[layer_prefix + "feed_forward.w1.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        ).T
        w_down = jnp.asarray(
            tensors[layer_prefix + "feed_forward.w2.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        ).T
        w_up = jnp.asarray(
            tensors[layer_prefix + "feed_forward.w3.weight"].to(
                torch_dtypes[config.dtype]
            ),
            dtype=config.dtype,
        ).T

        # Assign weights to the layer's parameter dictionary
        # The key 'layers_i' is automatically created by Flax for lists of modules.
        params[f"layer_{i}"] = {
            "attention_norm_weight": jnp.asarray(
                tensors[layer_prefix + "attention_norm.weight"].to(
                    torch_dtypes[config.dtype]
                ),
                dtype=config.dtype,
            ),
            "ffn_norm_weight": jnp.asarray(
                tensors[layer_prefix + "ffn_norm.weight"].to(
                    torch_dtypes[config.dtype]
                ),
                dtype=config.dtype,
            ),
            "wq": wq,
            "wk": wk,
            "wv": wv,
            "wo": wo,
            "w_gate": w_gate,
            "w_up": w_up,
            "w_down": w_down,
        }

    # Save the parameters to a .safetensors file
    output_path = Path(output_dir) / "model.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flatten_dict(params, sep="."), output_path)
    print(f"Model weights saved to {output_path}")
