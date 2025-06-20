"""
Functions to load model weights.
"""
import jax
import jax.numpy as jnp
from safetensors import safe_open
import flax
from pathlib import Path
import numpy as np

from .config import ModelConfig
from .model import LLaMA

def load_llama_weights(model_path: str) -> flax.core.FrozenDict:
    """
    Loads Llama3 model weights from .safetensors files into a Flax parameter structure
    that is compatible with the LLaMA model in model.py.

    Args:
        model_path: Path to the directory containing .safetensors files and config.json.

    Returns:
        A Flax FrozenDict containing the model parameters.
    """
    # Load all tensors from all safetensor files in the directory
    tensors = {}
    paths = list(Path(model_path).glob('*.safetensors'))
    if not paths:
        raise ValueError(f"No .safetensors files found in {model_path}")

    for filepath in paths:
        with safe_open(filepath, framework="flax") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    
    config = ModelConfig.from_json_file(model_path)

    # This will be the final Flax parameter dictionary, structured to match model.py
    params = {}

    # Token embeddings
    params['tok_embeddings'] = {'embedding': tensors['model.embed_tokens.weight']}

    # Final normalization
    params['norm_weight'] = tensors['model.norm.weight']

    # Language model head (output layer)
    # The nn.Dense layer in LLaMA expects a kernel of shape (features_in, features_out)
    # which is (dim, vocab_size). The embedding weight is (vocab_size, dim), so we transpose it.
    params['output'] = {'kernel': tensors['model.embed_tokens.weight'].T}

    # Transformer layers
    for i in range(config.n_layers):
        layer_prefix = f'model.layers.{i}.'
        
        # Get all weights from the tensor dict
        q_proj = tensors[layer_prefix + 'self_attn.q_proj.weight']
        k_proj = tensors[layer_prefix + 'self_attn.k_proj.weight']
        v_proj = tensors[layer_prefix + 'self_attn.v_proj.weight']
        o_proj = tensors[layer_prefix + 'self_attn.o_proj.weight']
        
        # Reshape attention weights to match the model's expected format (dim, n_heads, head_dim)
        wq = q_proj.T.reshape(config.dim, config.n_heads, config.head_dim)
        wk = k_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wv = v_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wo = o_proj.T

        # Get feed-forward weights
        gate_proj = tensors[layer_prefix + 'mlp.gate_proj.weight'].T
        up_proj = tensors[layer_prefix + 'mlp.up_proj.weight'].T
        down_proj = tensors[layer_prefix + 'mlp.down_proj.weight'].T

        # Assign weights to the layer's parameter dictionary
        # The key 'layers_i' is automatically created by Flax for lists of modules.
        params[f'layer_{i}'] = {
            'attention_norm_weight': tensors[layer_prefix + 'input_layernorm.weight'],
            'ffn_norm_weight': tensors[layer_prefix + 'post_attention_layernorm.weight'],
            'wq': wq,
            'wk': wk,
            'wv': wv,
            'wo': wo,
            'w1_gate': gate_proj,
            'w2_up': up_proj,
            'w3_down': down_proj,
        }

    return flax.core.freeze(params) 