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

def load_llama_weights(model_path: str, config: ModelConfig) -> flax.core.FrozenDict:
    """
    Loads Llama3 model weights from .safetensors files into a Flax parameter structure.

    Args:
        model_path: Path to the directory containing .safetensors files.
        config: The model configuration.

    Returns:
        A Flax FrozenDict containing the model parameters.
    """
    # Load all tensors from all safetensor files in the directory
    tensors = {}
    paths = list(Path(model_path).glob('*.safetensors'))
    if not paths:
        raise ValueError(f"No .safetensors files found in {model_path}")

    for filepath in paths:
        with safe_open(filepath, framework="jax") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # This will be the final Flax parameter dictionary
    params = {}

    # Token embeddings
    params['tok_embeddings'] = {'embedding': tensors['model.embed_tokens.weight']}

    # Final normalization
    params['norm_weight'] = tensors['model.norm.weight']

    # Language model head
    params['output'] = {'kernel': tensors['lm_head.weight'].T}

    # Transformer layers
    for i in range(config.n_layers):
        layer_prefix = f'model.layers.{i}.'
        
        # Normalization weights
        attention_norm_weight = tensors[layer_prefix + 'input_layernorm.weight']
        ffn_norm_weight = tensors[layer_prefix + 'post_attention_layernorm.weight']

        # Attention projection weights
        q_proj = tensors[layer_prefix + 'self_attn.q_proj.weight']
        k_proj = tensors[layer_prefix + 'self_attn.k_proj.weight']
        v_proj = tensors[layer_prefix + 'self_attn.v_proj.weight']
        o_proj = tensors[layer_prefix + 'self_attn.o_proj.weight']
        
        # Reshape and transpose attention weights to match our model's expected format
        wq = q_proj.T.reshape(config.dim, config.n_heads, config.head_dim)
        wk = k_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wv = v_proj.T.reshape(config.dim, config.n_kv_heads, config.head_dim)
        wo = o_proj.T

        # Feed-forward network weights
        gate_proj = tensors[layer_prefix + 'mlp.gate_proj.weight']
        up_proj = tensors[layer_prefix + 'mlp.up_proj.weight']
        down_proj = tensors[layer_prefix + 'mlp.down_proj.weight']

        # Transpose FFN weights
        w1_gate = gate_proj.T
        w2_up = up_proj.T
        w3_down = down_proj.T

        # Assign weights to the layer's parameter dictionary
        params[f'layers_{i}'] = {
            'attention_norm_weight': attention_norm_weight,
            'ffn_norm_weight': ffn_norm_weight,
            'attention': {
                'wq': wq,
                'wk': wk,
                'wv': wv,
                'wo': wo,
            },
            'feed_forward': {
                'w1_gate': w1_gate,
                'w2_up': w2_up,
                'w3_down': w3_down,
            }
        }

    return flax.core.freeze(params) 