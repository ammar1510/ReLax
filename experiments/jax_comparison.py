import torch
import jax
import jax.numpy as jnp
import numpy as np
import json
from dataclasses import dataclass
from typing import Optional
import flax.linen as nn
from safetensors.torch import load_file

from models.llama.model import LLaMa
from utils.kvcache import KVCache
import flax

jax.config.update("jax_default_matmul_precision", "float32")

@dataclass
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_hidden_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_seqlen: int
    head_dim: int
    dtype: jnp.dtype = jnp.bfloat16
    activation_fn = "silu"

def convert_weights(torch_weights, jax_config: ModelConfig):
    """Converts PyTorch weights to JAX/Flax format."""
    jax_params = {'params': {}}
    
    # Embedding
    jax_params['params']['tok_embeddings'] = {'embedding': jnp.asarray(torch_weights['tok_embeddings.weight'],dtype=jax_config.dtype)}
    
    # Transformer blocks
    for i in range(jax_config.n_layers):
        layer_prefix = f'layers.{i}.'
        jax_layer = {}
        
        # Attention weights
        wq = jnp.asarray(torch_weights[layer_prefix + 'attention.wq.weight'],dtype=jax_config.dtype).T.reshape(jax_config.dim, jax_config.n_heads, jax_config.head_dim)
        wk = jnp.asarray(torch_weights[layer_prefix + 'attention.wk.weight'],dtype=jax_config.dtype).T.reshape(jax_config.dim, jax_config.n_kv_heads, jax_config.head_dim)
        wv = jnp.asarray(torch_weights[layer_prefix + 'attention.wv.weight'],dtype=jax_config.dtype).T.reshape(jax_config.dim, jax_config.n_kv_heads, jax_config.head_dim)
        wo = jnp.asarray(torch_weights[layer_prefix + 'attention.wo.weight'],dtype=jax_config.dtype).T
        
        # Attention weights - flatten structure to match model expectations
        jax_layer['wq'] = wq
        jax_layer['wk'] = wk
        jax_layer['wv'] = wv
        jax_layer['wo'] = wo
        
        # Feed-forward weights
        w1 = jnp.asarray(torch_weights[layer_prefix + 'feed_forward.w1.weight'],dtype=jax_config.dtype).T
        w2 = jnp.asarray(torch_weights[layer_prefix + 'feed_forward.w2.weight'],dtype=jax_config.dtype).T
        w3 = jnp.asarray(torch_weights[layer_prefix + 'feed_forward.w3.weight'],dtype=jax_config.dtype).T

        # Feed-forward weights - flatten structure to match model expectations
        jax_layer['w1_gate'] = w1  # PyTorch w1 (gate) -> JAX w1_gate
        jax_layer['w2_up'] = w3    # PyTorch w3 (up) -> JAX w2_up  
        jax_layer['w3_down'] = w2  # PyTorch w2 (down) -> JAX w3_down
        
        # Normalization weights
        jax_layer['attention_norm_weight'] = jnp.asarray(torch_weights[layer_prefix + 'attention_norm.weight'],dtype=jax_config.dtype)
        jax_layer['ffn_norm_weight'] = jnp.asarray(torch_weights[layer_prefix + 'ffn_norm.weight'],dtype=jax_config.dtype)
        
        jax_params['params'][f'layer_{i}'] = flax.core.freeze(jax_layer)

    # Final normalization
    jax_params['params']['norm_weight'] = jnp.asarray(torch_weights['norm.weight'],dtype=jax_config.dtype)
    
    return flax.core.freeze(jax_params)

def load_and_run_jax_model():
    """
    This function loads the PyTorch weights, converts them for the JAX model,
    runs a forward pass, and saves the output logits for comparison.
    """
    config_path = "experiments/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = ModelConfig(
        dim=config['hidden_size'],
        n_layers=config['num_hidden_layers'],
        n_heads=config['num_attention_heads'],
        n_kv_heads=config['num_key_value_heads'],
        vocab_size=config['vocab_size'],
        ffn_hidden_dim=config['intermediate_size'],
        rms_norm_eps=config['rms_norm_eps'],
        rope_theta=config['rope_theta'],
        max_seqlen=2048,
        head_dim=config['head_dim'],
        dtype=jnp.float32
    )

    if not jax.devices('gpu'):
        raise RuntimeError("This script requires a GPU but JAX cannot find one.")
    
    torch_weights = load_file('mock_weights.safetensors', device='cuda')
    jax_params = convert_weights(torch_weights, model_config)
    
    model = LLaMa(model_config)
    
    batch_size = 1
    seq_len = 64
    torch.manual_seed(1337) # To match torch.randint behavior with the same seed
    torch_tokens = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    tokens = jnp.asarray(torch_tokens.numpy(), dtype=jnp.int32)
    
    # Initialize KVCache
    kv_cache = KVCache.new(model_config.n_layers, batch_size, seq_len, model_config.n_kv_heads, model_config.head_dim, dtype=model_config.dtype)
    
    # Run forward pass
    logits, _ = model.apply(jax_params, tokens, start_pos=0, kv_cache=kv_cache)
    
    np.save('jax_logits.npy', np.array(logits))

    print("JAX model executed and outputs saved to 'jax_logits.npy'.")
    print(f"Logits shape: {logits.shape}")

if __name__ == '__main__':
    load_and_run_jax_model() 