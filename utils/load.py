import jax
import jax.numpy as jnp
from safetensors import safe_open
from pathlib import Path
import json
from typing import Dict, Any, Optional

# The Flax parameter structure will be a nested dictionary mirroring the model
# e.g., {'tok_embeddings': {'embedding': ...}, 'layers': {'layers_0': {'attention': {'wq': {'kernel': ...}, /*wk,wv,wo...*/}, 'feed_forward': {'w1': {'kernel': ...}, /*w2,w3...*/}, 'attention_norm': {'scale': ...}, 'ffn_norm': {'scale': ...}}, /*, layers_1, etc.*/}, 'norm': {'scale': ...}, 'output': {'kernel': ...}}
FlaxModelParams = Dict[str, Any]

def load_llama_weights_safetensors(checkpoint_dir: str, device: Optional[jax.Device] = None) -> FlaxModelParams:
    """
    Loads Llama model weights from .safetensors files and converts them
    into the expected Flax parameter structure, loading onto the specified device.

    Args:
        checkpoint_dir: Path to the directory containing the .safetensors
                        files and the params.json configuration file.
        device: The JAX device (e.g., jax.devices('cpu')[0], jax.devices('gpu')[0])
                to load the tensors onto. Defaults to CPU if None.

    Returns:
        A nested dictionary representing the Flax model parameters on the target device.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # 1. Load model configuration (hyperparameters) from params.json
    params_path = checkpoint_path / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"params.json not found in {checkpoint_dir}")

    with open(params_path, 'r') as f:
        model_args = json.load(f)
        print(f"Loaded model arguments: {model_args}")
        # Extract necessary args for mapping (e.g., number of layers)
        n_layers = model_args.get("n_layers") # Key from our config/Llama 3 reference
        if n_layers is None:
             raise ValueError("Could not find number of layers ('n_layers') in params.json")


    # 2. Find all .safetensors files in the directory
    safetensor_files = list(checkpoint_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

    print(f"Found safetensors files: {safetensor_files}")

    # Determine target device string for safe_open
    device_str = "cpu" # Default
    if device:
        try:
            # JAX devices often have platform like 'cpu', 'gpu', 'tpu'
            device_str = device.platform
            print(f"Targeting device: {device} (Platform: {device_str})")
        except Exception as e:
            print(f"Warning: Could not determine platform from device object ({device}). Defaulting to CPU loading. Error: {e}")
            device_str = "cpu"
    else:
        print("No device specified, loading onto CPU.")


    loaded_params: Dict[str, jax.Array] = {}

    # 3. Iterate through files and load tensors onto the target device
    print(f"Loading tensors onto device: {device_str}...")
    for sf_file in safetensor_files:
        # Pass the device string ('cpu', 'gpu', 'tpu', etc.) to safe_open
        # Note: 'safetensors' handles framework integration.
        with safe_open(sf_file, framework="jax", device=device_str) as f:
            for key in f.keys():
                loaded_params[key] = f.get_tensor(key)
    print(f"Loaded {len(loaded_params)} tensors.")

    # 4. Map loaded tensor names to the Flax model structure defined in llama.py
    flax_params = {
        'tok_embeddings': {
            'embedding': loaded_params['model.embed_tokens.weight']
        },
        'norm': { # Final RMSNorm
            # Flax RMSNorm uses 'scale' for the weight parameter
            'scale': loaded_params['model.norm.weight']
        },
        # Initialize layers dict (Flax converts list attributes to dicts like layers_0)
        'layers': {},
        'output': { # LM Head (Flax Dense uses 'kernel')
            # Reuse the embedding matrix, transpose for [hidden_size, vocab_size]
            'kernel': loaded_params['model.embed_tokens.weight'].T
        }
    }

    # Populate the transformer layers
    for i in range(n_layers):
        # Flax layer key (matches attribute naming convention from lists)
        flax_layer_key = f'layers_{i}'
        # Corresponding HuggingFace layer prefix
        hf_layer_prefix = f'model.layers.{i}'

        # Check for missing keys defensively (using original HF names)
        required_hf_keys = [
            f'{hf_layer_prefix}.self_attn.q_proj.weight',
            f'{hf_layer_prefix}.self_attn.k_proj.weight',
            f'{hf_layer_prefix}.self_attn.v_proj.weight',
            f'{hf_layer_prefix}.self_attn.o_proj.weight',
            f'{hf_layer_prefix}.input_layernorm.weight', # Attention norm
            f'{hf_layer_prefix}.mlp.gate_proj.weight', # FeedForward w1
            f'{hf_layer_prefix}.mlp.up_proj.weight',   # FeedForward w3
            f'{hf_layer_prefix}.mlp.down_proj.weight', # FeedForward w2
            f'{hf_layer_prefix}.post_attention_layernorm.weight' # FeedForward norm
        ]
        for key in required_hf_keys:
            if key not in loaded_params:
                 raise KeyError(f"Missing required tensor key in loaded parameters: {key}")

        # Map parameters for the current layer
        flax_params['layers'][flax_layer_key] = {
            # Map to 'attention' field defined in LLaMA.setup()
            'attention': {
                # Flax Dense layers use 'kernel', transpose HF weights
                'wq': {'kernel': loaded_params[f'{hf_layer_prefix}.self_attn.q_proj.weight'].T},
                'wk': {'kernel': loaded_params[f'{hf_layer_prefix}.self_attn.k_proj.weight'].T},
                'wv': {'kernel': loaded_params[f'{hf_layer_prefix}.self_attn.v_proj.weight'].T},
                'wo': {'kernel': loaded_params[f'{hf_layer_prefix}.self_attn.o_proj.weight'].T},
            },
            # Map to 'attention_norm' field defined in LLaMA.setup()
            'attention_norm': {
                # Flax RMSNorm uses 'scale'
                'scale': loaded_params[f'{hf_layer_prefix}.input_layernorm.weight']
            },
            # Map to 'feed_forward' field defined in LLaMA.setup()
            'feed_forward': {
                # Flax Dense layers use 'kernel', transpose HF weights
                # Map HF names to standard Llama param names (w1, w3, w2)
                'w1': {'kernel': loaded_params[f'{hf_layer_prefix}.mlp.gate_proj.weight'].T}, # gate_proj -> w1
                'w3': {'kernel': loaded_params[f'{hf_layer_prefix}.mlp.up_proj.weight'].T},   # up_proj   -> w3
                'w2': {'kernel': loaded_params[f'{hf_layer_prefix}.mlp.down_proj.weight'].T}, # down_proj -> w2
            },
            # Map to 'ffn_norm' field defined in LLaMA.setup()
            'ffn_norm': {
                # Flax RMSNorm uses 'scale'
                'scale': loaded_params[f'{hf_layer_prefix}.post_attention_layernorm.weight']
            }
        }

    print("Parameter mapping complete.")
    # 5. Return the structured Flax parameters (which are already on the target device)
    return flax_params

# Example usage (commented out)
# if __name__ == '__main__':
#     # Replace with the actual path to your Llama 3 checkpoint directory
#     # (containing params.json and *.safetensors files)
#     # Example for a common HF Llama 3 8B format:
#     ckpt_dir = "path/to/Meta-Llama-3-8B"
#     try:
#         # Example: Load onto the first available GPU
#         target_device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
#         print(f"Attempting to load onto: {target_device}")
#         params = load_llama_weights_safetensors(ckpt_dir, device=target_device)
#         print(f"Successfully loaded and mapped parameters.")
# 
#         # Basic validation:
#         # Verify the device of a sample tensor (using the new structure)
#         sample_tensor = params['layers']['layers_0']['attention']['wq']['kernel']
#         print(f"Sample tensor device: {sample_tensor.device()}")
#         print(f"Top level keys: {list(params.keys())}") # Should include 'layers', 'tok_embeddings', 'norm', 'output'
#         print(f"Layer keys: {list(params['layers'].keys())}") # Should be like ['layers_0', 'layers_1', ...]
#         print(f"Number of layers loaded: {len(params['layers'])}")
# 
#     except FileNotFoundError as e:
#         print(e)
#     except KeyError as e:
#         print(f"Mapping Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}") 