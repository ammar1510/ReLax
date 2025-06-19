"""
Model configuration parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:

    # Architecture Config
    vocab_size: int 
    dim: int 
    ffn_hidden_dim: int  # FFN expansion size
    n_layers: int 
    n_heads: int 
    n_kv_heads: int  # Grouped Query Attention
    activation_fn: str 

    # Positional Embeddings Config
    max_seqlen: int 
    rope_theta: float 
    # rope_scaling: Optional[Dict] = field(default_factory=lambda: {"type": "dynamic", "factor": 32.0})
    # Note: RoPE scaling details often handled within the RoPE implementation itself based on sequence length,
    # rather than a static config dict. Keeping it simple for now.

    # Normalization Config
    rms_norm_eps: float 

    # Attention Config
    # head_dim is calculated: dim // n_heads = 3072 // 24 = 128

    mode:str = "inference"


    def __post_init__(self):
        # Ensure GQA constraints are met
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.dim // self.n_heads

    @classmethod
    def from_json_file(cls, model_path: str):
        """
        Loads model configuration from a config.json file.
        
        Args:
            model_path: Path to the directory containing config.json.
        
        Returns:
            An instance of ModelConfig.
        """
        config_path = Path(model_path) / 'config.json'
        if not config_path.is_file():
            raise ValueError(f"config.json not found in {model_path}")

        with open(config_path, 'r') as f:
            hf_config = json.load(f)

        # Mapping from Hugging Face config keys to our ModelConfig keys
        return cls(
            dim=hf_config['hidden_size'],
            n_layers=hf_config['num_hidden_layers'],
            n_heads=hf_config['num_attention_heads'],
            n_kv_heads=hf_config['num_key_value_heads'],
            ffn_hidden_dim=hf_config['intermediate_size'],
            vocab_size=hf_config['vocab_size'],
            rms_norm_eps=hf_config['rms_norm_eps'],
            rope_theta=hf_config.get('rope_theta', 10000.0),
            max_seqlen=hf_config['rope_scaling']['original_max_position_embeddings'],
            activation_fn=hf_config.get('hidden_act', 'silu')
        )