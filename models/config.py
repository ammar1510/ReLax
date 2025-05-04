"""
Model configuration parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ModelConfig:

    # Architecture Config
    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192  # FFN expansion size
    num_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8  # Grouped Query Attention
    hidden_act: str = "silu"

    # Positional Embeddings Config
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    # rope_type: str = "llama3" # Implicitly handled by RoPE implementation choice later
    # rope_scaling: Optional[Dict] = field(default_factory=lambda: {"type": "dynamic", "factor": 32.0})
    # Note: RoPE scaling details often handled within the RoPE implementation itself based on sequence length,
    # rather than a static config dict. Keeping it simple for now.

    # Normalization Config
    rms_norm_eps: float = 1e-05

    # Attention Config
    # head_dim is calculated: hidden_size // num_attention_heads = 3072 // 24 = 128
    attention_dropout: float = 0.0

    mode:str = "inference"

    def __post_init__(self):
        # Ensure GQA constraints are met
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads 