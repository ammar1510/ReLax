"""
Model configuration parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ModelConfig:

    # Architecture Config
    vocab_size: int = 128256
    dim: int = 3072
    ffn_hidden_dim: int = 8192  # FFN expansion size
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8  # Grouped Query Attention
    activation_fn: str = "silu"

    # Positional Embeddings Config
    max_seq_len: int = 131072
    rope_theta: float = 500000.0
    # rope_scaling: Optional[Dict] = field(default_factory=lambda: {"type": "dynamic", "factor": 32.0})
    # Note: RoPE scaling details often handled within the RoPE implementation itself based on sequence length,
    # rather than a static config dict. Keeping it simple for now.

    # Normalization Config
    rms_norm_eps: float = 1e-05

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