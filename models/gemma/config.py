"""
Gemma 4 model configuration parameters.
"""

from dataclasses import dataclass
from typing import Tuple
import json
from pathlib import Path


@dataclass(frozen=True)
class GemmaConfig:

    # Architecture
    vocab_size: int
    dim: int  # hidden_size
    ffn_hidden_dim: int  # intermediate_size
    n_layers: int
    n_heads: int  # num_attention_heads

    # Sliding attention
    n_kv_heads: int  # num_key_value_heads (for sliding layers)
    head_dim: int  # head_dim (for sliding layers)
    sliding_window: int
    sliding_rope_theta: float

    # Global attention
    n_global_kv_heads: int  # num_global_key_value_heads
    global_head_dim: int  # global_head_dim
    global_rope_theta: float
    global_partial_rotary_factor: float

    # Layer pattern
    layer_types: Tuple[str, ...]  # ("sliding_attention", ..., "full_attention", ...)

    # Activation & normalization
    activation_fn: str  # "gelu_pytorch_tanh"
    rms_norm_eps: float

    # Sequence length
    max_seqlen: int

    # Logit softcapping
    final_logit_softcapping: float

    # Embeddings
    tie_word_embeddings: bool

    # Dtype
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_heads % self.n_global_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_global_kv_heads ({self.n_global_kv_heads})"
            )

    @property
    def n_sliding_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "sliding_attention")

    @property
    def n_global_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def global_rotary_dim(self) -> int:
        return int(self.global_head_dim * self.global_partial_rotary_factor)

    @classmethod
    def from_json_file(cls, model_path: str):
        """
        Loads Gemma 4 config from a HuggingFace config.json file.

        Handles both standalone text configs and multimodal configs
        with a nested text_config.
        """
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise ValueError(f"config.json not found in {model_path}")

        with open(config_path, "r") as f:
            raw = json.load(f)

        # Handle multimodal wrapper: text params live under text_config
        if "text_config" in raw:
            cfg = raw["text_config"]
        else:
            cfg = raw

        rope_params = cfg.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        global_rope = rope_params.get("full_attention", {})

        return cls(
            vocab_size=cfg["vocab_size"],
            dim=cfg["hidden_size"],
            ffn_hidden_dim=cfg["intermediate_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            sliding_window=cfg["sliding_window"],
            sliding_rope_theta=sliding_rope.get("rope_theta", 10000.0),
            n_global_kv_heads=cfg["num_global_key_value_heads"],
            global_head_dim=cfg.get("global_head_dim", cfg["head_dim"]),
            global_rope_theta=global_rope.get("rope_theta", 1000000.0),
            global_partial_rotary_factor=global_rope.get(
                "partial_rotary_factor", 1.0
            ),
            layer_types=tuple(cfg["layer_types"]),
            activation_fn=cfg.get("hidden_activation", "gelu_pytorch_tanh"),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            max_seqlen=cfg.get("max_position_embeddings", 262144),
            final_logit_softcapping=cfg.get("final_logit_softcapping", 30.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            dtype=cfg.get("dtype", "bfloat16"),
        )
