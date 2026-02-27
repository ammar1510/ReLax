"""
Qwen3.5 MoE model configuration.

Supports the hybrid attention architecture with both full attention (GQA)
and linear attention (Gated DeltaNet) layers, plus Mixture-of-Experts FFN.
"""

from dataclasses import dataclass
from typing import Tuple
import json
from pathlib import Path


@dataclass(frozen=True)
class QwenConfig:

    # Core architecture
    vocab_size: int
    dim: int  # hidden_size
    n_layers: int
    rms_norm_eps: float

    # Full attention (GQA)
    n_heads: int
    n_kv_heads: int
    head_dim: int  # explicit (256), NOT dim // n_heads

    # Linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int

    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    # Hybrid attention pattern
    layer_types: Tuple[str, ...]  # ("linear_attention", ..., "full_attention", ...)
    attn_output_gate: bool

    # RoPE (full attention layers only)
    rope_theta: float
    partial_rotary_factor: float
    max_seqlen: int

    # Activation
    activation_fn: str

    mode: str = "inference"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        n_full = sum(1 for t in self.layer_types if t == "full_attention")
        n_linear = sum(1 for t in self.layer_types if t == "linear_attention")
        if n_full + n_linear != self.n_layers:
            raise ValueError(
                f"layer_types length ({n_full + n_linear}) must equal "
                f"n_layers ({self.n_layers})"
            )

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that get rotary embeddings."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def n_full_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def n_linear_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        """Total dimension going through the causal conv1d."""
        return self.linear_key_dim * 2 + self.linear_value_dim

    @classmethod
    def from_json_file(cls, model_path: str):
        """
        Load config from a HuggingFace config.json file.

        Handles the nested text_config structure used by Qwen3.5 MoE models.
        """
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            raise ValueError(f"config.json not found in {model_path}")

        with open(config_path, "r") as f:
            raw = json.load(f)

        # Qwen3.5 MoE nests text params under text_config
        tc = raw.get("text_config", raw)

        # Build layer_types from the config
        if "layer_types" in tc:
            layer_types = tuple(tc["layer_types"])
        else:
            # Fallback: construct from full_attention_interval
            interval = tc.get("full_attention_interval", 4)
            n_layers = tc["num_hidden_layers"]
            types = []
            for i in range(n_layers):
                if (i + 1) % interval == 0:
                    types.append("full_attention")
                else:
                    types.append("linear_attention")
            layer_types = tuple(types)

        # RoPE config
        rope_params = tc.get("rope_parameters", {})
        rope_theta = rope_params.get("rope_theta", tc.get("rope_theta", 10000000.0))
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)
        )

        return cls(
            vocab_size=tc["vocab_size"],
            dim=tc["hidden_size"],
            n_layers=tc["num_hidden_layers"],
            n_heads=tc["num_attention_heads"],
            n_kv_heads=tc.get("num_key_value_heads", 2),
            head_dim=tc.get("head_dim", 256),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=tc.get("linear_key_head_dim", 128),
            linear_num_key_heads=tc.get("linear_num_key_heads", 16),
            linear_num_value_heads=tc.get("linear_num_value_heads", 64),
            linear_value_head_dim=tc.get("linear_value_head_dim", 128),
            num_experts=tc.get("num_experts", 256),
            num_experts_per_tok=tc.get("num_experts_per_tok", 8),
            moe_intermediate_size=tc.get("moe_intermediate_size", 1024),
            shared_expert_intermediate_size=tc.get(
                "shared_expert_intermediate_size", 1024
            ),
            layer_types=layer_types,
            attn_output_gate=tc.get("attn_output_gate", True),
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            max_seqlen=tc.get("max_position_embeddings", 262144),
            activation_fn=tc.get("hidden_act", "silu"),
            dtype=tc.get("dtype", raw.get("torch_dtype", "bfloat16")),
        )
