"""
FSDP Configuration for distributed training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FSDPConfig:
    """Configuration for FSDP (Fully Sharded Data Parallel) training.

    Attributes:
        fsdp_axis_size: Number of devices for FSDP axis. -1 means all devices.
        tensor_parallel_size: Number of devices for tensor parallelism (future feature).

        compute_dtype: Data type for computations (e.g., 'bfloat16', 'float32').
        param_dtype: Data type for parameter storage (e.g., 'bfloat16', 'float32').

        checkpoint_dir: Directory for saving checkpoints.
        save_interval_steps: Save checkpoint every N steps.
        max_checkpoints_to_keep: Maximum number of checkpoints to retain.
        async_checkpointing: Whether to save checkpoints asynchronously (future feature).

        use_activation_checkpointing: Enable gradient checkpointing to save memory (future feature).
        activation_checkpoint_layers: Which layers to checkpoint. None means auto-select (future feature).

        coordinator_address: Address for multi-node coordinator (e.g., "10.0.0.1:1234").
        num_processes: Total number of processes across all nodes.
        process_id: ID of current process (0 to num_processes-1).
    """

    # Mesh configuration
    fsdp_axis_size: int = -1  # -1 = all devices
    tensor_parallel_size: int = 1

    # Mixed precision
    compute_dtype: str = 'bfloat16'
    param_dtype: str = 'bfloat16'

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_interval_steps: int = 1000
    max_checkpoints_to_keep: int = 3
    async_checkpointing: bool = False  # Future feature

    # Activation checkpointing (future feature)
    use_activation_checkpointing: bool = False
    activation_checkpoint_layers: Optional[List[int]] = None

    # Distributed training
    coordinator_address: Optional[str] = None
    num_processes: int = 1
    process_id: int = 0

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if self.fsdp_axis_size == 0:
            raise ValueError("fsdp_axis_size cannot be 0")

        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.compute_dtype not in ['float32', 'float16', 'bfloat16']:
            raise ValueError(f"Invalid compute_dtype: {self.compute_dtype}")

        if self.param_dtype not in ['float32', 'float16', 'bfloat16']:
            raise ValueError(f"Invalid param_dtype: {self.param_dtype}")

        if self.num_processes < 1:
            raise ValueError("num_processes must be >= 1")

        if self.process_id < 0 or self.process_id >= self.num_processes:
            raise ValueError(
                f"process_id ({self.process_id}) must be in range [0, {self.num_processes})"
            )

        if self.num_processes > 1 and self.coordinator_address is None:
            raise ValueError(
                "coordinator_address must be set for multi-node training (num_processes > 1)"
            )

    @property
    def is_distributed(self) -> bool:
        """Whether this is distributed multi-node training."""
        return self.num_processes > 1

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (process_id == 0)."""
        return self.process_id == 0
