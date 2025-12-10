import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as PS, Mesh
from jax.experimental import mesh_utils
import jax.tree_util as tree_util
from typing import Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)

# Legacy default mesh for backward compatibility
default_mesh = jax.make_mesh((len(jax.devices()), 1), ("fsdp", "tp"))


def mesh_sharding(param_type: str, mesh: Mesh = default_mesh) -> NamedSharding:
    pspec = get_partition_spec(param_type)
    return NamedSharding(mesh, pspec)


# ============================================================================
# Mesh Creation
# ============================================================================


def create_fsdp_mesh(
    fsdp_axis_size: int = -1,
    tensor_parallel_size: int = 1,
) -> Mesh:
    """
    Create device mesh for FSDP training.

    Axes:
    - 'fsdp': Fully Sharded Data Parallel axis (shards params, grads, opt_state)
    - 'tp': Tensor Parallel axis (for future model parallelism, currently unused)

    Args:
        fsdp_axis_size: Number of devices for FSDP axis. -1 means all devices.
        tensor_parallel_size: Number of devices for tensor parallelism. Default 1 (no TP).

    Returns:
        Mesh with shape (fsdp_axis_size, tensor_parallel_size) and axes ('fsdp', 'tp')

    Examples:
        >>> # 8 GPUs, pure FSDP
        >>> mesh = create_fsdp_mesh()  # shape (8, 1)
        >>>
        >>> # 16 GPUs across 2 nodes, pure FSDP
        >>> mesh = create_fsdp_mesh(fsdp_axis_size=16)
        >>>
        >>> # 16 GPUs with TP=2, FSDP=8
        >>> mesh = create_fsdp_mesh(tensor_parallel_size=2)  # shape (8, 2)
    """
    devices = jax.devices()
    num_devices = len(devices)

    if fsdp_axis_size == -1:
        fsdp_axis_size = num_devices // tensor_parallel_size

    if fsdp_axis_size * tensor_parallel_size != num_devices:
        raise ValueError(
            f"Invalid mesh configuration: fsdp_axis_size ({fsdp_axis_size}) * "
            f"tensor_parallel_size ({tensor_parallel_size}) = "
            f"{fsdp_axis_size * tensor_parallel_size}, but num_devices = {num_devices}"
        )

    mesh = jax.make_mesh((fsdp_axis_size, tensor_parallel_size), ("fsdp", "tp"))

    logger.info(f"Created FSDP mesh: shape={mesh.shape}, axes={mesh.axis_names}")
    logger.info(
        f"Devices: {num_devices} ({fsdp_axis_size} FSDP × {tensor_parallel_size} TP)"
    )

    return mesh


def create_data_parallel_mesh() -> Mesh:
    """
    Create simple data parallelism mesh (no FSDP).

    Returns:
        Mesh with all devices on 'data' axis.
    """
    devices = jax.devices()
    return jax.make_mesh((len(devices),), ("data",))


# ============================================================================
# FSDP Sharding Rules
# ============================================================================

# Sharding specifications for different parameter types
FSDP_SHARDING_RULES = {
    # Embeddings: shard along vocab dimension on FSDP axis
    "tok_embeddings": PS("fsdp", None),  # [vocab_size, dim]
    "embedding": PS("fsdp", None),  # Alternative name for embeddings
    # Attention weights
    "wq": PS(None, "fsdp", None),  # [dim, n_heads, head_dim] - shard heads
    "wk": PS(None, "fsdp", None),  # [dim, n_kv_heads, head_dim] - shard kv_heads
    "wv": PS(None, "fsdp", None),  # [dim, n_kv_heads, head_dim]
    "wo": PS("fsdp", None),  # [n_heads*head_dim, dim] - shard input dim
    # FFN weights - shard hidden dimension
    "w_gate": PS(None, "fsdp"),  # [dim, ffn_hidden_dim]
    "w_up": PS(None, "fsdp"),  # [dim, ffn_hidden_dim]
    "w_down": PS("fsdp", None),  # [ffn_hidden_dim, dim]
    # Norms: replicate (small, frequently accessed)
    "attention_norm_weight": PS(),  # [dim] - replicated
    "ffn_norm_weight": PS(),  # [dim] - replicated
    "norm_weight": PS(),  # [dim] - replicated
    # Output layer
    "output": PS(None, "fsdp"),  # [dim, vocab_size] - shard vocab
}


def get_partition_spec(param_type: str) -> PS:
    """
    Return a PartitionSpec for a parameter name.

    Args:
        param_type: string name, e.g. 'wq', 'tok_embeddings', 'norm_weight'

    Returns:
        A PartitionSpec for sharding the parameter.

    Raises:
        ValueError: if no sharding rule is defined for the parameter.
    """
    # Empty path → nothing to shard
    if len(param_type) == 0:
        return PS()

    # Explicit rule found
    if param_type in FSDP_SHARDING_RULES:
        return FSDP_SHARDING_RULES[param_type]

    # Norms → always replicate
    if "norm" in param_type.lower():
        return PS()

    # No rule found → this is an error
    raise ValueError(
        f"Sharding rule not found for parameter '{param_type}'. "
        f"Add a rule to FSDP_SHARDING_RULES or handle this name explicitly."
    )


def get_data_sharding_spec(shape: tuple, batch_axis: int = 0) -> PS:
    """
    Get PartitionSpec for data (shard batch dimension on FSDP axis).

    Args:
        shape: Shape of the array
        batch_axis: Which axis is the batch dimension (default 0)

    Returns:
        PartitionSpec with batch_axis sharded on 'fsdp'

    Examples:
        >>> get_data_sharding_spec((32, 512))  # [batch, seq_len]
        PartitionSpec('fsdp', None)
        >>> get_data_sharding_spec((32, 512, 768), batch_axis=0)
        PartitionSpec('fsdp', None, None)
    """
    spec = [None] * len(shape)
    spec[batch_axis] = "fsdp"
    return PS(*spec)


# ============================================================================
# Sharding Utilities
# ============================================================================


def shard_params(params: dict, mesh: Mesh) -> dict:
    """
    Apply FSDP sharding to parameters.

    Args:
        params: Parameter pytree (nested dict from Flax)
        mesh: Device mesh for sharding

    Returns:
        Sharded parameter pytree

    Example:
        >>> mesh = create_fsdp_mesh()
        >>> sharded_params = shard_params(params, mesh)
    """
    # Create partition specs for all parameters
    param_specs = tree_util.tree_map_with_path(
        lambda path, _: get_partition_spec(path), params
    )

    # Create NamedSharding for each param
    param_shardings = tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec), param_specs
    )

    # Transfer params to devices with sharding
    sharded_params = jax.device_put(params, param_shardings)

    return sharded_params


def with_fsdp_sharding(
    x: jax.Array,
    mesh: Mesh,
    spec: PS,
) -> jax.Array:
    """
    Apply sharding constraint to array.

    Args:
        x: Array to shard
        mesh: Device mesh
        spec: PartitionSpec for sharding

    Returns:
        Array with sharding constraint applied

    Example:
        >>> # Shard batch dimension
        >>> x_sharded = with_fsdp_sharding(x, mesh, PS('fsdp', None))
    """
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))
