"""
Utilities for multi-node distributed training with JAX.
"""

import jax
import jax.distributed as distributed
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def init_jax_distributed(
    coordinator_address: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
) -> None:
    """
    Initialize JAX for distributed multi-node training.

    Reads from environment variables if arguments not provided:
    - JAX_COORDINATOR_ADDRESS: Coordinator address (e.g., "10.0.0.1:1234")
    - JAX_NUM_PROCESSES: Total number of processes across all nodes
    - JAX_PROCESS_ID: This process's ID (0 to num_processes-1)

    Args:
        coordinator_address: Address of the coordinator process. If None, reads from
            environment variable JAX_COORDINATOR_ADDRESS.
        num_processes: Total number of processes. If None, reads from environment
            variable JAX_NUM_PROCESSES (default 1).
        process_id: ID of this process. If None, reads from environment variable
            JAX_PROCESS_ID (default 0).

    Example:
        >>> # Single node (no distributed setup needed)
        >>> init_jax_distributed()  # num_processes=1, process_id=0
        >>>
        >>> # Multi-node via environment variables
        >>> # On node 0: export JAX_COORDINATOR_ADDRESS="node0:1234"
        >>> #            export JAX_NUM_PROCESSES=4
        >>> #            export JAX_PROCESS_ID=0
        >>> init_jax_distributed()
        >>>
        >>> # Multi-node via arguments
        >>> init_jax_distributed(
        ...     coordinator_address="10.0.0.1:1234",
        ...     num_processes=4,
        ...     process_id=0,
        ... )
    """
    # Read from environment if not provided
    coordinator_address = coordinator_address or os.environ.get('JAX_COORDINATOR_ADDRESS')
    num_processes = num_processes or int(os.environ.get('JAX_NUM_PROCESSES', 1))
    process_id = process_id or int(os.environ.get('JAX_PROCESS_ID', 0))

    if num_processes > 1:
        if coordinator_address is None:
            raise ValueError(
                "coordinator_address must be provided (via argument or "
                "JAX_COORDINATOR_ADDRESS environment variable) for multi-node "
                "training (num_processes > 1)"
            )

        logger.info(
            f"Initializing JAX distributed: process {process_id}/{num_processes} "
            f"at {coordinator_address}"
        )

        distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )

        logger.info("JAX distributed initialization complete")
    else:
        logger.info("Single-process mode (no distributed initialization needed)")

    # Log device information
    log_device_info()


def get_process_info() -> Dict[str, any]:
    """
    Get information about the current process and devices.

    Returns:
        Dictionary containing:
        - process_id: Index of current process
        - process_count: Total number of processes
        - local_device_count: Number of devices local to this process
        - device_count: Total number of devices across all processes
        - devices: List of all devices
        - local_devices: List of devices local to this process

    Example:
        >>> info = get_process_info()
        >>> print(f"Process {info['process_id']}/{info['process_count']}")
        >>> print(f"Local devices: {info['local_device_count']}")
    """
    return {
        'process_id': jax.process_index(),
        'process_count': jax.process_count(),
        'local_device_count': jax.local_device_count(),
        'device_count': jax.device_count(),
        'devices': jax.devices(),
        'local_devices': jax.local_devices(),
    }


def log_device_info() -> None:
    """Log information about available devices."""
    info = get_process_info()

    logger.info("=" * 60)
    logger.info("JAX Device Information")
    logger.info("=" * 60)
    logger.info(f"Process: {info['process_id']}/{info['process_count']}")
    logger.info(f"Total devices across all processes: {info['device_count']}")
    logger.info(f"Local devices on this process: {info['local_device_count']}")
    logger.info(f"Device type: {info['local_devices'][0].platform if info['local_devices'] else 'N/A'}")
    logger.info("-" * 60)
    logger.info("Local devices:")
    for device in info['local_devices']:
        logger.info(f"  {device}")
    logger.info("=" * 60)


def is_distributed() -> bool:
    """Check if running in distributed mode (multiple processes)."""
    return jax.process_count() > 1


def is_main_process() -> bool:
    """Check if this is the main process (process_id == 0)."""
    return jax.process_index() == 0


def wait_for_all_processes() -> None:
    """
    Barrier synchronization across all processes.

    All processes wait until all have reached this point.

    Example:
        >>> # Process 0: do some initialization
        >>> if is_main_process():
        ...     initialize_something()
        >>>
        >>> # Wait for all processes
        >>> wait_for_all_processes()
        >>>
        >>> # Now all processes can proceed
    """
    if is_distributed():
        # Create a dummy all-reduce to synchronize
        dummy = jax.numpy.array([1.0])
        _ = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(
            dummy[None]
        )
        logger.debug("All processes synchronized")
