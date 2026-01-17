"""Multi-host synchronization server for distributed JAX inference.

This module provides SyncServer for coordinating multiple JAX processes
across different machines using JAX's distributed client.
"""

import json
from typing import Any

import jax
from jax._src import distributed


class SyncServer:
    """A network server for syncing between JAX processes in multi-process JAX setups.

    This class uses JAX's distributed client to provide barrier synchronization
    and broadcast operations across multiple processes/hosts.
    """

    CLIENT = None
    TIMEOUT_SEC = 600

    @staticmethod
    def _get_client():
        """Get or initialize the JAX distributed client."""
        if SyncServer.CLIENT is None:
            SyncServer.CLIENT = distributed.global_state.client
        return SyncServer.CLIENT

    @staticmethod
    def barrier(key: str, current_it: int) -> None:
        """Synchronize all processes at a barrier point.

        All processes must call this with the same key and iteration number.
        The method blocks until all processes reach this barrier.

        Args:
            key: Unique identifier for this barrier
            current_it: Iteration number (makes key unique per iteration)
        """
        client = SyncServer._get_client()
        if client is None or jax.process_count() == 1:
            return
        client.wait_at_barrier(key + str(current_it), timeout_in_ms=SyncServer.TIMEOUT_SEC * 1000)

    @staticmethod
    def broadcast(key: str, current_it: int, value: Any, is_source: bool = False, jsonify: bool = True) -> Any:
        """Broadcast a value from source process to all other processes.

        The source process sets the value, and all other processes receive it.

        Args:
            key: Unique identifier for this broadcast
            current_it: Iteration number (makes key unique per iteration)
            value: Value to broadcast (only used if is_source=True)
            is_source: Whether this process is the source of the broadcast
            jsonify: Whether to serialize value as JSON (default: True)

        Returns:
            The broadcast value (same on all processes after the call)
        """
        client = SyncServer._get_client()
        if client is None or jax.process_count() == 1:
            return value
        if is_source:
            client.key_value_set(key + str(current_it), json.dumps(value) if jsonify else value)
            return value
        else:
            value = client.blocking_key_value_get(key + str(current_it), SyncServer.TIMEOUT_SEC * 1000)
            return json.loads(value) if jsonify else value
