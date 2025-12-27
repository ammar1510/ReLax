"""Mesh and sharding helper utilities for distributed inference."""

from typing import Optional
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from utils.kvcache import KVCache


class MeshHelper:
    """Helper class for managing mesh placement and sharding operations."""

    @staticmethod
    def get_axis_name(mesh: Optional[Mesh]) -> Optional[str]:
        """Extract the first axis name from a mesh.

        Args:
            mesh: JAX mesh or None

        Returns:
            First axis name if mesh exists, otherwise None
        """
        if mesh is None or not mesh.axis_names:
            return None
        return mesh.axis_names[0]

    @staticmethod
    def batch_axis_spec(mesh: Optional[Mesh], rank: int, batch_axis: int) -> P:
        """Create a partition spec with batch axis sharded.

        Args:
            mesh: JAX mesh or None
            rank: Number of dimensions in the tensor
            batch_axis: Which axis to shard (0-indexed)

        Returns:
            PartitionSpec with batch_axis sharded along mesh axis
        """
        axis_name = MeshHelper.get_axis_name(mesh)
        if axis_name is None:
            return P()
        spec = [None] * rank
        spec[batch_axis] = axis_name
        return P(*spec)

    @staticmethod
    def put_on_mesh(value: jax.Array, mesh: Optional[Mesh], spec: P) -> jax.Array:
        """Place an array on a mesh with the given sharding spec.

        Args:
            value: Array to place
            mesh: Target mesh
            spec: Partition specification

        Returns:
            Array placed on mesh with sharding applied
        """
        if mesh is None:
            return value
        value = jax.device_put(value, NamedSharding(mesh, spec))
        return jax.block_until_ready(value)

    @staticmethod
    def place_kv_cache(cache: KVCache, mesh: Optional[Mesh]) -> KVCache:
        """Place a KV cache on a mesh with appropriate sharding.

        The cache is sharded along the batch dimension (axis 1 for k/v,
        axis 0 for seq_positions).

        Args:
            cache: KVCache to place
            mesh: Target mesh

        Returns:
            KVCache with all components placed on mesh
        """
        if mesh is None:
            return cache

        # Create sharding specs for k, v, and seq_positions
        k_spec = MeshHelper.batch_axis_spec(mesh, rank=len(cache.k.shape), batch_axis=1)
        v_spec = MeshHelper.batch_axis_spec(mesh, rank=len(cache.k.shape), batch_axis=1)
        pos_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.seq_positions.shape), batch_axis=0
        )

        return KVCache(
            k=MeshHelper.put_on_mesh(cache.k, mesh, k_spec),
            v=MeshHelper.put_on_mesh(cache.v, mesh, v_spec),
            seq_positions=MeshHelper.put_on_mesh(cache.seq_positions, mesh, pos_spec),
        )
