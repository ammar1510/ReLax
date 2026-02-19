"""Mesh and sharding helper utilities for distributed inference."""

from math import gamma
from typing import Optional, Any
import jax
import jax.lax as lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from numpy import format_float_positional

from utils.kvcache import KVCache


class MeshHelper:
    """Helper class for managing mesh placement and sharding operations."""

    @staticmethod
    def get_tp_axis(mesh: Optional[Mesh]) -> Optional[str]:
        """Get the tensor-parallel axis name ('tp' for 2D mesh, first axis for 1D)."""
        if mesh is None or not mesh.axis_names:
            return None
        return "tp" if "tp" in mesh.axis_names else mesh.axis_names[0]

    @staticmethod
    def allgather(array: jax.Array, mesh: Mesh) -> jax.Array:
        axis_name = MeshHelper.get_tp_axis(mesh)
        gather_fn = lambda x: jax.lax.all_gather(x, axis_name, tiled=True)
        sharded_gather = jax.shard_map(
            gather_fn,
            mesh=mesh,
            in_specs=PS(axis_name),
            out_specs=PS(),
            check_vma=False,
        )
        return sharded_gather(array)

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
    def batch_axis_spec(mesh: Optional[Mesh], rank: int, batch_axis: int) -> PS:
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
            return PS()
        spec = [None] * rank
        spec[batch_axis] = axis_name
        return PS(*spec)

    @staticmethod
    def put_on_mesh(value: jax.Array, mesh: Optional[Mesh], spec: PS) -> jax.Array:
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
    def place_kv_cache(
        cache: KVCache, mesh: Optional[Mesh], pspec: Optional[PS] = None
    ) -> KVCache:
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
        # k/v shape: [n_layers, bsz, n_kv_heads, max_seqlen, head_dim]
        # seq_positions shape: [bsz]
        if pspec is None:
            dp = "dp" if "dp" in mesh.axis_names else None
            tp = MeshHelper.get_tp_axis(mesh)
            if dp is not None:
                k_spec = PS(None, dp, tp, None, None)
                v_spec = PS(None, dp, tp, None, None)
                pos_spec = PS(dp)
            else:
                k_spec = MeshHelper.batch_axis_spec(
                    mesh, rank=len(cache.k.shape), batch_axis=1
                )
                v_spec = MeshHelper.batch_axis_spec(
                    mesh, rank=len(cache.k.shape), batch_axis=1
                )
                pos_spec = MeshHelper.batch_axis_spec(
                    mesh, rank=len(cache.seq_positions.shape), batch_axis=0
                )
        else:
            k_spec = pspec
            v_spec = pspec
            pos_spec = pspec

        return KVCache(
            k=MeshHelper.put_on_mesh(cache.k, mesh, k_spec),
            v=MeshHelper.put_on_mesh(cache.v, mesh, v_spec),
            seq_positions=MeshHelper.put_on_mesh(cache.seq_positions, mesh, pos_spec),
        )

    @staticmethod
    def param_sharding(x, name: str, mesh: Mesh):
        tp = MeshHelper.get_tp_axis(mesh)
        if tp is None or "norm" in name or "freqs_cis" in name:
            return PS()
        if any(k in name for k in ("wq", "wk", "wv", "embedding", "gate", "up")):
            spec = [None] * x.ndim
            spec[1] = tp
            return PS(*spec)
        if any(k in name for k in ("down", "output", "wo")):
            spec = [None] * x.ndim
            spec[0] = tp
            return PS(*spec)
        return PS()

    @staticmethod
    def shard_params(params: Any, mesh: Mesh) -> Any:
        """Apply parameter sharding to a pytree of parameters.

        Args:
            params: Pytree of parameters
            mesh: JAX mesh for sharding

        Returns:
            Pytree of parameters with sharding applied
        """

        def _get_key_name(key) -> str:
            """Extract string name from a JAX pytree path key."""
            if hasattr(key, "key"):  # DictKey
                return str(key.key)
            elif hasattr(key, "idx"):  # SequenceKey
                return str(key.idx)
            elif hasattr(key, "name"):  # GetAttrKey
                return str(key.name)
            return str(key)

        def shard_leaf(path, x):
            # Convert path to string name for sharding decisions
            name = "/".join(_get_key_name(k) for k in path)
            spec = MeshHelper.param_sharding(x, name, mesh)
            return MeshHelper.put_on_mesh(x, mesh, spec)

        return jax.tree.map_with_path(shard_leaf, params)
