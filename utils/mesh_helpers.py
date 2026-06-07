"""Mesh and sharding helper utilities for distributed inference."""

from typing import Optional, Any
import jax
import jax.lax as lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
import jax.numpy as jnp
import numpy as np

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
    def put_on_mesh(value: Any, mesh: Optional[Mesh], spec: PS) -> jax.Array:
        """Place an array on a mesh with the given sharding spec.

        Args:
            value: Array to place (can be numpy or jax)
            mesh: Target mesh
            spec: Partition specification

        Returns:
            Array placed on mesh with sharding applied
        """
        if mesh is None:
            return value

        value = jax.device_put(value, NamedSharding(mesh, spec))
        return value

    @staticmethod
    def create_sharded_zeros(
        shape: tuple, dtype: jnp.dtype, mesh: Mesh, spec: PS
    ) -> jax.Array:
        """Create a zero array directly sharded on mesh without allgather.

        Uses jax.make_array_from_callback to place shards directly on devices,
        avoiding the multihost assert_equal allgather that device_put triggers.
        """
        sharding = NamedSharding(mesh, spec)

        def _zeros_cb(index):
            shard_shape = [
                (s.stop or full) - (s.start or 0)
                for s, full in zip(index, shape)
            ]
            return np.zeros(shard_shape, dtype=dtype)

        return jax.make_array_from_callback(shape, sharding, _zeros_cb)

    @staticmethod
    def _kv_cache_specs(cache: KVCache, mesh: Mesh, pspec: Optional[PS] = None):
        """Compute partition specs for KV cache placement."""
        if pspec is not None:
            return pspec, pspec, pspec
        dp = "dp" if "dp" in mesh.axis_names else None
        tp = MeshHelper.get_tp_axis(mesh)
        if dp is not None:
            return PS(None, dp, tp, None, None), PS(None, dp, tp, None, None), PS(dp)
        k_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.k.shape), batch_axis=1
        )
        v_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.k.shape), batch_axis=1
        )
        pos_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.seq_positions.shape), batch_axis=0
        )
        return k_spec, v_spec, pos_spec
    @staticmethod
    def place_kv_cache(
        cache: KVCache, mesh: Optional[Mesh], pspec: Optional[PS] = None
    ) -> KVCache:
        """Re-shard an existing KV cache on a mesh, preserving data.

        Use this for re-sharding after cache mutations (e.g. prefill insertion).
        For initial zero cache creation, use init_kv_cache_on_mesh instead.
        """
        if mesh is None:
            return cache
        k_spec, v_spec, pos_spec = MeshHelper._kv_cache_specs(cache, mesh, pspec)
        return KVCache(
            k=MeshHelper.put_on_mesh(cache.k, mesh, k_spec),
            v=MeshHelper.put_on_mesh(cache.v, mesh, v_spec),
            seq_positions=MeshHelper.put_on_mesh(cache.seq_positions, mesh, pos_spec),
        )

    @staticmethod
    def place_hybrid_cache(cache, mesh: Optional[Mesh]):
        """Place a HybridCache on a mesh.

        KV cache portion gets standard KV sharding.
        DeltaNet state is replicated (batch-sharded only).
        """
        from utils.hybrid_cache import HybridCache, DeltaNetState

        if mesh is None:
            return cache

        kv = MeshHelper.place_kv_cache(cache.kv_cache, mesh)

        # DeltaNet state: batch-shard on axis 1 (shape: [layers, bsz, ...])
        state_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.deltanet_state.state.shape), batch_axis=1
        )
        conv_spec = MeshHelper.batch_axis_spec(
            mesh, rank=len(cache.deltanet_state.conv_state.shape), batch_axis=1
        )
        delta = DeltaNetState(
            state=MeshHelper.put_on_mesh(cache.deltanet_state.state, mesh, state_spec),
            conv_state=MeshHelper.put_on_mesh(cache.deltanet_state.conv_state, mesh, conv_spec),
        )

        return HybridCache(kv_cache=kv, deltanet_state=delta)

    @staticmethod
    def param_sharding(x, name: str, mesh: Mesh):
        tp = MeshHelper.get_tp_axis(mesh)
        ndim = len(x.shape) if not hasattr(x, "ndim") else x.ndim
        if tp is None or "norm" in name or "freqs_cis" in name or "shared_expert_gate" in name:
            return PS()
        if any(k in name for k in ("wq", "wk", "wv", "embedding", "gate", "up", "in_proj")):
            shard_axis = 1
            if shard_axis < ndim and x.shape[shard_axis] % mesh.shape[tp] == 0:
                spec = [None] * ndim
                spec[shard_axis] = tp
                return PS(*spec)
            return PS()
        if any(k in name for k in ("down", "output", "wo", "out_proj")):
            shard_axis = 0
            if shard_axis < ndim and x.shape[shard_axis] % mesh.shape[tp] == 0:
                spec = [None] * ndim
                spec[shard_axis] = tp
                return PS(*spec)
            return PS()
        return PS()

    @staticmethod
    def shard_params(params: Any, mesh: Mesh) -> Any:
        """Apply parameter sharding to a pytree of parameters.
        """

        def _get_key_name(key) -> str:
            if hasattr(key, "key"):
                return str(key.key)
            elif hasattr(key, "idx"):
                return str(key.idx)
            elif hasattr(key, "name"):
                return str(key.name)
            return str(key)

        def shard_leaf(path, x):
            name = "/".join(_get_key_name(k) for k in path)

            # Already a multi-host distributed array (e.g. from load_from_orbax with mesh).
            # astype preserves sharding; device_put would trigger allgather → OOM.
            if isinstance(x, jax.Array) and not x.is_fully_addressable:
                if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != jnp.bfloat16:
                    return x.astype(jnp.bfloat16)
                return x

            # numpy or host-local JAX array — place onto mesh via callback to avoid
            # the allgather that jax.device_put triggers on multi-host NamedShardings.
            x = np.asarray(x)
            if np.issubdtype(x.dtype, np.floating):
                import ml_dtypes
                x = x.astype(ml_dtypes.bfloat16)
            spec = MeshHelper.param_sharding(x, name, mesh)
            sharding = NamedSharding(mesh, spec)
            return jax.make_array_from_callback(x.shape, sharding, lambda idx: x[idx])

        # shard_params is a collective, all processes must call it
        params = jax.tree.map_with_path(shard_leaf, params)
        # Block until sharding is complete to ensure balanced memory before next steps
        return jax.block_until_ready(params)
