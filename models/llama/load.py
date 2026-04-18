"""
Functions to save/load model weights in orbax checkpoint format.
"""

import contextlib

from typing import Dict, Any
import numpy as np

from .config import ModelConfig


def save_orbax_weights(
    params: Dict[str, Any], checkpoint_path: str, mesh=None
) -> None:
    """
    Save ReLax params to an orbax checkpoint directory (local or GCS).

    When a mesh is provided, params are sharded before saving so each host
    writes only its local shards.
    """
    import orbax.checkpoint as ocp
    import jax

    if mesh is not None:
        from utils.mesh_helpers import MeshHelper
        params = MeshHelper.shard_params(params, mesh)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    with ocp.CheckpointManager(checkpoint_path, options=options) as mngr:
        mngr.save(0, args=ocp.args.StandardSave(params))
        mngr.wait_until_finished()
    if jax.process_index() == 0:
        print(f"Saved orbax checkpoint to {checkpoint_path}")


def load_from_orbax(
    checkpoint_path: str,
    mesh=None,
) -> Dict[str, Any]:
    """
    Load ReLax params from an orbax checkpoint (local or GCS), optionally sharding onto a mesh.

    When a mesh is provided, each host reads only its own shards by passing a
    target pytree of jax.ShapeDtypeStruct with sharding specs.
    """
    import orbax.checkpoint as ocp
    import jax

    ctx = mesh if mesh is not None else contextlib.nullcontext()
    with ctx:
        with ocp.CheckpointManager(checkpoint_path) as mngr:
            step = mngr.latest_step()

            if mesh is None:
                params = mngr.restore(step, args=ocp.args.StandardRestore(None))
                if jax.process_index() == 0:
                    print(f"Loaded orbax checkpoint from {checkpoint_path}")
                return params

            from jax.sharding import NamedSharding
            from utils.mesh_helpers import MeshHelper

            # Get array metadata (shapes/dtypes) without loading data
            item_meta = mngr.item_metadata(step)
            if jax.process_index() == 0:
                print(f"[load_from_orbax] item_meta type: {type(item_meta)}, value: {item_meta!r:.200}")

            def _get_key_name(key) -> str:
                if hasattr(key, "key"):
                    return str(key.key)
                elif hasattr(key, "idx"):
                    return str(key.idx)
                elif hasattr(key, "name"):
                    return str(key.name)
                return str(key)

            def _build_target(path, meta):
                name = "/".join(_get_key_name(k) for k in path)
                spec = MeshHelper.param_sharding(meta, name, mesh)
                sharding = NamedSharding(mesh, spec)
                return jax.ShapeDtypeStruct(meta.shape, meta.dtype, sharding=sharding)

            target = jax.tree.map_with_path(_build_target, item_meta)

            params = mngr.restore(step, args=ocp.args.StandardRestore(target))
            if jax.process_index() == 0:
                print(f"Loaded orbax checkpoint from {checkpoint_path} (sharded onto mesh {mesh.axis_names})")

            return params
