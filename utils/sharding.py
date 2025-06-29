import jax
from jax.sharding import NamedSharding, PartitionSpec as PS, Mesh
from typing import Optional

default_mesh = jax.make_mesh((2, 2), ("x", "y"))


def mesh_sharding(
    pspec: PS,
    mesh: Optional[Mesh] = None,
) -> NamedSharding:
    if mesh is None:
        mesh = default_mesh
    return NamedSharding(mesh, pspec)
