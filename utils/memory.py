import jax
from jax.tree_util import tree_leaves


def estimate_pytree_memory_footprint(pytree):
    """Estimates the memory footprint of a PyTree in bytes."""
    total_bytes = sum(leaf.nbytes for leaf in tree_leaves(pytree))
    return total_bytes


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into a human-readable string (KB, MB, GB, etc.)."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f}KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f}MB"
    elif num_bytes < 1024**4:
        return f"{num_bytes / 1024**3:.2f}GB"
    else:
        return f"{num_bytes / 1024**4:.2f}TB"
