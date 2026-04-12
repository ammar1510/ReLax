"""Protocol defining the cache interface expected by InferenceEngine.

Any cache class passed as ``cache_cls`` to InferenceEngine or ServingLoop
must implement these instance methods.  The factory classmethod ``new`` is
not enforced here because classmethod protocols with Self return types are
unwieldy in Python's type system; it is documented as a convention:

    cache_cls.new(config, bsz: int, max_seqlen: int, dtype=None) -> cache
"""

from typing import Protocol, runtime_checkable, Any
import jax


@runtime_checkable
class CacheProtocol(Protocol):
    """Structural protocol for KV cache classes used by the inference engine.

    All methods return a new cache instance (immutable / functional style,
    consistent with flax.struct.dataclass).
    """

    def slice(self, idx: int) -> "CacheProtocol":
        """Extract a single-sequence cache at batch index idx.

        Used by the engine to pull one prefill result out of a batched
        prefill before inserting it into a decode slot.
        """
        ...

    def init_on_mesh(self, mesh) -> "CacheProtocol":
        """Return a zero-initialised copy sharded on mesh.

        Use for initial allocation only — values are replaced with
        per-device zeros without triggering an allgather.
        """
        ...

    def place_on_mesh(self, mesh) -> "CacheProtocol":
        """Re-shard this cache onto mesh, preserving existing data.

        Use after mutating the cache (e.g. batch_insert) to restore
        correct sharding before the next JIT-compiled decode step.
        """
        ...

    def batch_insert(
        self,
        entries: list,
        slot_idxs: list[int],
        lens: list[int],
        next_tokens: list[Any],
        curr_tokens: jax.Array,
    ) -> tuple["CacheProtocol", jax.Array]:
        """Scatter prefill results into decode slots.

        Args:
            entries: Single-sequence caches from prefill (via slice()).
            slot_idxs: Decode slot indices to insert into.
            lens: Prefill sequence lengths for each entry.
            next_tokens: First generated token for each entry.
            curr_tokens: Current token array for the decode batch [batch, 1].

        Returns:
            Tuple of (updated cache, updated curr_tokens).
        """
        ...
