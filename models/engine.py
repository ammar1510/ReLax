"""Slot-based inference engine for efficient LLM serving.

This module implements a production-ready inference system with:
- Prefill: Processes prompts to populate initial KV cache
- Decode: Batches multiple sequences via slots for efficient generation
- Multi-host coordination via SyncServer
- Thread-safe request/response handling

Architecture:
    User Requests → Prefill Queue → prefill_step()
                                         ↓
                                    decode_step()
                                    (Slot-based batching)
                                         ↓
                                    results dict
"""

import sys
import threading
import dataclasses
from typing import Optional, Callable, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.sharding import Mesh, set_mesh
import numpy as np
from jax import jit

from jax import random

from utils.kvcache import KVCache
from utils.padding import take_nearest_bucket, pad_to_bucket, DEFAULT_PREFILL_BUCKETS
from utils.ops import build_attn_mask
from utils.mesh_helpers import MeshHelper
from models.sync_server import SyncServer
from sampling import greedy


########################################################################################################################
# Data Structures ######################################################################################################
########################################################################################################################


@dataclasses.dataclass
class EngineConfig:
    """Configuration for the serving loop.

    Attributes:
        decode_steps: Number of tokens to generate per decode step
        decode_batch_size: Maximum number of concurrent decode sequences
        prefill_batch_size: Maximum number of concurrent prefill sequences
        eos_tokens: Tuple of end-of-sequence token IDs
        token_pad_idx: Token ID used for padding
        max_decode_length: Maximum number of tokens to generate per request
    """

    sampler: Callable
    detokenize_fn: Callable
    decode_steps: int = 10
    decode_batch_size: int = 16
    prefill_batch_size: int = 16
    eos_tokens: tuple[int, ...] = ()
    token_pad_idx: int = 0
    max_decode_length: int = 256
    max_cache_seqlen: int = 1024
    rng_seed: int = 0


@dataclasses.dataclass
class UserRequestPrompt:
    """A user request for inference.

    Attributes:
        id: Unique request identifier
        text: Token IDs as a list (NOT jax.Array)
    """

    id: int
    text: list[int]


@dataclasses.dataclass
class DecodeResult:
    """Result for a single decode sequence.

    Attributes:
        id: Request ID
        token_list: Generated tokens
        tokens_decoded: Number of tokens decoded so far
        done: Whether generation is complete
    """

    id: int
    token_list: list[int]
    tokens_decoded: int = 0
    done: bool = False


@dataclasses.dataclass
class PrefillResult:
    """Result from prefill step, ready to be inserted into decode.

    Attributes:
        id: Request ID
        input: Input tokens as numpy array
        next_token: First generated token (from prefill)
        cache_entry: KV cache for this sequence
        len: Sequence length
    """

    id: int
    input: np.ndarray
    next_token: jax.Array
    cache_entry: Any
    len: int


@dataclasses.dataclass
class DecodeWork:
    """State for batched decode generation.

    Attributes:
        curr_tokens: Current tokens for each slot [B, 1]
        cache: Batch KV cache
        active_results: List of active decode results (None for inactive slots)
    """

    curr_tokens: jax.Array
    cache: Any
    active_results: list[Optional[DecodeResult]]


@dataclasses.dataclass
class PrefillWork:
    """State for prefill queue management.

    Attributes:
        to_prefill: Requests queued for prefill
        to_decode: Prefilled results waiting for decode slots
    """

    to_prefill: list[UserRequestPrompt]
    to_decode: list[PrefillResult]


########################################################################################################################
# InferenceEngine ######################################################################################################
########################################################################################################################


class InferenceEngine:
    """Inference engine managing prefill, slot-based generation, and serving.

    This class handles:
    - Prefill: Processing prompts to populate initial KV cache
    - Decode: Batched token generation across multiple slots
    - Slot management: Insert/remove sequences from the decode batch
    - Event loop: Multi-host coordination via SyncServer
    - Request/result tracking with thread-safe access

    Usage:
        engine = InferenceEngine(engine_cfg, model, params, mesh, is_server=True)

        # Add requests (server process only)
        engine.add_request(UserRequestPrompt(id=1, text=[1, 2, 3]))

        # Event loop (all processes)
        for _ in range(max_iterations):
            engine.serving_step()

            # Check results (server process)
            for id, result in engine.results.items():
                if result.done:
                    print(f"Completed: {result.token_list}")
    """

    def __init__(
        self,
        engine_cfg: EngineConfig,
        model,
        params: FrozenDict,
        mesh: Mesh,
        is_server: bool = False,
        verbose: bool = False,
        cache_factory: Optional[Callable] = None,
        cache_slicer: Optional[Callable] = None,
        cache_updater: Optional[Callable] = None,
        mask_cache_extractor: Optional[Callable] = None,
        place_cache: Optional[Callable] = None,
    ):
        """Initialize the inference engine.

        Args:
            engine_cfg: Serving configuration
            model: Model instance (LLaMa, Qwen, or any duck-typed model)
            params: Model parameters (Flax FrozenDict)
            mesh: JAX mesh for sharded computation
            is_server: Whether this process is the server (receives requests)
            verbose: Whether to print verbose logging
            cache_factory: (bsz, max_seqlen) -> cache. Defaults to KVCache.new().
            cache_slicer: (cache, idx) -> single_cache. Defaults to KVCache slicing.
            cache_updater: (decode_cache, entries, slot_idxs, lens, next_tokens, curr_tokens) -> (cache, tokens).
            mask_cache_extractor: (cache) -> kv_cache_for_mask. Defaults to identity.
            place_cache: (cache, mesh) -> cache. Defaults to MeshHelper.place_kv_cache.
        """
        self.engine_cfg = engine_cfg
        self.model = model
        self.mesh = mesh
        self.verbose = verbose
        self.rng_key = random.PRNGKey(engine_cfg.rng_seed)
        self.config = model.args

        # Place params on mesh
        self.params = jax.block_until_ready(
            MeshHelper.shard_params(params, mesh)
        )

        # Cache callbacks
        self.cache_factory = cache_factory or self._default_cache_factory
        self.cache_slicer = cache_slicer or self._default_cache_slicer
        self.cache_updater = cache_updater or self._default_cache_updater
        self.mask_cache_extractor = mask_cache_extractor or (lambda c: c)
        self.place_cache = place_cache or (lambda c, m: MeshHelper.place_kv_cache(c, m))
        self.init_cache = lambda c, m: MeshHelper.init_kv_cache_on_mesh(c, m)

        # JIT-compiled prefill core
        sample_fn = self.engine_cfg.sampler

        @jit
        def _jitted_prefill_core(params, tokens, true_lengths, cache, mask, rng_key):
            logits, updated_cache = self.model.apply(
                {"params": params},
                tokens,
                true_lengths,
                cache,
                mask,
            )
            indices = (true_lengths - 1)[:, None, None]  # [bsz, 1, 1]
            last_logits = jnp.take_along_axis(logits, indices, axis=1).squeeze(1)
            next_tokens = sample_fn(last_logits, rng_key)  # [bsz]
            return updated_cache, next_tokens

        self._jitted_prefill_core = _jitted_prefill_core

        # Initialize decode state
        kv_cache, tokens = self.init_decode_state()

        self.decode_work = DecodeWork(
            curr_tokens=tokens,
            cache=kv_cache,
            active_results=[None for _ in range(engine_cfg.decode_batch_size)],
        )

        # Create multistep decode function
        self.multistep_decode_fn = self.make_multistep_decode_fn()
        self._decode_call_count = 0

        # Prefill and request queues
        self.prefill_work = PrefillWork([], [])
        self.results = {}  # request_id -> DecodeResult
        self.pending_requests = []
        self.state_lock = threading.Lock()

        # Multi-host coordination
        self._it = 0
        self.roles = self._determine_roles(is_server, mesh)
        self.eos_tokens = np.array(engine_cfg.eos_tokens)

        print(f"[InferenceEngine] Initialized with roles: {self.roles}")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Cache defaults
    # ------------------------------------------------------------------

    def _default_cache_factory(self, bsz, max_seqlen):
        return KVCache.new(
            n_layers=self.config.n_layers,
            bsz=bsz,
            max_seqlen=max_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )

    def _default_cache_slicer(self, cache, idx):
        return KVCache(
            k=cache.k[:, idx : idx + 1, :, :, :],
            v=cache.v[:, idx : idx + 1, :, :, :],
            seq_positions=cache.seq_positions[idx : idx + 1],
        )

    def _default_cache_updater(self, decode_cache, entries, slot_idxs, lens, next_tokens, curr_tokens):
        idx = jnp.array(slot_idxs)
        new_positions = decode_cache.seq_positions.at[idx].set(jnp.array(lens))
        new_tokens = curr_tokens.at[idx, 0].set(jnp.array(next_tokens))
        new_k = decode_cache.k
        new_v = decode_cache.v
        for entry, slot_idx in zip(entries, slot_idxs):
            prefill_seqlen = entry.k.shape[3]
            new_k = new_k.at[:, slot_idx, :, :prefill_seqlen, :].set(entry.k[:, 0, :, :, :])
            new_v = new_v.at[:, slot_idx, :, :prefill_seqlen, :].set(entry.v[:, 0, :, :, :])
        return KVCache(k=new_k, v=new_v, seq_positions=new_positions), new_tokens

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def prefill(
        self,
        tokens: jax.Array,       # [bsz, bucket_size]
        true_lengths: jax.Array, # [bsz]
    ) -> Dict[str, any]:
        """Process a batch of prompts and return prefilled states.

        Args:
            tokens: Batched padded prompt tokens of shape [bsz, bucket_size]
            true_lengths: Actual (non-padded) lengths for each sequence [bsz]

        Returns:
            Dictionary with:
                - 'cache': Populated KVCache for batch [layers, bsz, ...]
                - 'next_tokens': First generated tokens [bsz]
                - 'seq_lengths': Current sequence lengths [bsz]
        """
        bsz, bucket_size = tokens.shape

        try:
            cache = self.cache_factory(bsz, max_seqlen=bucket_size)
        except TypeError:
            cache = self.cache_factory(bsz)
        cache = self.init_cache(cache, self.mesh)

        tokens = MeshHelper.put_on_mesh(
            tokens,
            self.mesh,
            MeshHelper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
        )
        true_lengths = MeshHelper.put_on_mesh(
            true_lengths,
            self.mesh,
            MeshHelper.batch_axis_spec(self.mesh, rank=1, batch_axis=0),
        )

        mask = build_attn_mask(bucket_size, self.mask_cache_extractor(cache), true_lengths)

        self.rng_key, subkey = random.split(self.rng_key)
        updated_cache, next_tokens = self._jitted_prefill_core(
            self.params,
            tokens,
            true_lengths,
            cache,
            mask,
            subkey,
        )

        return {
            "cache": updated_cache,
            "next_tokens": next_tokens,
            "seq_lengths": true_lengths,
        }

    def make_multistep_decode_fn(self):
        """Create a JIT-compiled multistep decode function.

        Returns:
            A JIT-compiled function:
                (curr_tokens, active_mask, params, cache, rng_key, steps, eos_tokens)
                -> ((curr_tokens, cache), output_tokens)
            where output_tokens has shape [batch, steps]
        """
        engine = self
        sample_fn = self.engine_cfg.sampler
        mask_extractor = self.mask_cache_extractor

        @partial(jax.jit, static_argnames=("steps", "eos_tokens"), donate_argnames=("cache",))
        def multistep_decode_fn(curr_tokens, active_mask, params, cache, rng_key, steps: int = 10, eos_tokens: tuple = ()):
            eos_ids = jnp.array(eos_tokens, dtype=curr_tokens.dtype) if eos_tokens else None

            def body(carry, _):
                curr_tokens, active_mask, cache, rng_key = carry

                true_lengths = active_mask.astype(jnp.int32)
                bsz, seqlen = curr_tokens.shape
                mask = build_attn_mask(seqlen, mask_extractor(cache), true_lengths)

                logits, updated_cache = engine.model.apply(
                    {"params": params},
                    curr_tokens,
                    true_lengths,
                    cache,
                    mask,
                )

                batch_logits = logits[:, 0, :]
                rng_key, subkey = random.split(rng_key)
                new_tokens = sample_fn(batch_logits, subkey)[:, None]

                updated_tokens = jnp.where(
                    active_mask[:, None],
                    new_tokens,
                    curr_tokens,
                )

                if eos_ids is not None:
                    hit_eos = jnp.any(new_tokens == eos_ids[None, :], axis=-1)  # [batch]
                    updated_mask = active_mask & ~hit_eos
                else:
                    updated_mask = active_mask

                updated_tokens = jax.lax.with_sharding_constraint(
                    updated_tokens, jax.typeof(curr_tokens).sharding.spec
                )

                return (updated_tokens, updated_mask, updated_cache, rng_key), new_tokens

            (final_tokens, _, final_cache, _), output_tokens = jax.lax.scan(
                body, (curr_tokens, active_mask, cache, rng_key), length=steps
            )

            # output_tokens shape: [steps, batch, 1] -> [batch, steps]
            output_tokens = output_tokens[:, :, 0].T

            return (final_tokens, final_cache), output_tokens

        return multistep_decode_fn

    def init_decode_state(self) -> tuple[Any, jax.Array]:
        """Initialize empty decode state with all slots inactive.

        Returns:
            Tuple of (cache, tokens) placed on the mesh
        """
        cache = self.cache_factory(self.engine_cfg.decode_batch_size, max_seqlen=self.engine_cfg.max_cache_seqlen)
        cache = self.init_cache(cache, self.mesh)

        tokens = jnp.zeros((self.engine_cfg.decode_batch_size, 1), dtype=jnp.int32)
        tokens = MeshHelper.put_on_mesh(
            tokens,
            self.mesh,
            MeshHelper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
        )

        return cache, tokens

    # ------------------------------------------------------------------
    # Serving loop
    # ------------------------------------------------------------------

    def _determine_roles(self, is_server: bool, mesh: Mesh) -> tuple:
        """Determine this process's roles for multi-host coordination.

        Roles:
            - server: Process that receives user requests
            - worker: Process has devices in the mesh
            - coordinator: Process with smallest mesh device (for broadcasts)
        """
        roles = ()
        if is_server:
            roles += ("server",)

        local_devices = set(d.id for d in jax.local_devices())
        mesh_devices = set(d.id for d in mesh.devices.flat)

        if local_devices & mesh_devices:
            roles += ("worker",)

        if any(d.id == min(mesh_devices) for d in jax.local_devices()):
            roles += ("coordinator",)

        return roles

    def _extract_individual_result(self, batched_result: Dict, idx: int) -> Dict:
        """Extract single-sequence result from batched prefill output."""
        single_cache = self.cache_slicer(batched_result["cache"], idx)
        return {
            "cache": single_cache,
            "next_token": batched_result["next_tokens"][idx],
            "seq_length": batched_result["seq_lengths"][idx],
        }

    def _update_cache_and_index(self, batch_updates: list):
        """Batch update cache and tokens for multiple slot insertions."""
        entries, slot_idxs, lens, next_tokens = map(list, zip(*batch_updates))

        new_cache, new_tokens = self.cache_updater(
            self.decode_work.cache, entries, slot_idxs, lens, next_tokens,
            self.decode_work.curr_tokens,
        )
        self.decode_work.cache = new_cache
        self.decode_work.curr_tokens = new_tokens

    def _check_done_sequences(self, output_tokens: np.ndarray) -> list[bool]:
        """Check which sequences are done (EOS or max length)."""
        done = []
        for i, result in enumerate(self.decode_work.active_results):
            if result is None:
                done.append(False)
                continue

            has_eos = np.any(output_tokens[i, :, None] == self.eos_tokens) if len(self.eos_tokens) > 0 else False
            is_max_length = result.tokens_decoded >= self.engine_cfg.max_decode_length

            done.append(bool(has_eos or is_max_length))

        return done

    def _update_results_and_evict(self, output_tokens_flat: list[int], output_mapping_flat: list[int], done: list[bool]):
        """Update results dict with new tokens and evict completed sequences."""
        for token, req_id in zip(output_tokens_flat, output_mapping_flat):
            if req_id >= 0 and req_id in self.results and not self.results[req_id].done:
                self.results[req_id].token_list.append(token)
                self.results[req_id].tokens_decoded += 1

        eos_set = set(self.eos_tokens.tolist()) if len(self.eos_tokens) > 0 else set()
        for i, result in enumerate(self.decode_work.active_results):
            if result is None:
                continue
            if done[i]:
                if eos_set:
                    for j, tok in enumerate(result.token_list):
                        if tok in eos_set:
                            result.token_list = result.token_list[:j]
                            break
                result.done = True
                self.decode_work.active_results[i] = None
                if self.verbose:
                    print(f"[InferenceEngine] Completed request {result.id} ({result.tokens_decoded} tokens)")
                    sys.stdout.flush()

    def _insert_prefills(self):
        """Insert pending prefill results into free decode slots."""
        if len(self.prefill_work.to_decode) == 0:
            return

        batch_updates = []
        for i, active_result in enumerate(self.decode_work.active_results):
            if active_result is not None:
                continue
            if len(self.prefill_work.to_decode) == 0:
                break

            result: PrefillResult = self.prefill_work.to_decode.pop(0)
            first_token = int(result.next_token)
            self.decode_work.active_results[i] = DecodeResult(result.id, [first_token], tokens_decoded=1)
            self.results[result.id] = self.decode_work.active_results[i]
            batch_updates.append((result.cache_entry, i, result.len, result.next_token))

        if len(batch_updates) > 0:
            self._update_cache_and_index(batch_updates)
            self.decode_work.cache = self.place_cache(self.decode_work.cache, self.mesh)
            self.decode_work.curr_tokens = MeshHelper.put_on_mesh(
                self.decode_work.curr_tokens,
                self.mesh,
                MeshHelper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
            )

    def decode_step(self):
        """Run multistep decode. Returns (output_tokens, output_mapping) or None if no active slots."""
        active_count = sum(1 for x in self.decode_work.active_results if x is not None)
        self._log(f"decode: active_slots={active_count}")
        if active_count == 0:
            return None

        active_mask = jnp.array([result is not None for result in self.decode_work.active_results])

        output_mapping = np.array(
            [getattr(result, "id", -1) for result in self.decode_work.active_results]
        )  # [batch]

        self.rng_key, decode_rng_key = random.split(self.rng_key)
        import time as _time
        _t0 = _time.time()
        with set_mesh(self.mesh):
            (final_tokens, final_cache), output_tokens = self.multistep_decode_fn(
                self.decode_work.curr_tokens,
                active_mask,
                self.params,
                self.decode_work.cache,
                decode_rng_key,
                steps=self.engine_cfg.decode_steps,
                eos_tokens=tuple(self.eos_tokens.tolist()),
            )
            jax.block_until_ready((final_tokens, final_cache, output_tokens))

            self.decode_work.curr_tokens = final_tokens
            self.decode_work.cache = final_cache

        self._decode_call_count += 1
        _elapsed = _time.time() - _t0
        _active = int(active_mask.sum())
        if self.verbose:
            print(f"[decode #{self._decode_call_count}] {_elapsed:.3f}s for {self.engine_cfg.decode_steps} steps, "
                  f"{_active}/{self.engine_cfg.decode_batch_size} active slots, "
                  f"{_elapsed/self.engine_cfg.decode_steps*1000:.1f}ms/token")
            sys.stdout.flush()

        return output_tokens, output_mapping

    def prefill_step(self):
        """One prefill iteration: batch prefill pending requests."""
        # Backpressure: don't prefill if too many results are waiting for decode slots
        if len(self.prefill_work.to_decode) >= self.engine_cfg.decode_batch_size:
            self._log(f"prefill: backpressure ({len(self.prefill_work.to_decode)} queued, skipping)")
            return

        prefill_batch = self.prefill_work.to_prefill[: self.engine_cfg.prefill_batch_size]
        self.prefill_work.to_prefill = self.prefill_work.to_prefill[len(prefill_batch):]

        if len(prefill_batch) == 0:
            self._log("prefill: no pending requests")
            return

        self._log(f"prefill: processing {len(prefill_batch)} requests")
        max_len = max(len(req.text) for req in prefill_batch)
        bucket_size = take_nearest_bucket(DEFAULT_PREFILL_BUCKETS, max_len)

        tokens_list = []
        true_lengths_list = []

        for req in prefill_batch:
            tokens = np.array(req.text)
            tokens_with_batch = tokens[None, :]  # [1, seqlen]
            padded = pad_to_bucket(tokens_with_batch, bucket_size, self.engine_cfg.token_pad_idx)
            tokens_list.append(padded)
            true_lengths_list.append(len(req.text))

        # Pad batch to prefill_batch_size so batch dim is divisible by dp axis
        actual_bsz = len(tokens_list)
        target_bsz = self.engine_cfg.prefill_batch_size
        for _ in range(target_bsz - actual_bsz):
            tokens_list.append(np.zeros_like(tokens_list[0]))
            true_lengths_list.append(0)

        batched_tokens = jnp.concatenate(tokens_list, axis=0)      # [target_bsz, bucket_size]
        batched_true_lengths = jnp.array(true_lengths_list, dtype=jnp.int32)  # [target_bsz]

        self._log(f"prefill: calling prefill (shape={batched_tokens.shape})")
        with set_mesh(self.mesh):
            prefill_result = self.prefill(batched_tokens, batched_true_lengths)
        self._log("prefill: done")

        for i, req in enumerate(prefill_batch):
            individual_result = self._extract_individual_result(prefill_result, i)
            new_decode = PrefillResult(
                req.id,
                np.array(req.text),
                individual_result["next_token"],
                individual_result["cache"],
                len(req.text),
            )
            self.prefill_work.to_decode.append(new_decode)

        if self.verbose:
            print(f"[InferenceEngine] Prefilled {len(prefill_batch)} requests")
            sys.stdout.flush()

    def _log(self, msg):
        if not self.verbose:
            return
        pid = jax.process_index()
        print(f"[P{pid}|it={self._it}] {msg}")
        sys.stdout.flush()

    def serving_step(self):
        """Main event loop step (call repeatedly).

        Coordinates all processes using SyncServer for multi-host setups.
        All multi-host sync is handled here; decode_step and prefill_step are pure compute.
        """
        self._log("entering barrier:serving_step")
        SyncServer.barrier("serving_step", self._it)
        self._it += 1

        # Broadcast new requests from server to all hosts
        if "server" in self.roles:
            with self.state_lock:
                requests = list(self.pending_requests)
                self.pending_requests = []
        else:
            requests = None

        self._log("entering broadcast:requests")
        requests = SyncServer.broadcast(
            "requests", self._it, requests, is_source="server" in self.roles
        )

        for req in requests or []:
            self.prefill_work.to_prefill.append(UserRequestPrompt(**req))

        # Insert prefills + decode (pure compute)
        self._log("entering insert_prefills")
        if "worker" in self.roles:
            self._insert_prefills()

        self._log("entering decode_step")
        decode_result = self.decode_step() if "worker" in self.roles else None

        # Sync decode output across hosts
        if decode_result is not None:
            output_tokens, output_mapping = decode_result

            self._log("entering barrier:decode_output")
            SyncServer.barrier("decode_output", self._it)

            if "worker" in self.roles:
                output_tokens = jax.block_until_ready(
                    jax.experimental.multihost_utils.process_allgather(output_tokens, tiled=True)
                )
                data = (
                    np.array(output_tokens).reshape(-1).tolist(),
                    np.repeat(output_mapping, self.engine_cfg.decode_steps).tolist(),
                    self._check_done_sequences(np.array(output_tokens)),
                )
            else:
                data = (None, None, None)

            self._log("entering broadcast:decode_tokens")
            output_tokens_flat, output_mapping_flat, done = SyncServer.broadcast(
                "decode_tokens", self._it, data,
                is_source="coordinator" in self.roles,
            )

            self._update_results_and_evict(output_tokens_flat, output_mapping_flat, done)

        # Prefill (pure compute)
        self._log("entering prefill_step")
        self.prefill_step()
        self._log("serving_step done")

    def pending_prefill_count(self) -> int:
        """Return the number of requests waiting to be prefilled."""
        return len(self.prefill_work.to_prefill)

    def add_request(self, request: UserRequestPrompt):
        """Add new request (thread-safe)."""
        with self.state_lock:
            self.pending_requests.append(dataclasses.asdict(request))

    def serve_forever(self, shutdown_signal: threading.Event):
        """Wrap serving_step in a background thread.

        Args:
            shutdown_signal: Event to signal shutdown
        """
        def serve_thread():
            try:
                while not shutdown_signal.is_set():
                    self.serving_step()
            except Exception as e:
                print(f"[InferenceEngine] Error: {e}")
                sys.stdout.flush()
            finally:
                shutdown_signal.set()
                print("[InferenceEngine] Stopped")
                sys.stdout.flush()

        thread = threading.Thread(target=serve_thread, daemon=True)
        thread.start()
        print("[InferenceEngine] Started in background thread")
        sys.stdout.flush()


