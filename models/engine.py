"""Async slot-based inference engine for efficient LLM serving.

This module implements a production-ready inference system with:
- Prefill thread: Processes prompts sequentially (one at a time)
- Generate thread: Batches multiple sequences via slots for efficient generation
- Streaming support via response queues
- Thread-safe request/response handling

Architecture:
    User Requests → Prefill Queue → Prefill Thread → Generate Queue
                                                          ↓
                                                    Generate Thread
                                                    (Slot-based batching)
                                                          ↓
                                                    Response Queues
"""

import sys
import threading
import dataclasses
from typing import Optional, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.sharding import Mesh, set_mesh
import numpy as np
from jax import jit
from jax.experimental.multihost_utils import process_allgather

from jax import random

from utils.padding import take_nearest_bucket, pad_to_bucket, DEFAULT_PREFILL_BUCKETS
from utils.mesh_helpers import MeshHelper
from utils.cache_protocol import CacheProtocol
from models.sync_server import SyncServer
from sampling import greedy



########################################################################################################################
# Serving Loop Data Structures (New API) ##############################################################################
########################################################################################################################


@dataclasses.dataclass
class ServingConfig:
    """Configuration for the serving loop.

    Attributes:
        decode_steps: Number of tokens to generate per decode step
        decode_batch_size: Maximum number of concurrent decode sequences
        prefill_batch_size: Maximum number of concurrent prefill sequences
        eos_tokens: Tuple of end-of-sequence token IDs
        token_pad_idx: Token ID used for padding
        max_decode_length: Maximum number of tokens to generate per request
    """

    decode_steps: int = 10
    decode_batch_size: int = 16
    prefill_batch_size: int = 16
    eos_tokens: tuple[int, ...] = ()
    token_pad_idx: int = 0
    max_decode_length: int = 64
    max_cache_seqlen: int = 1024
    sampler: callable = greedy
    rng_seed: int = 0


@dataclasses.dataclass
class UserRequestPrompt:
    """A user request for inference (serving loop API).

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


class InferenceEngine:
    """Core inference engine managing prefill and slot-based generation.

    This class handles:
    - Prefill: Processing prompts to populate initial KV cache
    - Generate: Batched token generation across multiple slots
    - Slot management: Insert/remove sequences from the decode batch

    The engine uses JIT-compiled functions for prefill and generate to maximize
    performance while maintaining flexibility through the slot mechanism.
    """

    def __init__(
        self,
        model,
        params: FrozenDict,
        mesh: Mesh,
        cache_cls: type[CacheProtocol],
        max_concurrent_slots: int = 8,
        pad_id: int = 0,
        sampler: Optional[callable] = None,
        rng_seed: int = 0,
        max_cache_seqlen: Optional[int] = None,
    ):
        """Initialize the inference engine.

        Args:
            model: Model instance (LLaMa, Qwen, Gemma, or any duck-typed model)
            params: Model parameters (Flax FrozenDict)
            mesh: JAX mesh for sharded computation
            cache_cls: Cache class implementing the cache protocol (KVCache, GemmaCache, HybridCache).
                       Must implement new(config, bsz, max_seqlen), slice(idx),
                       place_on_mesh(mesh), batch_insert(...).
            max_concurrent_slots: Maximum number of sequences to generate concurrently
            pad_id: Token ID used for padding
            sampler: Sample function (logits, key) -> token_ids (default: greedy)
            rng_seed: Seed for PRNG key used in stochastic sampling
            max_cache_seqlen: Override model's max_seqlen for KV cache allocation
        """
        self.model = model
        self.cache_cls = cache_cls
        self.max_slots = max_concurrent_slots
        self.pad_id = pad_id
        self.mesh = mesh
        self.sampler = sampler or greedy
        self.rng_key = random.PRNGKey(rng_seed)
        self.max_cache_seqlen = max_cache_seqlen or model.args.max_seqlen

        self.params = params

        # Cache model config for convenience
        self.config = model.args

        sample_fn = self.sampler

        # Create jitted core function for batched prefill logic
        @jit
        def _jitted_prefill_core(params, tokens, true_lengths, cache, rng_key):
            # Forward pass through model (positional args for model-agnostic call)
            logits, updated_cache = self.model.apply(
                {"params": params},
                tokens,
                true_lengths,
                cache,
            )

            # Extract logit at position (true_length - 1) for each sequence
            indices = (true_lengths - 1)[:, None, None]  # [bsz, 1, 1]
            last_logits = jnp.take_along_axis(logits, indices, axis=1).squeeze(1)

            next_tokens = sample_fn(last_logits, rng_key)  # [bsz]

            return updated_cache, next_tokens

        self._jitted_prefill_core = _jitted_prefill_core

    def prefill(
        self,
        tokens: jax.Array,  # [bsz, bucket_size]
        true_lengths: jax.Array,  # [bsz]
    ) -> Dict[str, any]:
        """Process a batch of prompts and return prefilled states.

        This method processes batched prompts in a single forward pass,
        populating the KV cache and sampling the first generated token for each.

        Args:
            tokens: Batched padded prompt tokens of shape [bsz, bucket_size]
            true_lengths: Actual (non-padded) lengths for each sequence [bsz]

        Returns:
            Dictionary with:
                - 'cache': Populated KVCache for batch [layers, bsz, ...]
                - 'next_tokens': First generated tokens [bsz]
                - 'seq_lengths': Current sequence lengths [bsz]

        Note: The core computation is JIT-compiled for efficiency.
        """
        bsz, bucket_size = tokens.shape

        cache = self.cache_cls.new(self.config, bsz, bucket_size, mesh=self.mesh)

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

        self.rng_key, subkey = random.split(self.rng_key)
        updated_cache, next_tokens = self._jitted_prefill_core(
            self.params,
            tokens,
            true_lengths,
            cache,
            subkey,
        )

        return {
            "cache": updated_cache,
            "next_tokens": next_tokens,  # [bsz]
            "seq_lengths": true_lengths,  # [bsz]
        }

    def make_multistep_decode_fn(self):
        """Create a JIT-compiled multistep decode function.

        This creates a function that generates N tokens per call using jax.lax.scan,
        which is more efficient than calling generate_batch N times separately.

        Returns:
            A JIT-compiled function with signature:
                (curr_tokens, active_mask, cache, steps) -> ((curr_tokens, cache), output_tokens)
            where output_tokens has shape [batch, steps]

        """
        engine = self

        # Capture callbacks for use inside JIT
        sample_fn = self.sampler

        @partial(jax.jit, static_argnames=("steps", "eos_tokens"), donate_argnames=("cache",))
        def multistep_decode_fn(curr_tokens, active_mask, params, cache, rng_key, steps: int = 10, eos_tokens: tuple = ()):
            """Generate multiple tokens in a single JIT-compiled call.

            Args:
                curr_tokens: Current tokens [batch, 1]
                active_mask: Boolean mask of active slots [batch]
                params: Model parameters (passed explicitly to avoid capture as constants)
                cache: KV cache
                rng_key: PRNG key for sampling (split each step)
                steps: Number of tokens to generate
                eos_tokens: Tuple of EOS token IDs; slots are masked out after generating one

            Returns:
                Tuple of ((curr_tokens, cache), output_tokens) where:
                    - curr_tokens: Updated current tokens [batch, 1]
                    - cache: Updated KV cache
                    - output_tokens: Generated tokens [batch, steps]
            """
            eos_ids = jnp.array(eos_tokens, dtype=curr_tokens.dtype) if eos_tokens else None

            def body(carry, _):
                curr_tokens, active_mask, cache, rng_key = carry

                true_lengths = active_mask.astype(jnp.int32)

                # Forward pass — params passed from outer scope (JIT arg, not closure capture)
                logits, updated_cache = engine.model.apply(
                    {"params": params},
                    curr_tokens,
                    true_lengths,
                    cache,
                )

                # Sample next tokens
                batch_logits = logits[:, 0, :]
                rng_key, subkey = random.split(rng_key)
                new_tokens = sample_fn(batch_logits, subkey)[:, None]

                # Disable slots that just produced an EOS token
                if eos_ids is not None:
                    hit_eos = jnp.any(new_tokens == eos_ids[None, :], axis=-1)  # [batch]
                    updated_mask = active_mask & ~hit_eos
                else:
                    updated_mask = active_mask

                # Freeze carry for inactive slots; slots that just hit EOS still emit new_tokens
                # (the EOS token itself), and will be frozen to it on subsequent steps.
                updated_tokens = jnp.where(
                    active_mask[:, None],
                    new_tokens,
                    curr_tokens,
                )

                # Reshard carry to match input sharding (prevents drift across scan iterations)
                updated_tokens = jax.lax.with_sharding_constraint(
                    updated_tokens, jax.typeof(curr_tokens).sharding.spec
                )

                return (updated_tokens, updated_mask, updated_cache, rng_key), updated_tokens

            (final_tokens, _, final_cache, _), output_tokens = jax.lax.scan(
                body, (curr_tokens, active_mask, cache, rng_key), length=steps
            )

            # output_tokens shape: [steps, batch, 1]
            # Transpose to [batch, steps] and squeeze last dim
            output_tokens = output_tokens[:, :, 0].T  # [batch, steps]

            return (final_tokens, final_cache), output_tokens

        return multistep_decode_fn

    def init_decode_state(self) -> tuple[Any, jax.Array]:
        """Initialize empty decode state with all slots inactive.

        Returns:
            Tuple of (cache, tokens) placed on the mesh
        """
        cache = self.cache_cls.new(self.config, self.max_slots, self.max_cache_seqlen, mesh=self.mesh)

        tokens = jnp.zeros((self.max_slots, 1), dtype=jnp.int32)
        tokens = MeshHelper.put_on_mesh(
            tokens,
            self.mesh,
            MeshHelper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
        )

        return cache, tokens


########################################################################################################################
# ServingLoop (New Event Loop-Based Serving) ##########################################################################
########################################################################################################################


class ServingLoop:
    """Event loop-based serving for multi-host inference.

    This class uses a pure event loop pattern that:
    - Eliminates threading complexity
    - Adds multi-host coordination via SyncServer
    - Provides flexible batching
    - Uses multistep decode for efficiency

    Usage:
        serving_loop = ServingLoop(serve_cfg, model, params, mesh, is_server=True)

        # Add requests (server process only)
        serving_loop.add_request(UserRequestPrompt(id=1, text=[1, 2, 3]))

        # Event loop (all processes). The server passes `should_stop`; it's
        # broadcast to all processes so everyone exits on the same iteration.
        while not serving_loop.serving_step(
            should_stop=serving_loop.done_count >= len(serving_loop.results)
        ):
            pass

            # Check results (server process)
            for id, result in serving_loop.results.items():
                if result.done:
                    print(f"Completed: {result.token_list}")
    """

    def __init__(
        self,
        serve_cfg: ServingConfig,
        model,
        params: FrozenDict,
        mesh: Mesh,
        cache_cls: type[CacheProtocol],
        is_server: bool = False,
        verbose: bool = False,
    ):
        """Initialize the serving loop.

        Args:
            serve_cfg: Serving configuration
            model: Model instance (LLaMa, Qwen, Gemma, or any duck-typed model)
            params: Model parameters
            mesh: JAX mesh for sharded computation
            cache_cls: Cache class implementing the cache protocol (forwarded to InferenceEngine)
            is_server: Whether this process is the server (receives requests)
        """
        self.serve_cfg = serve_cfg
        self.model = model
        self.mesh = mesh
        self.verbose = verbose

        # Initialize InferenceEngine for core operations
        self.engine = InferenceEngine(
            model=model,
            params=params,
            mesh=mesh,
            cache_cls=cache_cls,
            max_concurrent_slots=serve_cfg.decode_batch_size,
            pad_id=serve_cfg.token_pad_idx,
            sampler=serve_cfg.sampler,
            rng_seed=serve_cfg.rng_seed,
            max_cache_seqlen=serve_cfg.max_cache_seqlen,
        )

        # Initialize decode state
        kv_cache, tokens = self.engine.init_decode_state()

        # Setup decode work
        self.decode_work = DecodeWork(
            curr_tokens=tokens,
            cache=kv_cache,
            active_results=[None for _ in range(serve_cfg.decode_batch_size)],
        )

        # Create multistep decode function
        self.multistep_decode_fn = self.engine.make_multistep_decode_fn()
        self._decode_call_count = 0

        # Setup prefill work
        self.prefill_work = PrefillWork([], [])

        # Results tracking
        self.results = {}  # request_id -> DecodeResult
        self.done_count = 0  # incremented when a result is marked done
        self.pending_requests = []
        self.state_lock = threading.Lock()

        # Multi-host coordination
        self._it = 0
        self.roles = self._determine_roles(is_server, mesh)
        self.eos_tokens = np.array(serve_cfg.eos_tokens)

        print(f"[ServingLoop] Initialized with roles: {self.roles}")
        sys.stdout.flush()

    def warmup(self):
        """Trigger JIT compilation of prefill and decode with dummy inputs.

        Call this once before real inference to avoid compilation stalls
        during the first serving_step.
        """
        import time as _time
        bsz_prefill = self.serve_cfg.prefill_batch_size
        bsz_decode = self.serve_cfg.decode_batch_size

        # Warmup prefill
        print("[ServingLoop] Warming up prefill (JIT compiling)...")
        sys.stdout.flush()
        t0 = _time.time()

        dummy_tokens = jnp.ones((bsz_prefill, 64), dtype=jnp.int32)
        dummy_lengths = jnp.full((bsz_prefill,), 10, dtype=jnp.int32)
        with set_mesh(self.mesh):
            prefill_result = self.engine.prefill(dummy_tokens, dummy_lengths)
        jax.block_until_ready(prefill_result["next_tokens"])
        print(f"[ServingLoop] Prefill compiled in {_time.time() - t0:.1f}s")
        sys.stdout.flush()

        # Warmup decode
        print("[ServingLoop] Warming up decode (JIT compiling)...")
        sys.stdout.flush()
        t0 = _time.time()

        dummy_active_mask = jnp.ones(bsz_decode, dtype=bool)
        dummy_rng_key = random.PRNGKey(0)
        with set_mesh(self.mesh):
            self.multistep_decode_fn(
                self.decode_work.curr_tokens,
                dummy_active_mask,
                self.engine.params,
                self.decode_work.cache,
                dummy_rng_key,
                steps=self.serve_cfg.decode_steps,
                eos_tokens=tuple(self.eos_tokens.tolist()),
            )
        print(f"[ServingLoop] Decode compiled in {_time.time() - t0:.1f}s")
        sys.stdout.flush()

        # Re-initialize decode state (discard dummy results)
        self.decode_work.cache, self.decode_work.curr_tokens = self.engine.init_decode_state()

        print("[ServingLoop] Warmup complete")
        sys.stdout.flush()

    def _determine_roles(self, is_server: bool, mesh: Mesh) -> tuple:
        """Determine this process's roles for multi-host coordination.

        Roles:
            - server: Process that receives user requests
            - worker: Process has devices in the mesh
            - coordinator: Process with smallest mesh device (for broadcasts)

        Args:
            is_server: Whether this is the server process
            mesh: Computation mesh

        Returns:
            Tuple of role strings
        """
        roles = ()
        if is_server:
            roles += ("server",)

        local_devices = set(d.id for d in jax.local_devices())
        mesh_devices = set(d.id for d in mesh.devices.flat)

        if local_devices & mesh_devices:
            roles += ("worker",)

        # Coordinator = process with smallest device ID in mesh
        if any(d.id == min(mesh_devices) for d in jax.local_devices()):
            roles += ("coordinator",)

        return roles

    def _extract_individual_result(self, batched_result: Dict, idx: int) -> Dict:
        """Extract single-sequence result from batched prefill output.

        Args:
            batched_result: Dict with batched cache, next_tokens, seq_lengths
            idx: Index of sequence to extract

        Returns:
            Dict with single-sequence cache matching prefill() output format
        """
        single_cache = batched_result["cache"].slice(idx)

        return {
            "cache": single_cache,
            "next_token": batched_result["next_tokens"][idx],
            "seq_length": batched_result["seq_lengths"][idx],
        }

    def _update_cache_and_index(self, batch_updates: list):
        """Batch update cache and tokens for multiple slot insertions.

        All slot updates are applied in a single scatter (one copy of the full
        cache tensor) instead of one copy per slot.

        Args:
            batch_updates: List of (cache_entry, slot_idx, length, next_token) tuples
        """
        entries, slot_idxs, lens, next_tokens = map(list, zip(*batch_updates))

        new_cache, new_tokens = self.decode_work.cache.batch_insert(
            entries, slot_idxs, lens, next_tokens, self.decode_work.curr_tokens,
        )
        self.decode_work.cache = new_cache
        self.decode_work.curr_tokens = new_tokens

    def _update_results_and_evict(self, output_tokens: np.ndarray):
        """Update results dict with new tokens and evict completed sequences.

        Args:
            output_tokens: Generated tokens [batch, steps]
        """
        eos_set = set(self.eos_tokens.tolist()) if len(self.eos_tokens) > 0 else set()

        # Dispatch tokens, evicting slots that hit EOS or max length
        for i, (result, tokens) in enumerate(zip(self.decode_work.active_results, output_tokens)):
            if result is None:
                continue
            for token in tokens:
                if token in eos_set or result.tokens_decoded >= self.serve_cfg.max_decode_length:
                    result.done = True
                    self.done_count += 1
                    self.decode_work.active_results[i] = None
                    if self.verbose:
                        print(f"[ServingLoop] Completed request {result.id} ({result.tokens_decoded} tokens)")
                        sys.stdout.flush()
                    break
                self.results[result.id].token_list.append(token)
                result.tokens_decoded += 1

    def decode_step(self):
        """One decode iteration: insert pending prefills + run multistep decode."""
        self._log("decode: phase1 (insert prefills)")
        # Phase 1: Insert pending prefills into free slots
        if len(self.prefill_work.to_decode) > 0:
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

            # Batch update cache and tokens
            if "worker" in self.roles and len(batch_updates) > 0:
                self._update_cache_and_index(batch_updates)
                # Re-place on mesh to match warmup sharding (prevents JIT recompilation)
                self.decode_work.cache = self.decode_work.cache.place_on_mesh(self.mesh)
                self.decode_work.curr_tokens = MeshHelper.put_on_mesh(
                    self.decode_work.curr_tokens,
                    self.mesh,
                    MeshHelper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
                )

        # Phase 2: Run multistep decode (skip if all slots empty)
        active_count = sum(1 for x in self.decode_work.active_results if x is not None)
        self._log(f"decode: phase2 (active_slots={active_count})")
        if active_count == 0:
            self._log("decode: all slots empty, early return (SKIPPING phase3 sync!)")
            return

        # Build active_mask for multistep decode
        active_mask = jnp.array([result is not None for result in self.decode_work.active_results])


        self.engine.rng_key, decode_rng_key = random.split(self.engine.rng_key)
        import time as _time
        _t0 = _time.time()
        with set_mesh(self.mesh):
            (final_tokens, final_cache), output_tokens = self.multistep_decode_fn(
                self.decode_work.curr_tokens,
                active_mask,
                self.engine.params,
                self.decode_work.cache,
                decode_rng_key,
                steps=self.serve_cfg.decode_steps,
                eos_tokens=tuple(self.eos_tokens.tolist()),
            )
            jax.block_until_ready((final_tokens, final_cache, output_tokens))

            # Update decode work
            self.decode_work.curr_tokens = final_tokens
            self.decode_work.cache = final_cache

        self._decode_call_count += 1
        _elapsed = _time.time() - _t0
        _active = int(active_mask.sum())
        if self.verbose:
            print(f"[decode #{self._decode_call_count}] {_elapsed:.3f}s for {self.serve_cfg.decode_steps} steps, "
                  f"{_active}/{self.serve_cfg.decode_batch_size} active slots, "
                  f"{_elapsed/self.serve_cfg.decode_steps*1000:.1f}ms/token")
            sys.stdout.flush()

        # Phase 3: Process current iteration's output
        self._log("decode: entering barrier:decode_output")
        SyncServer.barrier("decode_output", self._it)
        if "worker" in self.roles:
            output_tokens = jax.block_until_ready(process_allgather(output_tokens, tiled=True))
            output_tokens = np.array(output_tokens).tolist()  # [batch, steps] as nested list
        else:
            output_tokens = None

        # Broadcast results to all processes
        self._log("decode: entering broadcast:decode_tokens")
        output_tokens, = SyncServer.broadcast(
            "decode_tokens",
            self._it,
            (output_tokens,),
            is_source="coordinator" in self.roles,
        )

        # Phase 4: Update results and evict completed sequences
        self._update_results_and_evict(output_tokens)

    def prefill_step(self):
        """One prefill iteration: batch prefill pending requests."""
        # Backpressure: don't prefill if too many results are waiting for decode slots
        if len(self.prefill_work.to_decode) >= self.serve_cfg.decode_batch_size:
            self._log(f"prefill: backpressure ({len(self.prefill_work.to_decode)} queued, skipping)")
            return

        # Process up to prefill_batch_size requests
        prefill_batch = self.prefill_work.to_prefill[: self.serve_cfg.prefill_batch_size]
        self.prefill_work.to_prefill = self.prefill_work.to_prefill[len(prefill_batch) :]

        if len(prefill_batch) == 0:
            self._log("prefill: no pending requests")
            return

        self._log(f"prefill: processing {len(prefill_batch)} requests")
        # Prepare batched inputs (pad to max length in batch)
        max_len = max(len(req.text) for req in prefill_batch)
        bucket_size = take_nearest_bucket(DEFAULT_PREFILL_BUCKETS, max_len)

        tokens_list = []
        true_lengths_list = []

        for req in prefill_batch:
            # Convert list to numpy array
            tokens = np.array(req.text)
            # Add batch dimension and pad
            tokens_with_batch = tokens[None, :]  # [1, seqlen]
            padded = pad_to_bucket(tokens_with_batch, bucket_size, self.engine.pad_id)
            tokens_list.append(padded)
            true_lengths_list.append(len(req.text))

        # Pad batch to prefill_batch_size so batch dim is divisible by dp axis
        actual_bsz = len(tokens_list)
        target_bsz = self.serve_cfg.prefill_batch_size
        for _ in range(target_bsz - actual_bsz):
            tokens_list.append(np.zeros_like(tokens_list[0]))
            true_lengths_list.append(0)

        batched_tokens = jnp.concatenate(tokens_list, axis=0)  # [target_bsz, bucket_size]
        batched_true_lengths = jnp.array(true_lengths_list, dtype=jnp.int32)  # [target_bsz]

        # Call prefill
        self._log(f"prefill: calling engine.prefill (shape={batched_tokens.shape})")
        with set_mesh(self.mesh):
            prefill_result = self.engine.prefill(batched_tokens, batched_true_lengths)
        self._log("prefill: engine.prefill done")

        # Extract individual results and add to decode queue
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
            print(f"[ServingLoop] Prefilled {len(prefill_batch)} requests")
            sys.stdout.flush()

    def _log(self, msg):
        if not self.verbose:
            return
        pid = jax.process_index()
        print(f"[P{pid}|it={self._it}] {msg}")
        sys.stdout.flush()

    def serving_step(self, should_stop: bool = False) -> bool:
        """Main event loop step (call repeatedly).

        This method coordinates all processes using SyncServer for multi-host setups.

        Args:
            should_stop: Only read on the server process. The server broadcasts
                it to all processes, so every process exits the loop at the same
                `_it` — no barrier-key divergence across processes.

        Returns:
            True when the server signalled stop (same value on every process).
        """
        # Sync requests from server process
        self._log("entering barrier:serving_step")
        SyncServer.barrier("serving_step", self._it)
        self._it += 1

        if "server" in self.roles:
            with self.state_lock:
                requests = list(self.pending_requests)
                self.pending_requests = []
            payload = {"requests": requests, "should_stop": bool(should_stop)}
        else:
            payload = None

        self._log("entering broadcast:requests")
        payload = SyncServer.broadcast(
            "requests", self._it, payload, is_source="server" in self.roles
        )

        # Add new requests to prefill queue
        for req in payload["requests"] or []:
            self.prefill_work.to_prefill.append(UserRequestPrompt(**req))

        # If the server signalled stop, exit before doing any more work this step.
        if payload["should_stop"]:
            self._log("serving_step: stop signalled, exiting loop")
            return True

        # Execute decode and prefill
        self._log("entering decode_step")
        self.decode_step()
        self._log("entering prefill_step")
        self.prefill_step()
        self._log("serving_step done")
        return False

    def pending_prefill_count(self) -> int:
        """Return the number of requests waiting to be prefilled."""
        return len(self.prefill_work.to_prefill)

    def add_request(self, request: UserRequestPrompt):
        """Add new request (thread-safe).

        Args:
            request: User request to add
        """
        with self.state_lock:
            self.pending_requests.append(dataclasses.asdict(request))

    def serve_forever(self, shutdown_signal: threading.Event):
        """Wrap serving_step in a background thread.

        The server process reads `shutdown_signal` each step and broadcasts the
        decision through `serving_step`, so every process exits the loop on the
        same `_it` — no per-process divergence on the shutdown flag.

        Args:
            shutdown_signal: Event the caller sets to request shutdown.
        """

        def serve_thread():
            try:
                while not self.serving_step(should_stop=shutdown_signal.is_set()):
                    pass
            except Exception as e:
                print(f"[ServingLoop] Error: {e}")
                sys.stdout.flush()
            finally:
                shutdown_signal.set()
                print("[ServingLoop] Stopped")
                sys.stdout.flush()

        thread = threading.Thread(target=serve_thread, daemon=True)
        thread.start()
        print("[ServingLoop] Started in background thread")
        sys.stdout.flush()
