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
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
from functools import partial
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.sharding import Mesh, set_mesh
import numpy as np
from jax import jit

from jax import random

from models.llama.model import LLaMa
from utils.kvcache import KVCache
from utils.padding import take_nearest_bucket, pad_to_bucket, DEFAULT_PREFILL_BUCKETS
from utils.ops import build_attn_mask
from utils.mesh_helpers import MeshHelper
from models.sync_server import SyncServer
from sampling import Sampler, GreedySampler


@dataclass
class DecodeState:
    """State for batched generation across slots.

    The decode state maintains a fixed-size batch dimension where each position
    is a "slot" that can be occupied by a generating sequence. This enables
    efficient batching of multiple concurrent requests.

    Attributes:
        kv_cache: Key-value cache for all slots
                  Shape: [layers, max_slots, max_seq_len, kv_heads, head_dim]
        tokens: Current token for each slot
                Shape: [max_slots, 1]
        active_mask: Boolean mask indicating which slots are actively generating
                     Shape: [max_slots]
        request_ids: Which request occupies each slot (None if slot is empty)
                     Length: max_slots
    """

    kv_cache: KVCache
    tokens: jax.Array  # [max_slots, 1]
    active_mask: jax.Array  # [max_slots] bool
    request_ids: List[Optional[str]]  # [max_slots]


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
    sampler: Sampler = dataclasses.field(default_factory=GreedySampler)
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
    cache: KVCache
    active_results: list[Optional[DecodeResult]]


@dataclasses.dataclass
class PrefillWork:
    """State for prefill queue management.

    Attributes:
        requests: New requests waiting to be triaged
        to_prefill: Requests queued for prefill
        to_decode: Prefilled results waiting for decode slots
    """

    requests: list[UserRequestPrompt]
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
        model: LLaMa,
        params: FrozenDict,
        mesh: Mesh,
        max_concurrent_slots: int = 8,
        pad_id: int = 0,
        buckets: Optional[List[int]] = None,
        sampler: Optional[Sampler] = None,
        rng_seed: int = 0,
        max_cache_seqlen: Optional[int] = None,
    ):
        """Initialize the inference engine.

        Args:
            model: LLaMa model instance
            params: Model parameters (Flax FrozenDict)
            mesh: JAX mesh for sharded computation
            max_concurrent_slots: Maximum number of sequences to generate concurrently
            pad_id: Token ID used for padding
            buckets: List of bucket sizes for prompt padding (default: power-of-2)
            sampler: Sampling strategy (default: GreedySampler)
            rng_seed: Seed for PRNG key used in stochastic sampling
            max_cache_seqlen: Override model's max_seqlen for KV cache allocation
        """
        self.model = model
        self.max_slots = max_concurrent_slots
        self.pad_id = pad_id
        self.buckets = buckets or DEFAULT_PREFILL_BUCKETS
        self.mesh = mesh
        self.sampler = sampler or GreedySampler()
        self.rng_key = random.PRNGKey(rng_seed)
        self.max_cache_seqlen = max_cache_seqlen or model.args.max_seqlen

        # Place params on mesh
        self.params = jax.block_until_ready(
            MeshHelper.shard_params(params, self.mesh)
        )

        # Cache model config for convenience
        self.config = model.args

        # Mesh helper for sharding operations
        self.mesh_helper = MeshHelper()

        # Create a JIT-compatible sample function from the sampler
        sample_fn = self.sampler.sample

        # Create jitted core function for batched prefill logic
        @jit
        def _jitted_prefill_core(params, tokens, true_lengths, kv_cache, mask, rng_key):
            """Jitable core logic for batched prefill.

            Args:
                params: Model parameters
                tokens: Batched tokens [bsz, seqlen]
                true_lengths: True lengths for each sequence [bsz]
                kv_cache: KV cache for batch
                mask: Attention mask
                rng_key: PRNG key for sampling

            Returns:
                Tuple of (updated_cache, next_tokens)
            """
            # Forward pass through model
            logits, updated_cache = self.model.apply(
                {"params": params},
                tokens,
                true_lengths=true_lengths,
                kv_cache=kv_cache,
                mask=mask,
            )

            # Batched logit extraction using take_along_axis
            # logits: [bsz, bucket_size, vocab_size]
            # Extract logit at position (true_length - 1) for each sequence
            indices = (true_lengths - 1)[:, None, None]  # [bsz, 1, 1]
            last_logits = jnp.take_along_axis(logits, indices, axis=1).squeeze(
                1
            )  # [bsz, vocab_size]

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

        kv_cache = KVCache.new(
            n_layers=self.config.n_layers,
            bsz=bsz,
            max_seqlen=self.max_cache_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )
        kv_cache = self.mesh_helper.place_kv_cache(kv_cache, self.mesh)

        tokens = self.mesh_helper.put_on_mesh(
            tokens,
            self.mesh,
            self.mesh_helper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
        )
        true_lengths = self.mesh_helper.put_on_mesh(
            true_lengths,
            self.mesh,
            self.mesh_helper.batch_axis_spec(self.mesh, rank=1, batch_axis=0),
        )

        mask = build_attn_mask(bucket_size, kv_cache, true_lengths)

        self.rng_key, subkey = random.split(self.rng_key)
        updated_cache, next_tokens = self._jitted_prefill_core(
            self.params,
            tokens,
            true_lengths,
            kv_cache,
            mask,
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

        # Capture sample_fn for use inside JIT
        sample_fn = self.sampler.sample

        @partial(jax.jit, static_argnames=("steps",), donate_argnames=("cache",))
        def multistep_decode_fn(curr_tokens, active_mask, params, cache, rng_key, steps: int = 10):
            """Generate multiple tokens in a single JIT-compiled call.

            Args:
                curr_tokens: Current tokens [batch, 1]
                active_mask: Boolean mask of active slots [batch]
                params: Model parameters (passed explicitly to avoid capture as constants)
                cache: KV cache
                rng_key: PRNG key for sampling (split each step)
                steps: Number of tokens to generate

            Returns:
                Tuple of ((curr_tokens, cache), output_tokens) where:
                    - curr_tokens: Updated current tokens [batch, 1]
                    - cache: Updated KV cache
                    - output_tokens: Generated tokens [batch, steps]
            """

            def body(carry, _):
                curr_tokens, cache, rng_key = carry

                # Compute true_lengths from active_mask (captured from closure)
                true_lengths = active_mask.astype(jnp.int32)

                # Build attention mask
                bsz, seqlen = curr_tokens.shape
                mask = build_attn_mask(seqlen, cache, true_lengths)

                # Forward pass — params passed from outer scope (JIT arg, not closure capture)
                logits, updated_cache = engine.model.apply(
                    {"params": params},
                    curr_tokens,
                    true_lengths,
                    cache,
                    mask,
                )

                # Sample next tokens
                batch_logits = logits[:, 0, :]
                rng_key, subkey = random.split(rng_key)
                new_tokens = sample_fn(batch_logits, subkey)
                if new_tokens.ndim == 1:
                    new_tokens = new_tokens[:, None]

                # Only update tokens for active slots
                updated_tokens = jnp.where(
                    active_mask[:, None],
                    new_tokens,
                    curr_tokens,
                )

                # Reshard carry to match input sharding (prevents drift across scan iterations)
                updated_tokens = jax.sharding.reshard(
                    updated_tokens, jax.typeof(curr_tokens).sharding.spec
                )

                return (updated_tokens, updated_cache, rng_key), new_tokens

            # Run scan for N steps — (curr_tokens, cache, rng_key) in carry
            (final_tokens, final_cache, final_rng_key), output_tokens = jax.lax.scan(
                body, (curr_tokens, cache, rng_key), length=steps
            )

            # output_tokens shape: [steps, batch, 1]
            # Transpose to [batch, steps] and squeeze last dim
            output_tokens = output_tokens[:, :, 0].T  # [batch, steps]

            return (final_tokens, final_cache), output_tokens

        return multistep_decode_fn

    def init_decode_state(self) -> DecodeState:
        """Initialize empty decode state with all slots inactive.

        Returns:
            Fresh DecodeState with empty KV cache and all slots inactive
        """
        # Initialize empty KV cache for all slots
        kv_cache = KVCache.new(
            n_layers=self.config.n_layers,
            bsz=self.max_slots,
            max_seqlen=self.max_cache_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )
        kv_cache = self.mesh_helper.place_kv_cache(kv_cache, self.mesh)

        # Initialize tokens (doesn't matter what they are since slots are inactive)
        tokens = jnp.zeros((self.max_slots, 1), dtype=jnp.int32)
        tokens = self.mesh_helper.put_on_mesh(
            tokens,
            self.mesh,
            self.mesh_helper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
        )

        # All slots start inactive
        active_mask = jnp.zeros(self.max_slots, dtype=bool)
        active_mask = self.mesh_helper.put_on_mesh(
            active_mask,
            self.mesh,
            self.mesh_helper.batch_axis_spec(self.mesh, rank=1, batch_axis=0),
        )

        # No requests assigned
        request_ids = [None] * self.max_slots

        return DecodeState(
            kv_cache=kv_cache,
            tokens=tokens,
            active_mask=active_mask,
            request_ids=request_ids,
        )


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

        # Event loop (all processes)
        for _ in range(max_iterations):
            serving_loop.serving_step()

            # Check results (server process)
            for id, result in serving_loop.results.items():
                if result.done:
                    print(f"Completed: {result.token_list}")
    """

    def __init__(
        self,
        serve_cfg: ServingConfig,
        model: LLaMa,
        params: FrozenDict,
        mesh: Mesh,
        is_server: bool = False,
    ):
        """Initialize the serving loop.

        Args:
            serve_cfg: Serving configuration
            model: LLaMa model instance
            params: Model parameters
            mesh: JAX mesh for sharded computation
            is_server: Whether this process is the server (receives requests)
        """
        self.serve_cfg = serve_cfg
        self.model = model
        self.mesh = mesh

        # Initialize InferenceEngine for core operations
        self.engine = InferenceEngine(
            model=model,
            params=params,
            mesh=mesh,
            max_concurrent_slots=serve_cfg.decode_batch_size,
            pad_id=serve_cfg.token_pad_idx,
            sampler=serve_cfg.sampler,
            rng_seed=serve_cfg.rng_seed,
            max_cache_seqlen=serve_cfg.max_cache_seqlen,
        )

        # Setup decode work
        self.decode_work = DecodeWork(
            curr_tokens=None,
            cache=None,
            active_results=[None for _ in range(serve_cfg.decode_batch_size)],
        )

        # Initialize decode state
        decode_state = self.engine.init_decode_state()
        self.decode_work.curr_tokens = decode_state.tokens
        self.decode_work.cache = decode_state.kv_cache

        # Create multistep decode function
        self.multistep_decode_fn = self.engine.make_multistep_decode_fn()
        self._decode_call_count = 0

        # Delayed EOS detection: stores (output_tokens, output_mapping) from previous iteration
        self.decode_output = (None, None)

        # Setup prefill work
        self.prefill_work = PrefillWork([], [], [])

        # Results tracking
        self.results = {}  # request_id -> DecodeResult
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
        cfg = self.engine.config
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
            )
        print(f"[ServingLoop] Decode compiled in {_time.time() - t0:.1f}s")
        sys.stdout.flush()

        # Re-initialize decode state (discard dummy results)
        decode_state = self.engine.init_decode_state()
        self.decode_work.curr_tokens = decode_state.tokens
        self.decode_work.cache = decode_state.kv_cache

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
        cache = batched_result["cache"]

        # Slice KV cache at batch dimension
        single_cache = KVCache(
            k=cache.k[:, idx : idx + 1, :, :, :],
            v=cache.v[:, idx : idx + 1, :, :, :],
            seq_positions=cache.seq_positions[idx : idx + 1],
        )

        return {
            "cache": single_cache,
            "next_token": batched_result["next_tokens"][idx],
            "seq_length": batched_result["seq_lengths"][idx],
        }

    def _update_cache_and_index(self, batch_updates: list):
        """Batch update cache and tokens for multiple slot insertions.

        Args:
            batch_updates: List of (cache_entry, slot_idx, length, next_token) tuples
        """
        entries, batch_idxs, lens, next_tokens = map(list, zip(*batch_updates))

        # Insert all cache entries into decode cache
        for entry, slot_idx, length, next_token in zip(entries, batch_idxs, lens, next_tokens):
            new_k = self.decode_work.cache.k.at[:, slot_idx : slot_idx + 1, :, :, :].set(entry.k)
            new_v = self.decode_work.cache.v.at[:, slot_idx : slot_idx + 1, :, :, :].set(entry.v)

            self.decode_work.cache = KVCache(
                k=new_k,
                v=new_v,
                seq_positions=self.decode_work.cache.seq_positions.at[slot_idx].set(length),
            )

            self.decode_work.curr_tokens = self.decode_work.curr_tokens.at[slot_idx, 0].set(next_token)

    def _check_done_sequences(self, output_tokens: np.ndarray) -> list[bool]:
        """Check which sequences are done (EOS or max length).

        Args:
            output_tokens: Generated tokens [batch, steps]

        Returns:
            List of booleans indicating done status for each slot
        """
        # Check for EOS tokens
        done = []
        for i, result in enumerate(self.decode_work.active_results):
            if result is None:
                done.append(False)
                continue

            # Check if any token in this row matches EOS
            has_eos = np.any(output_tokens[i, :, None] == self.eos_tokens) if len(self.eos_tokens) > 0 else False
            is_max_length = result.tokens_decoded >= self.serve_cfg.max_decode_length

            done.append(has_eos or is_max_length)

        return done

    def _update_results_and_evict(self, output_tokens_flat: list[int], output_mapping_flat: list[int], done: list[bool]):
        """Update results dict with new tokens and evict completed sequences.

        Args:
            output_tokens_flat: Flattened tokens [batch * steps]
            output_mapping_flat: Flattened request ID mapping [batch * steps]
            done: Done status for each slot
        """
        # Dispatch tokens to results via output_mapping
        for token, req_id in zip(output_tokens_flat, output_mapping_flat):
            if req_id >= 0 and req_id in self.results:
                self.results[req_id].token_list.append(token)
                self.results[req_id].tokens_decoded += 1

        # Evict completed sequences and truncate at EOS
        eos_set = set(self.eos_tokens.tolist()) if len(self.eos_tokens) > 0 else set()
        for i, result in enumerate(self.decode_work.active_results):
            if result is None:
                continue
            if done[i]:
                # Truncate token_list at first EOS token
                if eos_set:
                    for j, tok in enumerate(result.token_list):
                        if tok in eos_set:
                            result.token_list = result.token_list[:j]
                            break
                result.done = True
                self.decode_work.active_results[i] = None
                print(f"[ServingLoop] Completed request {result.id} ({result.tokens_decoded} tokens)")
                sys.stdout.flush()

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
                self.decode_work.active_results[i] = DecodeResult(result.id, [])
                self.results[result.id] = self.decode_work.active_results[i]
                batch_updates.append((result.cache_entry, i, result.len, result.next_token))

            # Batch update cache and tokens
            if "worker" in self.roles and len(batch_updates) > 0:
                self._update_cache_and_index(batch_updates)
                # Re-place on mesh to match warmup sharding (prevents JIT recompilation)
                self.decode_work.cache = self.engine.mesh_helper.place_kv_cache(
                    self.decode_work.cache, self.mesh
                )
                self.decode_work.curr_tokens = self.engine.mesh_helper.put_on_mesh(
                    self.decode_work.curr_tokens,
                    self.mesh,
                    self.engine.mesh_helper.batch_axis_spec(self.mesh, rank=2, batch_axis=0),
                )

        # Phase 2: Run multistep decode (skip if all slots empty)
        active_count = sum(1 for x in self.decode_work.active_results if x is not None)
        self._log(f"decode: phase2 (active_slots={active_count})")
        if active_count == 0:
            self._log("decode: all slots empty, early return (SKIPPING phase3 sync!)")
            return

        # Build active_mask for multistep decode
        active_mask = jnp.array([result is not None for result in self.decode_work.active_results])

        # Build output_mapping: maps [batch, steps] -> request IDs for result dispatch
        output_mapping = [
            [getattr(result, "id", -1) for result in self.decode_work.active_results]
        ] * self.serve_cfg.decode_steps
        output_mapping = np.array(output_mapping).T  # [batch, steps]

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
            )
            jax.block_until_ready((final_tokens, final_cache, output_tokens))

            # Update decode work
            self.decode_work.curr_tokens = final_tokens
            self.decode_work.cache = final_cache

        self._decode_call_count += 1
        _elapsed = _time.time() - _t0
        _active = int(active_mask.sum())
        print(f"[decode #{self._decode_call_count}] {_elapsed:.3f}s for {self.serve_cfg.decode_steps} steps, "
              f"{_active}/{self.serve_cfg.decode_batch_size} active slots, "
              f"{_elapsed/self.serve_cfg.decode_steps*1000:.1f}ms/token")
        sys.stdout.flush()

        # Phase 3: Delayed EOS detection — process PREVIOUS iteration's output
        # Swap current output with stored output (allows decode kernel to run async)
        self.decode_output, (output_tokens, output_mapping) = \
            (output_tokens, output_mapping), self.decode_output

        self._log(f"decode: phase3 (has_prev_output={output_tokens is not None})")
        if output_tokens is not None:
            self._log("decode: entering barrier:decode_output")
            SyncServer.barrier("decode_output", self._it)
            if "worker" in self.roles:
                output_tokens = jax.block_until_ready(jax.experimental.multihost_utils.process_allgather(output_tokens,tiled=True))
                output_tokens = np.array(output_tokens)  # [B, steps]
                done = self._check_done_sequences(output_tokens)
                output_tokens_flat = output_tokens.reshape(-1).tolist()
                output_mapping_flat = output_mapping.reshape(-1).tolist()
            else:
                output_tokens_flat, output_mapping_flat, done = None, None, None

            # Broadcast results to all processes
            self._log(f"decode: entering broadcast:decode_tokens (done={done})")
            output_tokens_flat, output_mapping_flat, done = SyncServer.broadcast(
                "decode_tokens",
                self._it,
                (output_tokens_flat, output_mapping_flat, done),
                is_source="coordinator" in self.roles,
            )

            # Phase 4: Update results and evict completed sequences
            self._update_results_and_evict(output_tokens_flat, output_mapping_flat, done)

    def prefill_step(self):
        """One prefill iteration: batch prefill pending requests."""
        # Triage new requests (move from requests to to_prefill)
        while len(self.prefill_work.requests) > 0:
            request = self.prefill_work.requests.pop(0)
            self.prefill_work.to_prefill.append(request)

        # Process up to prefill_batch_size requests
        prefill_batch = self.prefill_work.to_prefill[: self.serve_cfg.prefill_batch_size]
        self.prefill_work.to_prefill = self.prefill_work.to_prefill[len(prefill_batch) :]

        if len(prefill_batch) == 0:
            self._log("prefill: no pending requests")
            return

        self._log(f"prefill: processing {len(prefill_batch)} requests")
        # Prepare batched inputs (pad to max length in batch)
        max_len = max(len(req.text) for req in prefill_batch)
        bucket_size = take_nearest_bucket(self.engine.buckets, max_len)

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

        # Create batched inputs
        batched_tokens = jnp.concatenate(tokens_list, axis=0)  # [bsz, bucket_size]
        batched_true_lengths = jnp.array(true_lengths_list, dtype=jnp.int32)  # [bsz]

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
                len(req.text) - 1,
            )
            self.prefill_work.to_decode.append(new_decode)

        print(f"[ServingLoop] Prefilled {len(prefill_batch)} requests")
        sys.stdout.flush()

    def _log(self, msg):
        pid = jax.process_index()
        print(f"[P{pid}|it={self._it}] {msg}")
        sys.stdout.flush()

    def serving_step(self):
        """Main event loop step (call repeatedly).

        This method coordinates all processes using SyncServer for multi-host setups.
        """
        # Sync requests from server process
        self._log("entering barrier:serving_step")
        SyncServer.barrier("serving_step", self._it)
        self._it += 1

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

        # Add new requests to prefill queue
        for req in requests or []:
            self.prefill_work.requests.append(UserRequestPrompt(**req))

        # Execute decode and prefill
        self._log("entering decode_step")
        self.decode_step()
        self._log("entering prefill_step")
        self.prefill_step()
        self._log("serving_step done")

    def add_request(self, request: UserRequestPrompt):
        """Add new request (thread-safe).

        Args:
            request: User request to add
        """
        with self.state_lock:
            self.pending_requests.append(dataclasses.asdict(request))

    def serve_forever(self, shutdown_signal: threading.Event):
        """Optional: wrap serving_step in background thread.

        Args:
            shutdown_signal: Event to signal shutdown
        """

        def serve_thread():
            try:
                while not shutdown_signal.is_set():
                    self.serving_step()
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
