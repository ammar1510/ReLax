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
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.sharding import Mesh, PartitionSpec as PS
from jaxlib.xla_client import NamedSharding
import numpy as np
from jax import jit

from models.llama.model import LLaMa
from utils.kvcache import KVCache
from utils.padding import take_nearest_bucket, pad_to_bucket, DEFAULT_PREFILL_BUCKETS
from utils.ops import build_attn_mask
from utils.mesh_helpers import MeshHelper


@dataclass
class InferenceRequest:
    """A single inference request from a user.

    Attributes:
        request_id: Unique identifier for this request
        prompt_tokens: Input token IDs (1D array, unpadded)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = sharper, >1.0 = more random)
        top_k: Top-k sampling parameter
        eos_token_id: Token ID that signals end of sequence
        response_queue: Queue where responses will be sent (set by orchestrator)
    """

    request_id: str
    prompt_tokens: jax.Array  # [seqlen]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    eos_token_id: int = 2
    response_queue: Optional[queue.Queue] = None


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
        prefill_procs: List[int],
        generate_procs: List[int],
        prefill_mesh: Mesh,
        generate_mesh: Mesh,
        max_concurrent_slots: int = 8,
        pad_id: int = 0,
        buckets: Optional[List[int]] = None,
    ):
        """Initialize the inference engine.

        Args:
            model: LLaMa model instance
            params: Model parameters (Flax FrozenDict) - will be placed on both meshes
            prefill_mesh: Optional mesh for prefill operations
            generate_mesh: Optional mesh for generate operations (defaults to prefill_mesh)
            max_concurrent_slots: Maximum number of sequences to generate concurrently
            pad_id: Token ID used for padding
            buckets: List of bucket sizes for prompt padding (default: power-of-2)
        """
        self.model = model
        self.max_slots = max_concurrent_slots
        self.pad_id = pad_id
        self.buckets = buckets or DEFAULT_PREFILL_BUCKETS
        self.prefill_procs = prefill_procs
        self.generate_procs = generate_procs
        self.prefill_mesh = prefill_mesh
        self.generate_mesh = generate_mesh or prefill_mesh
        detokenize_mesh = Mesh(np.array(jax.devices()), axis_names=("i"))
        self.detokenize_sharding = NamedSharding(detokenize_mesh, PS())

        # Place params on both meshes for disaggregated inference
        self.prefill_params = jax.block_until_ready(
            MeshHelper.shard_params(params, self.prefill_mesh)
        )

        self.generate_params = jax.block_until_ready(
            MeshHelper.shard_params(params, self.generate_mesh)
        )

        # Cache model config for convenience
        self.config = model.args

        # Mesh helper for sharding operations
        self.mesh_helper = MeshHelper()

        # Create separate JIT-compiled functions for prefill and generate
        @jit
        def _jitted_prefill_apply(params, tokens, true_lengths, kv_cache, mask):
            return self.model.apply(
                {"params": params},
                tokens,
                true_lengths=true_lengths,
                kv_cache=kv_cache,
                mask=mask,
            )

        @jit
        def _jitted_generate_apply(params, tokens, true_lengths, kv_cache, mask):
            return self.model.apply(
                {"params": params},
                tokens,
                true_lengths=true_lengths,
                kv_cache=kv_cache,
                mask=mask,
            )

        self._jitted_prefill_apply = _jitted_prefill_apply
        self._jitted_generate_apply = _jitted_generate_apply

        # Create jitted core function for batched prefill logic
        @jit
        def _jitted_prefill_core(params, tokens, true_lengths, kv_cache, mask):
            """Jitable core logic for batched prefill.

            Args:
                params: Model parameters
                tokens: Batched tokens [bsz, seqlen]
                true_lengths: True lengths for each sequence [bsz]
                kv_cache: KV cache for batch
                mask: Attention mask

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

            # Greedy sampling (batched)
            next_tokens = jnp.argmax(last_logits, axis=-1)  # [bsz]

            return updated_cache, next_tokens

        self._jitted_prefill_core = _jitted_prefill_core

    def transfer_prefill_to_generate(
        self, prefill_result: Dict[str, any]
    ) -> Dict[str, any]:
        if self.prefill_mesh is None or self.generate_mesh is None:
            return prefill_result
        if self.prefill_mesh is self.generate_mesh:
            return prefill_result
        transferred = dict(prefill_result)
        transferred["cache"] = self.mesh_helper.place_kv_cache(
            prefill_result["cache"], self.generate_mesh, PS()
        )
        transferred["next_token"] = self.mesh_helper.put_on_mesh(
            prefill_result["next_token"], self.generate_mesh, PS()
        )
        transferred["seq_length"] = self.mesh_helper.put_on_mesh(
            prefill_result["seq_length"], self.generate_mesh, PS()
        )
        return transferred

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
            max_seqlen=self.config.max_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )
        kv_cache = self.mesh_helper.place_kv_cache(kv_cache, self.prefill_mesh)

        tokens = self.mesh_helper.put_on_mesh(
            tokens,
            self.prefill_mesh,
            self.mesh_helper.batch_axis_spec(self.prefill_mesh, rank=2, batch_axis=0),
        )
        true_lengths = self.mesh_helper.put_on_mesh(
            true_lengths,
            self.prefill_mesh,
            self.mesh_helper.batch_axis_spec(self.prefill_mesh, rank=1, batch_axis=0),
        )

        mask = build_attn_mask(bucket_size, kv_cache, true_lengths)

        updated_cache, next_tokens = self._jitted_prefill_core(
            self.prefill_params,
            tokens,
            true_lengths,
            kv_cache,
            mask,
        )

        return {
            "cache": updated_cache,
            "next_tokens": next_tokens,  # [bsz]
            "seq_lengths": true_lengths,  # [bsz]
        }

    def generate_batch(
        self,
        decode_state: DecodeState,
    ) -> Tuple[DecodeState, jax.Array]:
        """Generate one token for ALL active slots in a batch.

        This is the core batching mechanism - all active sequences generate
        one token in a single forward pass, maximizing GPU utilization.

        Args:
            decode_state: Current decode state with active slots

        Returns:
            Tuple of:
                - Updated decode state with new tokens and incremented positions
                - New tokens for each slot [max_slots, 1]

        Note: The model.apply step is JIT-compiled internally for efficiency.
        """
        # Forward pass with all slot tokens
        # tokens shape: [max_slots, 1]
        # true_lengths represents the actual (non-padded) input length for THIS step
        # For generation, we're processing 1 token at a time
        # Only active slots should have true_length=1, inactive slots should be 0
        true_lengths = decode_state.active_mask.astype(jnp.int32)

        # Build attention mask - pass seqlen directly instead of dummy tensor
        bsz, seqlen = decode_state.tokens.shape
        mask = build_attn_mask(seqlen, decode_state.kv_cache, true_lengths)

        # Forward pass (JIT-compiled)
        logits, updated_cache = self._jitted_generate_apply(
            self.generate_params,
            decode_state.tokens,
            true_lengths,
            decode_state.kv_cache,
            mask,
        )

        # Sample next token for each slot
        # logits shape: [max_slots, 1, vocab_size]
        # Take first (and only) position: [max_slots, vocab_size]
        batch_logits = logits[:, 0, :]

        # Greedy sampling (TODO: support temperature, top-k, etc.)
        new_tokens = jnp.argmax(batch_logits, axis=-1, keepdims=True)  # [max_slots, 1]

        # Only update tokens for active slots (inactive slots keep old tokens)
        updated_tokens = jnp.where(
            decode_state.active_mask[:, None],
            new_tokens,
            decode_state.tokens,
        )

        # Create updated decode state
        updated_state = DecodeState(
            kv_cache=updated_cache,
            tokens=updated_tokens,
            active_mask=decode_state.active_mask,
            request_ids=decode_state.request_ids,
        )

        return updated_state, new_tokens

    def insert_into_slot(
        self,
        prefill_result: Dict[str, any],
        decode_state: DecodeState,
        slot_idx: int,
        request_id: str,
    ) -> DecodeState:
        """Insert a prefilled sequence into a specific slot.

        This method takes a prefilled KV cache (from prefill()) and inserts it
        into the decode batch at the specified slot position.

        Args:
            prefill_result: Output from prefill() containing cache and first token
            decode_state: Current decode state
            slot_idx: Slot index to insert into (0 to max_slots-1)
            request_id: Request ID to associate with this slot

        Returns:
            Updated decode state with the new sequence inserted at slot_idx
        """
        # Extract from prefill result
        prefill_cache = prefill_result["cache"]
        next_token = prefill_result["next_token"]
        seq_length = prefill_result["seq_length"]

        # Prefill cache has shape [layers, 1, max_seq_len, kv_heads, head_dim]
        # Need to insert it at position slot_idx in the batch dimension

        # For KV cache: insert at batch position slot_idx
        # Shape: [layers, max_slots, max_seq_len, kv_heads, head_dim]

        # Extract the single-sequence cache
        # prefill_cache.k shape: [layers, 1, max_seq_len, kv_heads, head_dim]
        new_k = decode_state.kv_cache.k.at[:, slot_idx : slot_idx + 1, :, :, :].set(
            prefill_cache.k
        )
        new_v = decode_state.kv_cache.v.at[:, slot_idx : slot_idx + 1, :, :, :].set(
            prefill_cache.v
        )

        # Update cache with new K/V and set position for this slot
        updated_cache = KVCache(
            k=new_k,
            v=new_v,
            seq_positions=decode_state.kv_cache.seq_positions.at[slot_idx].set(
                seq_length
            ),
        )

        # Update token for this slot
        updated_tokens = decode_state.tokens.at[slot_idx, 0].set(next_token)

        # Mark slot as active
        updated_active_mask = decode_state.active_mask.at[slot_idx].set(True)

        # Update request IDs
        updated_request_ids = decode_state.request_ids.copy()
        updated_request_ids[slot_idx] = request_id

        return DecodeState(
            kv_cache=updated_cache,
            tokens=updated_tokens,
            active_mask=updated_active_mask,
            request_ids=updated_request_ids,
        )

    def remove_from_slot(
        self,
        decode_state: DecodeState,
        slot_idx: int,
    ) -> DecodeState:
        """Mark a slot as inactive when its sequence finishes.

        Args:
            decode_state: Current decode state
            slot_idx: Slot index to deactivate

        Returns:
            Updated decode state with slot marked as inactive
        """
        # Mark slot as inactive
        updated_active_mask = decode_state.active_mask.at[slot_idx].set(False)

        # Clear request ID
        updated_request_ids = decode_state.request_ids.copy()
        updated_request_ids[slot_idx] = None

        return DecodeState(
            kv_cache=decode_state.kv_cache,  # Cache can stay (will be overwritten)
            tokens=decode_state.tokens,
            active_mask=updated_active_mask,
            request_ids=updated_request_ids,
        )

    def init_decode_state(self) -> DecodeState:
        """Initialize empty decode state with all slots inactive.

        Returns:
            Fresh DecodeState with empty KV cache and all slots inactive
        """
        # Initialize empty KV cache for all slots
        kv_cache = KVCache.new(
            n_layers=self.config.n_layers,
            bsz=self.max_slots,
            max_seqlen=self.config.max_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )
        kv_cache = self.mesh_helper.place_kv_cache(kv_cache, self.generate_mesh)

        # Initialize tokens (doesn't matter what they are since slots are inactive)
        tokens = jnp.zeros((self.max_slots, 1), dtype=jnp.int32)
        tokens = self.mesh_helper.put_on_mesh(
            tokens,
            self.generate_mesh,
            self.mesh_helper.batch_axis_spec(self.generate_mesh, rank=2, batch_axis=0),
        )

        # All slots start inactive
        active_mask = jnp.zeros(self.max_slots, dtype=bool)
        active_mask = self.mesh_helper.put_on_mesh(
            active_mask,
            self.generate_mesh,
            self.mesh_helper.batch_axis_spec(self.generate_mesh, rank=1, batch_axis=0),
        )

        # No requests assigned
        request_ids = [None] * self.max_slots

        return DecodeState(
            kv_cache=kv_cache,
            tokens=tokens,
            active_mask=active_mask,
            request_ids=request_ids,
        )


class InferenceOrchestrator:
    """Orchestrates async prefill and generate threads for interactive serving.

    This class manages the complete inference pipeline:
    1. Receives requests via submit()
    2. Prefill thread: Processes prompts sequentially
    3. Generate thread: Batches multiple sequences via slots
    4. Returns results via response queues

    Threading Model:
        - Main thread: Accepts requests via submit()
        - Prefill thread: Processes prompts one at a time
        - Generate thread: Batches generation across slots

    Communication:
        - Prefill queue: Main → Prefill thread
        - Generate queue: Prefill → Generate thread
        - Response queues: Generate → User (one per request)
    """

    def __init__(
        self,
        engine: InferenceEngine,
        prefill_queue_size: int = 100,
        generate_queue_size: int = 100,
        max_prefill_batch_size: int = 2,
    ):
        """Initialize the orchestrator.

        Args:
            engine: InferenceEngine instance
            prefill_queue_size: Maximum size of prefill queue
            generate_queue_size: Maximum size of generate queue
            max_prefill_batch_size: Maximum number of requests to batch for prefill (fixed batch size)
        """
        self.engine = engine
        self.max_prefill_batch_size = max_prefill_batch_size

        # Request queues
        self._prefill_queue = queue.Queue(prefill_queue_size)
        self._transfer_backlog = queue.Queue(generate_queue_size)
        self._detokenize_backlog = queue.Queue(100)

        # Slot tracking - queue of available slot indices
        self._free_slots = queue.Queue(engine.max_slots)
        for i in range(engine.max_slots):
            self._free_slots.put(i)

        # Thread control
        self._running = False
        self._prefill_thread = None
        self._generate_thread = None
        self._detokenize_thread = None

    def start(self):
        """Start prefill and generate threads.

        This must be called before submitting requests.
        """
        if self._running:
            raise RuntimeError("Orchestrator is already running")

        self._running = True

        # Start threads
        self._prefill_thread = threading.Thread(
            target=self._prefill_loop,
            name="prefill-thread",
            daemon=True,
        )
        self._generate_thread = threading.Thread(
            target=self._generate_loop,
            name="generate-thread",
            daemon=True,
        )
        self._detokenize_thread = threading.Thread(
            target=self._detokenize_loop,
            name="detokenize-thread",
            daemon=True,
        )

        self._prefill_thread.start()
        self._generate_thread.start()
        self._detokenize_thread.start()

        print(f"[Orchestrator] Started with {self.engine.max_slots} slots")
        sys.stdout.flush()

    def stop(self, timeout: float = 5.0):
        """Stop all threads gracefully.

        Args:
            timeout: Maximum time to wait for threads to finish (seconds)
        """
        if not self._running:
            return

        print("[Orchestrator] Stopping...")
        sys.stdout.flush()

        self._running = False

        # Send shutdown signal to detokenize thread
        self._detokenize_backlog.put(None)

        # Wait for threads to finish
        self._prefill_thread.join(timeout=timeout)
        self._generate_thread.join(timeout=timeout)
        self._detokenize_thread.join(timeout=timeout)

        print("[Orchestrator] Stopped")
        sys.stdout.flush()

    def submit(self, request: InferenceRequest) -> queue.Queue:
        """Submit a request for inference and get a response queue.

        Args:
            request: InferenceRequest to process

        Returns:
            Queue that will receive response messages:
                - {"status": "generating", "token": <int>} for each token
                - {"status": "complete", "tokens": <list>, "request_id": <str>} when done

        Example:
            >>> request = InferenceRequest("req1", prompt_tokens, max_new_tokens=50)
            >>> response_queue = orchestrator.submit(request)
            >>> while True:
            ...     result = response_queue.get()
            ...     if result["status"] == "complete":
            ...         print(f"Generated: {result['tokens']}")
            ...         break
        """
        if not self._running:
            raise RuntimeError("Orchestrator is not running. Call start() first.")

        # Create response queue for this request
        response_queue = queue.Queue()
        request.response_queue = response_queue

        # Submit to prefill queue
        self._prefill_queue.put(request)

        return response_queue

    def _extract_individual_result(self, batched_result: Dict, idx: int) -> Dict:
        """Extract single-sequence result from batched prefill output.

        Args:
            batched_result: Dict with batched cache, next_tokens, seq_lengths
            idx: Index of sequence to extract

        Returns:
            Dict with single-sequence cache matching original prefill() output format
        """
        cache = batched_result["cache"]

        # Slice KV cache at batch dimension (axis 1)
        # cache.k: [layers, bsz, kv_heads, max_seqlen, head_dim]
        # Extract: [layers, 1, kv_heads, max_seqlen, head_dim]
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

    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests, padding to longest sequence length.

        Args:
            requests: List of InferenceRequests to process together
        """
        print(f"[Prefill] Processing batch of {len(requests)} requests")
        sys.stdout.flush()
        bsz = len(requests)

        # Find max length in batch
        max_length = max(len(req.prompt_tokens) for req in requests)

        # Find appropriate bucket for max length
        bucket_size = take_nearest_bucket(self.engine.buckets, max_length)

        # Stack tokens and true_lengths
        tokens_list = []
        true_lengths_list = []

        for req in requests:
            tokens_with_batch = req.prompt_tokens[None, :]  # [1, seqlen]
            padded = pad_to_bucket(tokens_with_batch, bucket_size, self.engine.pad_id)
            tokens_list.append(padded)
            true_lengths_list.append(len(req.prompt_tokens))

        # Create batched inputs
        batched_tokens = jnp.concatenate(tokens_list, axis=0)  # [bsz, bucket_size]
        batched_true_lengths = jnp.array(true_lengths_list, dtype=jnp.int32)  # [bsz]

        # Call batched prefill
        prefill_result = self.engine.prefill(batched_tokens, batched_true_lengths)
        print("[Prefill] Completed, sending to transfer backlog")
        sys.stdout.flush()

        # Unpack and send to transfer backlog
        for i, req in enumerate(requests):
            individual_result = self._extract_individual_result(prefill_result, i)
            individual_result["request"] = req
            self._transfer_backlog.put(individual_result)
        print("[Prefill] sent to transfer backlog.")
        sys.stdout.flush()

    def _prefill_loop(self):
        """Prefill thread: batch requests, padding to longest sequence."""
        print("[Prefill Thread] Started (batched mode)")
        sys.stdout.flush()

        pending_requests = []

        while self._running:
            # Phase 1: Try to get new request (non-blocking)
            try:
                request = self._prefill_queue.get(timeout=0.001)
                pending_requests.append(request)
            except queue.Empty:
                pass

            # Phase 2: Process batch when full
            if len(pending_requests) >= self.max_prefill_batch_size:
                try:
                    self._process_batch(pending_requests)
                except Exception as e:
                    # Send errors to individual response queues
                    for req in pending_requests:
                        req.response_queue.put(
                            {
                                "status": "error",
                                "error": str(e),
                                "request_id": req.request_id,
                            }
                        )

                # Clear processed batch
                pending_requests = []

            # Small sleep to prevent busy waiting
            if len(pending_requests) == 0:
                time.sleep(0.0001)

        print("[Prefill Thread] Stopped")
        sys.stdout.flush()

    def _generate_loop(self):
        """Generate thread: insert requests and unconditional generation."""
        print("[Generate Thread] Started")
        sys.stdout.flush()

        # Initialize decode state
        decode_state = self.engine.init_decode_state()

        # Initialize generate timestep counter
        generate_timestep = 0
        while self._running:
            # PHASE 1: Insert one request from transfer_backlog (if available and slot free)
            try:
                slot_idx = self._free_slots.get(block=False)
            except queue.Empty:
                slot_idx = None

            if slot_idx is not None:
                # Determine blocking behavior (JetStream pattern)
                # Block if ALL slots are inactive (ensures at least one request before generate)
                # Use int() to convert to Python scalar for comparison

                try:
                    prefill_result = self._transfer_backlog.get(block=True, timeout=1.0)

                    # CRITICAL: Transfer to generate mesh before insertion
                    prefill_result = self.engine.transfer_prefill_to_generate(
                        prefill_result
                    )
                    jax.block_until_ready(prefill_result)

                    request = prefill_result["request"]

                    if jax.process_index() in self.engine.generate_procs:
                        first_token = prefill_result["next_token"].item()
                    else:
                        first_token = None

                    # Insert into decode state
                    decode_state = self.engine.insert_into_slot(
                        prefill_result,
                        decode_state,
                        slot_idx,
                        request.request_id,
                    )
                    print(
                        f"[Generate] Inserted request '{request.request_id}' into slot {slot_idx}"
                    )
                    sys.stdout.flush()

                    # Send slot assignment to detokenize thread
                    # Message format: (slot_idx, request, first_token)
                    self._detokenize_backlog.put(
                        (slot_idx, request, first_token), block=True
                    )

                except queue.Empty:
                    # Put slot back if no request available
                    self._free_slots.put(slot_idx)
                    # Fall through to generate even if we timed out
                    # This ensures all devices execute generate_batch() together

            # PHASE 2: ALWAYS generate (no conditional) - ensures device sync
            decode_state, new_tokens = self.engine.generate_batch(decode_state)
            new_tokens_cpu = None
            if jax.process_index() in self.engine.generate_procs:
                new_tokens_gathered = MeshHelper.allgather(new_tokens, self.engine.generate_mesh)
                jax.block_until_ready(new_tokens_gathered)
                new_tokens_cpu = jax.device_get(new_tokens_gathered)

            # PHASE 3: Send generated tokens to detokenize thread
            # Message format: (generate_timestep, new_tokens)
            self._detokenize_backlog.put(
                (generate_timestep, new_tokens_cpu), block=True
            )
            generate_timestep += 1

        print("[Generate Thread] Stopped")
        sys.stdout.flush()

    def _detokenize_loop(self):
        """Detokenize thread: Process tokens, send responses, free slots."""
        print("[Detokenize Thread] Started")
        sys.stdout.flush()

        # Track active requests: slot_idx -> (request, generated_tokens_list)
        active_requests = {}

        while self._running:
            try:
                data = self._detokenize_backlog.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            if data is None:  # Shutdown signal
                break

            # Case 1: Slot assignment - (slot_idx, request, first_token)
            # Detect by checking if data[0] is an integer (slot index)
            if isinstance(data[0], int) and len(data) == 3:
                slot_idx, request, first_token = data
                print(
                    f"[Detokenize] Received first token for '{request.request_id}' in slot {slot_idx}"
                )
                sys.stdout.flush()

                # Initialize tracking for this slot
                active_requests[slot_idx] = (request, [first_token])

                # Send first token to user
                request.response_queue.put(
                    {"status": "generating", "token": first_token}
                )
                continue

            # Case 2: Generate step - (generate_timestep, new_tokens)
            # Detect by checking if data[0] is int and data[1] is jax.Array
            if isinstance(data[0], int) and len(data) == 2:
                generate_timestep, new_tokens = data

                # Process each active slot
                finished_slots = []

                for slot_idx, (request, tokens) in list(active_requests.items()):
                    if jax.process_index() in self.engine.generate_procs:
                        token = new_tokens[slot_idx, 0].item()
                    else:
                        token = None
                    tokens.append(token)

                    # Check for completion
                    is_eos = token == request.eos_token_id
                    is_max_length = len(tokens) >= request.max_new_tokens

                    if is_eos or is_max_length:
                        # Send final response
                        request.response_queue.put(
                            {
                                "status": "complete",
                                "tokens": tokens,
                                "request_id": request.request_id,
                                "finish_reason": "eos" if is_eos else "length",
                            }
                        )
                        print(
                            f"[Detokenize] Completed '{request.request_id}' ({len(tokens)} tokens, {'EOS' if is_eos else 'max length'})"
                        )
                        sys.stdout.flush()
                        finished_slots.append(slot_idx)
                    else:
                        # Send streaming update
                        request.response_queue.put(
                            {"status": "generating", "token": token}
                        )

                # Free finished slots
                for slot_idx in finished_slots:
                    del active_requests[slot_idx]
                    self._free_slots.put(slot_idx)
                    print(f"[Detokenize] Freed slot {slot_idx}")
                    sys.stdout.flush()

        print("[Detokenize Thread] Stopped")
        sys.stdout.flush()
