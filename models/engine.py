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

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from models.llama.model import LLaMa
from utils.kvcache import KVCache
from utils.padding import take_nearest_bucket, pad_to_bucket, DEFAULT_PREFILL_BUCKETS
from utils.ops import build_attn_mask
from functools import partial
from jax import jit


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
        max_concurrent_slots: int = 8,
        pad_id: int = 0,
        buckets: Optional[List[int]] = None,
    ):
        """Initialize the inference engine.

        Args:
            model: LLaMa model instance
            params: Model parameters (Flax FrozenDict)
            max_concurrent_slots: Maximum number of sequences to generate concurrently
            pad_id: Token ID used for padding
            buckets: List of bucket sizes for prompt padding (default: power-of-2)
        """
        self.model = model
        self.params = params
        self.max_slots = max_concurrent_slots
        self.pad_id = pad_id
        self.buckets = buckets or DEFAULT_PREFILL_BUCKETS

        # Cache model config for convenience
        self.config = model.args

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

    def prefill(
        self,
        tokens: jax.Array,  # [1, bucket_size]
        true_length: int,
    ) -> Dict[str, any]:
        """Process a prompt and return prefilled state.

        This method processes the entire prompt in a single forward pass,
        populating the KV cache and sampling the first generated token.

        Args:
            tokens: Padded prompt tokens of shape [1, bucket_size]
            true_length: Actual (non-padded) length of the prompt

        Returns:
            Dictionary with:
                - 'cache': Populated KVCache for this sequence
                - 'next_token': First generated token (sampled from position true_length-1)
                - 'seq_length': Current sequence length (= true_length)

        Note: The model.apply step is JIT-compiled internally for efficiency.
        """
        # Initialize cache for single sequence
        kv_cache = KVCache.new(
            n_layers=self.config.n_layers,
            bsz=1,
            max_seqlen=self.config.max_seqlen,
            kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=jnp.dtype(self.config.dtype),
        )

        # Create true_lengths array
        true_lengths = jnp.array([true_length], dtype=jnp.int32)

        # Build attention mask - pass seqlen directly instead of dummy tensor
        bsz, seqlen = tokens.shape
        mask = build_attn_mask(seqlen, kv_cache, true_lengths)

        # Forward pass through entire prompt (JIT-compiled)
        logits, updated_cache = self._jitted_prefill_apply(
            self.params,
            tokens,
            true_lengths,
            kv_cache,
            mask,
        )

        # Extract logits at the last real token position
        # logits shape: [1, seqlen, vocab_size]
        last_logits = logits[0, true_length - 1]  # [vocab_size]

        # Sample next token (greedy for now - argmax)
        next_token = jnp.argmax(last_logits)

        return {
            "cache": updated_cache,
            "next_token": next_token,
            "seq_length": true_length,
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
            self.params,
            decode_state.tokens,
            true_lengths,
            decode_state.kv_cache,
            mask,
        )

        # Sample next token for each slot
        # logits shape: [max_slots, 1, vocab_size]
        # Take first (and only) position: [max_slots, vocab_size]
        batch_logits = logits[:, 0, :]

        # # Debug: Print cache positions to diagnose repetition
        # print(f"[DEBUG] Cache positions before: {decode_state.kv_cache.seq_positions}")
        # print(f"[DEBUG] Cache positions after: {updated_cache.seq_positions}")
        # print(f"[DEBUG] Tokens being fed: {decode_state.tokens.flatten()[:5]}")
        # print(f"[DEBUG] Top 5 logits: {batch_logits[0, :5]}")

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
        layers, max_slots, max_seq_len, kv_heads, head_dim = (
            decode_state.kv_cache.k.shape
        )

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

        # Initialize tokens (doesn't matter what they are since slots are inactive)
        tokens = jnp.zeros((self.max_slots, 1), dtype=jnp.int32)

        # All slots start inactive
        active_mask = jnp.zeros(self.max_slots, dtype=bool)

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
    ):
        """Initialize the orchestrator.

        Args:
            engine: InferenceEngine instance
            prefill_queue_size: Maximum size of prefill queue
            generate_queue_size: Maximum size of generate queue
        """
        self.engine = engine

        # Request queues
        self._prefill_queue = queue.Queue(prefill_queue_size)
        self._generate_queue = queue.Queue(generate_queue_size)

        # Slot tracking - queue of available slot indices
        self._free_slots = queue.Queue(engine.max_slots)
        for i in range(engine.max_slots):
            self._free_slots.put(i)

        # Thread control
        self._running = False
        self._prefill_thread = None
        self._generate_thread = None

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

        self._prefill_thread.start()
        self._generate_thread.start()

        print(f"[Orchestrator] Started with {self.engine.max_slots} slots")

    def stop(self, timeout: float = 5.0):
        """Stop all threads gracefully.

        Args:
            timeout: Maximum time to wait for threads to finish (seconds)
        """
        if not self._running:
            return

        print("[Orchestrator] Stopping...")
        self._running = False

        # Wait for threads to finish
        self._prefill_thread.join(timeout=timeout)
        self._generate_thread.join(timeout=timeout)

        print("[Orchestrator] Stopped")

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

    def _prefill_loop(self):
        """Prefill thread: process prompts one at a time."""
        print("[Prefill Thread] Started")

        while self._running:
            try:
                # Get request from prefill queue (with timeout to check _running)
                request = self._prefill_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Pad prompt to bucket
                prompt_tokens = request.prompt_tokens
                true_length = len(prompt_tokens)
                bucket_size = take_nearest_bucket(self.engine.buckets, true_length)

                # Add batch dimension and pad
                tokens_with_batch = prompt_tokens[None, :]  # [1, seqlen]
                padded_tokens = pad_to_bucket(
                    tokens_with_batch, bucket_size, self.engine.pad_id
                )

                # Run prefill
                prefill_result = self.engine.prefill(padded_tokens, true_length)

                # Add request metadata
                prefill_result["request"] = request

                # Send to generate queue
                self._generate_queue.put(prefill_result)

            except Exception as e:
                # Send error to response queue
                request.response_queue.put(
                    {
                        "status": "error",
                        "error": str(e),
                        "request_id": request.request_id,
                    }
                )

        print("[Prefill Thread] Stopped")

    def _generate_loop(self):
        """Generate thread: batch multiple sequences via slots."""
        print("[Generate Thread] Started")

        # Initialize decode state
        decode_state = self.engine.init_decode_state()

        # Track active requests: slot_idx -> (request, generated_tokens)
        active_requests: Dict[int, Tuple[InferenceRequest, List[int]]] = {}

        while self._running:
            # PHASE 1: Fill free slots with new prefilled requests
            try:
                while not self._free_slots.empty():
                    # Try to get a free slot
                    slot_idx = self._free_slots.get(block=False)

                    try:
                        # Try to get a prefilled request
                        prefill_result = self._generate_queue.get(block=False)

                        # Insert into slot
                        request = prefill_result["request"]
                        decode_state = self.engine.insert_into_slot(
                            prefill_result,
                            decode_state,
                            slot_idx,
                            request.request_id,
                        )

                        # Track this request
                        first_token = prefill_result["next_token"].item()
                        active_requests[slot_idx] = (request, [first_token])

                        # Send first token to response queue
                        request.response_queue.put(
                            {
                                "status": "generating",
                                "token": first_token,
                            }
                        )

                    except queue.Empty:
                        # No prefilled requests available, put slot back
                        self._free_slots.put(slot_idx)
                        break
            except queue.Empty:
                # No free slots
                pass

            # PHASE 2: Generate one token for all active slots
            if jnp.any(decode_state.active_mask):
                try:
                    decode_state, new_tokens = self.engine.generate_batch(decode_state)

                    # Process results for each active slot
                    finished_slots = []

                    for slot_idx, (request, tokens) in list(active_requests.items()):
                        if decode_state.active_mask[slot_idx]:
                            token = new_tokens[slot_idx, 0].item()
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
                                        "finish_reason": (
                                            "eos" if is_eos else "length"
                                        ),
                                    }
                                )
                                finished_slots.append(slot_idx)
                            else:
                                # Send streaming update
                                request.response_queue.put(
                                    {
                                        "status": "generating",
                                        "token": token,
                                    }
                                )

                    # Remove finished sequences and free slots
                    for slot_idx in finished_slots:
                        decode_state = self.engine.remove_from_slot(
                            decode_state, slot_idx
                        )
                        del active_requests[slot_idx]
                        self._free_slots.put(slot_idx)

                except Exception as e:
                    print(f"[Generate Thread] Error during generation: {e}")
            else:
                time.sleep(0.001)

        print("[Generate Thread] Stopped")
