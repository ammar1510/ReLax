"""
Efficient inference engine with queue-based architecture for high-throughput serving.

Supports variable-length sequence processing with continuous batching and slot management.
Non-streaming implementation - returns complete responses after full generation.
"""

import asyncio
import dataclasses
import functools
import logging
import queue
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple
from concurrent import futures
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from models.llama.model import LLaMa
from models.llama.config import ModelConfig  
from models.llama.load import load_llama_weights
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache
from sampling import TopPSampler
from utils.memory import estimate_pytree_memory_footprint, format_bytes


# Configure logging
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


@dataclass
class InferenceRequest:
    """A single inference request with completion signaling."""
    
    request_id: str
    prompt: str  
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Internal state
    tokens: Optional[jax.Array] = None
    seq_length: Optional[int] = None
    result: Optional[str] = None
    
    # Completion signaling
    complete_event: threading.Event = field(default_factory=threading.Event)
    
    # Timing metadata
    enqueue_time: Optional[float] = None
    prefill_start_time: Optional[float] = None  
    prefill_end_time: Optional[float] = None
    generation_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None
    complete_time: Optional[float] = None

    @classmethod
    def create(cls, prompt: str, max_tokens: int, temperature: float = 1.0, top_p: float = 1.0) -> "InferenceRequest":
        """Create a new inference request with unique ID."""
        return cls(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enqueue_time=time.perf_counter(),
        )


@dataclass
class GenerationSlot:
    """Represents a slot in the generation batch."""
    
    slot_id: int
    request: Optional[InferenceRequest] = None
    prefill_cache: Optional[Any] = None  # KV cache from prefill
    current_length: int = 0
    tokens_generated: int = 0
    is_complete: bool = False
    
    def is_free(self) -> bool:
        """Check if this slot is available for a new request."""
        return self.request is None
    
    def reset(self):
        """Reset slot for reuse."""
        self.request = None
        self.prefill_cache = None
        self.current_length = 0
        self.tokens_generated = 0
        self.is_complete = False


class InferenceThread(threading.Thread):
    """Thread that terminates the program if it encounters an error."""
    
    def run(self):
        try:
            super().run()
        except Exception as e:
            logging.error(f"Thread {self.name} encountered fatal error: {e}")
            traceback.print_exc()
            # Terminate the program - inference threads are critical
            os.kill(os.getpid(), signal.SIGKILL)


class InferenceEngine:
    """High-performance inference engine with queue-based architecture."""
    
    def __init__(
        self,
        model: LLaMa,
        params: Any,
        tokenizer: Tokenizer,
        max_batch_size: int = 8,
        max_prefill_length: int = 2048,
        prefill_batch_size: int = 4,
        max_prefill_wait_time: float = 0.01,
    ):
        """Initialize the inference engine.
        
        Args:
            model: The LLaMA model instance
            params: Model parameters
            tokenizer: Tokenizer instance
            max_batch_size: Maximum number of concurrent generation slots
            max_prefill_length: Maximum length for prefill sequences
            prefill_batch_size: Maximum number of requests to batch in prefill
            max_prefill_wait_time: Maximum time to wait for prefill batch (seconds)
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_prefill_length = max_prefill_length
        self.prefill_batch_size = prefill_batch_size
        self.max_prefill_wait_time = max_prefill_wait_time
        
        # Initialize generation slots
        self.slots = [GenerationSlot(i) for i in range(max_batch_size)]
        self.free_slots = queue.Queue(maxsize=max_batch_size)
        for i in range(max_batch_size):
            self.free_slots.put(i)
        
        # Queues for request processing pipeline
        self.prefill_queue = queue.Queue()
        self.transfer_queue = queue.Queue()  
        self.completion_queue = queue.Queue()
        
        # Initialize KV cache for generation batch
        self.generation_cache = KVCache.new(
            n_layers=model.args.n_layers,
            bsz=max_batch_size,
            max_seqlen=max_prefill_length + self.get_max_generation_length(),
            kv_heads=model.args.n_kv_heads,
            head_dim=model.args.head_dim,
            dtype=getattr(jnp, model.args.dtype.lower()),
        )
        
        # Initialize sampler with default parameters
        self.sampler = TopPSampler(p=0.9, temperature=1.0)
        
        # Random key for sampling
        self.rng_key = jax.random.PRNGKey(42)
        
        # Control flag
        self.is_running = True
        
        # Create and start threads
        self.threads = [
            InferenceThread(target=self._prefill_worker, name="prefill-worker", daemon=True),
            InferenceThread(target=self._transfer_worker, name="transfer-worker", daemon=True), 
            InferenceThread(target=self._generation_worker, name="generation-worker", daemon=True),
            InferenceThread(target=self._completion_worker, name="completion-worker", daemon=True),
        ]
        
        for thread in self.threads:
            thread.start()
        
        logging.info(f"InferenceEngine initialized with {max_batch_size} generation slots")
    
    def _create_empty_cache(self, batch_size: int) -> KVCache:
        """Create an empty KV cache for the given batch size."""
        return KVCache.new(
            n_layers=self.model.args.n_layers,
            bsz=batch_size,
            max_seqlen=self.max_prefill_length + self.get_max_generation_length(),
            kv_heads=self.model.args.n_kv_heads,
            head_dim=self.model.args.head_dim,
            dtype=getattr(jnp, self.model.args.dtype.lower()),
        )
    
    def _sample_next_tokens(self, logits: jax.Array) -> jax.Array:
        """Sample next tokens from batch logits using configured sampler."""
        # Update RNG key for each sampling step
        self.rng_key, subkey = jax.random.split(self.rng_key)
        
        # Use batch-aware sampling - sampler handles full batch at once
        return self.sampler.sample(logits, subkey)
    
    def _prepare_generation_batch(self, active_slots: List[GenerationSlot]) -> Tuple[jax.Array, jax.Array]:
        """Prepare input tokens and sequence lengths for generation step."""
        batch_tokens = []
        
        for slot in active_slots:
            if hasattr(slot, 'last_token') and slot.last_token is not None:
                # Use the last generated token
                batch_tokens.append([slot.last_token])
            elif hasattr(slot.request, 'first_token') and slot.request.first_token is not None:
                # Use first token from prefill
                batch_tokens.append([slot.request.first_token])
            elif slot.request.tokens is not None and len(slot.request.tokens) > 0:
                # Fallback to last token from prefill tokens
                batch_tokens.append([int(slot.request.tokens[-1])])
            else:
                # Last resort: BOS token
                batch_tokens.append([self.tokenizer.bos_id])
        
        # Convert to JAX array [batch_size, 1] - all sequences have length 1 for generation
        batch_tokens_array = jnp.array(batch_tokens)
        seq_lengths_array = jnp.ones(len(active_slots), dtype=jnp.int32)
        
        return batch_tokens_array, seq_lengths_array
    
    def get_max_generation_length(self) -> int:
        """Calculate maximum possible generation length."""
        # This is a conservative estimate - could be made configurable
        return 1024
    
    def generate(self, prompt: str, max_tokens: int, temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate text for a single prompt (blocking call).
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text completion
        """
        request = InferenceRequest.create(prompt, max_tokens, temperature, top_p)
        
        # Submit to prefill queue
        self.prefill_queue.put(request)
        
        # Block until completion
        request.complete_event.wait()
        
        if request.result is None:
            raise RuntimeError(f"Request {request.request_id} completed without result")
        
        return request.result
    
    def generate_batch(self, prompts: List[str], max_tokens: int, 
                      temperature: float = 1.0, top_p: float = 1.0) -> List[str]:
        """Generate text for multiple prompts concurrently.
        
        Args:
            prompts: List of input text prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated text completions
        """
        requests = [
            InferenceRequest.create(prompt, max_tokens, temperature, top_p)
            for prompt in prompts
        ]
        
        # Submit all requests to prefill queue
        for request in requests:
            self.prefill_queue.put(request)
        
        # Wait for all to complete
        results = []
        for request in requests:
            request.complete_event.wait()
            if request.result is None:
                raise RuntimeError(f"Request {request.request_id} completed without result")
            results.append(request.result)
        
        return results
    
    def stop(self):
        """Stop the inference engine and all worker threads."""
        logging.info("Stopping inference engine...")
        self.is_running = False
        
        # Send stop signals to all queues
        self.prefill_queue.put(None)
        self.transfer_queue.put(None)
        self.completion_queue.put(None)
        
        # Wait for threads to stop
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logging.warning(f"Thread {thread.name} did not stop gracefully")
        
        logging.info("Inference engine stopped")
    
    def _prefill_worker(self):
        """Worker thread for batched prefill processing."""
        logging.info("Prefill worker started")
        
        while self.is_running:
            try:
                # Collect a batch of requests
                batch_requests = self._collect_prefill_batch()
                
                if not batch_requests:
                    continue
                
                if len(batch_requests) == 1 and batch_requests[0] is None:
                    # Stop signal received
                    break
                
                # Process the entire batch together
                self._process_prefill_batch(batch_requests)
                
            except Exception as e:
                logging.error(f"Error in prefill worker: {e}")
                continue
        
        logging.info("Prefill worker stopped")
    
    def _collect_prefill_batch(self) -> List[InferenceRequest]:
        """Collect a batch of prefill requests with timeout."""
        batch = []
        start_time = time.perf_counter()
        
        while len(batch) < self.prefill_batch_size:
            remaining_time = self.max_prefill_wait_time - (time.perf_counter() - start_time)
            if remaining_time <= 0:
                break
            
            try:
                request = self.prefill_queue.get(timeout=remaining_time)
                if request is None:  # Stop signal
                    if not batch:  # No requests collected yet
                        return [None]
                    else:
                        # Put stop signal back and process current batch
                        self.prefill_queue.put(None)
                        break
                
                batch.append(request)
                
            except queue.Empty:
                break
        
        return batch
    
    def _process_prefill_batch(self, batch_requests: List[InferenceRequest]):
        """Process a batch of prefill requests together for high throughput."""
        if not batch_requests:
            return
        
        batch_start_time = time.perf_counter()
        
        # Tokenize all prompts in the batch
        all_tokens = []
        seq_lengths = []
        
        for request in batch_requests:
            request.prefill_start_time = batch_start_time
            
            # Tokenize the prompt
            tokens = self.tokenizer.encode(request.prompt, bos=True, eos=False, allowed_special="all")
            
            if len(tokens) > self.max_prefill_length:
                logging.warning(f"Request {request.request_id} prompt too long, truncating")
                tokens = tokens[:self.max_prefill_length]
            
            all_tokens.append(tokens)
            seq_lengths.append(len(tokens))
        
        # Pad sequences to max length in batch
        max_len = max(seq_lengths)
        padded_tokens = []
        
        for tokens in all_tokens:
            if len(tokens) < max_len:
                # Pad with tokenizer pad_id
                pad_length = max_len - len(tokens)
                padded = tokens + [self.tokenizer.pad_id] * pad_length
            else:
                padded = tokens
            padded_tokens.append(padded)
        
        # Convert to JAX arrays
        batch_tokens = jnp.array(padded_tokens)  # [batch_size, max_len]
        batch_seq_lengths = jnp.array(seq_lengths)  # [batch_size]
        
        # TODO: Process batch through model
        # batch_embeddings = self.model.token_embeddings(batch_tokens)
        # batch_kv_caches = self.model.forward_prefill(batch_embeddings, batch_seq_lengths)
        # batch_first_tokens = self.model.generate_first_tokens(batch_kv_caches)
        
        batch_end_time = time.perf_counter()
        
        # Distribute results back to individual requests
        for i, request in enumerate(batch_requests):
            request.tokens = jnp.array(all_tokens[i])
            request.seq_length = seq_lengths[i]
            request.prefill_end_time = batch_end_time
            
            # TODO: Assign actual prefill results
            # request.prefill_cache = batch_kv_caches[i] 
            # request.first_token = batch_first_tokens[i]
            
            # Move to transfer queue
            self.transfer_queue.put(request)
        
        batch_time = batch_end_time - batch_start_time
        logging.info(
            f"Processed prefill batch: {len(batch_requests)} requests, "
            f"max_len={max_len}, time={batch_time*1000:.2f}ms"
        )
    
    def _transfer_worker(self):
        """Worker thread for transferring prefilled requests to generation slots."""
        logging.info("Transfer worker started")
        
        while self.is_running:
            try:
                request = self.transfer_queue.get(timeout=1.0)
                if request is None:  # Stop signal
                    break
                
                # Wait for a free slot
                slot_id = self.free_slots.get()
                slot = self.slots[slot_id]
                
                # Assign request to slot
                slot.request = request
                slot.current_length = request.seq_length
                slot.tokens_generated = 0
                slot.is_complete = False
                
                request.generation_start_time = time.perf_counter()
                
                logging.info(f"Assigned request {request.request_id} to slot {slot_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in transfer worker: {e}")
                continue
        
        logging.info("Transfer worker stopped")
    
    def _generation_worker(self):
        """Worker thread for generation stepping."""
        logging.info("Generation worker started")
        
        step_count = 0
        
        while self.is_running:
            try:
                # Check if we have any active slots
                active_slots = [slot for slot in self.slots if not slot.is_free() and not slot.is_complete]
                
                if not active_slots:
                    time.sleep(0.01)  # No active requests, short sleep
                    continue
                
                # Perform a generation step for all active slots
                self._generation_step(active_slots, step_count)
                step_count += 1
                
                # Check for completed requests
                for slot in active_slots:
                    if slot.is_complete:
                        self.completion_queue.put(slot)
                
            except Exception as e:
                logging.error(f"Error in generation worker: {e}")
                time.sleep(0.1)
                continue
        
        logging.info("Generation worker stopped")
    
    def _generation_step(self, active_slots: List[GenerationSlot], step_count: int):
        """Perform one generation step for active slots."""
        # This is a simplified version - full implementation would need to handle
        # variable-length batching properly with our enhanced grouped_query_attention
        
        # TODO: Implement actual generation logic using:
        # - self.model with variable-length sequences
        # - self.generation_cache.update_batch()
        # - self.sampler for token selection
        # - Proper handling of EOS tokens and max_tokens
        
        # For now, just mark slots as complete after some tokens
        for slot in active_slots:
            slot.tokens_generated += 1
            slot.current_length += 1
            
            if (slot.tokens_generated >= slot.request.max_tokens or 
                slot.tokens_generated >= 10):  # Temporary limit for testing
                slot.is_complete = True
                slot.request.generation_end_time = time.perf_counter()
        
        logging.debug(f"Generation step {step_count} completed for {len(active_slots)} slots")
    
    def _completion_worker(self):
        """Worker thread for completing finished requests.""" 
        logging.info("Completion worker started")
        
        while self.is_running:
            try:
                slot = self.completion_queue.get(timeout=1.0)
                if slot is None:  # Stop signal
                    break
                
                request = slot.request
                request.complete_time = time.perf_counter()
                
                # TODO: Implement proper detokenization of generated sequence
                # For now, just return a placeholder
                request.result = f"Generated response for: {request.prompt[:50]}..."
                
                # Signal completion
                request.complete_event.set()
                
                # Free the slot
                slot.reset()
                self.free_slots.put(slot.slot_id)
                
                # Log timing information
                total_time = request.complete_time - request.enqueue_time
                prefill_time = request.prefill_end_time - request.prefill_start_time
                generation_time = request.generation_end_time - request.generation_start_time
                
                logging.info(
                    f"Completed request {request.request_id}: "
                    f"total={total_time*1000:.2f}ms, "
                    f"prefill={prefill_time*1000:.2f}ms, "
                    f"generation={generation_time*1000:.2f}ms"
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in completion worker: {e}")
                continue
        
        logging.info("Completion worker stopped")