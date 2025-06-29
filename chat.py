from dataclasses import dataclass
from functools import partial
from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
    Optional,
    Dict,
    Any,
    Callable,
    Tuple,
)

import jax
import jax.numpy as jnp

from models.llama.model import LLaMA
from models.llama.tokenizer import Tokenizer
from utils.kvcache import KVCache
from sampling import Sampler, GreedySampler

# --- Type Definitions ---

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

# --- State Management ---


@dataclass
class ChatState:
    """Holds the state of a single, ongoing conversation."""

    dialog_history: Dialog
    kv_cache: KVCache
    sequence_length: int = 0


# --- Core Chat Engine ---


class ChatEngine:
    """
    A stateless engine for driving conversational chat with a Llama model.
    It operates on ChatState objects and is designed to be JIT-friendly.
    """

    def __init__(
        self,
        model: LLaMA,
        params: Dict,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
        cache_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_dtype = cache_dtype

        # --- Pre-compile JAX functions ---
        self._jitted_model_step = partial(self._model_step, self)

        if jax.default_backend() != "cpu":
            self._jitted_model_step = jax.jit(
                self._jitted_model_step, static_argnames=("self",)
            )

    def _create_kv_cache(self) -> KVCache:
        """Helper to initialize a new, empty KVCache."""
        return KVCache(
            batch_size=1,  # Chat is always batch size 1
            max_seq_len=self.max_seq_len,
            n_layers=self.model.args.n_layers,
            n_kv_heads=self.model.args.n_kv_heads,
            head_dim=self.model.args.head_dim,
            dtype=self.cache_dtype,
        )

    def start_new_conversation(self, system_prompt: Optional[str] = None) -> ChatState:
        """
        Creates and returns a new ChatState, warming up the KVCache with either
        a system prompt or just the BOS token.
        """
        kv_cache = self._create_kv_cache()
        initial_dialog: Dialog = []

        if system_prompt:
            # A system prompt is provided, so pre-process it.
            initial_dialog.append({"role": "system", "content": system_prompt})
            system_message_tokens = self._encode_message(initial_dialog[0])
            prompt_tokens = [self.tokenizer.bos_id] + system_message_tokens
        else:
            # No system prompt, just start with the BOS token.
            prompt_tokens = [self.tokenizer.bos_id]

        prompt_jnp = jnp.array([prompt_tokens], dtype=jnp.int32)

        # Run the model to "warm up" the cache with the initial tokens.
        _, updated_kv_cache = self._jitted_model_step(
            tokens=prompt_jnp, kv_cache=kv_cache, start_pos=0
        )

        return ChatState(
            dialog_history=initial_dialog,  # Will be empty if no system prompt
            kv_cache=updated_kv_cache,
            sequence_length=prompt_jnp.shape[1],
        )

    def _model_step(
        self,
        tokens: jnp.ndarray,
        kv_cache: KVCache,
        start_pos: int,
    ) -> Tuple[jnp.ndarray, KVCache]:
        """
        Processes a sequence of tokens and returns the logits for the *next* token
        along with the updated KVCache.
        """
        logits, updated_kv_cache = self.model.apply(
            {"params": self.params}, tokens, start_pos=start_pos, kv_cache=kv_cache
        )
        # The logits for the next token are always at the last position of the sequence.
        return logits[:, -1, :], updated_kv_cache

    def _encode_message(self, message: Message) -> List[int]:
        """Encodes a single message with Llama 3 header, content, and end-of-turn."""
        tokens = []
        # Add header start token
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # Add role tokens (no BOS/EOS)
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # Add header end token and newline
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        # Add content tokens (no BOS/EOS)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        # Add end-of-turn token
        tokens.append(self.tokenizer.eot_id)
        return tokens

    def generate_response(
        self,
        user_message: str,
        state: ChatState,
        sampler: Sampler,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[str, ChatState]:
        """
        Generates a model response to a user message, given a conversation state.
        This implementation is optimized to only process new tokens.
        """
        new_user_message: Message = {"role": "user", "content": user_message}
        updated_dialog = state.dialog_history + [new_user_message]

        # 2. Determine the tokens to process. This includes the new user message
        # and the header for the assistant's turn, which prompts the model.
        user_tokens = self._encode_message(new_user_message)
        assistant_prompt_tokens = (
            [self.tokenizer.special_tokens["<|start_header_id|>"]]
            + self.tokenizer.encode("assistant", bos=False, eos=False)
            + [self.tokenizer.special_tokens["<|end_header_id|>"]]
            + self.tokenizer.encode("\n\n", bos=False, eos=False)
        )
        tokens_to_process = user_tokens + assistant_prompt_tokens
        tokens_for_step = jnp.array([tokens_to_process], dtype=jnp.int32)

        # 3. Initialize generation state
        current_seq_len = state.sequence_length
        kv_cache = state.kv_cache
        generated_tokens = []

        # 4. Autoregressive generation loop
        while current_seq_len < self.max_seq_len:
            # Prevent generating past the model's maximum sequence length
            if current_seq_len + tokens_for_step.shape[1] > self.max_seq_len:
                print(f"Warning: max sequence length ({self.max_seq_len}) reached.")
                break

            # Run the model for one step. On the first iteration, this processes
            # the entire prompt. On subsequent iterations, it processes a single token.
            next_token_logits, kv_cache = self._jitted_model_step(
                tokens=tokens_for_step,
                kv_cache=kv_cache,
                start_pos=current_seq_len,
            )
            current_seq_len += tokens_for_step.shape[1]

            # Sample the next token from the logits.
            rng_key, sample_key = jax.random.split(rng_key)
            next_token = sampler.sample(next_token_logits, sample_key)

            # Stop if the model produces an end-of-turn token.
            if next_token.item() in [self.tokenizer.eot_id, self.tokenizer.eos_id]:
                break

            # Add the generated token to the list and prepare the next token for the next iteration.
            generated_tokens.append(next_token.item())
            tokens_for_step = next_token.reshape(1, 1)

        # 5. Decode response and finalize state
        response_text = self.tokenizer.decode(generated_tokens)
        final_dialog = updated_dialog + [
            {"role": "assistant", "content": response_text}
        ]

        new_state = ChatState(
            dialog_history=final_dialog,
            kv_cache=kv_cache,
            sequence_length=current_seq_len,
        )

        return response_text, new_state


# --- Example Usage ---
if __name__ == "__main__":
    # This section is for conceptual demonstration and testing.
    # It requires a mock or real Llama model, Tokenizer, and params.

    # --- Mockups for demonstration ---
    class MockModelArgs:
        n_layers = 2
        n_kv_heads = 2
        head_dim = 64
        vocab_size = 128256

    class MockLlamaModel:
        def __init__(self):
            self.args = MockModelArgs()

        def apply(self, params_dict, tokens, start_pos, kv_cache):
            mock_vocab_size = self.args.vocab_size
            logits = (
                jnp.ones((tokens.shape[0], tokens.shape[1], mock_vocab_size)) * -10.0
            )
            logits = logits.at[:, :, 5].set(10.0)  # Make token '5' likely
            return logits, kv_cache

    class MockTokenizer:
        def __init__(self):
            self.bos_id = 128000
            self.eos_id = 128001
            self.pad_id = 0
            self.eot_id = 128009
            self.special_tokens = {
                "<|start_header_id|>": 128006,
                "<|end_header_id|>": 128007,
            }
            self.vocab_size = 128256

        def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
            tokens = [ord(c) for c in text[:10]]
            return tokens

        def decode(self, token_ids: List[int]) -> str:
            # Mock decoding, ignoring special tokens for simplicity in output
            return "".join([f"<T{tid}>" for tid in token_ids])

    # --- Setup Mocks ---
    mock_model_instance = MockLlamaModel()
    mock_tokenizer_instance = MockTokenizer()
    mock_params = {}

    # --- Test ChatEngine ---
    engine = ChatEngine(
        model=mock_model_instance,
        tokenizer=mock_tokenizer_instance,
        params=mock_params,
        max_seq_len=128,
    )

    print("--- Starting Conversation 1 (Pirate Bot) ---")
    # Start a new conversation for a pirate bot
    pirate_state = engine.start_new_conversation(system_prompt="You are a pirate.")

    # User sends a message
    print("User: Ahoy there!")
    rng_key = jax.random.PRNGKey(0)
    pirate_response, pirate_state = engine.generate_response(
        user_message="Ahoy there!",
        state=pirate_state,
        sampler=GreedySampler(),
        rng_key=rng_key,
    )
    print(f"Pirate Bot: {pirate_response}")
    print(f"Dialog History: {pirate_state.dialog_history}")
    print(f"Sequence Length: {pirate_state.sequence_length}\n")

    # --- Starting Conversation 2 (Poet Bot) ---
    print("--- Starting Conversation 2 (Poet Bot) ---")
    poet_state = engine.start_new_conversation(system_prompt="You are a poet.")

    # User sends a message
    print("User: Tell me a short poem.")
    rng_key, poet_key = jax.random.split(rng_key)
    poet_response, poet_state = engine.generate_response(
        user_message="Tell me a short poem.",
        state=poet_state,
        sampler=GreedySampler(),
        rng_key=poet_key,
    )
    print(f"Poet Bot: {poet_response}")
    print(f"Dialog History: {poet_state.dialog_history}")
    print(f"Sequence Length: {poet_state.sequence_length}\n")

    # --- Continuing Conversation 1 ---
    print("--- Continuing Conversation 1 (Pirate Bot) ---")
    print("User: Where's the treasure?")
    rng_key, pirate_key_2 = jax.random.split(rng_key)
    pirate_response_2, pirate_state = engine.generate_response(
        user_message="Where's the treasure?",
        state=pirate_state,  # Pass the *updated* pirate_state
        sampler=GreedySampler(),
        rng_key=pirate_key_2,
    )
    print(f"Pirate Bot: {pirate_response_2}")
    print(f"Final Pirate Dialog: {pirate_state.dialog_history}")
    print(f"Final Pirate Sequence Length: {pirate_state.sequence_length}")
