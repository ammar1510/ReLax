# Placeholder for Chat Interface

from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
    Optional,
    Dict,
    Any,
    Callable,
)

# Import the Tokenizer class (assuming it stays in tokenizer_utils.py)
from tokenizer import Tokenizer

# Define types for chat messages, needed by ChatFormat
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class ChatFormat:
    """Formats chat prompts according to Llama 3's expected structure."""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        """Encodes the header part of a message (role)."""
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"],)
        # Encode role string - ensure no BOS/EOS added here
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"],)
        # Encode the double newline after header - ensure no BOS/EOS added here
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        """Encodes a single message (header + content + end of turn)."""
        tokens = self.encode_header(message)
        # Encode content string - ensure no BOS/EOS added here
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        # Add the end-of-turn token
        tokens.append(self.tokenizer.eot_id)
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        """Encodes a full dialog into a single list of token IDs, ready for the model."""
        tokens = []
        # Must start with BOS token
        tokens.append(self.tokenizer.bos_id)
        for message in dialog:
            tokens.extend(self.encode_message(message))

        # Add the start of an assistant message header for the model to complete.
        # The actual generation picks up from here.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


from typing import List, Optional, Dict, Any # Added Any for placeholder if needed
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from models.llama.model import LLaMA
from utils.kvcache import KVCache 
from load_utils import FlaxModelParams 
from sampling import Sampler, GreedySampler, CategoricalSampler, TopKSampler, TopPSampler

class ConversationChat:
    """
    Manages a conversation with the Llama model, including dialog history,
    KVCache, and token generation.
    """
    def __init__(self, 
                 model: LLaMA, 
                 tokenizer: Tokenizer,
                 chat_formatter: ChatFormat, # Use the ChatFormat protocol
                 params: dict, 
                 chat_max_seq_len: int = 2048, # Default max sequence length for chat
                 cache_dtype: jnp.dtype = jnp.bfloat16 # Default dtype for KVCache
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_formatter = chat_formatter # Store the formatter
        self.params = params
        self.chat_max_seq_len = chat_max_seq_len # Max sequence length for KVCache
        self.cache_dtype = cache_dtype # Store dtype for KVCache re-initialization
        
        self.dialog: List[Dict[str, str]] = []
        
        # KVCache and sequence length will be initialized/reset by start_new_conversation
        self.current_kv_cache: Optional[KVCache] = None
        self.current_sequence_length: int = 0
        self.pending_next_token_logits: Optional[jnp.ndarray] = None


        # Initialize KVCache upon creation for the first conversation
        self.start_new_conversation()


    # def _initialize_kv_cache(self):
    #     """Initializes or re-initializes the KVCache."""
    #     # This method is now part of start_new_conversation
    #     # batch_size is 1 for chat inference
    #     # n_layers, n_kv_heads, head_dim are from model.args
    #     # dtype is now passed via constructor
    #     self.current_kv_cache = KVCache(
    #         batch_size=1, # Standard for chat
    #         max_seq_len=self.chat_max_seq_len, 
    #         n_layers=self.model.args.n_layers,
    #         n_kv_heads=self.model.args.n_kv_heads,
    #         head_dim=self.model.args.head_dim,
    #         dtype=self.cache_dtype # USE stored dtype
    #     )
    #     self.current_sequence_length = 0

    def get_dialog_string(self) -> str:
        """Returns the current dialog as a formatted string."""
        # This might be too simplistic. The chat_formatter should ideally handle this.
        # For now, let's assume the formatter is used before tokenization.
        # If a human-readable string is needed, the formatter might offer a method.
        # This is a placeholder for how one might view the dialog.
        # For actual model input, self.chat_formatter.encode_dialog_prompt(self.dialog) is key.
        
        # Simple concatenation for display, not for model processing
        full_dialog = ""
        for message in self.dialog:
            full_dialog += f"{message['role'].capitalize()}: {message['content']}\\n"
        return full_dialog.strip()

    def clear_dialog(self):
        """Clears the dialog history and resets the KVCache."""
        # This is effectively what start_new_conversation does now
        self.start_new_conversation()


    def start_new_conversation(self, 
                               system_prompt_text: Optional[str] = None,
                               first_user_message_text: Optional[str] = None):
        """
        Starts a new conversation, optionally with a system prompt and a first user message.
        Resets dialog, KVCache, sequence length, and pending logits.
        Processes initial prompts if provided.
        """
        self.dialog = [] 
        if system_prompt_text:
            self.dialog.append({"role": "system", "content": system_prompt_text})
        if first_user_message_text:
            self.dialog.append({"role": "user", "content": first_user_message_text})
        
        self.current_kv_cache = KVCache(
            batch_size=1,
            max_seq_len=self.chat_max_seq_len, 
            n_layers=self.model.args.n_layers,
            n_kv_heads=self.model.args.n_kv_heads,
            head_dim=self.model.args.head_dim,
            dtype=self.cache_dtype
        )
        self.current_sequence_length = 0
        self.pending_next_token_logits = None

        if self.dialog: # If there's any initial prompt (system or user)
            error_message = self._process_prompt_and_update_cache()
            if error_message:
                # Handle error, e.g., log it or raise an exception
                # For now, print it, this might indicate an issue with empty initial prompts
                print(f"Error during initial prompt processing: {error_message}")


    def add_user_message(self, text: str):
        """Adds a user message to the dialog history and processes it."""
        if not text.strip():
            print("Warning: User message is empty or whitespace only. Not adding to dialog.")
            return # Or handle as an error/special case if needed

        self.dialog.append({"role": "user", "content": text})
        error_message = self._process_prompt_and_update_cache()
        if error_message:
            # This error means the user's message (even if non-empty) resulted in no tokens
            # or some other processing issue. The model won't be able to respond.
            # We might want to remove the last user message from dialog or flag an error state.
            print(f"Error processing user message: {error_message}. Model may not be able to respond.")
            # Consider how to signal this to the caller or if dialog should be reverted.
            # For now, dialog includes the message, but pending_next_token_logits might be None.

    def _process_prompt_and_update_cache(self) -> Optional[str]:
        """
        Encodes the current dialog, updates KVCache with new tokens,
        and stores logits for the next token.
        This is called after a new system/user message is added.
        Returns an error message string if processing fails, None otherwise.
        """
        if not self.dialog:
            # Should not happen if called after adding a message, but as a safeguard.
            self.pending_next_token_logits = None
            return "Dialog is empty, cannot process."

        prompt_tokens_list = self.chat_formatter.encode_dialog_prompt(self.dialog)

        if not prompt_tokens_list:
            # This implies the entire dialog (including the new message) encoded to nothing.
            # This is a more severe issue.
            print("Warning: Empty token list from chat_formatter for non-empty dialog.")
            self.pending_next_token_logits = None
            return "<Error: Could not encode dialog to tokens>"

        full_prompt_jnp = jnp.array([prompt_tokens_list], dtype=jnp.int32)

        if self.current_sequence_length == 0:
            # First processing run for this (segment of) dialog
            tokens_for_context_update = full_prompt_jnp
            start_pos_for_context_update = 0
        else:
            # Incremental update
            if full_prompt_jnp.shape[1] <= self.current_sequence_length:
                print(f"Warning/Debug: full_prompt_jnp.shape[1] ({full_prompt_jnp.shape[1]}) "
                      f"<= self.current_sequence_length ({self.current_sequence_length}). "
                      "This might indicate an issue or an empty new message part effectively.")
                # If no new tokens, we cannot get new logits.
                # The generation must rely on previously stored pending_next_token_logits
                # or it's an error if those are also None.
                # For now, if this happens, we effectively have no new context to process.
                # self.pending_next_token_logits will remain as it was.
                # This case is tricky: if the user *did* add a message but it tokenized to nothing
                # that was already part of the sequence, what should happen?
                # We should probably clear pending_next_token_logits if it's stale.
                # For now, let's assume if no new tokens, no update to logits.
                # This implies generate_model_response will fail if pending_next_token_logits is None.
                self.pending_next_token_logits = None # Clear stale logits
                return "<Info: No new tokens to process from the latest message>"
            
            tokens_for_context_update = full_prompt_jnp[:, self.current_sequence_length:]
            start_pos_for_context_update = self.current_sequence_length

        if tokens_for_context_update.shape[1] == 0:
            # This can happen if the slicing results in empty, e.g. after the warning above.
            print("Info: tokens_for_context_update is empty. No KV cache update or logit calculation will occur.")
            # self.pending_next_token_logits should reflect that no new info is available.
            # If it was already None, it stays None. If it had old logits, they should be invalidated.
            self.pending_next_token_logits = None # Ensure it's clear if no new tokens processed
            return "<Info: No effective new tokens to process for KV Cache>"

        # Process new context tokens
        logits, updated_kv_cache = self.model.apply(
            {'params': self.params},
            tokens_for_context_update,
            start_pos=start_pos_for_context_update,
            kv_cache=self.current_kv_cache
        )
        self.current_kv_cache = updated_kv_cache
        self.current_sequence_length += tokens_for_context_update.shape[1]
        self.pending_next_token_logits = logits[:, -1, :] # Logits for the *next* token
        return None # Success


    def generate_model_response(self, 
                                max_new_tokens: int, 
                                sampler: Sampler, 
                                rng_key: jax.random.PRNGKey) -> str:
        """
        Generates a response from the model using previously processed context.
        Relies on self.pending_next_token_logits being set by _process_prompt_and_update_cache.
        """
        if self.pending_next_token_logits is None:
            # This means no valid context was processed to produce initial logits for generation.
            # Could be due to an empty initial prompt, or user message that tokenized to nothing new.
            print("Error: pending_next_token_logits is None. Cannot start generation. "
                  "Ensure a valid prompt/message was processed.")
            return "<Error: Model has no starting point for generation. Process a prompt first.>"

        if self.current_kv_cache is None:
             # Should be initialized by start_new_conversation
            print("Error: KVCache not initialized. Cannot generate.")
            return "<Error: KVCache is not initialized>"


        next_token_logits_to_use = self.pending_next_token_logits
        self.pending_next_token_logits = None # Consume the logits

        generated_ids_this_turn = []
        
        # Sample the first token using the pre-computed next_token_logits
        rng_key, sample_key = jax.random.split(rng_key)
        current_token_for_loop_input = sample_next_token( # Use the standalone utility
            next_token_logits_to_use, sampler, sample_key
        )
        
        if current_token_for_loop_input.ndim > 0 : 
            current_token_for_loop_input = current_token_for_loop_input.squeeze() 

        generated_ids_this_turn.append(current_token_for_loop_input.item())

        # Autoregressive generation loop for subsequent tokens
        for _ in range(max_new_tokens - 1):
            # Check for End-of-Turn token
            # This requires tokenizer to have eot_id attribute, or equivalent.
            # Assuming Llama3 specific EOT handling might be done by tokenizer or chat_formatter
            # For a generic approach, we'd need a way to get this ID.
            # Let's assume tokenizer provides this.
            if hasattr(self.tokenizer, 'eot_id') and \
               current_token_for_loop_input.item() == self.tokenizer.eot_id:
                # print(f"Info: End-of-turn token {self.tokenizer.eot_id} encountered.")
                break 
            # Fallback: check for EOS if EOT is not specific or present
            elif current_token_for_loop_input.item() == self.tokenizer.eos_id:
                # print(f"Info: End-of-sequence token {self.tokenizer.eos_id} encountered.")
                break


            rng_key, sample_key = jax.random.split(rng_key)
            input_token_jnp = jnp.array([[current_token_for_loop_input.item()]], dtype=jnp.int32)

            logits, updated_kv_cache = self.model.apply(
                {'params': self.params},
                input_token_jnp, # Shape (1, 1) for single token input
                start_pos=self.current_sequence_length, 
                kv_cache=self.current_kv_cache
            )
            self.current_kv_cache = updated_kv_cache
            self.current_sequence_length += 1 # We processed one token
            next_token_logits_loop = logits[:, -1, :] 
            
            current_token_for_loop_input = sample_next_token( # Use the standalone utility
                next_token_logits_loop, sampler, sample_key
            )
            if current_token_for_loop_input.ndim > 0 :
                current_token_for_loop_input = current_token_for_loop_input.squeeze()
            generated_ids_this_turn.append(current_token_for_loop_input.item())

        decoded_response = self.tokenizer.decode(generated_ids_this_turn)
        
        # Add assistant's response to dialog *after* full generation
        self.dialog.append({"role": "assistant", "content": decoded_response})
        
        # IMPORTANT: After the assistant responds, the KVCache and sequence length
        # are now up-to-date with the assistant's tokens.
        # The *next* user message will then be processed by _process_prompt_and_update_cache,
        # which will correctly slice the full dialog tokens from self.current_sequence_length.
        
        return decoded_response

    # The _sample_next_token method is removed as we use the standalone generation_utils.sample_next_token

# Example Usage (Conceptual - requires model, tokenizer, params, etc.)
if __name__ == '__main__':
    # This section is for conceptual demonstration and testing.
    # It requires a mock or real Llama model, Tokenizer, and params.

    # --- Mockups for demonstration ---
    class MockModelArgs:
        n_layers = 2
        n_kv_heads = 2
        head_dim = 64
        # vocab_size = 32000 # Not directly needed by ConversationChat init for Llama 3 style

    class MockLlamaModel:
        def __init__(self):
            self.args = MockModelArgs()

        def apply(self, params_dict, tokens, start_pos, kv_cache):
            # Mock logits: batch_size, seq_len, vocab_size
            # For Llama 3, vocab_size is often around 128256
            # Let's use a smaller mock vocab for simplicity.
            mock_vocab_size = 1000 
            
            # Logits shape: (batch_size, input_sequence_length, vocab_size)
            # input_sequence_length is tokens.shape[1]
            # batch_size is tokens.shape[0], typically 1 for chat
            
            # Simple mock: return uniform logits or logits pointing to next token + 1
            # For a more realistic mock, it should depend on input tokens.
            # For now, let's make it pick token ID 5 as the most probable.
            logits = jnp.ones((tokens.shape[0], tokens.shape[1], mock_vocab_size)) * -10.0 
            # Make token '5' highly probable for all positions in the output sequence.
            logits = logits.at[:, :, 5].set(10.0) 

            # Mock KVCache update (doesn't need to be functional for this test, just return something)
            # In a real scenario, updated_kv_cache would be a new KVCache object or modified one.
            # For this mock, we can just return the passed kv_cache.
            return logits, kv_cache 

    class MockTokenizer:
        def __init__(self):
            self.bos_id = 1
            self.eos_id = 2
            self.pad_id = 0
            # For Llama 3, eot_id is important for chat.
            # e.g. <|eot_id|> (end of turn)
            self.eot_id = 128009 # Example Llama 3 EOT ID, replace with actual if known/different
            self.vocab_size = 1000 # Matching mock_vocab_size in MockLlamaModel


        def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
            # Simple mock: return list of ASCII values + some offset, or fixed tokens
            tokens = [ord(c) % self.vocab_size for c in text[:10]] # Limit length
            if bos:
                tokens = [self.bos_id] + tokens
            if eos: # EOS for user input, not for assistant generation start typically
                tokens = tokens + [self.eos_id]
            return tokens if tokens else [self.bos_id] # Ensure not empty, at least BOS

        def decode(self, token_ids: List[int]) -> str:
            # Simple mock: convert IDs back to chars or fixed strings
            # Filter out special tokens for decoding display
            filtered_ids = [tid for tid in token_ids if tid not in [self.bos_id, self.eos_id, self.eot_id, self.pad_id]]
            return "".join([chr(tid % 256) if tid < 256 else f"<T{tid}>" for tid in filtered_ids])
    
    class MockLlama32ChatFormat(ChatFormat):
        def __init__(self, tokenizer: MockTokenizer):
            self.tokenizer = tokenizer

        def encode_dialog_prompt(self, dialog: List[Dict[str, str]]) -> List[int]:
            # Simplified mock of Llama 3.2 chat format
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            # {{ system_prompt }}<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>
            # {{ user_message }}<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>
            # {{ assistant_response }}<|eot_id|> ...
            # For the prompt, it ends before the assistant's turn starts.
            
            # These are placeholder token IDs. In reality, they are specific strings tokenized.
            # For this mock, we'll use arbitrary integers not conflicting with BOS/EOS/EOT
            # if the tokenizer's encode method doesn't produce them from strings.
            # A better mock tokenizer would tokenize "<|start_header_id|>", etc.
            
            # Let's assume tokenizer.encode handles the text part.
            # We need to manually add structure if tokenizer doesn't do full templating.

            # For simplicity, this mock will just join content with basic separators.
            # A real implementation needs the special tokens of Llama 3.
            
            tokenized_prompt = []
            if not dialog: # Should not happen if used correctly
                return [self.tokenizer.bos_id] # Start with BOS if empty

            # Always start with BOS for the whole conversation turn
            # The Llama3 format usually starts with <|begin_of_text|>
            # which might be what tokenizer.bos_id represents if configured for Llama3.
            # Let's assume self.tokenizer.bos_id is <|begin_of_text|>
            tokenized_prompt.append(self.tokenizer.bos_id)

            for message in dialog:
                role = message["role"]
                content = message["content"]
                
                # Mocking the special tokens like <|start_header_id|>role<|end_header_id|>
                # This is highly simplified. Real Llama3 format uses specific token sequences.
                # For this mock, we'll just encode the role string and content string.
                # A proper ChatFormat would use tokenizer.encode for "<|start_header_id|>", "system", etc.
                
                role_tokens = self.tokenizer.encode(f"{role}: ", bos=False, eos=False)
                content_tokens = self.tokenizer.encode(content, bos=False, eos=False) # No BOS/EOS for content parts
                
                tokenized_prompt.extend(role_tokens)
                tokenized_prompt.extend(content_tokens)
                
                # Add <|eot_id|> after system and user messages in the prompt
                # For the last message (if user), it's part of the prompt.
                # If the last message is assistant, it shouldn't be in the prompt being built *for* next assistant response.
                if role == "system" or role == "user":
                    tokenized_prompt.append(self.tokenizer.eot_id)
            
            # If the dialog ends with a user or system message,
            # the prompt is ready for the assistant to start generating.
            # The Llama3 format expects <|start_header_id|>assistant<|end_header_id|> to precede assistant generation.
            # This should be implicitly handled by the model or the generation loop starting point.
            # The `encode_dialog_prompt` should produce tokens UP TO where the assistant starts.
            # The first token generated by the model will be the beginning of its actual response.

            # The current mock just appends EOT after user/system.
            # For generating an assistant response, the prompt should probably end with the user's message + EOT,
            # then the model starts generating. The "assistant" header might be implicitly expected by the model
            # or should be added as the *first part* of the assistant's generated sequence if not baked in.
            # This detail is crucial for correct Llama3 chat finetunes.

            # For now, this mock is very basic.
            # A key point is that self.tokenizer.encode(..., eos=True) for user message
            # might add a general EOS, not the specific Llama3 EOT.
            # The Llama3ChatFormat should handle these special tokens.

            return tokenized_prompt if tokenized_prompt else [self.tokenizer.bos_id]


    # --- Setup Mocks ---
    mock_model_instance = MockLlamaModel()
    mock_tokenizer_instance = MockTokenizer()
    # Params dict can be empty for this mock as MockLlamaModel.apply doesn't use it.
    mock_params = {} 
    mock_chat_formatter = MockLlama32ChatFormat(mock_tokenizer_instance)


    # --- Test ConversationChat ---
    chat_interface = ConversationChat(
        model=mock_model_instance,
        tokenizer=mock_tokenizer_instance,
        chat_formatter=mock_chat_formatter,
        params=mock_params,
        chat_max_seq_len=128 # Small for testing
    )

    print("Starting new conversation with a system prompt.")
    # start_new_conversation is called in __init__, so KVCache is ready.
    # We can call it again to set a system prompt.
    chat_interface.start_new_conversation(system_prompt_text="You are a helpful bot.")
    print(f"Dialog after system prompt: {chat_interface.get_dialog_string()}")
    # At this point, _process_prompt_and_update_cache has run for the system prompt.
    # chat_interface.pending_next_token_logits should be set.

    print("\\nAdding user message...")
    chat_interface.add_user_message("Hello there!")
    print(f"Dialog after user message: {chat_interface.get_dialog_string()}")
    # _process_prompt_and_update_cache has run again for the user message.
    # chat_interface.pending_next_token_logits should be updated.

    print("\\nGenerating model response...")
    # Create a dummy RNG key for JAX
    key = jax.random.PRNGKey(0)
    # Use a simple GreedySampler for testing
    greedy_sampler = GreedySampler()
    
    response = chat_interface.generate_model_response(
        max_new_tokens=10, 
        sampler=greedy_sampler, 
        rng_key=key
    )
    print(f"Model Response: {response}")
    print(f"Dialog after model response: {chat_interface.get_dialog_string()}")
    print(f"Current KVCache sequence length: {chat_interface.current_sequence_length}")

    print("\\nAdding another user message...")
    chat_interface.add_user_message("How are you?")
    print(f"Dialog after second user message: {chat_interface.get_dialog_string()}")

    print("\\nGenerating another model response...")
    key, gen_key = jax.random.split(key)
    response2 = chat_interface.generate_model_response(
        max_new_tokens=15,
        sampler=greedy_sampler,
        rng_key=gen_key
    )
    print(f"Second Model Response: {response2}")
    print(f"Final Dialog: {chat_interface.get_dialog_string()}")
    print(f"Final KVCache sequence length: {chat_interface.current_sequence_length}")

    print("\\nTesting empty user message handling in add_user_message:")
    chat_interface.add_user_message("   ") # Empty or whitespace only
    # We expect a warning and no change in dialog or pending logits from this.
    # So, trying to generate again should fail or use stale state if not handled carefully.
    # The current _process_prompt_and_update_cache should return an error/info message
    # and clear pending_next_token_logits if input is effectively empty.
    
    print("\\nAttempting to generate after an effectively empty user message (should use prior state or error):")
    key, gen_key_after_empty = jax.random.split(key)
    # Since add_user_message for "   " should result in _process_prompt_and_update_cache
    # clearing pending_next_token_logits (or returning an error that leads to it),
    # the following generate call should error out.
    response_after_empty = chat_interface.generate_model_response(
        max_new_tokens=5,
        sampler=greedy_sampler,
        rng_key=gen_key_after_empty
    )
    print(f"Response after empty user input: {response_after_empty}") # Expecting an error message

    print("\\nStarting a completely new conversation (resets KVCache and dialog):")
    chat_interface.start_new_conversation(system_prompt_text="New session: You are a poet.")
    chat_interface.add_user_message("Tell me a short poem.")
    print(f"Dialog for new conversation: {chat_interface.get_dialog_string()}")
    key, poetry_key = jax.random.split(key)
    poem_response = chat_interface.generate_model_response(
        max_new_tokens=20,
        sampler=greedy_sampler,
        rng_key=poetry_key
    )
    print(f"Poem Response: {poem_response}")
    print(f"Final dialog for poetry session: {chat_interface.get_dialog_string()}")
    print(f"Final KVCache sequence length for poetry session: {chat_interface.current_sequence_length}")
