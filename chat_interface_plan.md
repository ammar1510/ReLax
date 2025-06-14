# Plan for Refactoring Chat Interface with KVCache Management

This document outlines the plan to refactor `chat.py` by introducing a `ConversationChat` class to manage chat history, KVCache state, and model interaction for a JAX-based LLaMA model.

## I. KVCache Considerations

- The existing `KVCache` class/structure (defined or used in `models/kvcache.py` and `models/llama.py`) is assumed to be suitable for iterative updates.
- The `LLaMA.__call__` method in `models/llama.py` is expected to return `(logits, updated_kv_cache)`. This `updated_kv_cache` will be the state maintained by `ConversationChat` across generation turns.
- The `ConversationChat` class will be responsible for initializing and resetting the `KVCache` object (e.g., when a new system prompt is set or a new chat session begins).

## II. `ConversationChat` Class (to be implemented in `chat.py`)

This class will encapsulate the chat logic, history, and KV cache state.

```python
from typing import List, Optional, Dict
import jax
import jax.numpy as jnp
from tokenizer_utils import Tokenizer # Assuming Tokenizer class
from models.llama import LLaMA, ModelArgs # Assuming LLaMA model and ModelArgs
from models.kvcache import KVCache # Assuming KVCache class/type
from load_utils import FlaxModelParams # Assuming type for model weights
from chat import ChatFormat, Dialog # From existing chat.py
from sampling import temperature_scale, sample_top_k, sample_top_p # For sampling strategies
# NEW: Import Sampler interface and concrete samplers
from sampling import Sampler, GreedySampler, CategoricalSampler, TopKSampler, TopPSampler 

class ConversationChat:
    def __init__(self, 
                 tokenizer: Tokenizer, 
                 model: LLaMA, 
                 params: FlaxModelParams):
        self.tokenizer = tokenizer
        self.model = model
        self.params = params # Loaded model weights

        self.chat_formatter = ChatFormat(self.tokenizer)
        self.dialog: Dialog = [] # Stores {"role": ..., "content": ...}

        # KVCache and sequence length for managing its state
        self.current_kv_cache: Optional[KVCache] = None
        self.current_sequence_length: int = 0
        
    def start_new_conversation(self, system_prompt_text: Optional[str] = None):
        """Starts a new conversation, optionally with a system prompt, and resets KVCache."""
        self.dialog = [] # Clear previous dialog
        if system_prompt_text:
            self.dialog.append({"role": "system", "content": system_prompt_text})
        # Resetting KVCache because the context changes fundamentally
        self.current_kv_cache = None 
        self.current_sequence_length = 0

    def add_user_message(self, text: str):
        """Adds a user message to the dialog history."""
        self.dialog.append({"role": "user", "content": text})
        # KVCache will be updated/used when generate_model_response is called

    def _sample_next_token(self, 
                           logits: jax.Array, 
                           sampler: Sampler, # MODIFIED: Sampler object
                           key: jax.random.PRNGKey) -> jax.Array:
        """Samples a token from the logits distribution using the provided sampler."""
        # Ensure logits are effectively rank 1 for samplers (remove batch dim if present)
        if logits.ndim > 1:
            logits = logits.squeeze(axis=0) 
        
        return sampler.sample(logits, key)

    def generate_model_response(self, 
                                max_new_tokens: int, 
                                sampler: Sampler, # MODIFIED: Sampler object
                                rng_key: jax.random.PRNGKey) -> str:
        """
        Generates a response from the model based on the current dialog.
        Manages KVCache updates for efficient autoregressive generation.
        """
        if self.current_kv_cache is None:
            self.current_sequence_length = 0

        prompt_tokens_list = self.chat_formatter.encode_dialog_prompt(self.dialog)
        prompt_jnp = jnp.array([prompt_tokens_list], dtype=jnp.int32)

        tokens_for_context_update = prompt_jnp[:, self.current_sequence_length:]
        start_pos_for_context_update = self.current_sequence_length
        
        logits, updated_kv_cache = self.model.apply(
            {'params': self.params},
            tokens_for_context_update,
            start_pos=start_pos_for_context_update,
            kv_cache=self.current_kv_cache
        )
        self.current_kv_cache = updated_kv_cache
        self.current_sequence_length += tokens_for_context_update.shape[1]
        next_token_logits = logits[:, -1, :] 

        generated_ids_this_turn = []
        # Sample the first token
        rng_key, sample_key = jax.random.split(rng_key)
        current_token_for_loop_input = self._sample_next_token(
            next_token_logits, sampler, sample_key # MODIFIED
        )
        generated_ids_this_turn.append(current_token_for_loop_input.item())

        for _ in range(max_new_tokens - 1):
            # Check for End-of-Turn token (e.g., <|eot_id|> for Llama 3 chat)
            if current_token_for_loop_input.item() == self.tokenizer.eot_id:
                break 

            rng_key, sample_key = jax.random.split(rng_key)
            input_token_jnp = jnp.array([[current_token_for_loop_input.item()]], dtype=jnp.int32)

            logits, updated_kv_cache = self.model.apply(
                {'params': self.params},
                input_token_jnp,
                start_pos=self.current_sequence_length, 
                kv_cache=self.current_kv_cache
            )
            self.current_kv_cache = updated_kv_cache
            self.current_sequence_length += 1
            next_token_logits = logits[:, -1, :] 
            current_token_for_loop_input = self._sample_next_token(
                next_token_logits, sampler, sample_key # MODIFIED
            )
            generated_ids_this_turn.append(current_token_for_loop_input.item())

        decoded_response = self.tokenizer.decode(generated_ids_this_turn)
        self.dialog.append({"role": "assistant", "content": decoded_response})
        return decoded_response

```

## III. Main Chat Interface (Updates in `chat.py`)

The `main()` function in `chat.py` will be updated to use the `ConversationChat` class.

```python
# Inside main() function in chat.py (conceptual)

# ... (argparse or Hydra config setup) ...
# args = parser.parse_args() or cfg = hydra_config
# Generation parameters might include:
# args.temperature (float)
# args.sampler_name (str, e.g., "greedy", "categorical", "top_k", "top_p")
# args.top_k (int, relevant if sampler_name is "top_k")
# args.top_p (float, relevant if sampler_name is "top_p")

# 1. Load ModelArgs (e.g., from params.json in model_dir)
#    model_args = ModelArgs(**loaded_params_json) 

# 2. Initialize Tokenizer
#    tokenizer = Tokenizer(model_path=tokenizer_load_path)

# 3. Initialize Flax LLaMA Model
#    llama_flax_model = LLaMA(model_args)

# 4. Load Model Weights (using load_utils.py)
#    model_params: FlaxModelParams = load_llama_weights_safetensors(args.model_dir, device=jax.devices()[0])

# 5. Initialize ConversationChat
#    chat_session = ConversationChat(
#        model_args=model_args,
#        tokenizer=tokenizer,
#        flax_model=llama_flax_model,
#        model_params=model_params
#    )

# 6. Start New Conversation (with system prompt from args/config)
#    system_prompt = args.system_prompt if hasattr(args, 'system_prompt') else None
#    chat_session.start_new_conversation(system_prompt_text=system_prompt)
#    if system_prompt:
#        print(f"System: {system_prompt}")

# 7. Initialize JAX PRNGKey for generation
#    key = jax.random.PRNGKey(seed=0) # Or some other seed

# 8. Create Sampler instance based on args
#    sampler: Sampler
#    if args.sampler_name == "greedy" or args.temperature == 0.0:
#        sampler = GreedySampler()
#    elif args.sampler_name == "categorical":
#        sampler = CategoricalSampler(temperature=args.temperature)
#    elif args.sampler_name == "top_k":
#        if args.top_k is None:
#            raise ValueError("top_k must be provided for 'top_k' sampler.")
#        sampler = TopKSampler(k=args.top_k, temperature=args.temperature)
#    elif args.sampler_name == "top_p":
#        if args.top_p is None:
#            raise ValueError("top_p must be provided for 'top_p' sampler.")
#        sampler = TopPSampler(p=args.top_p, temperature=args.temperature)
#    else: # Default or error
#        print(f"Warning: Unknown sampler_name '{args.sampler_name}'. Defaulting to Categorical.")
#        sampler = CategoricalSampler(temperature=args.temperature if args.temperature > 0 else 1.0)

# 9. Chat Loop
#    print("Starting chat. Type 'quit', 'exit', or 'bye' to end.")
#    while True:
#        user_input = input("You: ")
#        if user_input.lower() in ["quit", "exit", "bye"]:
#            print("Exiting chat.")
#            break
#
#        chat_session.add_user_message(user_input)
#
#        key, gen_key = jax.random.split(key) # New key for this generation
#        assistant_response = chat_session.generate_model_response(
#            max_new_tokens=args.max_new_tokens,
#            sampler=sampler, # MODIFIED: Pass sampler object
#            rng_key=gen_key
#        )
#        print(f"Assistant: {assistant_response}")

# ... (rest of main, if __name__ == "__main__":) ...
```