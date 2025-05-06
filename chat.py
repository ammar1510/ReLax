# Placeholder for Chat Interface

from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)

# Import the Tokenizer class (assuming it stays in tokenizer_utils.py)
from tokenizer_utils import Tokenizer

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

# TODO: Add main chat loop, argument parsing, model loading, inference call etc. 