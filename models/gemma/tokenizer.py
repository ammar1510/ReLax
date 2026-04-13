"""Gemma tokenizer using the HuggingFace tokenizers library.

Loads from tokenizer.json (HF fast tokenizer format).

Requires: pip install tokenizers
"""

from typing import List


class GemmaTokenizer:
    """Tokenizer wrapper for Gemma models using tokenizer.json."""

    def __init__(self, tokenizer_path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers is required for Gemma tokenization.\n"
                "Install with: pip install tokenizers"
            )

        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.no_truncation()
        self.tok.no_padding()

        self.vocab_size: int = self.tok.get_vocab_size()

        # Special token IDs (Gemma 4 format)
        self.bos_id: int = self.tok.token_to_id("<bos>")
        self.eos_id: int = self.tok.token_to_id("<eos>")
        self.pad_id: int = self.tok.token_to_id("<pad>") or 0
        self.start_of_turn_id: int = self.tok.token_to_id("<|turn>")
        self.end_of_turn_id: int = self.tok.token_to_id("<turn|>")

        # Thinking mode: <|think|> enables thinking in the system prompt;
        # generated thought is wrapped in <|channel>thought\n...\n<channel|>
        self.think_token_id: int | None = self.tok.token_to_id("<|think|>")
        self.start_channel_id: int | None = self.tok.token_to_id("<|channel>")
        self.end_channel_id: int | None = self.tok.token_to_id("<channel|>")

        # Stop on both eos and end-of-turn
        self.stop_ids: tuple[int, ...] = (self.eos_id, self.end_of_turn_id)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        tokens: List[int] = self.tok.encode(text, add_special_tokens=False).ids
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tok.decode(tokens, skip_special_tokens=True)
