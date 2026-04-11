"""Gemma tokenizer using SentencePiece.

Requires: pip install sentencepiece
"""

from typing import List


class GemmaTokenizer:
    """Minimal SentencePiece wrapper for Gemma models."""

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece is required for Gemma tokenization.\n"
                "Install with: pip install sentencepiece"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        self.vocab_size: int = self.sp.vocab_size()
        self.bos_id: int = self.sp.bos_id()
        self.eos_id: int = self.sp.eos_id()

        # pad_id returns -1 if not defined; fall back to 0
        raw_pad = self.sp.pad_id()
        self.pad_id: int = raw_pad if raw_pad >= 0 else 0

        # Conversation turn tokens
        self.start_of_turn_id: int = self.sp.piece_to_id("<start_of_turn>")
        self.end_of_turn_id: int = self.sp.piece_to_id("<end_of_turn>")

        # eos_tokens for ServingConfig — stop on both eos and end-of-turn
        self.stop_ids: tuple[int, ...] = (self.eos_id, self.end_of_turn_id)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        tokens: List[int] = self.sp.encode(text, out_type=int)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        # Filter out special tokens that SentencePiece can't decode cleanly
        filtered = [t for t in tokens if t < self.vocab_size]
        return self.sp.decode(filtered)
