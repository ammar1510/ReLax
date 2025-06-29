# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer based on Llama 3 reference.
    """

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000
    MAX_NO_WHITESPACES_CHARS = 25_000

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model file (.tiktoken).

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            raise FileNotFoundError(f"Tokenizer model not found at {model_path}")

        mergeable_ranks = load_tiktoken_bpe(str(model_path_obj))
        num_base_tokens = len(mergeable_ranks)

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2+i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name=model_path_obj.name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.vocab_size: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id: int = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]

        # Tokens that signify the end of a model's turn
        self.stop_tokens = {
            self.eos_id,
            self.eot_id,
        }

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs, handling potential Tiktoken length limits.

        Args:
            s (str): The input string.
            bos (bool): Whether to prepend the beginning-of-text token.
            eos (bool): Whether to append the end-of-text token.
            allowed_special: Handling of special tokens (refer to Tiktoken docs).
            disallowed_special: Handling of special tokens (refer to Tiktoken docs).

        Returns:
            List[int]: The encoded token IDs.
        """
        assert type(s) is str
        # TIKTOKEN_MAX_ENCODE_CHARS = 400_000 # Moved to class attribute
        # MAX_NO_WHITESPACES_CHARS = 25_000 # Moved to class attribute

        substrs = (
            substr
            for i in range(0, len(s), self.TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + self.TIKTOKEN_MAX_ENCODE_CHARS], self.MAX_NO_WHITESPACES_CHARS
            )
        )

        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.
        """
        # Typecast is safe per Tiktoken docs.
        return self.model.decode(cast(List[int], t))

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.vocab_size

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits a string into segments of consecutive characters that are either all spaces or all non-spaces.

        Args:
            s (str): The input string.
            max_consecutive_slice_len (int): The maximum length of a consecutive segment.

        Yields:
            str: The next segment of the string.
        """

        if not s:  # Handle empty string case
            yield ""
            return

        current_slice_len = 0
        # Initialize based on the first character.
        current_slice_is_space = s[0].isspace()
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:  # Type of char changed
                yield s[slice_start:i]  # Yield the segment that just finished
                slice_start = i  # New segment starts at i
                current_slice_len = 1  # Length of new segment is 1 (s[i])
                current_slice_is_space = is_now_space  # Update type for the new segment
            else:  # Same type as before
                # The length of the current segment including s[i] is (i - slice_start + 1)
                current_slice_len = i - slice_start + 1
                if current_slice_len > max_consecutive_slice_len:
                    # Yield the part that has hit max_consecutive_slice_len.
                    # This is s[slice_start : slice_start + max_consecutive_slice_len].
                    # Since s[i] is the character that made it (slice_start + max_consecutive_slice_len),
                    # the end of the slice to yield is i.
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1  # Reset for the new segment starting at i
                    # current_slice_is_space remains the same for the new segment starting at i

        # Yield any remaining part of the string
        if slice_start < len(s):
            yield s[slice_start:]
