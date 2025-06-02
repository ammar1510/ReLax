import pytest
import os
from pathlib import Path
from typing import List

# Adjust the import path according to your project structure
# This assumes 'tokenizer_utils.py' is in the parent directory of 'tests'
from tokenizer_utils import Tokenizer


# Helper to create a dummy tiktoken file for tests
def create_dummy_model_file(model_path: Path, content: str):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w") as f:
        f.write(content)
    return model_path

# Fixture for the Tokenizer
@pytest.fixture(scope="module")
def tokenizer_model_path(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("data")
    model_file = model_dir / "dummy_model.tiktoken"

    # A BPE ranks file where tokens align with pat_str behavior.
    # "token"   -> dG9rZW4=
    # "1"       -> MQ==
    # "2"       -> Mg==
    # " "       -> IA==
    # "b"       -> Yg==
    # "c"       -> Yw==
    # "a"       -> YQ== (Corrected from ZQ==)
    # "<"       -> PA==
    # "|"       -> fA==
    # "eot"     -> ZW90
    # "_"       -> Xw==
    # "id"      -> aWQ=
    # ">"       -> Pg==
    content = """
dG9rZW4= 0
MQ== 1
Mg== 2
IA== 3
Yg== 4
Yw== 5
YQ== 6
PA== 7
fA== 8
ZW90 9
Xw== 10
aWQ= 11
Pg== 12
"""
    create_dummy_model_file(model_file, content)
    return str(model_file)

@pytest.fixture(scope="module")
def tokenizer(tokenizer_model_path):
    return Tokenizer(model_path=tokenizer_model_path)

# --- Test Cases ---

def test_tokenizer_initialization(tokenizer: Tokenizer, tokenizer_model_path: str):
    """Tests basic tokenizer initialization and properties."""
    assert tokenizer.model is not None
    assert tokenizer.model.name == Path(tokenizer_model_path).name
    assert tokenizer.vocab_size > 0
    assert tokenizer.bos_id is not None
    assert tokenizer.eos_id is not None
    assert tokenizer.eot_id is not None
    assert tokenizer.pad_id == -1 # As defined in Tokenizer

    # Check if special tokens are loaded
    assert "<|begin_of_text|>" in tokenizer.special_tokens
    assert tokenizer.special_tokens["<|begin_of_text|>"] == tokenizer.bos_id
    assert "<|end_of_text|>" in tokenizer.special_tokens
    assert tokenizer.special_tokens["<|end_of_text|>"] == tokenizer.eos_id
    assert "<|eot_id|>" in tokenizer.special_tokens
    assert tokenizer.special_tokens["<|eot_id|>"] == tokenizer.eot_id

    # Check num_reserved_special_tokens
    expected_num_special_tokens = 10 + (tokenizer.num_reserved_special_tokens - 5 - 5)
    assert len(tokenizer.special_tokens) == expected_num_special_tokens

    # Test vocab size (base tokens + special tokens)
    # The dummy model now has 13 base tokens.
    num_base_tokens = 13
    assert tokenizer.get_vocab_size() == num_base_tokens + len(tokenizer.special_tokens)
    assert tokenizer.vocab_size == num_base_tokens + len(tokenizer.special_tokens)


def test_tokenizer_missing_model_file():
    """Tests that Tokenizer raises FileNotFoundError for a missing model file."""
    with pytest.raises(FileNotFoundError):
        Tokenizer(model_path="non_existent_model.tiktoken")

def test_encode_simple_string(tokenizer: Tokenizer):
    """Tests encoding a simple string."""
    # "token1 token2" -> "token", "1", " ", "token", "2"
    # Ranks: token=0, 1=1, space=3, 2=2
    encoded_tokens = tokenizer.encode("token1 token2", bos=False, eos=False)
    assert encoded_tokens == [0, 1, 3, 0, 2]

    decoded_text = tokenizer.decode(encoded_tokens)
    assert decoded_text == "token1 token2"

    # "b c" -> "b", " ", "c"
    # Ranks: b=4, space=3, c=5
    encoded_tokens_abc = tokenizer.encode("b c", bos=False, eos=False)
    assert encoded_tokens_abc == [4, 3, 5]
    assert tokenizer.decode(encoded_tokens_abc) == "b c"

    # " a" -> " ", "a"
    # Ranks: space=3, a=6
    encoded_a = tokenizer.encode(" a", bos=False, eos=False)
    assert encoded_a == [3, 6]
    assert tokenizer.decode(encoded_a) == " a"


def test_encode_with_bos_eos(tokenizer: Tokenizer):
    """Tests encoding with BOS and EOS tokens."""
    text = "b" # token "b" is rank 4
    encoded_tokens_bos = tokenizer.encode(text, bos=True, eos=False)
    assert encoded_tokens_bos == [tokenizer.bos_id, 4]

    encoded_tokens_eos = tokenizer.encode(text, bos=False, eos=True)
    assert encoded_tokens_eos == [4, tokenizer.eos_id]

    encoded_tokens_bos_eos = tokenizer.encode(text, bos=True, eos=True)
    assert encoded_tokens_bos_eos == [tokenizer.bos_id, 4, tokenizer.eos_id]

def test_decode_simple(tokenizer: Tokenizer):
    """Tests decoding a simple list of token IDs."""
    # Corresponds to "token1 token2" with the current dummy model
    # "token"=0, "1"=1, " "=3, "token"=0, "2"=2
    tokens = [0, 1, 3, 0, 2]
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == "token1 token2"

    tokens_with_special = [tokenizer.bos_id, 0, 1, 3, 0, 2, tokenizer.eos_id]
    decoded_text_special = tokenizer.decode(tokens_with_special)
    expected_decode_special = "<|begin_of_text|>token1 token2<|end_of_text|>"
    assert decoded_text_special == expected_decode_special


def test_get_vocab_size(tokenizer: Tokenizer):
    """Tests the get_vocab_size method."""
    num_base_tokens = 13 # Updated to 13 base tokens
    expected_num_special_tokens = 10 + (tokenizer.num_reserved_special_tokens - 5 - 5)
    assert tokenizer.get_vocab_size() == num_base_tokens + expected_num_special_tokens

def test_split_whitespaces_or_nonwhitespaces_helper():
    """Tests the _split_whitespaces_or_nonwhitespaces static method."""
    split_fn = Tokenizer._split_whitespaces_or_nonwhitespaces
    max_len = 5

    assert list(split_fn("", max_len)) == [""]
    assert list(split_fn("abc", max_len)) == ["abc"]
    assert list(split_fn("   ", max_len)) == ["   "]
    assert list(split_fn("abcdefgh", max_len)) == ["abcde", "fgh"]
    assert list(split_fn("        ", max_len)) == ["     ", "   "]

    s = "abc  defghij klmno p"
    expected = ["abc", "  ", "def", "ghi", "j", " ", "klm", "no", " ", "p"]
    assert list(split_fn(s, 3)) == expected

    s2 = "aaaaabbbbbccccc"
    assert list(split_fn(s2, 5)) == ["aaaaa", "bbbbb", "ccccc"]

    s3 = "     #####"
    assert list(split_fn(s3, 3)) == ["   ", "  ", "###", "##"]

    s4 = "a b c d e f"
    assert list(split_fn(s4, 1)) == ["a", " ", "b", " ", "c", " ", "d", " ", "e", " ", "f"]


def test_encode_long_string_splitting(tokenizer: Tokenizer):
    """
    Tests encoding of long strings that should trigger splitting logic
    both by TIKTOKEN_MAX_ENCODE_CHARS and MAX_NO_WHITESPACES_CHARS.
    """
    original_max_no_ws_chars = Tokenizer.MAX_NO_WHITESPACES_CHARS
    original_tiktoken_max_chars = Tokenizer.TIKTOKEN_MAX_ENCODE_CHARS
    Tokenizer.MAX_NO_WHITESPACES_CHARS = 5

    try:
        # "b" is rank 4
        text_long_no_ws = "b" * 7
        expected_tokens = [4] * 7
        encoded = tokenizer.encode(text_long_no_ws, bos=False, eos=False)
        assert encoded == expected_tokens
        assert tokenizer.decode(encoded) == text_long_no_ws

        # " " is rank 3
        text_long_ws = " " * 7
        expected_tokens_ws = [3] * 7
        encoded_ws = tokenizer.encode(text_long_ws, bos=False, eos=False)
        assert encoded_ws == expected_tokens_ws
        assert tokenizer.decode(encoded_ws) == text_long_ws

        Tokenizer.TIKTOKEN_MAX_ENCODE_CHARS = 3
        Tokenizer.MAX_NO_WHITESPACES_CHARS = 5 # Larger than TIKTOKEN_MAX_ENCODE_CHARS

        # "b b b" -> b=4, space=3. Encodes to [4,3,4,3,4]
        text_tik_split = "b b b"
        encoded_tik_split = tokenizer.encode(text_tik_split, bos=False, eos=False)
        assert encoded_tik_split == [4, 3, 4, 3, 4]
        assert tokenizer.decode(encoded_tik_split) == text_tik_split

        Tokenizer.MAX_NO_WHITESPACES_CHARS = 2 # Smaller than TIKTOKEN_MAX_ENCODE_CHARS=3
        # "bbb" -> b=4. Encodes to [4,4,4]
        text_complex_split = "bbb"
        encoded_complex = tokenizer.encode(text_complex_split, bos=False, eos=False)
        assert encoded_complex == [4, 4, 4]
        assert tokenizer.decode(encoded_complex) == text_complex_split

    finally:
        Tokenizer.MAX_NO_WHITESPACES_CHARS = original_max_no_ws_chars
        Tokenizer.TIKTOKEN_MAX_ENCODE_CHARS = original_tiktoken_max_chars


def test_special_tokens_values(tokenizer: Tokenizer):
    """Test that special token IDs are assigned correctly and are unique."""
    num_base_tokens = 13 # Updated to 13 base tokens
    # Ensure the tokens exist (values are checked for uniqueness below)
    assert "<|begin_of_text|>" in tokenizer.special_tokens
    assert "<|end_of_text|>" in tokenizer.special_tokens
    assert "<|reserved_special_token_0|>" in tokenizer.special_tokens
    assert "<|eot_id|>" in tokenizer.special_tokens
    assert "<|reserved_special_token_5|>" in tokenizer.special_tokens

    # Check that all special token IDs (which include bos, eos, eot via the dict) are unique.
    special_token_values = list(tokenizer.special_tokens.values())
    assert len(special_token_values) == len(set(special_token_values))


def test_stop_tokens_set(tokenizer: Tokenizer):
    """Test the stop_tokens set."""
    assert tokenizer.stop_tokens == {tokenizer.eos_id, tokenizer.eot_id}


def test_encode_empty_string(tokenizer: Tokenizer):
    """Tests encoding of an empty string with bos/eos flags."""
    assert tokenizer.encode("", bos=False, eos=False) == []
    assert tokenizer.encode("", bos=True, eos=False) == [tokenizer.bos_id]
    assert tokenizer.encode("", bos=False, eos=True) == [tokenizer.eos_id]
    assert tokenizer.encode("", bos=True, eos=True) == [tokenizer.bos_id, tokenizer.eos_id]


def test_encode_special_token_handling(tokenizer: Tokenizer):
    """Tests encoding with allowed_special and disallowed_special parameters."""
    special_token_str = "<|eot_id|>"
    special_token_id = tokenizer.eot_id

    # Test 1: Allowed special token
    assert tokenizer.encode(special_token_str, bos=False, eos=False, allowed_special="all") == [special_token_id]
    assert tokenizer.encode(special_token_str, bos=False, eos=False, allowed_special={special_token_str}) == [special_token_id]

    # Test 2: Unallowed special token (default behavior)
    # With the dummy model, tiktoken will likely try to break down "<|eot_id|>" into characters.
    # Since these characters ('<', '|', 'e', etc.) are not in the dummy model's base vocabulary
    # and the dummy model doesn't have byte-level tokens, tiktoken.encode will likely raise an error.
    # This error confirms that the special token isn't being treated as special if not allowed,
    # and also isn't silently passing as unknown bytes if the model can't handle arbitrary bytes.
    # The exact error might be tiktoken-version dependent (e.g. KeyError, ValueError).
    # For robust testing, we check if *any* exception is raised when encoding unallowed special tokens
    # that cannot be represented by the limited dummy model.
    # --- UPDATED TEST LOGIC ---
    # Now that '<', '|', 'eot', '_', 'id', '>' are in the dummy vocab,
    # encoding "<|eot_id|>" with allowed_special=set() should succeed and produce their tokens.
    # String "<|eot_id|>" tokenizes as: '<' (7), '|' (8), 'eot' (9), '_' (10), 'id' (11), then the second '|' (8) from input, then '>' (12).
    expected_tokens_for_disallowed_special = [7, 8, 9, 10, 11, 8, 12]
    actual_tokens = tokenizer.encode(special_token_str, bos=False, eos=False, allowed_special=set())
    assert actual_tokens == expected_tokens_for_disallowed_special

    # Test 3: Disallowed special token
    # If a special token is explicitly disallowed, tiktoken should raise an error.
    with pytest.raises(ValueError): # tiktoken.encode raises ValueError if disallowed_special is violated
        tokenizer.encode(special_token_str, bos=False, eos=False, allowed_special="all", disallowed_special={special_token_str})
    
    with pytest.raises(ValueError):
        tokenizer.encode(special_token_str, bos=False, eos=False, allowed_special="all", disallowed_special="all")

    # Test 4: Encoding regular text that happens to look like a special token pattern, when special tokens are not allowed.
    # The string "<eot>" is different from "<|eot_id|>" and its parts are in the dummy vocab.
    text_like_special = "<eot>"
    # Expected: '<' (7), 'eot' (9), '>' (12)
    expected_text_like_special_tokens = [7, 9, 12]
    assert tokenizer.encode(text_like_special, bos=False, eos=False, allowed_special=set()) == expected_text_like_special_tokens

    # Test 5: Behavior with a mix of normal text and special tokens
    text_mixed = "token1 <|eot_id|> b"
    # Ranks: token=0, 1=1, space=3, b=4. <|eot_id|> = special_token_id
    # "token1 <|eot_id|> b" -> "token", "1", " ", "<|eot_id|>", " ", "b"
    expected_mixed_tokens = [0, 1, 3, special_token_id, 3, 4]
    encoded_mixed = tokenizer.encode(text_mixed, bos=False, eos=False, allowed_special="all")
    assert encoded_mixed == expected_mixed_tokens
    assert tokenizer.decode(encoded_mixed) == "token1 <|eot_id|> b" # tiktoken should reconstruct this if special token is known

    # What if the special token is not surrounded by spaces?
    text_mixed_no_space = "token1<|eot_id|>b"
    # "token1<|eot_id|>b" -> "token", "1", "<|eot_id|>", "b"
    expected_mixed_no_space = [0, 1, special_token_id, 4]
    encoded_mixed_no_space = tokenizer.encode(text_mixed_no_space, bos=False, eos=False, allowed_special="all")
    assert encoded_mixed_no_space == expected_mixed_no_space
    assert tokenizer.decode(encoded_mixed_no_space) == "token1<|eot_id|>b" 