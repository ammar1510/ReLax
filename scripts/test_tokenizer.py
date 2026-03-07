"""Quick sanity check for the tokenizer."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.llama.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Directory containing original/tokenizer.model")
    args = parser.parse_args()

    tokenizer_path = Path(args.model_path) / "original" / "tokenizer.model"
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer(str(tokenizer_path))
    print(f"Vocab size: {tokenizer.vocab_size}")

    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text, bos=True, eos=False)
    decoded = tokenizer.decode(tokens)
    print(f"Input:   {text}")
    print(f"Tokens:  {tokens}")
    print(f"Decoded: {decoded}")

    assert decoded == text, f"Round-trip mismatch: {decoded!r} != {text!r}"
    print("Tokenizer OK!")


if __name__ == "__main__":
    main()
