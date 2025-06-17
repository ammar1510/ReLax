import fire
import torch
from typing import List

from tests.torch_ops import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 64,
    max_batch_size: int = 1,
):
    """
    Text generation script for the Llama model.
    """
    # Load the model and tokenizer
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device='cuda',
    )
    model = llama.model
    model.eval()

    # Set up the random seed for reproducibility
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)

    print("Model and tokenizer loaded. Enter prompts for text generation (or 'quit' to exit).")

    while True:
        prompt = input(">> ")
        if prompt.lower() == 'quit':
            break

        prompts: List[str] = [prompt]
        
        result = llama.generate(
            prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        print(f"Prompt: {prompt}")
        print(f"Generated: {result}")
        print("-" * 20)

if __name__ == "__main__":
    fire.Fire(main) 