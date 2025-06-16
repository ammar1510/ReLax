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
    max_batch_size: int = 4,
    flash: bool = True,
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
        flash=flash,
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
        
        results = llama.text_completion(
            prompts,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for result in results:
            print(f"   {result['generation']}")
            print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main) 