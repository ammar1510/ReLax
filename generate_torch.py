import fire
import torch
import time
from typing import List

from experiments.torch_llama import Llama
from models.llama.tokenizer import Tokenizer

def generate(
    llama: Llama,
    prompt: str,
    max_gen_len: int,
    temperature: float,
    top_p: float,
    rng_seed: int,
):
    """
    PyTorch-based text generation function.
    """
    # Create a generator for sampling
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(rng_seed)

    # Use the text_completion method from the Llama wrapper
    results = llama.text_completion(
        prompts=[prompt],
        sample_rng=sample_rng,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return results[0]['generation']


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
    max_batch_size: int = 1, # for single prompt generation
    seed: int = 1,
    max_gen_len: int = 500,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    """
    Entry point for running the Llama PyTorch model for text generation.
    """
    system_prompt = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
    user_prompt = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    print("Loading model and tokenizer...")
    start_time = time.time()

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=True,
        seed=seed,
    )
    
    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")

    # Generate text
    result = generate(
        llama=llama,
        prompt=prompt,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        rng_seed=seed,
    )

    # Print the result
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 20)
    
if __name__ == "__main__":
    fire.Fire(main) 