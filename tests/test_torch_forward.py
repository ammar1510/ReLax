import numpy as np
import torch
from pathlib import Path
import argparse

# PyTorch model components
from experiments.torch_llama import Llama as Llama_wrapper
from models.llama.tokenizer import Tokenizer


def test_torch_forward_pass(model_path: str, output_file: str = "torch_output.txt", max_gen_len: int = 256):
    """Test PyTorch model text generation with greedy sampling and save output to file."""

    model_path = Path(model_path)
    print(f"Loading model from {model_path}")

    # Test configuration
    test_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Device selection for PyTorch
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        TORCH_XLA_AVAILABLE = True
    except ImportError:
        TORCH_XLA_AVAILABLE = False
        torch_xla = None

    if TORCH_XLA_AVAILABLE:
        torch_device = torch_xla.device()
        print(f"Using PyTorch XLA device: {torch_device}")
    elif torch.cuda.is_available():
        torch_device = "cuda"
        print("Using CUDA device")
    else:
        torch_device = "cpu"
        print("Using CPU device")

    # Use bfloat16 for consistency with typical inference
    torch_dtype = torch.bfloat16

    print("\n" + "="*80)
    print("LOADING PYTORCH MODEL")
    print("="*80)

    # Load PyTorch model using the Llama.build method
    torch_original_path = model_path / "original"
    tokenizer_path = torch_original_path / "tokenizer.model"

    if not torch_original_path.exists():
        # Try alternative path
        torch_original_path = model_path
        tokenizer_path = model_path / "tokenizer.model"

    print(f"Loading from: {torch_original_path}")
    print(f"Tokenizer: {tokenizer_path}")

    # Calculate max_seq_len needed
    max_seq_len = len(test_prompt.split()) * 2 + max_gen_len  # Rough estimate
    max_seq_len = min(max_seq_len, 8192)  # Cap at model max
    
    llama_wrapper = Llama_wrapper.build(
        ckpt_dir=str(torch_original_path),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=max_seq_len,
        max_batch_size=1,
        flash=False,
    )
    torch_model = llama_wrapper.model
    torch_model.eval()

    # Move to target device if needed
    if hasattr(torch_device, 'type') and torch_device.type == 'xla':
        # XLA device is already set in build
        pass
    elif isinstance(torch_device, str) and torch_device != str(torch_model.tok_embeddings.weight.device):
        torch_model = torch_model.to(torch_device)

    print("✓ PyTorch model loaded successfully")

    print("\n" + "="*80)
    print("PREPARING INPUT")
    print("="*80)

    # Initialize tokenizer
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    # Tokenize input text
    prompt_tokens = tokenizer.encode(test_prompt, bos=True, eos=False)
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    print("\n" + "="*80)
    print("GENERATING TEXT WITH GREEDY SAMPLING")
    print("="*80)

    # Get device for tensor creation
    if hasattr(torch_device, 'type') and torch_device.type == 'xla':
        device_for_tensor = torch_device
    elif isinstance(torch_device, str):
        device_for_tensor = torch_device
    else:
        device_for_tensor = torch_device

    # Initialize KV caches in all layers
    torch_params = torch_model.params
    total_len = len(prompt_tokens) + max_gen_len
    total_len = min(total_len, max_seq_len)
    
    print(f"Initializing KV caches for max sequence length: {total_len}")
    from experiments.torch_llama import KVCache as KVCache_torch
    for i in range(torch_params.n_layers):
        layer_dtype = torch_model.layers[i].attention.wq.weight.dtype
        layer_device = torch_model.layers[i].attention.wq.weight.device
        torch_model.layers[i].attention.cache = KVCache_torch(
            batch_size=1,
            seq_length=total_len,
            n_kv_heads=torch_params.n_kv_heads,
            head_dim=torch_params.dim // torch_params.n_heads,
            dtype=layer_dtype,
            device=layer_device,
        )
    print(f"✓ KV caches initialized in all layers")

    # Convert prompt tokens to tensor
    tokens = torch.tensor([prompt_tokens], device=device_for_tensor, dtype=torch.long)  # [1, prompt_len]
    start_pos = 0

    # Prefill: process the prompt
    print(f"Prefilling with {len(prompt_tokens)} tokens...")
    with torch.no_grad():
        logits = torch_model.forward_inference(tokens, start_pos)
        # Synchronize XLA operations if using XLA
        if TORCH_XLA_AVAILABLE:
            torch_xla.sync()  # Synchronize XLA operations
    print(f"✓ Prefill complete")

    # Generate tokens autoregressively with greedy sampling
    generated_tokens = []
    stop_tokens = {tokenizer.eos_id, tokenizer.eot_id}
    
    print(f"Generating up to {max_gen_len} tokens...")
    current_pos = len(prompt_tokens)
    
    for step in range(max_gen_len):
        # Get logits for the last token position
        # logits shape: [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # Greedy sampling: argmax
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        # Synchronize XLA operations before materializing with .item()
        if TORCH_XLA_AVAILABLE:
            torch_xla.sync()  # Synchronize XLA operations
        
        next_token_val = int(next_token.item())
        
        # Stop if we hit a stop token
        if next_token_val in stop_tokens:
            print(f"  Stopped at step {step} (stop token: {next_token_val})")
            break
        
        generated_tokens.append(next_token_val)
        
        # Prepare next token for forward pass
        next_token_tensor = torch.tensor([[next_token_val]], device=device_for_tensor, dtype=torch.long)  # [1, 1]
        
        # Forward pass with single new token
        with torch.no_grad():
            logits = torch_model.forward_inference(next_token_tensor, current_pos)
            # Synchronize XLA operations after forward pass
            if TORCH_XLA_AVAILABLE:
                torch_xla.sync()  # Synchronize XLA operations
        
        current_pos += 1
        
        if (step + 1) % 10 == 0:
            print(f"  Generated {step + 1} tokens...")

    print(f"✓ Generation complete: {len(generated_tokens)} tokens generated")

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)
    full_text = test_prompt + generated_text

    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)

    # Save generated text to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"✓ Saved generated text to {output_path}")
    print(f"\nGenerated text ({len(generated_tokens)} tokens):")
    print(f"  {generated_text}")
    print(f"\nFull text (prompt + generated):")
    print(f"  {full_text}")

    # Clean up PyTorch caches
    for block in torch_model.layers:
        block.attention.cache = None

    print("\n" + "="*80)
    print("PYTORCH GENERATION COMPLETE")
    print("="*80)

    return {
        "output_file": str(output_path),
        "prompt": test_prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "num_tokens": len(generated_tokens),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test PyTorch LLaMA model text generation with greedy sampling"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory. Should have 'original' subdirectory with .pth file and tokenizer.model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="torch_output.txt",
        help="Output file path for saving generated text (default: torch_output.txt)"
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PYTORCH TEXT GENERATION TEST")
    print("="*80)

    results = test_torch_forward_pass(args.model_path, args.output_file, args.max_gen_len)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Output saved to: {results['output_file']}")
    print(f"Generated {results['num_tokens']} tokens")
    print("="*80)