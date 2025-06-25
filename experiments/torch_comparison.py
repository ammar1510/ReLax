import torch
import json
from pathlib import Path
import numpy as np
from safetensors.torch import save_file

from experiments.torch_llama import ModelArgs, Transformer

def create_mock_torch_model_and_inputs():
    """
    This function creates a mock LLaMa model using PyTorch, runs a forward pass
    with mock inputs, and saves the model weights and output logits.
    """
    config_path = 'artifacts/weights/Llama-3.2-3B/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_args = ModelArgs(
        dim=config['hidden_size'],
        n_layers=config['num_hidden_layers'],
        n_heads=config['num_attention_heads'],
        n_kv_heads=config['num_key_value_heads'],
        vocab_size=config['vocab_size'],
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=config['rms_norm_eps'],
        rope_theta=config['rope_theta'],
        max_seq_len=2048,
        max_batch_size=1,
        flash=False,
    )

    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        raise ValueError("CPU is not supported for this experiment.")

    model = Transformer(model_args).to(device)
    model.eval()

    batch_size = 1
    seq_len = 64
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)

    # The forward_inference function handles the necessary attention mask and positional encodings.
    logits = model.forward_inference(tokens, start_pos=0)

    # Save outputs
    np.save('torch_logits.npy', logits.cpu().detach().numpy())
    save_file(model.state_dict(), 'mock_weights.safetensors')

    print("PyTorch model executed and outputs saved to 'torch_logits.npy' and 'mock_weights.safetensors'.")
    print(f"Logits shape: {logits.shape}")


if __name__ == '__main__':
    create_mock_torch_model_and_inputs() 