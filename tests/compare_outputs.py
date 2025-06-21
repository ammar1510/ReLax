import numpy as np
import os

def compare_logits():
    output_dir = "output"
    jax_logits_path = os.path.join(output_dir, "jax_logits.npy")
    torch_logits_path = os.path.join(output_dir, "torch_logits.npy")

    if not os.path.exists(jax_logits_path) or not os.path.exists(torch_logits_path):
        print("Logit files not found. Please run generate_jax_output.py and generate_torch_output.py first.")
        return

    jax_logits = np.load(jax_logits_path)
    torch_logits = np.load(torch_logits_path)

    print("Comparing JAX and PyTorch logits...")
    print(f"JAX logits shape: {jax_logits.shape}")
    print(f"PyTorch logits shape: {torch_logits.shape}")

    if jax_logits.shape != torch_logits.shape:
        print("\nShapes do not match!")
        return
        
    if np.allclose(jax_logits, torch_logits, atol=1e-5, rtol=1e-5):
        print("\n✅ Logits are equal (within tolerance).")
    else:
        print("\n❌ Logits are NOT equal.")
        absolute_diff = np.abs(jax_logits - torch_logits)
        relative_diff = absolute_diff / (np.abs(torch_logits) + 1e-9) # add epsilon to avoid division by zero
        
        print(f"  - Maximum absolute difference: {np.max(absolute_diff)}")
        print(f"  - Mean absolute difference: {np.mean(absolute_diff)}")
        print(f"  - Maximum relative difference: {np.max(relative_diff)}")
        print(f"  - Mean relative difference: {np.mean(relative_diff)}")

if __name__ == "__main__":
    compare_logits() 