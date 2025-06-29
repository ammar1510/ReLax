import numpy as np

def compare_logits(file1="jax_logits.npy", file2="torch_logits.npy"):
    """
    Loads two .npy files containing logits and compares them.

    Args:
        file1 (str): Path to the first .npy file.
        file2 (str): Path to the second .npy file.
    """
    try:
        logits1 = np.load(file1)
        logits2 = np.load(file2)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both logit files exist.")
        return

    print(f"Shapes: {logits1.shape} vs {logits2.shape}")

    if logits1.shape != logits2.shape:
        print("Shapes do not match. Cannot perform detailed comparison.")
        return

    try:
        np.testing.assert_allclose(logits1, logits2, rtol=1e-4, atol=1e-4)
        print("✅ Logits are very close.")
    except AssertionError as e:
        print("❌ Logits are different.")
        print(e)

    # Print some stats about the differences
    abs_diff = np.abs(logits1 - logits2)
    rel_diff = abs_diff / (np.abs(logits2) + 1e-9) # Avoid division by zero
    
    print("\n--- Difference Statistics ---")
    print(f"  Max absolute difference: {np.max(abs_diff):.6f}")
    print(f"  Mean absolute difference: {np.mean(abs_diff):.6f}")
    print(f"  Max relative difference: {np.max(rel_diff):.6f}")
    print(f"  Mean relative difference: {np.mean(rel_diff):.6f}")


if __name__ == "__main__":
    compare_logits() 