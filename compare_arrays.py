import numpy as np
import sys

def compare_npy_files(file1, file2):
    """
    Loads two .npy files and compares them for equality.
    """
    try:
        arr1 = np.load(file1)
        arr2 = np.load(file2)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return

    print(f"File 1 ({file1}) dtype: {arr1.dtype}")
    print(f"File 2 ({file2}) dtype: {arr2.dtype}")

    if str(arr1.dtype) == '|V2':
        try:
            from ml_dtypes import bfloat16
            print("Interpreting file 1 as bfloat16 and converting to float32 for comparison.")
            arr1 = arr1.view(bfloat16).astype('float32')
        except ImportError:
            print("\nError: The first array seems to be bfloat16, but the 'ml_dtypes' library is not installed.")
            print("Please install it to handle this data type: pip install ml_dtypes")
            return

    if arr1.shape != arr2.shape:
        arr1 = arr1.reshape(-1)
        arr2 = arr2.reshape(-1)
        print(f"Arrays have different shapes: {arr1.shape} vs {arr2.shape}")

    if np.allclose(arr1, arr2):
        print("Arrays are equal (within floating point tolerance).")
    else:
        print("Arrays are not equal.")
        diff = np.abs(arr1 - arr2)
        print(f"Maximum absolute difference: {np.max(diff)}")
        print(f"Mean absolute difference: {np.mean(diff)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_arrays.py <file1.npy> <file2.npy>")
    else:
        compare_npy_files(sys.argv[1], sys.argv[2]) 