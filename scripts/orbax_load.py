import os
import glob
import gc
import numpy as np
from safetensors import safe_open
import orbax.checkpoint as ocp
import ml_dtypes # Required for bfloat16 numpy support

def pytorch_key_to_jax_path(pt_key: str):
    """Converts a flat PyTorch key to a JAX tuple path."""
    pt_key = pt_key.replace(".weight", ".kernel")
    return pt_key.split(".")

def insert_into_pytree(tree: dict, path: list, value):
    """Recursively inserts a value into a nested dictionary."""
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = value

def main():
    safetensor_files = glob.glob("/path/to/downloaded/model/*.safetensors")
    
    # We need a temporary directory on a drive with ~200GB of free space
    temp_dir = "./temp_npy_weights"
    os.makedirs(temp_dir, exist_ok=True)

    print("--- PASS 1: Extracting tensors to local disk ---")
    for file in safetensor_files:
        with safe_open(file, framework="np", device="cpu") as f:
            for key in f.keys():
                # 1. Load a SINGLE tensor into RAM (~a few MBs to ~2GB max)
                tensor = f.get_tensor(key)
                
                # 2. Transpose 2D matrices for JAX
                if len(tensor.shape) == 2 and "embed" not in key:
                    tensor = tensor.transpose()

                # 3. Cast to bfloat16
                tensor = tensor.astype(ml_dtypes.bfloat16)
                
                # 4. Save to local disk as a temporary .npy file
                safe_filename = key.replace(".", "_") + ".npy"
                temp_filepath = os.path.join(temp_dir, safe_filename)
                np.save(temp_filepath, tensor)
                
                # 5. IMMEDIATELY free the RAM!
                del tensor
        
        # Force garbage collection just to be safe
        gc.collect() 
        print(f"Processed {file}")

    print("--- PASS 2: Building Lazy PyTree and Saving to Orbax ---")
    jax_pytree = {}
    
    # Re-iterate through the files (or you could have saved a map of keys in Pass 1)
    for file in safetensor_files:
        with safe_open(file, framework="np", device="cpu") as f:
            for key in f.keys():
                safe_filename = key.replace(".", "_") + ".npy"
                temp_filepath = os.path.join(temp_dir, safe_filename)
                
                # THE MAGIC TRICK: mmap_mode="r"
                # This returns a lazy pointer to the disk file, taking almost 0 RAM.
                lazy_tensor = np.load(temp_filepath, mmap_mode="r")
                
                jax_path = pytorch_key_to_jax_path(key)
                insert_into_pytree(jax_pytree, jax_path, lazy_tensor)

    print("Lazy PyTree constructed! Current RAM usage is tiny.")
    
    # Point this to your GCS Bucket
    gcs_checkpoint_dir = 'gs://my-bucket/100B-model-orbax'
    
    mngr = ocp.CheckpointManager(gcs_checkpoint_dir)
    
    print(f"Streaming chunks from local disk to {gcs_checkpoint_dir}...")
    mngr.save(
        step=0, 
        args=ocp.args.StandardSave(jax_pytree)
    )
    
    mngr.wait_until_finished()
    print("Upload complete!")

    # Cleanup local disk
    # import shutil
    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
