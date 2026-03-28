# save_single_tpu.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
import os

# 1. Define the Toy Llama Model
class ToyLlama(nn.Module):
    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.vocab_size, self.hidden_size, name='tok_embeddings')(x)
        x = nn.Dense(self.hidden_size, name='attn_proj')(x)
        return x

def main():
    # 2. Instantiate model and generate actual weights locally
    model = ToyLlama(vocab_size=32000, hidden_size=4096)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    
    print("Initializing model weights on single TPU...")
    # These weights sit entirely on the single TPU's memory
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 3. Configure the CheckpointManager pointing to GCS
    gcs_checkpoint_dir = 'gs://model-weights-1510/toy-llama'
    
    # Options like how many checkpoints to keep
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    
    # 4. Save the weights using CheckpointManager
    with ocp.CheckpointManager(gcs_checkpoint_dir, options=options) as mngr:
        step = 100 # Example step number
        
        print(f"Uploading checkpoint step {step} to GCS...")
        # StandardSave is the modern Orbax wrapper for saving PyTrees
        mngr.save(
            step, 
            args=ocp.args.StandardSave(variables)
        )
        
        # Wait for the async save to finish
        mngr.wait_until_finished()
        
    print("Upload complete!")

if __name__ == "__main__":
    main()
