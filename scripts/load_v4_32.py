# load_v4_32.py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.linen as nn
import orbax.checkpoint as ocp

# 1. Initialize multi-host environment (REQUIRED for v4-32)
jax.distributed.initialize()

# Define the identical Toy Llama Model
class ToyLlama(nn.Module):
    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.vocab_size, self.hidden_size, name='tok_embeddings')(x)
        x = nn.Dense(self.hidden_size, name='attn_proj')(x)
        return x

def main():
    # 2. Setup the Distributed Device Mesh (16 devices total)
    devices = jax.devices()
    print(f"Worker {jax.process_index()} sees {len(jax.local_devices())} local devices.")
    mesh = Mesh(devices, axis_names=('tp',))

    # 3. Setup dummy inputs and get Abstract Shapes
    model = ToyLlama(vocab_size=32000, hidden_size=4096)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)

    # Lazily evaluate shapes to prevent OOM
    abstract_vars = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy_input)

    # 4. Define the target sharding for the v4-32
    def get_sharding(val):
         return NamedSharding(mesh, P('tp'))
         
    sharding_tree = jax.tree_util.tree_map(get_sharding, abstract_vars)

    # 5. Tell Orbax how to shard the parameters upon restoration
    restore_args_tree = jax.tree_util.tree_map(
        lambda s: ocp.RestoreArgs(sharding=s),
        sharding_tree
    )

    # 6. Initialize CheckpointManager
    gcs_checkpoint_dir = 'gs://model-weights-1510/toy-llama'
    step = 100 # The step we uploaded earlier
    
    with ocp.CheckpointManager(gcs_checkpoint_dir) as mngr:
        # Check if the step exists in GCS
        if step not in mngr.all_steps():
            raise ValueError(f"Step {step} not found in {gcs_checkpoint_dir}")

        print(f"Worker {jax.process_index()}: Downloading and sharding weights...")
        
        # 7. Restore into the mesh
        with mesh:
            # StandardRestore uses the abstract_vars as a structure guide
            # and restore_args_tree tells it how to shard across the mesh.
            restored_vars = mngr.restore(
                step,
                args=ocp.args.StandardRestore(
                    item=abstract_vars, 
                    restore_args=restore_args_tree
                )
            )

    print(f"Worker {jax.process_index()}: Successfully loaded sharded weights!")
    
    # Verify sharding (only printing from worker 0 to avoid terminal spam)
    if jax.process_index() == 0:
        embed_sharding = restored_vars['params']['tok_embeddings']['embedding'].sharding
        print(f"Embedding Sharding: {embed_sharding}")

if __name__ == "__main__":
    main()
