import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.linen as nn
import orbax.checkpoint as ocp

# 1. Initialize multi-host environment
jax.distributed.initialize()

# Define your model architecture
class ToyLlama(nn.Module):
    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.vocab_size, self.hidden_size, name='tok_embeddings')(x)
        x = nn.Dense(self.hidden_size, name='attn_proj')(x)
        return x

def main():
    mesh = Mesh(jax.devices(), axis_names=('tp',))
    gcs_checkpoint_dir = 'gs://model-weights-1510/toy-llama'
    step = 100 
    
    # 2. Get zero-memory pure abstract shapes 
    # (No metadata contamination from the old single-device checkpoint)
    model = ToyLlama(vocab_size=32000, hidden_size=4096)
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    abstract_vars = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy_input)

    # 3. Inject the v4-32 Sharding directly into the abstract shapes
    def apply_sharding(var):
        return jax.ShapeDtypeStruct(
            shape=var.shape,
            dtype=var.dtype,
            sharding=NamedSharding(mesh, P('tp'))
        )
        
    abstract_target = jax.tree_util.tree_map(apply_sharding, abstract_vars)

    # 4. Restore the single item directly
    with ocp.CheckpointManager(gcs_checkpoint_dir) as mngr:
        print(f"Worker {jax.process_index()}: Downloading and sharding weights...")
        
        with mesh:
            # We pass abstract_target directly to item. No Composite needed.
            restored_vars = mngr.restore(
                step,
                args=ocp.args.StandardRestore(item=abstract_target)
            )

    print(f"Worker {jax.process_index()}: Successfully loaded sharded weights!")

if __name__ == "__main__":
    main()
