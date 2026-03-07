import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp

# 1. Initialize multi-host environment
jax.distributed.initialize()

# (Include your ToyLlama definition here)

def main():
    mesh = Mesh(jax.devices(), axis_names=('tp',))
    gcs_checkpoint_dir = 'gs://your-bucket-name/llama-orbax-ckpts'
    step = 100 
    
    # 2. Get zero-memory abstract shapes 
    # (This traces the shapes instantly without running a forward pass or using memory)
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

    # 4. Restore the specific "default" item using Composite
    with ocp.CheckpointManager(gcs_checkpoint_dir) as mngr:
        print(f"Worker {jax.process_index()}: Downloading and sharding weights...")
        
        with mesh:
            restored = mngr.restore(
                step,
                args=ocp.args.Composite(
                    default=ocp.args.StandardRestore(item=abstract_target)
                )
            )
            
    # Extract the weights from the restored default item
    restored_vars = restored.default

    print(f"Worker {jax.process_index()}: Successfully loaded sharded weights!")

if __name__ == "__main__":
    main()
