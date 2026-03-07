import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import orbax.checkpoint as ocp

# 1. Initialize multi-host environment
jax.distributed.initialize()

def main():
    mesh = Mesh(jax.devices(), axis_names=('tp',))
    gcs_checkpoint_dir = 'gs://model-weights-1510/toy-llama'
    step = 100 
    
    with ocp.CheckpointManager(gcs_checkpoint_dir) as mngr:
        print(f"Worker {jax.process_index()}: Reading checkpoint metadata from GCS...")
        metadata = mngr.item_metadata(step)
        
        # Extract the underlying PyTree dictionary from the metadata
        tree_meta = metadata.tree if hasattr(metadata, 'tree') else metadata

        # 2. BUILD ABSTRACT TARGET WITH EMBEDDED SHARDING
        def process_metadata_leaf(meta):
            if hasattr(meta, 'shape') and hasattr(meta, 'dtype'):
                # FIX: We now pass the sharding directly into the JAX structure!
                return jax.ShapeDtypeStruct(
                    shape=meta.shape, 
                    dtype=meta.dtype,
                    sharding=NamedSharding(mesh, P('tp'))
                )
            else:
                return meta

        # Walk through the metadata tree and build our target PyTree
        abstract_target = jax.tree.map(process_metadata_leaf, tree_meta)

        print(f"Worker {jax.process_index()}: Downloading and sharding weights...")
        
        # 3. RESTORE USING THE TARGET STRUCTURE
        with mesh:
            # FIX: No more restore_args_tree! Orbax reads the sharding directly 
            # from the abstract_target.
            restored_vars = mngr.restore(
                step,
                args=ocp.args.StandardRestore(abstract_target)
            )

    print(f"Worker {jax.process_index()}: Successfully loaded sharded weights!")

if __name__ == "__main__":
    main()
