import torch
import jax
import jax.numpy as jnp
from jax.dlpack import from_dlpack

# 1. Set a seed for PyTorch on the GPU for deterministic generation.
torch.cuda.manual_seed(42)

# 2. Generate a large random tensor on the GPU with PyTorch.
torch_tensor_gpu = torch.randn(1000, 1000, device='cuda')

# 3. Create a DLPack capsule from the PyTorch tensor. This does not copy data.
dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor_gpu)
print(dlpack_capsule)

# 4. Consume the DLPack capsule in JAX to create a JAX array.
# This is a zero-copy operation; the JAX array points to the same GPU memory.
jax_array_gpu = from_dlpack(dlpack_capsule)

# Now `torch_tensor_gpu` and `jax_array_gpu` point to the same data on the GPU.
# You can verify they have the same values.
print("PyTorch tensor on device:", torch_tensor_gpu.device)
print("JAX array on device:", jax_array_gpu.device)

# To verify, let's compare the sum of both arrays.
# Note: JAX operations are asynchronous, so we block until computation is done.
print("Sum of torch tensor:", torch.sum(torch_tensor_gpu).item())
print("Sum of jax array:", jnp.sum(jax_array_gpu).item())

# The JAX array is read-only by default after creation from DLPack
# to prevent accidental modification from two frameworks, which can cause
# unexpected behavior. If you need to modify it in JAX, you can copy it.
jax_array_gpu_copy = jnp.copy(jax_array_gpu)