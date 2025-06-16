import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze

# 1. A simple Flax Module
class SimpleMultiplier(nn.Module):
    """A simple module that multiplies input by a kernel parameter."""

    def setup(self):
        """Initializes the parameter in setup, like the LLaMA model."""
        self.kernel = self.param(
            'kernel',
            nn.initializers.ones,
            (1,)  # Shape is known for this test
        )

    def __call__(self, x):
        """Uses the parameter in the forward pass."""
        return x * self.kernel

def test_parameter_loading_and_application():
    """
    Tests if a Flax module correctly uses externally supplied parameters
    via the `apply` method, instead of its initialized values.
    """
    key = jax.random.PRNGKey(0)
    input_data = jnp.ones((1,), dtype=jnp.float32) * 2.0 # Input is 2.0

    # Initialize the model. This will set the internal `kernel` to 1.0.
    model = SimpleMultiplier()
    initial_params = model.init(key, input_data)['params']

    # Run the model with its own initialized parameters.
    # Expected output: 2.0 * 1.0 = 2.0
    output_with_initial_params = model.apply({'params': initial_params}, input_data)

    print(f"Initial kernel value: {initial_params['kernel']}")
    print(f"Output with initial params (2.0 * 1.0): {output_with_initial_params}")
    
    # 2. Simulate loading parameters with a different value.
    # We create a new parameter dictionary where 'kernel' is 5.0.
    loaded_kernel_value = jnp.ones((1,), dtype=jnp.float32) * 5.0
    loaded_params = freeze({'kernel': loaded_kernel_value})
    
    # 3. Apply the model with the new "loaded" parameters.
    # If this works correctly, the model should use 5.0, not 1.0.
    # Expected output: 2.0 * 5.0 = 10.0
    output_with_loaded_params = model.apply({'params': loaded_params}, input_data)
    
    print(f"Loaded kernel value: {loaded_params['kernel']}")
    print(f"Output with loaded params (2.0 * 5.0): {output_with_loaded_params}")

    # 4. Assert correctness
    assert jnp.allclose(output_with_initial_params, jnp.array([2.0])), "Model did not use initial param correctly."
    assert jnp.allclose(output_with_loaded_params, jnp.array([10.0])), "Model did not use loaded param correctly."

    print("\\nTest passed! The model correctly uses the externally supplied parameters.")

# To run the test:
if __name__ == '__main__':
    test_parameter_loading_and_application() 