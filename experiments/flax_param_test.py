import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.random import PRNGKey

class SimpleModule(nn.Module):
    """
    A simple Flax module to demonstrate parameter initialization.
    """
    def setup(self):
        """
        Declare two parameters with the same shape and initializer.
        """
        # Declare the first parameter, 'param_a'
        self.param_a = self.param(
            'param_a', 
            nn.initializers.normal(stddev=0.02), # Use a standard normal initializer
            (2, 3)                     # Shape of the parameter array
        )
        
        # Declare the second parameter, 'param_b', with the exact same setup
        self.param_b = self.param(
            'param_b', 
            nn.initializers.normal(stddev=0.02), # Use the same initializer
            (2, 3)                     # Use the same shape
        )

    def __call__(self, x):
        # This module doesn't need to do anything, 
        # as we are only interested in its parameters.
        return x

def verify_parameter_initialization():
    """
    Initializes the SimpleModule and verifies that the two parameters
    have different values.
    """
    print("--- Verifying Flax Parameter Initialization ---")

    # 1. Create a master PRNGKey. This is the source of randomness.
    master_key = PRNGKey(0)
    print(f"Master PRNGKey created: {master_key}")

    # 2. Instantiate the module.
    model = SimpleModule()

    # 3. Initialize the model's parameters.
    #    Flax uses the master_key to derive unique keys for each `self.param` call.
    #    We need a dummy input to determine the shapes of intermediate arrays,
    #    though it's not strictly necessary for this simple model.
    dummy_input = jnp.ones((1,))
    params = model.init(master_key, dummy_input)['params']
    
    print("\nModel parameters initialized.")
    print("Initialized 'param_a':\n", params['param_a'])
    print("Initialized 'param_b':\n", params['param_b'])

    # 4. Verify that the parameters are not equal.
    #    jnp.array_equal() will return False if the arrays are different.
    are_arrays_equal = jnp.array_equal(params['param_a'], params['param_b'])
    
    print("\n--- Verification Result ---")
    if not are_arrays_equal:
        print("✅ SUCCESS: `param_a` and `param_b` are NOT equal.")
        print("This confirms that Flax splits the PRNGKey for each parameter initialization.")
    else:
        print("❌ FAILURE: `param_a` and `param_b` are equal.")
        print("This would indicate that the same PRNGKey was used for both initializations.")

if __name__ == "__main__":
    verify_parameter_initialization() 