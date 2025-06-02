import jax
import jax.numpy as jnp
from jax import random
from jax import jit
import abc # For Abstract Base Class

"""Sampling functions for the model.
This entire file needs to be JIT compatible.
"""

#@jit
def temperature_scale(logits: jax.Array, temperature: float) -> jax.Array:
    """
    Scales logits by temperature.

    Args:
        logits: The input logits array (usually shape [vocab_size]).
        temperature: The sampling temperature. Lower values make the distribution
                     sharper, higher values make it flatter. Must be positive.

    Returns:
        The scaled logits.
    """
    # Ensure temperature is positive to avoid division by zero or negative values
    # Small epsilon added for numerical stability if temperature is very close to 0
    safe_temperature = jnp.maximum(temperature, 1e-6)
    return logits / safe_temperature

#@jit
def sample_top_k(logits: jax.Array, k: int, key: random.PRNGKey) -> jax.Array:
    """
    Samples from the top-k logits.

    Filters the logits to keep only the top k values, then samples from the
    resulting distribution.

    Args:
        logits: The input logits array (usually shape [vocab_size]).
        k: The number of top logits to consider.
        key: JAX PRNG key for random sampling.

    Returns:
        The sampled token ID (scalar integer array).
    """
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
    # Sample from the filtered distribution
    sampled_index = random.categorical(key, top_k_logits)
    # Map the sampled index back to the original vocabulary index
    sampled_token_id = jnp.take(top_k_indices, sampled_index)
    return sampled_token_id

#@jit
def sample_top_p(logits: jax.Array, p: float, key: random.PRNGKey, temperature: float = 1.0) -> jax.Array:
    """
    Samples using nucleus sampling (top-p).

    Sorts logits, calculates the cumulative probability, filters out tokens
    whose cumulative probability exceeds p, scales by temperature, and then samples.

    Args:
        logits: The input logits array (usually shape [vocab_size]).
        p: The cumulative probability threshold for nucleus sampling.
        key: JAX PRNG key for random sampling.
        temperature: Sampling temperature, applied before sampling.

    Returns:
        The sampled token ID (scalar integer array).
    """
    # Apply temperature scaling first
    scaled_logits = temperature_scale(logits, temperature)

    # Sort logits in descending order
    sorted_indices = jnp.argsort(scaled_logits)[::-1]
    sorted_logits = scaled_logits[sorted_indices]

    # Calculate cumulative probabilities
    probs = jax.nn.softmax(sorted_logits)
    cumulative_probs = jnp.cumsum(probs, axis=-1)

    # Find indices where cumulative probability exceeds p
    # We include the first element always, and then elements where the *previous*
    # cumulative probability was less than p.
    indices_to_remove = cumulative_probs > p
    # Shift right and insert True at the beginning
    indices_to_remove = jnp.roll(indices_to_remove, 1)
    indices_to_remove = indices_to_remove.at[0].set(False)

    # Set logits for removed tokens to negative infinity
    # Need to use the *original* indices to modify the *scaled* logits
    # We create a mask based on the sorted order removal criteria
    # Use scatter update to set the logits of removed tokens
    # First, create a large negative value
    neg_inf = jnp.finfo(scaled_logits.dtype).min
    # Create an array of updates, setting removed indices to neg_inf
    updates = jnp.where(indices_to_remove, neg_inf, sorted_logits)
    # Scatter these updates back to the original logit positions based on sorted_indices
    # Need to create the inverse permutation of sorted_indices
    scatter_indices = jnp.argsort(sorted_indices)
    logits_filtered = updates[scatter_indices]


    # Sample from the filtered distribution
    sampled_token_id = random.categorical(key, logits_filtered)
    return sampled_token_id 

class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, logits: jax.Array, key: random.PRNGKey) -> jax.Array:
        """Samples a token from the logits distribution."""
        pass

class GreedySampler(Sampler):
    """Selects the token with the highest probability (argmax)."""
    def sample(self, logits: jax.Array, key: random.PRNGKey) -> jax.Array:
        # key is not used for greedy, but kept for interface consistency
        return jnp.argmax(logits, axis=-1)

class CategoricalSampler(Sampler):
    """Samples from the distribution after applying temperature."""
    def __init__(self, temperature: float):
        if temperature <= 0:
            # Allow temperature of 0 to mean greedy, but it's better to use GreedySampler for that.
            # For strict categorical sampling, temperature should be positive.
            # Let's keep the strict interpretation here.
            raise ValueError("Temperature must be positive for CategoricalSampler.")
        self.temperature = temperature

    def sample(self, logits: jax.Array, key: random.PRNGKey) -> jax.Array:
        scaled_logits = temperature_scale(logits, self.temperature)
        return random.categorical(key, scaled_logits, axis=-1)

class TopKSampler(Sampler):
    """Samples from the top-k tokens after applying temperature."""
    def __init__(self, k: int, temperature: float):
        if k <= 0:
            raise ValueError("k must be positive for TopKSampler.")
        # temperature can be 0 for greedy sampling from top-k
        if temperature < 0:
            raise ValueError("Temperature cannot be negative.")
        self.k = k
        self.temperature = temperature

    def sample(self, logits: jax.Array, key: random.PRNGKey) -> jax.Array:
        if self.temperature == 0.0:
            # Greedy sampling from the top-k
            # Get top-k logits and their original indices
            # Note: jax.lax.top_k might not be JIT-friendly with dynamic k from self.k inside a JITted sample
            # but here sample itself is not JITted directly.
            top_k_logits_values, top_k_indices = jax.lax.top_k(logits, self.k)
            # Find the index of the max logit *within* the top_k_logits_values
            idx_in_top_k = jnp.argmax(top_k_logits_values)
            # Return the original token index
            return top_k_indices[idx_in_top_k]
        
        scaled_logits = temperature_scale(logits, self.temperature)
        # The existing sample_top_k function expects (potentially scaled) logits, k, and a key.
        return sample_top_k(scaled_logits, self.k, key)

class TopPSampler(Sampler):
    """Samples using nucleus (top-p) sampling. Temperature is applied internally by sample_top_p."""
    def __init__(self, p: float, temperature: float):
        if not (0 < p <= 1.0):
            raise ValueError("p must be in (0, 1] for TopPSampler.")
        if temperature <= 0: # sample_top_p internally handles temperature > 0
            raise ValueError("Temperature must be positive for TopPSampler if using the helper.")
            # If a temp of 0 was desired with top-p, logic would need to adapt.
            # The current sample_top_p scales with temperature and expects it > 0.
        self.p = p
        self.temperature = temperature

    def sample(self, logits: jax.Array, key: random.PRNGKey) -> jax.Array:
        # The existing sample_top_p function handles temperature scaling internally.
        return sample_top_p(logits, self.p, key, temperature=self.temperature) 