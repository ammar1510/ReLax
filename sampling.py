import jax
import jax.numpy as jnp
from jax import random
from jax import jit

"""Sampling functions for the model.
This entire file needs to be JIT compatible.

All sample functions follow the signature: (logits, key) -> token_ids
For functions with extra params (temperature, k, p), use functools.partial
to bind them before passing to the engine.
"""


@jit
def temperature_scale(logits: jax.Array, temperature: float) -> jax.Array:
    safe_temperature = jnp.maximum(temperature, 1e-6)
    return logits / safe_temperature


def greedy(logits: jax.Array, key: jax.Array) -> jax.Array:
    return jnp.argmax(logits, axis=-1)


def categorical(logits: jax.Array, key: jax.Array, temperature: float = 1.0) -> jax.Array:
    scaled_logits = temperature_scale(logits, temperature)
    return random.categorical(key, scaled_logits, axis=-1)


def top_k(logits: jax.Array, key: jax.Array, k: int = 10, temperature: float = 1.0) -> jax.Array:
    if temperature == 0.0:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
        idx_in_top_k = jnp.argmax(top_k_logits)
        return top_k_indices[idx_in_top_k]
    scaled_logits = temperature_scale(logits, temperature)
    top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, k)
    sampled_index = random.categorical(key, top_k_logits)
    return jnp.take(top_k_indices, sampled_index)


def top_p(logits: jax.Array, key: jax.Array, p: float = 0.9, temperature: float = 1.0) -> jax.Array:
    scaled_logits = temperature_scale(logits, temperature)

    sorted_indices = jnp.flip(jnp.argsort(scaled_logits, axis=-1), axis=-1)
    sorted_logits = jnp.take_along_axis(scaled_logits, sorted_indices, axis=-1)

    probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative_probs = jnp.cumsum(probs, axis=-1)

    indices_to_remove = cumulative_probs > p
    indices_to_remove = jnp.roll(indices_to_remove, 1, axis=-1)
    indices_to_remove = indices_to_remove.at[:, 0].set(False)

    neg_inf = jnp.finfo(scaled_logits.dtype).min
    updates = jnp.where(indices_to_remove, neg_inf, sorted_logits)

    scatter_indices = jnp.argsort(sorted_indices, axis=-1)
    logits_filtered = jnp.take_along_axis(updates, scatter_indices, axis=-1)

    return random.categorical(key, logits_filtered, axis=-1)
