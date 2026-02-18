"""Base trainer class for ReLax models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict


@struct.dataclass
class TrainState:
    """Training state with parameters, optimizer state, and step counter.

    This is a custom TrainState that works with JAX transformations.
    Unlike Flax's TrainState, this is designed for RL training where
    we may need to track additional state like reference model logprobs.
    """
    step: int
    params: FrozenDict[str, Any]
    opt_state: Any

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be updated.

        Returns:
            An updated instance of `self` with `step` incremented, new `params`,
            and updated `opt_state` and any additional keyword arguments.
        """
        updates, new_opt_state = kwargs.pop('tx').update(
            grads, self.opt_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new training state.

        Args:
            apply_fn: Usually set to `model.apply()`. Not used in this implementation
                     but kept for compatibility with Flax's TrainState API.
            params: Model parameters.
            tx: An Optax optimizer.
            **kwargs: Additional dataclass fields.

        Returns:
            A `TrainState` instance.
        """
        opt_state = tx.init(params)
        return cls(
            step=0,
            params=params,
            opt_state=opt_state,
            **kwargs,
        )


class Trainer(ABC):
    """Abstract base class for model trainers.

    Provides common functionality for training loops, checkpointing,
    and logging. Subclasses should implement train_step and any
    task-specific methods.
    """

    def __init__(
        self,
        model: Any,
        params: FrozenDict,
        optimizer: optax.GradientTransformation,
        seed: int = 42,
    ):
        """Initialize trainer.

        Args:
            model: The model to train (Flax module).
            params: Initial model parameters.
            optimizer: Optax optimizer.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.optimizer = optimizer
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        # Create training state
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )

    @abstractmethod
    def train_step(self, state: TrainState, batch: Dict[str, jax.Array], rng: jax.random.PRNGKey) -> tuple[TrainState, Dict[str, float]]:
        """Performs a single training step.

        Args:
            state: Current training state.
            batch: Batch of training data.
            rng: Random number generator key.

        Returns:
            Tuple of (updated_state, metrics_dict).
        """
        pass

    def get_rng(self) -> jax.random.PRNGKey:
        """Get a new random key."""
        self.rng, subkey = jax.random.split(self.rng)
        return subkey

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        raise NotImplementedError("Checkpoint saving not yet implemented")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        raise NotImplementedError("Checkpoint loading not yet implemented")
