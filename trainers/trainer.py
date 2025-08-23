import abc
import dataclasses
from functools import partial
from typing import Any, Callable

import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


# Define a custom TrainState using flax.struct.dataclass for JAX transformations
@flax.struct.dataclass
class TrainState:
    """A dataclass to hold the state of training.

    Attributes:
        params: The model's parameters (weights).
        opt_state: The state of the optimizer (e.g., momentum).
        step: The current training step.
    """

    params: optax.Params
    opt_state: optax.OptState
    step: int


class Trainer(abc.ABC):
    """Abstract base class for all trainers."""

    def __init__(self, model: Any, optimizer: optax.GradientTransformation):
        """Initializes the Trainer.

        Args:
            model: The Flax model to be trained.
            optimizer: The Optax optimizer.
        """
        self.model = model
        self.optimizer = optimizer

    @abc.abstractmethod
    def compute_loss(self, params, batch) -> jax.Array:
        """Computes the loss for a batch.

        Args:
            params: Model parameters.
            batch: Training batch.

        Returns:
            Loss value.
        """
        pass

    @partial(jax.jit, static_argnames=["self"])
    def train_step(self, state: TrainState, batch) -> tuple[TrainState, jax.Array]:
        """Performs a single, JIT-compiled training step."""

        def loss_fn(params):
            return self.compute_loss(params, batch)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        updates, new_opt_state = self.optimizer.update(
            grads, state.opt_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)

        new_state = dataclasses.replace(
            state, params=new_params, opt_state=new_opt_state, step=state.step + 1
        )
        return new_state, loss

    @abc.abstractmethod
    def train(self, train_loader: Any, num_epochs: int, state: TrainState) -> TrainState:
        """The main training loop.

        Args:
            train_loader: The data loader for the training set.
            num_epochs: The total number of epochs to train for.
            state: The initial training state.

        Returns:
            Final training state.
        """
        pass
