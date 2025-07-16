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
    def train(self, train_loader: Any, num_epochs: int, state: TrainState):
        """The main training loop.

        Args:
            train_loader: The data loader for the training set.
            num_epochs: The total number of epochs to train for.
            state: The initial training state.
        """
        pass
