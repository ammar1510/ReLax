from typing import Any, Dict
from functools import partial
import dataclasses

import jax
import optax

from .trainer import Trainer, TrainState


class SFTTrainer(Trainer):
    """A trainer for Supervised Fine-Tuning (SFT)."""

    def __init__(self, model: Any, optimizer: optax.GradientTransformation):
        """Initializes the SFTTrainer.

        Args:
            model: The Flax model to be trained.
            optimizer: The Optax optimizer.
        """
        super().__init__(model, optimizer)

    def compute_loss(self, params, batch: Dict[str, jax.Array]) -> jax.Array:
        """Computes the softmax cross-entropy loss for SFT.

        Args:
            params: The model's parameters.
            batch: A dictionary containing 'input_ids' and 'labels'.

        Returns:
            The mean loss for the batch, ignoring padding.
        """
        logits = self.model.apply({"params": params}, batch["input_ids"])

        # Calculate loss, ignoring padding
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["labels"]
        )

        # Create a mask to ignore padding tokens (assuming padding label is 0)
        loss_mask = batch["labels"] > 0

        # Compute mean loss over non-padded tokens
        masked_loss = loss * loss_mask
        mean_loss = masked_loss.sum() / loss_mask.sum()

        return mean_loss

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

    def train(self, train_loader: Any, num_epochs: int, state: TrainState):
        """The main training loop.

        Args:
            train_loader: The data loader for the training set.
            num_epochs: The total number of epochs to train for.
            state: The initial training state.
        """
        for epoch in range(num_epochs):
            for batch in train_loader:
                state, loss = self.train_step(state, batch)

                if state.step % 100 == 0:
                    print(f"Epoch {epoch}, Step {state.step}, Loss: {loss:.4f}")

        return state
