from typing import Any, Dict

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

    def create_train_state(
        self, key: jax.random.PRNGKey, dummy_input: jax.Array
    ) -> TrainState:
        """Creates the initial TrainState for SFT.

        Args:
            key: A JAX random key for parameter initialization.
            dummy_input: A batch of dummy data to initialize the model structure.

        Returns:
            An initial TrainState.
        """
        params = self.model.init(key, dummy_input)["params"]
        opt_state = self.optimizer.init(params)
        return TrainState(params=params, opt_state=opt_state, step=0)

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
