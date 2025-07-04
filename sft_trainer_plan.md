### Plan for `SFTTrainer`

The `SFTTrainer`'s primary responsibility is to provide the specific logic for Supervised Fine-Tuning, namely: how to initialize the model's state and how to compute the standard cross-entropy loss.

1.  **File Location**: The new class will reside in `trainers/sft_trainer.py`.

2.  **Class Definition**:
    *   It will be named `SFTTrainer`.
    *   It will inherit directly from the `Trainer` base class we just created (`from .trainer import Trainer`).
    *   Its `__init__` method will simply call `super().__init__` to pass the model and optimizer up to the parent class.

3.  **Implementation of Abstract Methods**: This is where the core work happens. The `SFTTrainer` will provide concrete implementations for the two abstract methods defined in `Trainer`.

    *   **`create_train_state(self, key: jax.random.PRNGKey, dummy_input: jax.Array) -> TrainState`**:
        *   This method will be responsible for creating the initial `TrainState`.
        *   It will initialize the model's parameters by calling `self.model.init(key, dummy_input)`. This requires a `dummy_input` tensor with the correct shape (e.g., `(batch_size, seq_len)`) to tell JAX how to construct the model's weights.
        *   It will then initialize the optimizer state based on these newly created parameters using `self.optimizer.init(params)`.
        *   Finally, it will bundle the `params`, `opt_state`, and an initial `step` of `0` into a `TrainState` object and return it.

    *   **`compute_loss(self, params, batch: dict) -> jax.Array`**:
        *   This method defines the heart of SFT: the loss function.
        *   It will receive the current `params` and a `batch` of data. The `batch` is expected to be a dictionary containing at least `'input_ids'` and `'labels'`.
        *   It will perform a forward pass of the model to get the logits: `logits = self.model.apply({'params': params}, batch['input_ids'])`.
        *   It will calculate the cross-entropy loss between the model's `logits` and the ground-truth `labels`. A standard function like `optax.softmax_cross_entropy_with_integer_labels` is ideal here.
        *   **Important**: It must correctly handle padding. Language model batches are often padded to the same length. The loss should only be computed on the actual tokens, not the padding. This is typically done by creating a `loss_mask` where padded tokens in the `labels` (often represented by `0` or `-100`) are masked out, and then taking the mean of the loss over the unmasked tokens.
        *   It will return the final, single scalar value for the loss.

---

### Implementation Prompts

Here is the step-by-step implementation plan, broken down into structured prompts.

#### Part 1: Create `trainers/sft_trainer.py` and Class Skeleton

*   **Goal**: Create the new file and the `SFTTrainer` class definition.
*   **Prompt**: "Create a new file at `trainers/sft_trainer.py`. In this file, import `jax`, `optax`, and any other necessary libraries. From `.trainer`, import the `Trainer` and `TrainState` classes. Define a new class `SFTTrainer` that inherits from `Trainer`. Its `__init__` method should just call `super().__init__(model, optimizer)`."

#### Part 2: Implement `create_train_state`

*   **Goal**: Implement the logic for initializing the training state.
*   **Prompt**: "In the `SFTTrainer` class, provide a concrete implementation for the `create_train_state` method. It should accept `self`, a JAX random `key`, and a `dummy_input` tensor. Use the `key` and `dummy_input` to initialize the model's parameters with `self.model.init`. Then, initialize the optimizer state using `self.optimizer.init`. Return a new `TrainState` object containing the initialized `params`, `opt_state`, and a `step` counter set to `0`."

#### Part 3: Implement `compute_loss`

*   **Goal**: Implement the supervised fine-tuning loss calculation.
*   **Prompt**: "Implement the `compute_loss` method in `SFTTrainer`. This method will receive the model `params` and a `batch` dictionary.
    1.  Extract the `input_ids` and `labels` from the batch.
    2.  Perform a forward pass with `self.model.apply` to get the `logits`.
    3.  Use `optax.softmax_cross_entropy_with_integer_labels` to get the raw, per-token loss.
    4.  Create a boolean `loss_mask` to identify non-padding tokens in the `labels` (assume padded labels are `0`).
    5.  Calculate the final mean loss by multiplying the raw loss by the mask and dividing by the total number of non-padded tokens.
    6.  Return the resulting scalar loss." 