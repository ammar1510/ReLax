### Plan for the Base Trainer

The base `Trainer` will be designed as an abstract class that handles the generic, boilerplate parts of a JAX-based training loop. Specific trainers like `SFTTrainer` or `GRPOTrainer` will inherit from it and only need to implement the logic that is unique to their training algorithm (i.e., how the loss is computed).

Here's the planned structure:

1.  **`TrainState` Dataclass**: Following functional programming principles, we'll use a state-passing style. A `TrainState` will be a simple dataclass that bundles together all the stateful objects that change during training. This includes:
    *   `params`: The model's trainable weights.
    *   `opt_state`: The state of the optimizer (e.g., momentum vectors).
    *   `step`: The current training step count.

2.  **`Trainer` Abstract Class**: This will be the core component in `trainers/trainer.py`.
    *   **`__init__`**: It will be initialized with the model object, an `optax` optimizer, and any static training configurations (e.g., number of epochs).
    *   **`train()`**: This will be the main public method that a user calls. It will orchestrate the entire training process: initializing the `TrainState`, looping over epochs and data batches, and calling the internal `train_step` for each batch. It will also handle logging and checkpointing.
    *   **`train_step()`**: This will be a JIT-compiled internal function that performs a single gradient update for one batch of data. It will take the current `TrainState` and a batch, and return the updated `TrainState` and the loss for that step. Crucially, the actual loss calculation will be delegated to an abstract method.
    *   **Abstract Methods**: To ensure modularity, the `Trainer` will define abstract methods that subclasses are required to implement:
        *   `compute_loss()`: This is the most important one. It defines how to calculate the loss for a given set of `params` and a `batch`. An `SFTTrainer` will implement this with cross-entropy loss, while a `GRPOTrainer` will implement the more complex advantage calculation and policy update logic here.
        *   `create_train_state()`: This will handle the initial creation of the model's parameters and the optimizer state.

### Token Sampling in Training

*   **For Supervised Fine-Tuning (SFT):** No, token sampling is **not** used. During SFT, we use a technique called "teacher forcing." The model is trained to predict the next token in a sequence given all the *previous ground-truth tokens* from the dataset. The loss is then calculated between the model's predicted probability distribution (logits) and the single correct next token. The goal is to maximize the likelihood of the known correct answer.

*   **For Reinforcement Learning (RL/GRPO):** Yes, token sampling **is** used. In these methods, the model (the "policy") acts as an agent and must explore by generating diverse responses to a prompt. We sample from its output distribution to create a group of candidate responses. These responses are then evaluated by a reward function, and the resulting scores are used to update the model's policy.

So, for our base `Trainer`, we'll assume the SFT case, but the `compute_loss` abstraction will allow the `GRPOTrainer` to introduce its sampling-based logic later.

---

### Implementation Prompts

Here is the implementation plan broken down into a series of structured prompts.

#### Part 1: Create `trainers/trainer.py` and `TrainState`

*   **Goal**: Create the file and the `TrainState` dataclass.
*   **Prompt**: "Create a new file at `trainers/trainer.py`. Inside this file, import necessary libraries like `abc`, `flax`, `jax`, and `optax`. Define a `TrainState` using `@flax.struct.dataclass`. This class will be a container for the dynamic components of our training loop and should hold three attributes: `params` for the model's weights, `opt_state` for the optimizer's internal state, and `step` to count the training iterations."

#### Part 2: Define the Abstract `Trainer` Skeleton

*   **Goal**: Create the main `Trainer` class with its abstract methods.
*   **Prompt**: "In the same file, `trainers/trainer.py`, define an abstract base class named `Trainer` using `abc.ABC`. Its `__init__` method should accept and store a `model` instance and an `optax` optimizer. Declare three abstract methods using the `@abc.abstractmethod` decorator:
    1.  `create_train_state(self, key: jax.random.PRNGKey) -> TrainState`
    2.  `compute_loss(self, params, batch) -> jax.Array`
    3.  `train_step(self, state: TrainState, batch) -> tuple[TrainState, jax.Array]`"

#### Part 3: Implement the Main `train` Loop

*   **Goal**: Implement the user-facing `train` method that orchestrates the training.
*   **Prompt**: "Add a public method `train(self, train_loader, num_epochs, key)` to the `Trainer` class. This method should first initialize the training state by calling `self.create_train_state(key)`. Then, it should contain the main training loop that iterates for `num_epochs`. Inside this loop, iterate through the provided `train_loader`. For each batch, call `self.train_step(state, batch)` to execute one step of training and receive the updated state. Include a basic print statement to log the loss periodically (e.g., every 100 steps)."

#### Part 4: Implement the `train_step` Logic

*   **Goal**: Implement the generic, JIT-compiled logic for a single gradient update step.
*   **Prompt**: "Now, implement the `train_step` method in the `Trainer` class. This implementation will not be abstract. It should perform the following steps:
    1.  Use `jax.value_and_grad` to create a function that computes both the loss and the gradients with respect to the parameters. The loss function passed to `value_and_grad` should simply call `self.compute_loss(state.params, batch)`.
    2.  Call this new function to get the loss value and the gradients.
    3.  Use `self.optimizer.update()` to apply the computed gradients to the current `params` and `opt_state`, yielding the new `params` and `opt_state`.
    4.  Create and return a new `TrainState` with the updated values and an incremented step counter. Return the loss as well.
    5.  Finally, decorate this method with `@partial(jax.jit, static_argnames=['self'])` to enable JAX's just-in-time compilation for maximum performance." 