### Plan for `GRPOTrainer`

The `GRPOTrainer` will be a concrete implementation of our abstract `Trainer` and will encapsulate the logic for Group Relative Policy Optimization. Its main job is to orchestrate the sample generation, reward scoring, and the unique loss calculation that defines GRPO.

1.  **File Location**: All logic will be contained in a new file, `trainers/grpo_trainer.py`.

2.  **Class Definition**:
    *   The class will be named `GRPOTrainer` and will inherit from `Trainer` (`from .trainer import Trainer, TrainState`).
    *   Its `__init__` method will be more detailed, requiring the essential components for a RL setup:
        *   `policy_model`: The model we are actively training (the actor).
        *   `ref_model`: A frozen, read-only copy of the policy model used as a stable reference for calculating the KL-divergence penalty.
        *   `optimizer`: The standard Optax optimizer.
        *   `reward_fn`: A crucial, user-provided function. This function will be responsible for taking a list of generated text responses and returning a list of scalar reward scores.
        *   `grpo_config`: A dataclass or dictionary to hold GRPO-specific hyperparameters, such as `kl_coeff` (the weight for the KL penalty), `num_samples_per_prompt`, and `temperature` for generation.

3.  **Implementation of Abstract Methods**:

    *   **`create_train_state(self, key: jax.random.PRNGKey, dummy_input: jax.Array) -> TrainState`**:
        *   This method will initialize the `TrainState` for the `policy_model`.
        *   It will use the provided `dummy_input` to initialize the policy model's parameters.
        *   It will then create and return a `TrainState` instance containing the policy parameters, the corresponding optimizer state, and a starting step count of 0. The reference model parameters are not part of this state as they are not trained.

    *   **`compute_loss(self, params, batch: dict) -> jax.Array`**:
        *   This is the heart of the `GRPOTrainer` and will contain the full GRPO logic in a single, JIT-compatible function. The process is as follows:
        *   **1. Sample Generation**: For each prompt in the input `batch`, the function will use the `policy_model` (with the current `params`) to autoregressively generate a "group" of `N` responses (`N` is from `grpo_config`). This step requires sampling (e.g., using `jax.random.categorical`) to ensure a diverse set of responses for exploration.
        *   **2. Reward Scoring**: All generated responses are passed to the `self.reward_fn` to obtain a scalar reward for each.
        *   **3. Advantage Calculation**: For each group of responses corresponding to a single prompt, the function will calculate the mean reward. The "advantage" for each individual response is then computed as `its_reward - group_mean_reward`. This value represents how much better or worse a given response was compared to the policy's average performance on that prompt.
        *   **4. Policy Loss**: The function will calculate the log-probability of generating each of the sampled responses according to the current policy. The policy loss is then `mean(-advantages * log_probabilities)`. This update rule encourages the model to produce more responses with positive advantages.
        *   **5. KL-Divergence Penalty**: The function will also calculate the log-probabilities of the same sampled responses under the frozen `ref_model`. The KL divergence between the policy and reference distributions is calculated (`policy_log_probs - ref_log_probs`). This term penalizes the policy for straying too far from the trusted reference model, ensuring training stability.
        *   **6. Final Loss**: The total loss is a weighted sum of the policy loss and the KL penalty: `total_loss = policy_loss + self.grpo_config.kl_coeff * kl_divergence`. This final scalar value is what gets returned.

---

### Implementation Prompts

Here is the plan broken down into structured prompts for implementation.

#### Part 1: Create `trainers/grpo_trainer.py` and Class Skeleton

*   **Goal**: Create the file and the basic `GRPOTrainer` class.
*   **Prompt**: "Create a new file at `trainers/grpo_trainer.py`. Import necessary libraries like `jax`, `optax`, and `flax`. From `.trainer`, import `Trainer` and `TrainState`. Define a new class `GRPOTrainer` inheriting from `Trainer`. Its `__init__` method should accept and store a `policy_model`, `ref_model`, `optimizer`, a `reward_fn`, and a `grpo_config` object."

#### Part 2: Implement `create_train_state`

*   **Goal**: Implement the state initialization logic for the policy model.
*   **Prompt**: "In `GRPOTrainer`, implement the `create_train_state` method. It takes a `key` and `dummy_input`. It should initialize the `policy_model`'s parameters and the optimizer state, then return a `TrainState` object for the policy."

#### Part 3: Implement `compute_loss` (The Core GRPO Logic)

*   **Goal**: Implement the full, multi-step GRPO loss function.
*   **Prompt**: "Implement the `compute_loss` method in `GRPOTrainer`. This method receives the policy `params` and a `batch` of prompts.
    1.  **Sample Generation**: Write a generation loop (ideally with `jax.lax.scan`) to autoregressively sample `N` responses for each prompt from the `policy_model`.
    2.  **Reward Scoring**: Apply `self.reward_fn` to the generated samples to get their rewards.
    3.  **Advantage Calculation**: For each group of samples, compute the mean reward and then the advantage for each sample (`advantage = reward - mean_reward`).
    4.  **Log-Probabilities**: Calculate the log-probabilities of the generated samples under both the current `policy_model` and the frozen `self.ref_model`.
    5.  **Policy Loss**: Compute the policy loss as the mean of `-advantages * policy_log_probs`.
    6.  **KL Divergence**: Compute the mean KL divergence as `mean(policy_log_probs - ref_log_probs)`.
    7.  **Total Loss**: Combine them: `total_loss = policy_loss + self.grpo_config.kl_coeff * kl_divergence`. Return this total loss." 