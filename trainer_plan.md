### Plan for Modular Training Framework

#### 1. High-Level Goal

To create a new `trainers` directory that is modular and extensible. This will allow for the implementation of various training schemes like Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) while keeping the core logic clean and separated. Configuration and data loading responsibilities will be handled by the main executable scripts to keep the trainer components focused solely on the training logic.

#### 2. Proposed Directory Structure

```
ReLax/
├── trainers/
│   ├── __init__.py
│   ├── trainer.py       # Abstract Base Class for all trainers
│   ├── sft_trainer.py   # Supervised Fine-Tuning Trainer
│   ├── rl_trainer.py    # Reinforcement Learning Trainer (e.g., PPO)
│   └── grpo_trainer.py  # GRPO Trainer
│
├── run_sft.py           # Main executable script to launch SFT
├── run_rl.py            # Main executable script to launch RL
└── run_grpo.py          # Main executable script to launch GRPO
```

#### 3. Component Breakdown

*   **`trainers/trainer.py`**:
    *   This will define an abstract base class, `Trainer`, which will serve as the foundation for all specific trainer implementations.
    *   It will manage the generic training loop (`for epoch in epochs... for batch in dataloader...`).
    *   It will handle common tasks:
        *   Initializing the model, optimizer, and learning rate scheduler.
        *   Orchestrating checkpoint saving and loading (model weights, optimizer state).
        *   Logging metrics.
    *   It will define an abstract `compute_loss` method that each subclass (`SFTTrainer`, `RLTrainer`, etc.) must implement according to its specific logic.
    *   All configurations (learning rate, batch size, etc.) will be passed directly to the trainer's constructor.

*   **`trainers/sft_trainer.py`**:
    *   Will contain the `SFTTrainer` class, inheriting from `trainers.trainer.Trainer`.
    *   It will provide a concrete implementation of the `compute_loss` method for Supervised Fine-Tuning.
    *   The loss will be a standard cross-entropy calculation between the model's output logits and the target labels, JIT-compiled for performance.

*   **`trainers/rl_trainer.py`**:
    *   Will contain the `RLTrainer` class, also inheriting from `trainers.trainer.Trainer`.
    *   This trainer will be designed for more complex algorithms like PPO (Proximal Policy Optimization).
    *   It will manage multiple models: the policy (actor), a value function (critic), and potentially a reward model.
    *   Its `compute_loss` implementation will calculate the PPO objective function, managing an experience buffer for policy updates.

*   **`trainers/grpo_trainer.py`**:
    *   Will contain the `GRPOTrainer` class, inheriting from `trainers.trainer.Trainer`.
    *   This trainer will implement the Group Relative Policy Optimization algorithm, a memory-efficient alternative to PPO.
    *   It will not require a separate value/critic model.
    *   The `compute_loss` method will involve:
        1.  Generating a group of responses for each prompt in a batch.
        2.  Scoring each response with one or more reward functions.
        3.  Calculating the advantage for each response relative to the group's average reward.
        4.  Computing the final loss, which includes a KL-divergence penalty against a reference model to ensure training stability.

*   **`run_sft.py`, `run_rl.py`, and `run_grpo.py` (in root directory)**:
    *   These will be the user-facing executable scripts for launching training jobs.
    *   They will use a library like `fire` for parsing command-line arguments.
    *   Their responsibilities will include:
        1.  Loading the model(s) (e.g., policy, reference model) and tokenizer.
        2.  Loading and preparing the dataset (including tokenization and batching).
        3.  Defining or loading the reward function(s) for RL/GRPO trainers.
        4.  Initializing the appropriate trainer (`SFTTrainer`, `RLTrainer`, or `GRPOTrainer`) with all necessary hyperparameters.
        5.  Calling the `trainer.train()` method to start the training process. 