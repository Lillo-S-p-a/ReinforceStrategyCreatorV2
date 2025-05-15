+++
id = "ADR-20250515-RLlib-Ray-PyTorch-Integration"
title = "Integrate RLlib, Ray, and PyTorch for Parallel Trading Agent Training"
status = "accepted"
created_date = "2025-05-15"
updated_date = "2025-05-15"
tags = ["architecture", "rllib", "ray", "pytorch", "parallel-training", "trading-agent", "refactor"]
related_docs = [".ruru/tasks/REFACTOR_RLlib_Ray_Pytorch/TASK-ARCH-20250515-144808.md"]
+++

# Integrate RLlib, Ray, and PyTorch for Parallel Trading Agent Training

## Context

The existing trading agent project uses a basic training loop and a custom trading environment. To significantly speed up training iteration cycles and enable more efficient hyperparameter tuning, there is a need to leverage parallel processing. Ray and RLlib are chosen for distributed execution and reinforcement learning framework management, with PyTorch as the deep learning backend.

The current system structure involves:
- `train.py`: Manages data loading, environment creation, agent training loop, and basic logging.
- `reinforcestrategycreator/trading_environment.py`: Defines the custom Gymnasium environment for trading simulation.
- `reinforcestrategycreator/rl_agent.py`: (Implicitly used by the current setup, likely a simple agent or placeholder).

The goal is to refactor the system to fully utilize RLlib's parallel training capabilities with the existing custom environment and a PyTorch-based agent.

## Decision

The decision is to refactor the project to integrate Ray, RLlib, and PyTorch for parallel training of the trading agent.

Key architectural decisions:

1.  **Leverage Existing `TradingEnv`:** The current `reinforcestrategycreator/trading_environment.py` will be adapted to ensure full compatibility with RLlib's parallel workers. This involves verifying statelessness between episodes and thread-safety. The environment will be registered with RLlib using `ray.tune.registry.register_env`.
2.  **Utilize RLlib's PyTorch Agents:** Instead of a custom agent implementation from scratch, we will configure RLlib to use its built-in PyTorch-based agents (starting with DQN as per the current `train.py` setup, but allowing for easy switching to others like PPO, A2C, etc., via configuration). This provides optimized implementations and handles model distribution and synchronization across workers.
3.  **Ray for Parallel Execution:** Ray will manage the distributed execution. RLlib's `num_env_runners` parameter will control the number of parallel environment instances collecting experience.
4.  **Data Handling:** Historical market data will be loaded on the driver and passed to each environment worker via the `env_config`. Workers will collect trajectories and send them back to the driver for training updates.
5.  **Logging and Callbacks:** The existing `DatabaseLoggingCallbacks` will be modified to correctly handle and aggregate metrics and trade data from multiple parallel episodes running on different workers. This will likely involve using Ray's built-in mechanisms for collecting data from workers or ensuring the callback logic is robust to concurrent calls.
6.  **Configuration:** Training parameters, environment settings, and agent hyperparameters will be managed through RLlib's `AlgorithmConfig` object, passed to the `config.build_algo()` method.

## Consequences

*   **Positive:**
    *   Significant speedup in training iteration cycles due to parallel environment interaction.
    *   Easier experimentation with different RLlib algorithms and hyperparameters.
    *   Leveraging battle-tested, optimized implementations from RLlib and PyTorch.
    *   Improved scalability for training on larger datasets or with more complex models.
*   **Negative:**
    *   Requires significant refactoring of the existing training loop and potentially the environment/agent interaction logic.
    *   Increased complexity in debugging due to distributed nature.
    *   Potential challenges in correctly aggregating metrics and logging from parallel workers.
    *   Requires installation and configuration of Ray, RLlib, and PyTorch dependencies.
*   **Neutral:**
    *   The core trading logic within the environment remains largely the same.
    *   The database schema for logging may need minor adjustments to accommodate parallel episode tracking if not already designed for it.

## Alternatives Considered

*   **Manual Parallelization:** Implementing parallel environment execution manually using Python's `multiprocessing` or `threading`.
    *   *Rejected:* This would be significantly more complex to manage, especially with model synchronization and data aggregation, compared to using a dedicated framework like Ray/RLlib.
*   **Using a Different RL Framework:** Exploring other frameworks like Stable Baselines3 or Acme.
    *   *Rejected:* While viable, RLlib is specifically designed for large-scale distributed RL and integrates tightly with Ray, making it a more suitable choice for the stated goal of parallel training for quick iteration. The project already has some Ray/RLlib components, making this a more natural progression.

## Status

Accepted. This ADR documents the planned architectural approach for the refactoring task.

## Follow-up Actions

*   Proceed with the implementation steps outlined in the related MDTM task.
*   Thoroughly test the parallel training setup.
*   Document the new architecture and usage.