+++
id = "ADR-001"
title = "Parallel Training Strategy for Reinforcement Learning Agents"
status = "proposed"
decision_date = "2025-05-13"
authors = ["core-architect"]
template_schema_doc = ".ruru/templates/toml-md/07_adr.README.md"
affected_components = ["train.py", "reinforcestrategycreator/rl_agent.py", "reinforcestrategycreator/trading_environment.py"]
tags = ["parallelization", "reinforcement-learning", "training", "architecture", "ray", "rllib"]
# supersedes_adr = ""
+++

# ADR-001: Parallel Training Strategy for Reinforcement Learning Agents

**Status:** proposed

**Date:** 2025-05-13

## Context 🤔

*   The current training process for reinforcement learning agents, as implemented in [`train.py`](../../train.py), executes training episodes sequentially. This leads to long training times, hindering rapid iteration and development.
*   The primary bottleneck is the sequential nature of running `TradingEnv` simulations and collecting experiences for the `StrategyAgent`.
*   Constraints include the existing codebase structure (Python, TensorFlow/Keras for the agent) and the need for robust data logging to a database.
*   Alternatives considered include using Dask for general-purpose parallelism or Python's `multiprocessing` module.

## Decision ✅ / ❌

*   We will adopt **Ray** (specifically its **RLlib** library) to implement a parallel training strategy for the reinforcement learning agents.
*   The architecture will feature a central learner (`StrategyAgent`) and multiple Ray actors as rollout workers, each running an instance of `TradingEnv`.

## Rationale / Justification 💡

*   **Ray/RLlib Specialization:** RLlib is purpose-built for distributed reinforcement learning. It provides high-level APIs and abstractions that handle many complexities of parallel environment execution, experience collection, and distributed learning algorithms. This significantly reduces the custom implementation effort compared to Dask or `multiprocessing`.
*   **Scalability:** Ray is designed for scalability, allowing the system to efficiently utilize multiple cores on a single machine and potentially scale to multiple machines in the future if needed.
*   **Algorithm Support:** RLlib supports a wide range of RL algorithms, including DQN (which our `StrategyAgent` uses), and provides implementations for common patterns like Ape-X.
*   **Efficiency:** Ray's actor model and object store are optimized for distributed applications, which should lead to efficient experience collection and policy updates.
*   **Community & Ecosystem:** Ray has a large and active community, providing good documentation, examples, and support.

**Trade-offs:**
*   **Learning Curve:** Introducing Ray/RLlib adds a new framework to the project, which will have a learning curve for the team.
*   **Complexity:** While RLlib simplifies many aspects, distributed systems inherently add a layer of complexity for debugging and deployment compared to a single-process application.
*   **Resource Usage:** Running multiple environments in parallel will consume more CPU and memory resources.

## Consequences / Implications ➡️

*   **Code Changes:**
    *   [`train.py`](../../train.py) will need significant modification to initialize Ray, configure RLlib, define how `StrategyAgent` and `TradingEnv` are used within RLlib's framework (e.g., custom environment registration, agent configuration).
    *   [`reinforcestrategycreator/rl_agent.py`](../../reinforcestrategycreator/rl_agent.py) might require minor adjustments to be compatible with RLlib's agent APIs, or RLlib's DQN implementation might be used directly, configured with our network architecture.
    *   [`reinforcestrategycreator/trading_environment.py`](../../reinforcestrategycreator/trading_environment.py) will need to be wrapped or adapted to conform to RLlib's environment interface (likely `gym.Env` compatibility is sufficient, but registration with Ray is needed).
*   **Infrastructure:** No immediate changes to infrastructure are required if running on a single multi-core machine. Future scaling to a cluster would require Ray cluster setup.
*   **Database Logging:**
    *   Logging from multiple parallel environments needs careful consideration. Options include:
        1.  Workers send log data to a central Ray actor responsible for DB writes.
        2.  RLlib's built-in logging/callback mechanisms might be leveraged to collect and write data.
        3.  Each worker manages its own DB session (requires careful handling of connections and potential for contention if not managed well).
    *   The `run_id` and `episode_id` generation and linkage will need to be coordinated.
*   **Model Saving:** The central learner component within RLlib will handle model checkpointing and saving.
*   **Development Workflow:** Debugging parallel applications can be more complex. Ray provides tools like the Ray Dashboard to aid this.
*   **Performance:** Expected significant reduction in overall training time due to parallel episode execution.

## Alternatives Considered (Optional Detail) 📝

*   **Dask:**
    *   **Description:** General-purpose parallel computing library. Could be used with `dask.delayed` to parallelize episode runs.
    *   **Pros:** Flexible, good for general Python parallelism.
    *   **Cons:** Requires more manual implementation of RL-specific logic (experience aggregation, agent learning synchronization, policy distribution) compared to RLlib. Less out-of-the-box support for RL patterns.
*   **Python `multiprocessing`:**
    *   **Description:** Standard library for process-based parallelism. `Pool` could manage worker processes running episodes.
    *   **Pros:** No external dependencies beyond Python itself. Full control.
    *   **Cons:** Most manual effort. Requires careful implementation of inter-process communication for experiences and model updates, and synchronization. Prone to GIL issues if not managed correctly for CPU-bound tasks within a process (though separate processes for envs bypass this for the envs themselves).

## Related Links 🔗 (Optional)

*   Task: [`.ruru/tasks/ARCH_ParallelTraining/TASK-ARCH-20250513-090800.md`](../../.ruru/tasks/ARCH_ParallelTraining/TASK-ARCH-20250513-090800.md)
*   Ray Documentation: [https://docs.ray.io/](https://docs.ray.io/)
*   RLlib Documentation: [https://docs.ray.io/en/latest/rllib/index.html](https://docs.ray.io/en/latest/rllib/index.html)