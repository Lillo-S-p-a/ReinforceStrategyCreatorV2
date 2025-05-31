# 13. Appendix

### 13.1. Glossary of Terms

*   **Agent:** In reinforcement learning, the entity that learns to make decisions by interacting with an environment.
*   **Artifact Store:** A system for storing and versioning outputs of the MLOps pipeline, such as models, datasets, and results.
*   **Benchmark:** A standard or point of reference against which the performance of a trading strategy or model can be compared.
*   **Checkpointing:** The process of saving the state of a model during training, allowing it to be resumed later or loaded for inference.
*   **Configuration Management:** The process of managing and controlling settings and parameters for the pipeline and its components.
*   **Datadog:** A monitoring and analytics platform used for tracking application performance and infrastructure health.
*   **Deployment:** The process of making a trained model operational, e.g., for paper trading or live trading.
*   **DQN (Deep Q-Network):** A type of reinforcement learning algorithm that uses a deep neural network to approximate the Q-value function.
*   **Environment (RL):** In reinforcement learning, the external system with which the agent interacts, providing observations and receiving actions. For trading, this is typically a market simulation.
*   **Episode:** A complete sequence of interactions between an agent and an environment, from an initial state to a terminal state or a maximum number of steps.
*   **Epsilon-Greedy:** An exploration strategy in reinforcement learning where the agent chooses a random action with probability epsilon and the greedy (best known) action with probability 1-epsilon.
*   **Feature Engineering:** The process of creating new input variables (features) for a machine learning model from raw data.
*   **Gamma (Discount Factor):** A parameter in reinforcement learning that determines the present value of future rewards.
*   **Hyperparameter Optimization (HPO):** The process of automatically finding the best set of hyperparameters for a model to maximize its performance.
*   **Hyperparameters:** Parameters that are set before the learning process begins and are not learned by the model itself (e.g., learning rate, number of hidden layers).
*   **Live Trading:** Deploying a trading strategy to execute real trades with actual capital in a live market.
*   **Max Drawdown:** The largest percentage decline from a peak to a trough in an investment's value during a specific period.
*   **MDTM (Markdown-Driven Task Management):** A system for managing tasks using Markdown files with TOML frontmatter.
*   **Model Factory:** A design pattern used to create instances of different model classes based on configuration, without exposing the instantiation logic to the client.
*   **Orchestrator (`ModelPipeline`):** The central component that manages the execution flow and coordination of different stages in the pipeline.
*   **Paper Trading:** Simulating trades based on live or historical market data without using real money, to test a strategy's performance.
*   **Pipeline Stage:** A distinct, modular component of the overall pipeline responsible for a specific task (e.g., Data Ingestion, Training, Evaluation).
*   **Profit Factor:** Gross profit divided by gross loss for a trading strategy.
*   **Ray Tune:** A Python library for hyperparameter tuning at any scale.
*   **Reinforcement Learning (RL):** A type of machine learning where agents learn to make a sequence of decisions by interacting with an environment to maximize a cumulative reward.
*   **Replay Buffer (Experience Replay):** A component in some RL algorithms (like DQN) that stores past experiences (state, action, reward, next state) which are then sampled to train the model.
*   **Sharpe Ratio:** A measure of risk-adjusted return, calculated as the average return earned in excess of the risk-free rate per unit of volatility or total risk.
*   **TensorBoard:** A visualization toolkit for TensorFlow (and other frameworks like PyTorch) used to inspect and understand ML experiments and graphs.
*   **TOML (Tom's Obvious, Minimal Language):** A configuration file format designed to be easy to read due to its simple semantics.
*   **Total Return:** The overall gain or loss of an investment over a specific period, expressed as a percentage.
*   **Win Rate:** The percentage of trades that result in a profit.
*   **YAML (YAML Ain't Markup Language):** A human-readable data serialization standard often used for configuration files.

### 13.2. Full Configuration Reference (`pipeline.yaml`)

*(This section should ideally contain a verbatim copy or a well-structured summary of all possible parameters in `pipeline.yaml`, along with their descriptions, data types, and default values if applicable. For brevity in this generation, we will reference the detailed breakdown already provided in Section 3.2.)*

Please refer to **[Section 3.2: `pipeline.yaml`: Detailed Explanation](./03_configuration_management.md#32-pipelineyaml-detailed-explanation)** for a comprehensive breakdown of all configuration parameters within the base `pipeline.yaml` file. This includes details on the `data`, `model`, `training`, `evaluation`, `deployment`, `monitoring`, and `artifact_store` sections.