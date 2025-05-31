# 3. Configuration Management (`ConfigManager`, `ConfigLoader`)

The ReinforceStrategyCreator Pipeline employs a hierarchical and flexible configuration system, managed by components likely named `ConfigManager` and `ConfigLoader` (or similar, based on `src/config/` directory). This system allows for base configurations that can be overridden by environment-specific settings.

### 3.1. Overview of Configuration System
The pipeline configurations are primarily stored in YAML files. The core idea is to have:
*   **Base Configurations:** Located in `configs/base/`, these files define default settings for all aspects of the pipeline. The main pipeline configuration is typically `pipeline.yaml`.
*   **Environment-Specific Configurations:** Located in `configs/environments/` (e.g., `configs/environments/development.yaml`, `configs/environments/production.yaml`). These files can override any settings from the base configuration, allowing for tailored setups for different deployment or testing scenarios.
*   **Pipeline Definitions:** The `pipelines_definition.yaml` file specifies the sequence of stages in a given pipeline and the modules/classes that implement them.

The system loads the base configuration first, then merges any environment-specific configurations on top, providing a final, consolidated configuration object for the pipeline run.

### 3.2. `pipeline.yaml`: Detailed Explanation
The `configs/base/pipeline.yaml` file is the central place for configuring a pipeline run. Below is a detailed breakdown of its main sections and parameters:

*   **Top-Level Settings:**
    *   `name`: A descriptive name for the pipeline (e.g., "reinforcement_learning_trading_pipeline").
    *   `version`: The version of the pipeline configuration (e.g., "1.0.0").
    *   `environment`: The current operating environment (e.g., "development", "production"). This often dictates which environment-specific overrides are applied.
    *   `random_seed`: An integer seed (e.g., `42`) used to initialize random number generators for reproducibility.

*   **3.2.1. `data` Section:** Configures data sources, symbols, date ranges, caching, and initial transformations.
    *   `source_id`: A unique identifier for the data source (e.g., "dummy_csv_data").
    *   `source_type`: The type of data source (e.g., "csv", "api").
    *   `source_path`: Filesystem path to the data if `source_type` is "csv" (e.g., `"../dummy_data.csv"`). Relative paths are often resolved from the configuration file's location or a project root.
    *   `api_endpoint`: URL for the data API if `source_type` is "api".
    *   `api_key`: API key for accessing the data source, often loaded from environment variables (e.g., `"${DATA_API_KEY}"`).
    *   `symbols`: A list of financial instrument symbols to fetch data for (e.g., `["AAPL", "GOOGL"]`).
    *   `start_date`: The start date for data retrieval (e.g., "2020-01-01").
    *   `end_date`: The end date for data retrieval (e.g., "2023-12-31").
    *   `cache_enabled`: Boolean (e.g., `true`) to enable/disable caching of downloaded/processed data.
    *   `cache_dir`: Directory path for storing cached data (e.g., `"./cache/data"`).
    *   `validation_enabled`: Boolean (e.g., `true`) to enable/disable initial data validation steps.
    *   `transformation`: Contains sub-configurations for data transformations.
        *   `add_technical_indicators`: Boolean (e.g., `false`) to control the addition of common technical indicators.

*   **3.2.2. `model` Section:** Defines the RL model type and its architectural hyperparameters.
    *   `model_type`: Specifies the type of RL model to use (e.g., "DQN", "PPO", "A2C").
    *   `hyperparameters`: A dictionary of model-specific hyperparameters.
        *   `hidden_layers`: List defining the architecture of neural network hidden layers (e.g., `[256, 128, 64]`).
        *   `activation`: Activation function for hidden layers (e.g., "relu").
        *   `dropout_rate`: Dropout rate for regularization (e.g., `0.2`).
    *   `checkpoint_dir`: Directory to save model checkpoints (e.g., `"./checkpoints"`).
    *   `save_frequency`: How often (e.g., number of episodes or epochs) to save a checkpoint (e.g., `10`).
    *   `load_checkpoint`: Path to a specific checkpoint file to load and resume training or for inference (e.g., `null` or `"./checkpoints/run_XYZ/checkpoint_epoch_N"`).

*   **3.2.3. `training` Section:** Configures the training process for the RL model.
    *   `episodes`: Number of training episodes (e.g., `100`).
    *   `batch_size`: Batch size for training updates (e.g., `32`).
    *   `learning_rate`: Learning rate for the optimizer (e.g., `0.001`).
    *   `gamma`: Discount factor for future rewards (e.g., `0.99`).
    *   `epsilon_start`: Initial value for epsilon in epsilon-greedy exploration (e.g., `1.0`).
    *   `epsilon_end`: Final value for epsilon (e.g., `0.01`).
    *   `epsilon_decay`: Decay rate for epsilon per step/episode (e.g., `0.995`).
    *   `replay_buffer_size`: Size of the experience replay buffer (e.g., `10000`).
    *   `target_update_frequency`: How often to update the target network in Q-learning based models (e.g., `100` steps/episodes).
    *   `validation_split`: Proportion of data to use for validation during training (e.g., `0.2`).
    *   `early_stopping_patience`: Number of epochs/episodes to wait for improvement before stopping training (e.g., `10`).
    *   `use_tensorboard`: Boolean (e.g., `true`) to enable TensorBoard logging.
    *   `log_dir`: Directory for storing TensorBoard logs and other training logs (e.g., `"./logs"`).

*   **3.2.4. `evaluation` Section:** Defines how trained models are evaluated.
    *   `metrics`: List of performance metrics to calculate (e.g., `["sharpe_ratio", "total_return", "max_drawdown"]`).
    *   `benchmark_symbols`: List of symbols to use as benchmarks (e.g., `["SPY"]`).
    *   `test_episodes`: Number of episodes to run for evaluation on the test set (e.g., `10`).
    *   `save_results`: Boolean (e.g., `true`) to save evaluation results.
    *   `results_dir`: Directory to store evaluation results (e.g., `"./results"`).
    *   `generate_plots`: Boolean (e.g., `true`) to generate performance plots.
    *   `report_formats`: List of formats for the evaluation report (e.g., `["html", "markdown"]`).

*   **3.2.5. `deployment` Section:** Configures parameters for deploying the model (e.g., paper or live trading).
    *   `mode`: Deployment mode (e.g., "paper_trading", "live_trading").
    *   `api_endpoint`: API endpoint for the trading platform, often loaded from environment variables (e.g., `"${TRADING_API_ENDPOINT}"`).
    *   `api_key`: API key for the trading platform, often loaded from environment variables (e.g., `"${TRADING_API_KEY}"`).
    *   `max_positions`: Maximum number of concurrent positions (e.g., `10`).
    *   `position_size`: Fraction of capital to allocate per position (e.g., `0.1` for 10%).
    *   `risk_limit`: Maximum risk per trade or overall (e.g., `0.02` for 2% of capital).
    *   `update_frequency`: How often the deployed model should make decisions or update (e.g., `"1h"`, `"1d"`).

*   **3.2.6. `monitoring` Section:** Configures integration with monitoring services like Datadog.
    *   `enabled`: Boolean (e.g., `true`) to enable/disable monitoring.
    *   `datadog_api_key`: Datadog API key, typically from an environment variable (e.g., `"${DATADOG_API_KEY}"`).
    *   `datadog_app_key`: Datadog Application key, typically from an environment variable (e.g., `"${DATADOG_APP_KEY}"`).
    *   `metrics_prefix`: Prefix for metrics sent to Datadog (e.g., "model_pipeline").
    *   `log_level`: Logging level for the pipeline (e.g., "INFO", "DEBUG").
    *   `alert_thresholds`: Dictionary defining thresholds for alerts.
        *   `sharpe_ratio_min`: Minimum acceptable Sharpe ratio (e.g., `0.5`).
        *   `max_drawdown_max`: Maximum acceptable drawdown (e.g., `0.2`).
        *   `error_rate_max`: Maximum acceptable error rate (e.g., `0.05`).

*   **3.2.7. `artifact_store` Section:** Configures how and where pipeline artifacts (models, datasets, results) are stored.
    *   `type`: Type of artifact store backend (e.g., "local", "s3", "gcs", "azure").
    *   `root_path`: Root path for the artifact store (e.g., `"./artifacts"` for local storage).
    *   `versioning_enabled`: Boolean (e.g., `true`) to enable versioning of artifacts.
    *   `metadata_backend`: Backend for storing artifact metadata (e.g., "json", "sqlite", "postgres").
    *   `cleanup_policy`: Defines rules for automatic artifact cleanup.
        *   `enabled`: Boolean (e.g., `false`) to enable/disable cleanup.
        *   `max_versions_per_artifact`: Maximum number of versions to keep per artifact (e.g., `10`).
        *   `max_age_days`: Maximum age in days for an artifact before it's eligible for cleanup (e.g., `90`).

### 3.3. `pipelines_definition.yaml`: Defining Pipeline Stages
The `configs/base/pipelines_definition.yaml` file defines the structure of one or more named pipelines. Each pipeline consists of a sequence of stages.

Example structure:
```yaml
pipelines:
  full_cycle_pipeline: # Name of the pipeline
    stages:
      - name: "DataIngestion" # User-friendly name for the stage
        module: "reinforcestrategycreator_pipeline.src.pipeline.stages.data_ingestion" # Python module path
        class: "DataIngestionStage" # Class name within the module
        config: {} # Optional: Stage-specific config overrides. If empty, uses global config sections.
      # ... other stages like FeatureEngineering, Training, Evaluation
```
This file allows for flexible pipeline construction by defining which stages run and in what order. Each stage entry specifies:
*   `name`: A descriptive name for the stage instance.
*   `module`: The Python module path where the stage's class is defined.
*   `class`: The name of the class implementing the stage's logic.
*   `config`: An optional dictionary for stage-specific configuration overrides. If empty or not provided, the stage typically draws its configuration from relevant global sections in `pipeline.yaml` (e.g., the "DataIngestion" stage would use the `data` section).

### 3.4. Environment-Specific Configurations
To tailor pipeline runs for different environments (e.g., development, staging, production), you can create YAML files in the `configs/environments/` directory (e.g., `configs/environments/production.yaml`). These files can override any parameter defined in the base `pipeline.yaml`. For instance, `production.yaml` might specify a different `api_endpoint` for data or trading, disable caching, or use a more robust `artifact_store` like "s3".

The pipeline's configuration loader is responsible for merging these environment-specific files over the base configuration when the pipeline is initialized with a specific environment context.

### 3.5. Accessing Configuration in Pipeline Stages
Within each pipeline stage (e.g., `DataIngestionStage`, `TrainingStage`), the consolidated configuration object (resulting from the merge of base and environment-specific configs) is typically made available. Stages can then access their relevant parameters (e.g., `config.data.source_path`, `config.training.learning_rate`) to guide their execution.

This structured approach to configuration management ensures that the pipeline is flexible, reproducible, and adaptable to various operational requirements.