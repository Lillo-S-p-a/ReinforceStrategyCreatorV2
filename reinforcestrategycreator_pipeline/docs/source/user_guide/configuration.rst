Pipeline Configuration
======================

The ReinforceStrategyCreator Pipeline uses a hierarchical configuration system based on YAML files, managed by the ``ConfigManager`` component (see :ref:`config_manager_api` for API details). This allows for a base configuration with environment-specific and experiment-specific overrides. The ``ConfigManager`` is responsible for loading these files, merging them in the correct order, and providing access to configuration values throughout the pipeline.

Configuration Files
-------------------

The main configuration files are located in the ``reinforcestrategycreator_pipeline/configs/`` directory:

*   **Base Configurations**: Found in ``reinforcestrategycreator_pipeline/configs/base/``.
    *   ``pipeline.yaml``: This is the primary configuration file, defining default settings for most aspects of the pipeline, including data sources, model choices, training parameters, evaluation metrics, deployment strategies, monitoring, and artifact storage.
    *   ``data.yaml``: Can provide more detailed or alternative configurations for data sources, preprocessing, feature engineering, and validation. Settings here can be referenced or merged by the ``data`` section in ``pipeline.yaml`` or loaded directly by the ``DataManager`` for specific tasks.
    *   ``models.yaml``: Can contain specific configurations for different model architectures (e.g., DQN, PPO, A2C) and their default hyperparameters. These can be referenced by the ``model`` section in ``pipeline.yaml``.
    *   ``deployment.yaml``: May offer more granular settings for various deployment targets or strategies, complementing the ``deployment`` section in ``pipeline.yaml``.
    *   ``hpo.yaml``: Contains configurations specific to Hyperparameter Optimization (HPO) runs, such as search spaces, optimization algorithms, and trial settings.
*   **Environment Configurations**: Found in ``reinforcestrategycreator_pipeline/configs/environments/``. These files (e.g., ``development.yaml``, ``staging.yaml``, ``production.yaml``) override the base settings for specific operational environments.
*   **Experiment Configurations**: (Potentially in ``reinforcestrategycreator_pipeline/configs/experiments/``) Users can define specific experiment configurations that can override both base and environment settings.

The pipeline loads the base configuration first, then merges the active environment's configuration, and finally, if an experiment configuration is specified, it merges those settings.


Key Configuration Sections (from ``pipeline.yaml``)
-------------------------------------------------

The following are the primary sections found in the base ``pipeline.yaml``:


General Pipeline Settings
~~~~~~~~~~~~~~~~~~~~~~~~~
These settings define the overall behavior and identification of the pipeline run.

*   ``name``: (String) The name of the pipeline (e.g., "reinforcement_learning_trading_pipeline").
*   ``version``: (String) The version of the pipeline configuration (e.g., "1.0.0").
*   ``environment``: (String) The active environment (e.g., "development", "production"). This determines which environment-specific configuration file is loaded.
*   ``random_seed``: (Integer) A seed value for random number generators to ensure reproducibility (e.g., 42).


Data Configuration (``data``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section defines where and how to get the data for the pipeline.

*   ``source_type``: (String) The type of data source. Examples: "api", "csv", "database".
*   ``api_endpoint``: (String) URL for the data API if ``source_type`` is "api".
*   ``api_key``: (String) API key for the data source. Can use environment variables (e.g., "${DATA_API_KEY}").
*   ``symbols``: (List of Strings) List of financial symbols to fetch data for (e.g., ["AAPL", "GOOGL"]).
*   ``start_date``: (String) Start date for the data (e.g., "2020-01-01").
*   ``end_date``: (String) End date for the data (e.g., "2023-12-31").
*   ``cache_enabled``: (Boolean) Whether to cache downloaded data (e.g., true).
*   ``cache_dir``: (String) Directory to store cached data (e.g., "./cache/data").
*   ``validation_enabled``: (Boolean) Whether to perform data validation (e.g., true).

Feature Engineering and Transformation (within ``data`` or ``data.yaml``)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
The pipeline supports various data transformation and feature engineering steps, configured typically within the ``data`` section of ``pipeline.yaml`` or in a dedicated ``data.yaml``.

*   ``preprocessing_steps``: (List of Objects) Defines a sequence of preprocessing actions. Each object might specify a type of operation (e.g., "handle_missing", "normalize", "scale") and its parameters.
    *   Example: ``{ "type": "handle_missing", "strategy": "interpolate" }``
    *   Example: ``{ "type": "normalize", "method": "min_max" }``
*   ``feature_engineering``: (List of Objects) Defines features to be created.
    *   Example: ``{ "type": "moving_average", "windows": [20, 50] }``
    *   Example: ``{ "type": "rsi", "period": 14 }``
*   These configurations are used by the ``DataTransformer`` component within the ``DataManager``.


Model Configuration (``model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines the type of model to be used and its specific hyperparameters.

*   ``model_type``: (String) The type of reinforcement learning model (e.g., "DQN", "PPO", "A2C").
*   ``hyperparameters``: (Object) A nested structure containing model-specific hyperparameters.
    *   Example for DQN: ``hidden_layers: [256, 128, 64]``, ``activation: "relu"``, ``dropout_rate: 0.2``.
*   ``checkpoint_dir``: (String) Directory to save model checkpoints (e.g., "./checkpoints").
*   ``save_frequency``: (Integer) How often (in episodes or steps) to save a checkpoint.
*   ``load_checkpoint``: (String or Null) Path to a specific checkpoint file to load and resume training or for inference. If null, starts fresh.


Training Configuration (``training``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parameters related to the model training process.

*   ``episodes``: (Integer) Number of episodes to train the agent.
*   ``batch_size``: (Integer) Batch size for training.
*   ``learning_rate``: (Float) Learning rate for the optimizer.
*   ``gamma``: (Float) Discount factor for future rewards.
*   ``epsilon_start``: (Float) Initial value for epsilon in epsilon-greedy exploration.
*   ``epsilon_end``: (Float) Final value for epsilon.
*   ``epsilon_decay``: (Float) Decay rate for epsilon.
*   ``replay_buffer_size``: (Integer) Size of the experience replay buffer.
*   ``target_update_frequency``: (Integer) How often to update the target network (for models like DQN).
*   ``validation_split``: (Float) Fraction of data to use for validation during training (e.g., 0.2).
*   ``early_stopping_patience``: (Integer) Number of epochs to wait for improvement before stopping training early.
*   ``use_tensorboard``: (Boolean) Whether to log training progress to TensorBoard.
*   ``log_dir``: (String) Directory for TensorBoard logs and other training logs.

Hyperparameter Optimization (HPO) Configuration (``hpo.yaml`` or within ``training`` section)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Settings for HPO are typically defined in ``configs/base/hpo.yaml`` or can be part of the ``training`` section in ``pipeline.yaml`` if HPO is integrated into a standard training run.

*   ``enabled``: (Boolean) Whether to perform HPO.
*   ``optimizer``: (String) The HPO library/algorithm to use (e.g., "ray_tune", "optuna").
*   ``num_samples``: (Integer) Number of different hyperparameter sets to try.
*   ``max_concurrent_trials``: (Integer) Maximum number of HPO trials to run in parallel (if supported by the optimizer).
*   ``metric_to_optimize``: (String) The primary metric to optimize (e.g., "validation_sharpe_ratio").
*   ``optimization_direction``: (String) "maximize" or "minimize" the ``metric_to_optimize``.
*   ``search_space``: (Object) Defines the hyperparameters to tune and their possible ranges or values.
    *   Example for Ray Tune:

        .. code-block:: yaml

           search_space:
             learning_rate: {"tune.loguniform": [0.0001, 0.01]}
             batch_size: {"tune.choice": [32, 64, 128]}
             model_specific_param: {"tune.uniform": [0.1, 0.9]}
*   ``early_stopping_policy``: (Object) Configuration for HPO trial early stopping (e.g., ASHA, MedianStoppingRule).
*   These settings are used by the ``HPOptimizer`` component.


Evaluation Configuration (``evaluation``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settings for evaluating the trained model's performance.

*   ``metrics``: (List of Strings) Performance metrics to calculate (e.g., ["sharpe_ratio", "total_return"]).
*   ``benchmark_symbols``: (List of Strings) Symbols to use as benchmarks (e.g., ["SPY"]).
*   ``test_episodes``: (Integer) Number of episodes to run for evaluation on the test set.
*   ``save_results``: (Boolean) Whether to save evaluation results.
*   ``results_dir``: (String) Directory to save evaluation results and reports.
*   ``generate_plots``: (Boolean) Whether to generate performance plots.


Deployment Configuration (``deployment``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parameters for deploying the trained model.

*   ``mode``: (String) Deployment mode (e.g., "paper_trading", "live_trading", "batch_inference").
*   ``api_endpoint``: (String) API endpoint for the trading platform (if applicable).
*   ``api_key``: (String) API key for the trading platform. Can use environment variables.
*   ``max_positions``: (Integer) Maximum number of concurrent positions.
*   ``position_size``: (Float) Fraction of capital to allocate per position.
*   ``risk_limit``: (Float) Maximum risk per trade or overall.
*   ``update_frequency``: (String) How often the deployed model should update/trade (e.g., "1h", "1d").


Monitoring Configuration (``monitoring``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settings for monitoring the pipeline and deployed models.

*   ``enabled``: (Boolean) Whether monitoring is enabled.
*   ``datadog_api_key``: (String) Datadog API key (if using Datadog). Uses environment variables.
*   ``datadog_app_key``: (String) Datadog App key. Uses environment variables.
*   ``metrics_prefix``: (String) Prefix for metrics sent to the monitoring system (e.g., "model_pipeline").
*   ``log_level``: (String) Logging level (e.g., "INFO", "DEBUG").
*   ``alert_thresholds``: (Object) Thresholds for triggering alerts on key metrics.
    *   Example: ``sharpe_ratio_min: 0.5``, ``max_drawdown_max: 0.2``.


Artifact Storage Configuration (``artifact_store``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines how and where pipeline artifacts (models, datasets, results) are stored.

*   ``type``: (String) Type of artifact store (e.g., "local", "s3", "gcs", "azure").
*   ``root_path``: (String) Root path for the artifact store (e.g., "./artifacts" for local).
*   ``versioning_enabled``: (Boolean) Whether to version artifacts.
*   ``metadata_backend``: (String) Backend for storing artifact metadata (e.g., "json", "sqlite").
*   ``cleanup_policy``: (Object) Policy for cleaning up old artifacts.
    *   ``enabled``: (Boolean)
    *   ``max_versions_per_artifact``: (Integer)
    *   ``max_age_days``: (Integer)

Environment Variables
---------------------
Some sensitive values like API keys (e.g., ``${DATA_API_KEY}``, ``${TRADING_API_KEY}``, ``${DATADOG_API_KEY}``) are specified as placeholders to be resolved from environment variables at runtime. This is a security best practice to avoid hardcoding secrets in configuration files. Ensure these environment variables are set in the execution environment of the pipeline.

Overriding Configurations
-------------------------
To override base configurations for a specific environment (e.g., production), you would modify the corresponding file in ``configs/environments/``. For example, to change the ``api_endpoint`` for production data, you would set it in ``configs/environments/production.yaml``:

.. code-block:: yaml

   # In configs/environments/production.yaml
   data:
     api_endpoint: "https://api.production.example.com/data"
     cache_enabled: false # Example: disable cache for production

Similarly, experiment-specific configurations can be created to override settings for particular runs. The ``ConfigManager`` component of the pipeline is responsible for loading and merging these configurations in the correct order.