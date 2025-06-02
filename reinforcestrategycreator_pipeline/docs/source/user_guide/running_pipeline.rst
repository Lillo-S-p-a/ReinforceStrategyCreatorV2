Running the Pipeline
====================

This section explains the general approach to running the ReinforceStrategyCreator Pipeline, focusing on a typical training workflow. For more complex, multi-stage operations (e.g., a full data-preprocessing, training, evaluation, and deployment sequence), the ``ModelPipeline`` orchestrator would be used, typically invoked from a custom script.

General Workflow
----------------

Running a pipeline, such as a model training process, generally involves the following steps:

1.  **Prepare Configuration**: Ensure your YAML configuration files (base, environment, and any experiment-specific files) are correctly set up as described in the :doc:`configuration` guide. This includes defining data sources, model parameters, training settings, etc.

2.  **Create an Entry Point Script**: You will typically write a Python script to drive the pipeline. This script will:
    *   Initialize necessary components like the ``ConfigManager``, ``DataManager``, ``ArtifactStore``, and ``ModelRegistry``.
    *   Instantiate the relevant pipeline engine (e.g., ``TrainingEngine`` for training, ``EvaluationEngine`` for evaluation).
    *   Invoke the engine's primary method (e.g., ``train()``, ``evaluate()``).

Example: Running a Training Pipeline
-------------------------------------

The following illustrates how to run a model training process using the ``TrainingEngine``. Refer to ``examples/training_engine_example.py`` for a runnable version.

.. code-block:: python

   import logging
   from src.training import TrainingEngine, LoggingCallback, ModelCheckpointCallback
   from src.config.manager import ConfigManager
   from src.data.manager import DataManager # If using DataManager for data loading
   from src.artifact_store.local_adapter import LocalArtifactStore # Example store
   from src.models.registry import ModelRegistry # If saving models to registry

   # --- 1. Basic Setup ---
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   # --- 2. Initialize Core Components ---
   # Configuration Manager: Point to your config directory and select environment
   config_manager = ConfigManager(config_dir="./configs", environment="development")

   # Artifact Store: (Example: Local storage)
   artifact_store = LocalArtifactStore(base_path="./artifacts")

   # Model Registry: (For saving and versioning trained models)
   model_registry = ModelRegistry(artifact_store=artifact_store)

   # Data Manager: (Handles loading and preparing data based on config)
   # This assumes 'data' section in your config points to actual data sources.
   # For simplicity, example below might pass data directly or use a simpler data loader.
   data_manager = DataManager(config_manager=config_manager, artifact_store=artifact_store)

   # Training Engine:
   engine = TrainingEngine(
       model_registry=model_registry,
       artifact_store=artifact_store,
       data_manager=data_manager, # Optional, if data_config uses source_id
       checkpoint_dir="./checkpoints/my_training_run" # For saving checkpoints
   )

   # --- 3. Define Configurations for the Training Run ---
   # These can be loaded from ConfigManager or defined directly in the script
   # For clarity, defined directly here, but typically fetched via config_manager.get_config('model'), etc.

   model_config = {
       "model_type": "DQNModel",  # Must be a registered model type
       "name": "my_dqn_model_v1",
       "hyperparameters": {
           "learning_rate": 0.001,
           "hidden_layers": [128, 64],
           # ... other model-specific hyperparameters
       }
   }

   # Data configuration:
   # Option 1: Using DataManager by specifying a source_id from your data.yaml
   data_config_from_manager = {
       "source_id": "my_feature_engineered_data", # Defined in your data.yaml
       "params": { # Optional parameters for the data source
           "validation_split": 0.15 # Example: override validation split for this run
       }
   }
   # Option 2: Providing data directly (e.g., pandas DataFrames)
   # train_df = ... load your training DataFrame ...
   # val_df = ... load your validation DataFrame ...
   # data_config_direct = {
   #     "train_data": train_df,
   #     "val_data": val_df # Optional, if not using validation_split from training_config
   # }

   training_config = {
       "epochs": 50,
       "batch_size": 64,
       "validation_split": 0.2, # Used if val_data not in data_config
       "save_checkpoints": True,
       "checkpoint_frequency": "epoch", # Or an integer for steps
       "save_best_only": True,
       "monitor": "val_loss", # Metric to monitor for saving best model and early stopping
       "log_frequency": "epoch",
       "verbose": 1
       # Add data_info if you want to version the dataset used for this model
       # "data_info": { "dataset_name": "processed_stock_data", "version": "2.1" }
   }

   # Callbacks (Optional)
   callbacks = [
       LoggingCallback(log_frequency="epoch", verbose=2),
       ModelCheckpointCallback(
           checkpoint_dir=engine.checkpoint_dir, # Uses engine's dir
           save_best_only=True,
           monitor=training_config["monitor"],
           mode="min" # for loss, "max" for accuracy/reward
       ),
       # EarlyStoppingCallback(monitor=training_config["monitor"], patience=5, mode="min")
   ]

   # --- 4. Run Training ---
   logger.info(f"Starting training for model: {model_config['name']}")
   try:
       result = engine.train(
           model_config=model_config,
           data_config=data_config_from_manager, # Or data_config_direct
           training_config=training_config,
           callbacks=callbacks
       )

       if result.get("success"):
           logger.info("Training completed successfully!")
           logger.info(f"Model ID: {result.get('model_id')}")
           logger.info(f"Epochs trained: {result.get('epochs_trained')}")
           logger.info(f"Final metrics: {result.get('final_metrics')}")
           # Access full history: result.get('history')
       else:
           logger.error(f"Training failed: {result.get('error', 'Unknown error')}")

   except Exception as e:
       logger.exception(f"An error occurred during the training script: {e}")


Running Different Pipeline Operations
-------------------------------------

*   **Evaluation**:
    To evaluate a trained model, you'll use the ``EvaluationEngine``. This involves loading the model (from the ``ModelRegistry`` or a checkpoint) and providing evaluation data.

    .. code-block:: python

       # (Assuming ConfigManager, ArtifactStore, ModelRegistry, DataManager are initialized as in the Training example)
       from reinforcestrategycreator_pipeline.src.evaluation.engine import EvaluationEngine

       evaluation_engine = EvaluationEngine(
           config_manager=config_manager,
           model_registry=model_registry,
           data_manager=data_manager,
           artifact_store=artifact_store
       )

       # Configuration for the evaluation run
       # Typically fetched from ConfigManager, e.g., config_manager.get_config('my_evaluation_task')
       evaluation_run_config = {
           "model_id": "my_dqn_model_v1", # ID of the model in ModelRegistry
           "model_version": "1.0.0",     # Specific version to evaluate
           # OR, if loading from a checkpoint directly:
           # "model_checkpoint_path": "./checkpoints/my_training_run/best_model.pt",
           # "model_config": model_config, # The config used to train the checkpointed model

           "evaluation_data_config": {     # How to get evaluation data
               "source_id": "my_test_dataset", # Key from data.yaml or data section
               # "test_data": test_df # Or provide DataFrame directly
           },
           "metrics_to_calculate": ["sharpe_ratio", "total_return", "max_drawdown"],
           "benchmarks_to_compare": ["buy_and_hold_SPY"], # Optional
           "generate_report": True,
           "report_output_dir": "./artifacts/evaluation_reports"
       }

       logger.info(f"Starting evaluation for model: {evaluation_run_config.get('model_id')}")
       try:
           results = evaluation_engine.evaluate_model(
               model_id=evaluation_run_config.get("model_id"),
               model_version=evaluation_run_config.get("model_version"),
               # model_checkpoint_path=evaluation_run_config.get("model_checkpoint_path"),
               # model_config_for_checkpoint=evaluation_run_config.get("model_config"),
               evaluation_data_config=evaluation_run_config["evaluation_data_config"],
               metrics_to_calculate=evaluation_run_config["metrics_to_calculate"],
               benchmarks_to_compare=evaluation_run_config.get("benchmarks_to_compare"),
               generate_report=evaluation_run_config.get("generate_report", True),
               report_output_dir=evaluation_run_config.get("report_output_dir")
           )
           logger.info(f"Evaluation completed. Report saved at: {results.get('report_path')}")
           logger.info(f"Metrics: {results.get('metrics')}")
       except Exception as e:
           logger.exception(f"An error occurred during evaluation: {e}")

*   **Hyperparameter Optimization (HPO)**:
    The ``HPOptimizer`` (from ``src.training.hpo_optimizer``) is used to find the best set of hyperparameters for a given model type and dataset. It typically integrates with a library like Ray Tune.

    .. code-block:: python

       # (Assuming ConfigManager, TrainingEngine, DataManager, ArtifactStore are initialized)
       from reinforcestrategycreator_pipeline.src.training.hpo_optimizer import HPOptimizer

       # HPO configuration, usually from hpo.yaml or a dedicated section in pipeline.yaml
       # Fetched via config_manager.get_config('my_hpo_task_settings')
       hpo_config = {
           "optimizer_type": "ray_tune", # or "optuna"
           "model_type": "DQNModel",     # Model type to optimize
           "search_space": {             # Defines ranges for hyperparameters
               "learning_rate": {"tune.loguniform": [0.00001, 0.001]},
               "model_specific.hidden_layers": {"tune.choice": [[64,32], [128,64], [256,128]]},
               # ... other hyperparameters ...
           },
           "num_samples": 50, # Number of trials
           "metric_to_optimize": "val_sharpe_ratio", # Metric reported by TrainingEngine
           "optimization_mode": "max", # "max" or "min"
           "hpo_specific_params": { # Parameters for Ray Tune itself
               "time_budget_s": 3600, # Optional: time limit for HPO
               # "resources_per_trial": {"cpu": 1, "gpu": 0.25} # Optional
           }
       }

       # Data configuration for the HPO trials (how to get training/validation data)
       hpo_data_config = {
           "source_id": "my_hpo_dataset", # Key from data.yaml
           "params": {"validation_split": 0.2}
       }

       # Base model config (non-tuned parameters, model name prefix for trials)
       base_model_config_for_hpo = {
           "name": "hpo_dqn_trial", # Name prefix for trial models
           # ... any fixed hyperparameters ...
       }

       # Base training config for HPO trials (epochs, batch_size if not tuned, etc.)
       base_training_config_for_hpo = {
           "epochs": 20, # Shorter epochs for HPO trials usually
           # ... other fixed training params ...
       }

       hpo_optimizer = HPOptimizer(
           training_engine=engine, # The TrainingEngine instance
           config_manager=config_manager # For loading HPO specific configs if needed
       )

       logger.info(f"Starting HPO for model type: {hpo_config['model_type']}")
       try:
           best_hyperparameters, best_trial_result = hpo_optimizer.optimize(
               model_type=hpo_config["model_type"],
               search_space_config=hpo_config["search_space"], # Can be dict or key to config
               data_config=hpo_data_config,
               base_model_config=base_model_config_for_hpo,
               base_training_config=base_training_config_for_hpo,
               hpo_params=hpo_config # Pass the full HPO config for optimizer settings
           )
           logger.info(f"HPO completed.")
           logger.info(f"Best hyperparameters: {best_hyperparameters}")
           logger.info(f"Best trial result ({hpo_config['metric_to_optimize']}): {best_trial_result.get('metric_value')}")
           logger.info(f"Best trial ID: {best_trial_result.get('trial_id')}")
           # The best model from HPO might be registered by HPOptimizer or TrainingEngine
           # or you might retrain with best_hyperparameters.
       except Exception as e:
           logger.exception(f"An error occurred during HPO: {e}")

*   **Deployment**:
    The ``DeploymentManager`` (from ``src.deployment.manager``) is responsible for packaging a trained model and deploying it to a target environment, such as a paper trading setup. The ``PaperTradingDeployer`` (from ``src.deployment.paper_trading``) provides a specialized interface for paper trading.

    .. code-block:: python

       # (Assuming ConfigManager, ArtifactStore, ModelRegistry are initialized)
       from reinforcestrategycreator_pipeline.src.deployment.manager import DeploymentManager, DeploymentStrategy
       from reinforcestrategycreator_pipeline.src.deployment.paper_trading import PaperTradingDeployer

       deployment_manager = DeploymentManager(
           model_registry=model_registry,
           artifact_store=artifact_store,
           deployment_root="./deployments" # Root for storing deployment packages and state
       )

       # Example: Deploying to a paper trading environment
       paper_trading_deployer = PaperTradingDeployer(
           deployment_manager=deployment_manager, # Can be optional if it has its own logic
           model_registry=model_registry,
           artifact_store=artifact_store,
           config_manager=config_manager, # For paper trading specific configs
           paper_trading_root="./paper_trading_simulations"
       )

       # Configuration for the deployment
       # Typically fetched from ConfigManager
       deployment_run_config = {
           "model_id": "my_dqn_model_v1",
           "model_version": "1.0.0", # Version to deploy
           "deployment_target_name": "my_paper_trading_env_alpha",
           "paper_trading_specific_config": { # Params for PaperTradingDeployer
               "initial_capital": 100000,
               "data_feed_config_key": "live_data_feed_config", # Key to data source for paper trading
               "broker_config_key": "paper_broker_config"
           }
           # For DeploymentManager directly:
           # "target_environment_config": { ... }
           # "strategy": DeploymentStrategy.DIRECT # or BLUE_GREEN, CANARY
       }

       logger.info(f"Starting deployment for model: {deployment_run_config['model_id']}")
       try:
           # Using PaperTradingDeployer
           simulation_id = paper_trading_deployer.deploy_to_paper_trading(
               model_id=deployment_run_config["model_id"],
               model_version=deployment_run_config["model_version"],
               simulation_config=deployment_run_config["paper_trading_specific_config"],
               deployment_name=deployment_run_config["deployment_target_name"]
           )
           logger.info(f"Paper trading deployment successful. Simulation ID: {simulation_id}")

           # Alternatively, using DeploymentManager for a generic deployment
           # package_path = deployment_manager.package_model(
           #     model_id=deployment_run_config["model_id"],
           #     model_version=deployment_run_config["model_version"]
           # )
           # deployment_id = deployment_manager.deploy_package(
           #     package_path=package_path,
           #     target_environment_name=deployment_run_config["deployment_target_name"],
           #     deployment_config=deployment_run_config.get("target_environment_config"),
           #     strategy=deployment_run_config.get("strategy", DeploymentStrategy.DIRECT)
           # )
           # logger.info(f"Generic deployment successful. Deployment ID: {deployment_id}")

       except Exception as e:
           logger.exception(f"An error occurred during deployment: {e}")

    For more comprehensive details on deployment procedures, packaging, and setting up environments, please refer to the :doc:`../deployment_guide`.

*   **Full Orchestrated Pipeline**: For sequences of these operations, the ``ModelPipeline`` class (from ``src.pipeline.orchestrator``) is the primary tool.

Running a Full Orchestrated Pipeline with ``ModelPipeline``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``ModelPipeline`` orchestrator allows you to define and run complex, multi-stage workflows. These workflows are defined in your ``pipeline.yaml`` (or an environment-specific override) under a top-level ``pipelines`` key. Each named pipeline consists of a list of stages, and each stage has a type (which maps to a stage class in ``src.pipeline.stages``) and its specific configuration.

**1. Define Your Pipeline in Configuration (e.g., ``configs/base/pipeline.yaml``):**

.. code-block:: yaml

   # In configs/base/pipeline.yaml (or an environment override)
   name: "reinforcement_learning_trading_pipeline"
   version: "1.0.0"
   # ... other global configs ...

   pipelines:
     my_full_training_pipeline: # A unique name for this pipeline definition
       description: "A complete pipeline that ingests data, trains a model, and evaluates it."
       stages:
         - name: "DataIngestion" # User-defined name for this step
           type: "DataIngestionStage" # Maps to a class in src.pipeline.stages
           config:
             data_key: "financial_timeseries_AAPL" # Key from the 'data' section or data.yaml
             # ... other stage-specific configs ...

         - name: "FeatureEngineering"
           type: "FeatureEngineeringStage"
           config:
             input_data_key_in_context: "raw_data_AAPL" # Assumes DataIngestionStage put data here
             output_data_key_in_context: "processed_data_AAPL"
             # ... feature engineering steps ...

         - name: "ModelTraining"
           type: "TrainingStage"
           config:
             model_config_key: "my_dqn_config" # Key from 'model_configs' or models.yaml
             data_input_key_in_context: "processed_data_AAPL"
             training_params_key: "default_training_params" # Key from 'training_params' or training.yaml
             # ... other training stage configs ...

         - name: "ModelEvaluation"
           type: "EvaluationStage"
           config:
             model_id_key_in_context: "trained_model_id" # Assumes TrainingStage put model_id here
             evaluation_data_key_in_context: "processed_data_AAPL" # Or a specific test set key
             evaluation_params_key: "default_evaluation_params"
             # ... other evaluation stage configs ...

     another_pipeline_for_deployment:
       description: "Deploys a pre-trained model."
       stages:
         - name: "ModelDeployment"
           type: "DeploymentStage"
           config:
             model_id: "my_production_model_v2" # Specific model ID to deploy
             model_version: "2.1.0"
             deployment_target_key: "paper_trading_prod_env"
             # ... other deployment configs ...

   # You would also need to define 'data_sources', 'model_configs', 'training_params', etc.
   # elsewhere in your YAML configuration files, which these pipeline stages would reference.

**2. Create an Entry Point Script to Run the Orchestrated Pipeline:**

.. code-block:: python

   import logging
   from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
   from reinforcestrategycreator_pipeline.src.pipeline.orchestrator import ModelPipeline

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def run_orchestrated_pipeline(pipeline_name: str, config_env: str = "development"):
       logger.info(f"Attempting to run orchestrated pipeline: '{pipeline_name}' in '{config_env}' environment.")
       try:
           # Initialize ConfigManager
           # Ensure paths are correct relative to your script's execution location
           config_manager = ConfigManager(
               config_dir="./configs", # Adjust if your script is elsewhere
               environment=config_env
           )

           # Instantiate the ModelPipeline orchestrator
           pipeline_orchestrator = ModelPipeline(
               pipeline_name=pipeline_name,
               config_manager=config_manager
           )

           # Run the pipeline
           logger.info(f"Starting pipeline: {pipeline_name}")
           final_context = pipeline_orchestrator.run()

           if final_context.get_metadata("pipeline_status") == "completed":
               logger.info(f"Pipeline '{pipeline_name}' completed successfully.")
               # You can access results or artifacts from the final_context if stages added them
               # e.g., final_model_id = final_context.get_data("final_model_id")
           else:
               error_message = final_context.get_metadata("error_message", "Unknown error")
               logger.error(f"Pipeline '{pipeline_name}' failed: {error_message}")

       except Exception as e:
           logger.exception(f"An critical error occurred while trying to run pipeline '{pipeline_name}': {e}")

   if __name__ == "__main__":
       # Example: Run the full training pipeline defined in pipeline.yaml
       run_orchestrated_pipeline(pipeline_name="my_full_training_pipeline", config_env="development")

       # Example: Run a deployment pipeline
       # run_orchestrated_pipeline(pipeline_name="another_pipeline_for_deployment", config_env="production")

The ``ModelPipeline`` uses a ``PipelineExecutor`` internally to run each defined stage. Each stage receives a ``PipelineContext`` object, allowing stages to pass data and state to subsequent stages (e.g., a data ingestion stage might put a DataFrame into the context, which a training stage then retrieves).

Command-Line Interface (CLI) and Scripting
------------------------------------------

Currently, the ``ReinforceStrategyCreator Pipeline`` primarily relies on Python scripts as entry points for executing various operations (training, evaluation, HPO, deployment, or full orchestrated pipelines) as demonstrated in the examples above.

Users are encouraged to:

1.  **Create Custom Entry-Point Scripts**: Adapt the provided examples (like those in the ``examples/`` directory or the snippets in this guide) to create their own scripts for specific tasks. These scripts would initialize the ``ConfigManager``, load the desired configurations, instantiate the necessary engine(s) or the ``ModelPipeline`` orchestrator, and then invoke their respective ``run()`` or execution methods.
2.  **Parameterize Scripts**: Use libraries like ``argparse`` in their custom scripts to accept command-line arguments for things like environment selection, pipeline name, specific configuration file paths, or key parameters to override.

    Example of a parameterized script structure:

    .. code-block:: python

       import argparse
       # ... other imports from previous examples (ConfigManager, ModelPipeline, logger setup) ...

       # Assuming run_orchestrated_pipeline(pipeline_name, config_env) is defined as in the
       # "Running a Full Orchestrated Pipeline with ModelPipeline" section above.

       def main():
           parser = argparse.ArgumentParser(description="Run ReinforceStrategyCreator Pipeline operations.")
           parser.add_argument(
               "--pipeline",
               required=True,
               help="Name of the pipeline to run (defined in pipeline.yaml)"
           )
           parser.add_argument(
               "--env",
               default="development",
               help="Configuration environment (e.g., development, production)"
           )
           # Add other arguments as needed, for example:
           # parser.add_argument("--experiment-config", help="Path to an experiment-specific YAML config")
           args = parser.parse_args()

           # You might want to pass experiment_config_path to ConfigManager if using it
           # config_manager = ConfigManager(config_dir="./configs", environment=args.env, experiment_config_path=args.experiment_config)

           run_orchestrated_pipeline(pipeline_name=args.pipeline, config_env=args.env)

       if __name__ == "__main__":
           main()

    This script could then be run from your terminal like:

    .. code-block:: bash

       python your_pipeline_runner_script.py --pipeline my_full_training_pipeline --env production

3.  **Use a Task Runner (Optional)**: For more complex workflows or frequent tasks, consider using a task runner like Invoke, Make, or even simple shell scripts to encapsulate common pipeline execution commands with different configurations.

If a more formal, centralized CLI tool is developed for the pipeline in the future, this section will be updated with its specific commands and usage instructions. For now, direct scripting provides the most flexibility for running and orchestrating pipeline tasks.