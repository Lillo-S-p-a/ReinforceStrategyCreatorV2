"""Example script for running Hyperparameter Optimization (HPO) for DQN model."""

import logging
import sys
from pathlib import Path
import json

# Add src to path
# Assuming the script is in reinforcestrategycreator_pipeline/examples/
# and src is reinforcestrategycreator_pipeline/src/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Set PYTHONPATH environment variable for Ray workers
import os
current_pythonpath = os.environ.get('PYTHONPATH', '')
new_pythonpath = f"{project_root}:{current_pythonpath}" if current_pythonpath else str(project_root)
os.environ['PYTHONPATH'] = new_pythonpath

# Import ray early to configure runtime environment for workers
import ray

from src.training import TrainingEngine, HPOptimizer
from src.models.factory import ModelFactory
from src.models.registry import ModelRegistry # Not strictly needed for this script if not registering final model
from src.artifact_store.local_adapter import LocalFileSystemStore as LocalArtifactStore
from src.data.manager import DataManager
from src.config.loader import ConfigLoader
from src.config.manager import ConfigManager
from src.config.models import PipelineConfig # For type hinting pipeline_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce verbosity of some libraries if needed
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


def run_dqn_hpo():
    """Runs HPO for the DQN model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting DQN HPO process...")

    # Base path for configurations relative to this script
    # Script is in ReinforceStrategyCreatorV2/reinforcestrategycreator_pipeline/examples/
    # Configs are in ReinforceStrategyCreatorV2/reinforcestrategycreator_pipeline/configs/
    project_root = Path(__file__).resolve().parent.parent 
    config_base_path = project_root / "configs" / "base"
    
    # Initialize components
    logger.info("Initializing components...")
    model_factory = ModelFactory()
    # Using relative paths from the project root (ReinforceStrategyCreatorV2)
    # as this script will likely be run from there.
    artifact_store_path = project_root / "artifacts" / "hpo_dqn"
    model_registry_path = project_root / "model_registry" / "hpo_dqn"
    checkpoints_path = project_root / "checkpoints" / "hpo_dqn"
    hpo_results_path = project_root / "hpo_results" / "dqn"

    artifact_store_path.mkdir(parents=True, exist_ok=True)
    model_registry_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    hpo_results_path.mkdir(parents=True, exist_ok=True)

    artifact_store = LocalArtifactStore(root_path=str(artifact_store_path))
    model_registry = ModelRegistry(artifact_store=LocalArtifactStore(str(model_registry_path)))
    
    training_engine = TrainingEngine(
        model_factory=model_factory,
        model_registry=model_registry, # Optional for HPO, but good practice
        artifact_store=artifact_store,
        checkpoint_dir=str(checkpoints_path)
    )
    
    hpo_optimizer = HPOptimizer(
        training_engine=training_engine,
        artifact_store=artifact_store, # HPO might save trial artifacts
        results_dir=str(hpo_results_path)
    )
    
    # Load configurations
    logger.info("Loading configurations...")
    config_loader = ConfigLoader(base_path=str(project_root)) # Ensure loader looks from project root

    try:
        # Path to pipeline.yaml relative to project_root
        pipeline_config_path = config_base_path / "pipeline.yaml"
        pipeline_config_dict = config_loader.load_yaml(str(pipeline_config_path))
        pipeline_config: PipelineConfig = PipelineConfig(**pipeline_config_dict)
        logger.info(f"Pipeline config loaded from {pipeline_config_path}")

        # Create ConfigManager instance for DataManager
        config_manager = ConfigManager(
            config_dir=project_root / "configs",
            environment=None
        )
        # Set the loaded config in the manager
        config_manager._config = pipeline_config
        config_manager._raw_config = pipeline_config_dict

        # Path to hpo.yaml relative to project_root
        hpo_config_path = config_base_path / "hpo.yaml"
        hpo_config = config_loader.load_yaml(str(hpo_config_path))
        logger.info(f"HPO config loaded from {hpo_config_path}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Prepare Data using DataManager
    logger.info("Preparing data...")
    try:
        # Ensure data_source_id is correctly referenced from pipeline_config.data
        data_manager = DataManager(config_manager=config_manager, artifact_store=artifact_store)
        # get_data() should return a dict like {"train_data": (X_train, y_train), "val_data": (X_val, y_val)}
        # or whatever format TrainingEngine expects.
        # For RL, it might be an environment or preprocessed data.
        # Assuming DataManager handles the creation of appropriate train/val splits or environments
        # based on pipeline_config.data.
        # The HPO example uses a simple dict with "train_data" and "val_data" keys.
        # We need to ensure our DataManager provides data in a compatible format for the TrainingEngine.
        # For now, let's assume TrainingEngine can handle the direct output of data_manager.get_data()
        # or that HPOptimizer's _objective function adapts it.
        # The hpo_example.py creates sample data in the format:
        # {"train_data": (X_train, y_train), "val_data": (X_val, y_val)}
        # Let's assume our DataManager's get_data method will provide something similar
        # or that the TrainingEngine is adapted for it.
        # For RL, data_config for training engine might just be env_id or env_config
        
        # The TrainingEngine.train method expects data_config.
        # For HPO, this data_config will be passed to each trial.
        # We need to ensure the DataManager provides data in a way that can be used.
        # Let's assume the HPOptimizer's objective function will handle data loading if needed per trial,
        # or that we pass a data configuration that the TrainingEngine understands.
        # The example passes `data_config=create_sample_data()` to `hpo_optimizer.optimize`.
        # This `data_config` is then passed to `training_engine.train` within the HPO loop.
        
        # For a real scenario, data_manager.get_data() might return a dataset object or path.
        # The HPO objective function would then use this.
        # For simplicity, and aligning with the task to use actual project data,
        # we'll pass the data section of the pipeline_config to the HPO,
        # and assume the TrainingEngine or HPO objective can use it.
        
        # The HPOptimizer's `_objective` function in `src/training/hpo_optimizer.py`
        # calls `self.training_engine.train(model_config=current_model_config, data_config=self.data_config, ...)`
        # So, `self.data_config` is set when `optimize` is called.
        # We need to provide the data here.
        
        # Register the data source from pipeline config before loading
        # The DataManager needs to know about the data source configuration
        data_source_config = pipeline_config.data.model_dump()
        data_manager.register_source(
            source_id="yfinance_spy",
            source_type="yfinance",
            config=data_source_config
        )
        logger.info("Data source 'yfinance_spy' registered successfully")
        
        # Let's fetch the data once. This might be large.
        # Alternatively, HPO trials could load data themselves if memory is an issue.
        # For now, load once.
        processed_data = data_manager.load_data(source_id="yfinance_spy") # Load data from registered source
        # Ensure processed_data is in the format expected by TrainingEngine for HPO
        # e.g., {"train_data": train_df, "val_data": val_df} or specific environment objects
        # The example uses tuples: (features, labels). This needs to align with DQN model.
        # For DQN, it's usually an environment.
        # The TrainingEngine for DQN should be able to take an env_id or env_config.
        # Let's assume data_config for HPO will be the 'data' section of pipeline_config.
        # The TrainingEngine's train method for DQN should then use this to instantiate the env.

        data_for_hpo = pipeline_config.data.model_dump() # Pass the Pydantic model as dict

        logger.info("Data prepared successfully.")
    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        sys.exit(1)

    # Model configuration for DQN
    # Base model config; HPO will override hyperparameters
    model_config_hpo = {
        "model_type": "DQN",  # Ensure this matches a registered model type in ModelFactory
        "name": "dqn_hpo_tuned", # Name for the HPO experiment's model instances
        "hyperparameters": {
            # Default values, will be overridden by HPO search space
            # These are from pipeline.yaml, but HPO will search over them
            "learning_rate": pipeline_config.training.learning_rate, # Example
            "buffer_size": pipeline_config.training.replay_buffer_size, # Example
            "batch_size": pipeline_config.training.batch_size, # Example
            # Add other DQN specific params if they are part of the base model_config
            # and not solely defined by HPO search space.
            # Usually, the HPO search space defines all tunable params.
        }
        # Add any other necessary fields for DQN model initialization if not covered by HPO
        # e.g., network architecture if not part of hyperparameters being tuned
    }

    # Training configuration for each HPO trial
    # These are settings for how each individual model in the HPO search is trained
    training_config_hpo = {
        "episodes": 50,  # Number of episodes per HPO trial (adjust as needed, less than full training)
        "max_steps_per_episode": 200, # Optional: if your env needs it
        # "batch_size": 32, # This will be tuned by HPO from search_space
        "validation_interval": 5, # How often to run validation during a trial
        "early_stopping_patience": 5, # For HPO trials
        "save_checkpoints": False, # Usually false for HPO trials to save space/time
        "save_best_only": False,
        "monitor_metric": "episode_reward_mean", # Metric to monitor for early stopping and for HPO
        "monitor_mode": "max" # "max" for reward, "min" for loss
    }
    logger.info(f"Training config for HPO trials: {training_config_hpo}")


    # Use DQN search space and parameter mappings from hpo_config
    dqn_search_space = hpo_config["search_spaces"]["dqn"]
    dqn_param_mapping = hpo_config["param_mappings"]["dqn"]
    logger.info(f"DQN Search Space: {json.dumps(dqn_search_space, indent=2)}")
    logger.info(f"DQN Param Mapping: {json.dumps(dqn_param_mapping, indent=2)}")

    # Get HPO experiment preset (e.g., "quick_test")
    # Task specifies "quick_test" initially
    experiment_preset_name = "quick_test" 
    experiment_settings = hpo_config["experiments"].get(experiment_preset_name)
    if not experiment_settings:
        logger.error(f"Experiment preset '{experiment_preset_name}' not found in hpo.yaml.")
        sys.exit(1)
    
    logger.info(f"Using HPO experiment preset: '{experiment_preset_name}'")
    logger.info(f"Preset settings: {json.dumps(experiment_settings, indent=2)}")

    # Run optimization
    logger.info("Starting HPO optimization...")
    try:
        results = hpo_optimizer.optimize(
            model_config=model_config_hpo,
            data_config=data_for_hpo, # This is pipeline_config.data dict
            training_config=training_config_hpo,
            param_space=dqn_search_space,
            param_mapping=dqn_param_mapping,
            metric=training_config_hpo["monitor_metric"], # e.g., "episode_reward_mean" or "val_loss"
            mode=training_config_hpo["monitor_mode"],    # "max" for reward, "min" for loss
            name=f"dqn_hpo_{experiment_preset_name}", # Unique name for the HPO run
            **experiment_settings # Unpack preset settings (num_trials, search_algorithm, etc.)
        )
        
        logger.info("HPO optimization finished.")
        
        results_file_path = results.get("results_file_path", "N/A")
        logger.info(f"HPO results file saved at: {results_file_path}")
        logger.info(f"Best parameters found: {json.dumps(results['best_params'], indent=2)}")
        logger.info(f"Best score ({results['metric_name']}): {results['best_score']:.4f}")
        
        # Analyze results
        logger.info("Analyzing HPO results...")
        analysis = hpo_optimizer.analyze_results(results_file_path) # Pass path or results object
        
        logger.info(f"\n--- HPO Run Summary ({results.get('name', 'N/A')}) ---")
        logger.info(f"Total trials: {analysis.get('total_trials', 'N/A')}")
        logger.info(f"Best trial ID: {analysis.get('best_trial_id', 'N/A')}")
        logger.info(f"Best score ({results['metric_name']}): {analysis.get('best_score', 'N/A'):.4f}")
        
        logger.info("\nTop K Trials:")
        for i, trial in enumerate(analysis.get("top_k_trials", [])):
            logger.info(f"  Rank {i+1}: Score={trial.get(results['metric_name'], trial.get('metric_value', 'N/A')):.4f} - Params={json.dumps(trial.get('params', {}))}")

        logger.info(f"\nParameter Importance (if available):")
        param_importance = analysis.get("parameter_importance")
        if param_importance:
            for param, importance in param_importance.items():
                logger.info(f"  {param}: {importance:.3f}")
        else:
            logger.info("  Parameter importance not available for this search algorithm/scheduler.")
            
        # Get best model configuration
        best_model_full_config = hpo_optimizer.get_best_model_config(
            base_model_config=model_config_hpo, # The initial model_config used for HPO
            param_mapping=dqn_param_mapping,
            # results_or_path=results # Pass the results object directly
        )
        logger.info(f"\nBest model full configuration based on HPO:")
        logger.info(json.dumps(best_model_full_config, indent=2))

    except ImportError as e:
        logger.error(f"ImportError during HPO: {e}. Ensure Ray[tune] and Optuna are installed.")
        logger.error("Try: pip install 'ray[tune]' optuna")
    except Exception as e:
        logger.error(f"Error during HPO optimization: {e}", exc_info=True)

if __name__ == "__main__":
    setup_logging()
    run_dqn_hpo()