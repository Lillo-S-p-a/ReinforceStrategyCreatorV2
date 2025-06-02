"""Training stage implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd # Added import
import json
import pickle
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactType
# Imports for RL TrainingEngine
from reinforcestrategycreator_pipeline.src.models.factory import ModelFactory
from reinforcestrategycreator_pipeline.src.training.engine import TrainingEngine
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager # For type hinting
from reinforcestrategycreator_pipeline.src.data.manager import DataManager # Import DataManager
from reinforcestrategycreator_pipeline.src.monitoring.service import MonitoringService # Added
 
 
class TrainingStage(PipelineStage):
    """
    Stage responsible for training machine learning models.
    
    This stage handles:
    - Loading preprocessed training data
    - Initializing models based on configuration
    - Training models with specified hyperparameters
    - Tracking training metrics
    - Saving trained models as artifacts
    """
    
    def __init__(self, name: str = "training", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training stage.
        
        Args:
            name: Stage name
            config: Stage configuration containing:
                - model_type: Type of model to train
                - model_config: Model-specific configuration
                - training_config: Training parameters (epochs, batch_size, etc.)
                - validation_split: Fraction of data for validation
                - early_stopping: Early stopping configuration
                - checkpoint_config: Model checkpointing settings
        """
        super().__init__(name, config or {})
        # Configs will be fetched from global config via context in setup
        self.global_model_config: Optional[Dict[str, Any]] = None
        self.global_data_config: Optional[Dict[str, Any]] = None # For RL, this might define env behavior
        self.global_training_config: Optional[Dict[str, Any]] = None
        
        self.model_factory: Optional[ModelFactory] = None
        self.training_engine: Optional[TrainingEngine] = None
        self.artifact_store = None # Will be fetched from context
        self.monitoring_service: Optional[MonitoringService] = None # Added
 
        # These will be populated by the TrainingEngine's results
        self.trained_model = None
        self.training_history = None
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the training stage."""
        self.logger.info(f"Setting up {self.name} stage")
        
        # Get ConfigManager and ArtifactStore from context
        config_manager: Optional[ConfigManager] = context.get("config_manager")
        self.artifact_store = context.get("artifact_store")

        if not config_manager:
            raise ValueError("ConfigManager not found in pipeline context. Cannot proceed with TrainingStage setup.")

        # Fetch global configurations
        # The stage-specific self.config can be used for overrides if needed later
        pipeline_cfg = config_manager.get_config()
        self.global_model_config = pipeline_cfg.model.model_dump() if pipeline_cfg.model else {}
        self.global_data_config = pipeline_cfg.data.model_dump() if pipeline_cfg.data else {}
        self.global_training_config = pipeline_cfg.training.model_dump() if pipeline_cfg.training else {}

        self.logger.info(f"Global Model Config: {self.global_model_config}")
        self.logger.info(f"Global Data Config: {self.global_data_config}")
        self.logger.info(f"Global Training Config: {self.global_training_config}")

        # Initialize ModelFactory and TrainingEngine
        # ModelFactory needs the path to model implementations.
        # Assuming models are in reinforcestrategycreator_pipeline.src.models.implementations
        # ModelFactory now auto-registers from a fixed relative path.
        self.model_factory = ModelFactory()
        
        # Instantiate DataManager
        # DataManager needs config_manager and artifact_store.
        # It will use data_config.cache_dir from global config by default.
        data_manager = DataManager(
            config_manager=config_manager,
            artifact_store=self.artifact_store
        )
        self.logger.info("DataManager instantiated.")
        
        # TrainingEngine requires model_factory, artifact_store, and optionally data_manager
        self.training_engine = TrainingEngine(
            model_factory=self.model_factory,
            artifact_store=self.artifact_store,
            data_manager=data_manager # Pass the DataManager instance
        )
        self.logger.info("ModelFactory, DataManager, and TrainingEngine initialized.")

        # The 'processed_features' check might be less relevant for RL if the
        # environment handles its own data based on data_config.
        # For now, let's keep it, as some RL envs might take pre-processed data.
        processed_features = context.get("processed_features")
        if not isinstance(processed_features, pd.DataFrame): # Allow empty if it's just a signal
            self.logger.warning("'processed_features' in context is not a DataFrame. RL environment might handle data loading.")
        elif processed_features.empty:
             self.logger.warning("'processed_features' DataFrame in context is empty.")
        else:
            self.logger.info(f"Received 'processed_features' with shape {processed_features.shape}")

        # Get monitoring service from context
        self.monitoring_service = context.get("monitoring_service") # Added
        if self.monitoring_service: # Added
            self.logger.info("MonitoringService retrieved from context.") # Added
        else: # Added
            self.logger.warning("MonitoringService not found in context. Monitoring will be disabled for this stage.") # Added
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the training stage using the RL TrainingEngine.
        
        Args:
            context: Pipeline context. Expected to have ConfigManager.
                     'processed_features' might be used by the environment.
            
        Returns:
            Updated pipeline context with trained model and history.
        """
        self.logger.info(f"Running {self.name} stage using RL TrainingEngine.")

        if self.monitoring_service: # Added
            self.monitoring_service.log_event(event_type=f"{self.name}.started", description=f"Stage {self.name} started.") # Added
        
        try: # Added
            if not self.training_engine:
                raise RuntimeError("TrainingEngine not initialized in TrainingStage. Call setup() first.")
            # This check should be inside the try block as well
            if not self.global_model_config or not self.global_data_config or not self.global_training_config: # Indented
                # These should have been populated in setup()
                raise RuntimeError("Global configurations (model, data, training) not properly loaded in TrainingStage setup().") # Indented
 
            self.logger.info("Calling TrainingEngine.train()...") # Indented
            # The TrainingEngine's train method is expected to return a dictionary # Indented
            # containing at least 'trained_model' and 'history'. # Indented
            training_start_time = datetime.now() # Indented
            training_result = self.training_engine.train( # Indented
                model_config=self.global_model_config, # Indented
                data_config=self.global_data_config, # Indented
                training_config=self.global_training_config # Indented
            ) # Indented
            training_duration_seconds = (datetime.now() - training_start_time).total_seconds() # Indented
            
            self.trained_model = training_result.get("model") # Changed key from "trained_model" to "model" # Indented
            self.training_history = training_result.get("history", {}) # Indented
            
            # Check if training was successful before assuming model is present # Indented
            if not training_result.get("success", False) or not self.trained_model: # Indented
                error_msg = training_result.get("error", "Unknown error from TrainingEngine") # Indented
                self.logger.error(f"TrainingEngine reported failure or did not return a model. Error: {error_msg}") # Indented
                # Re-raise or handle error appropriately. For now, let the original ValueError be raised if model is None. # Indented
                if not self.trained_model: # This check will now be for the "model" key # Indented
                     raise ValueError(f"TrainingEngine.train() did not return a 'model' or training failed. Reported error: {error_msg}") # Indented
 
            self.logger.info(f"TrainingEngine finished. Trained model type: {type(self.trained_model)}") # Indented
            
            # Store model and training results in context # Indented
            context.set("trained_model", self.trained_model) # Indented
            context.set("training_history", self.training_history) # Indented
            context.set("model_type", self.global_model_config.get("model_type")) # From global config # Indented
            context.set("model_config", self.global_model_config) # Full model config used # Indented
            
            # Store training metadata # Indented
            metadata = { # Indented
                "model_type": self.global_model_config.get("model_type"), # Indented
                "training_duration_seconds": training_duration_seconds, # Indented
                "final_metrics": self._get_final_metrics_from_history(), # Indented
                "hyperparameters_used": self.global_training_config # Indented
            } # Indented
            context.set("training_metadata", metadata) # Indented

            if self.monitoring_service: # Added
                self.monitoring_service.log_metric(f"{self.name}.training_duration_seconds", training_duration_seconds) # Added
                # Log metrics from history
                # Assuming history is a list of dicts, where each dict is an epoch
                if isinstance(self.training_history, list): # Added
                    for i, epoch_metrics in enumerate(self.training_history): # Added
                        if isinstance(epoch_metrics, dict): # Added
                            for metric_name, metric_value in epoch_metrics.items(): # Added
                                if isinstance(metric_value, (int, float)): # Added
                                    self.monitoring_service.log_metric(f"{self.name}.epoch_{i}.{metric_name}", metric_value) # Added
                elif isinstance(self.training_history, dict): # Handle dict of lists case from _get_final_metrics_from_history # Added
                    for metric_name, values_list in self.training_history.items(): # Added
                        if isinstance(values_list, list): # Added
                             for i, metric_value in enumerate(values_list): # Added
                                if isinstance(metric_value, (int, float)): # Added
                                    self.monitoring_service.log_metric(f"{self.name}.{metric_name}.step_{i}", metric_value) # Added
                # Log final metrics specifically if they are different or summarized
                final_metrics_log = self._get_final_metrics_from_history() # Added
                for fm_name, fm_value in final_metrics_log.items(): # Added
                    self.monitoring_service.log_metric(f"{self.name}.final.{fm_name}", fm_value) # Added
            
            # Save model artifact if artifact store is available # Indented
            if self.artifact_store: # Indented
                self._save_model_artifact(context) # Indented
            
            self.logger.info(f"Training stage completed. Model type: {self.global_model_config.get('model_type')}") # Indented
            self.logger.info(f"Final metrics from history: {metadata['final_metrics']}") # Indented

            if self.monitoring_service: # Added
                # Checkpoint saved event is implicitly handled by TrainingEngine if it logs,
                # otherwise, if _save_model_artifact is the main checkpoint, log here.
                # For now, assume TrainingEngine might log its own checkpoints.
                # We log the final model save as a 'checkpoint.saved' type event.
                if context.get("trained_model_artifact_id"): # Added
                    self.monitoring_service.log_event( # Added
                        event_type=f"{self.name}.checkpoint.saved", # Added
                        description=f"Trained model artifact saved: {context.get('trained_model_artifact_id')}", # Added
                        level="info", # Added
                        context={"artifact_id": context.get("trained_model_artifact_id"), "version": context.get("trained_model_version")} # Added
                    ) # Added
                self.monitoring_service.log_event(event_type=f"{self.name}.completed", description=f"Stage {self.name} completed successfully.", level="info") # Added
            
            return context # Indented
            
        except Exception as e:
            self.logger.error(f"Error in RL TrainingStage run: {str(e)}", exc_info=True)
            if self.monitoring_service: # Added
                self.monitoring_service.log_event(event_type=f"{self.name}.failed", description=f"Stage {self.name} failed: {str(e)}", level="error", context={"error_details": str(e)}) # Added
            # Set pipeline status to failed in context? Executor might do this.
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after training."""
        self.logger.info(f"Tearing down {self.name} stage")
        # Any specific cleanup for TrainingEngine or models can go here
        # For example, if TrainingEngine created temp files not managed by artifact store.
        # RLlib might also have cleanup needs if not handled by its shutdown.

    def _get_final_metrics_from_history(self) -> Dict[str, float]:
        """Extract final metrics from RL training history."""
        if not self.training_history:
            self.logger.warning("Attempted to get final metrics, but training_history is empty.")
            return {}
        
        final_metrics = {}
        # TrainingEngine's history format might vary.
        # Common RLlib pattern: history is a list of dicts, each dict is an epoch/iteration.
        if isinstance(self.training_history, list) and self.training_history:
            last_epoch_metrics = self.training_history[-1]
            if isinstance(last_epoch_metrics, dict):
                for key, value in last_epoch_metrics.items():
                    if isinstance(value, (int, float)): # Only numeric metrics
                        final_metrics[key] = value # Use original key
        elif isinstance(self.training_history, dict): # Or, dict of lists
             for key, values_list in self.training_history.items():
                if isinstance(values_list, list) and values_list and isinstance(values_list[-1], (int, float)):
                    final_metrics[key] = values_list[-1]
        else:
            self.logger.warning(f"Unrecognized training_history format: {type(self.training_history)}. Cannot extract final metrics.")

        # Example: if 'episode_reward_mean' is a key in the last epoch's metrics
        # if 'episode_reward_mean' in final_metrics:
        #    pass # It's already there
        
        self.logger.debug(f"Extracted final metrics: {final_metrics}")
        return final_metrics
        
    def _save_model_artifact(self, context: PipelineContext) -> None:
        """Save the trained RL model as an artifact."""
        if not self.trained_model:
            self.logger.warning("No trained_model available in TrainingStage, skipping artifact saving.")
            return
        if not self.artifact_store:
            self.logger.warning("Artifact store not available in TrainingStage, skipping artifact saving.")
            return
            
        temp_model_save_dir = None # Initialize for finally block
        try:
            model_name_from_config = self.global_model_config.get("name", self.global_model_config.get("model_type", "rl_model"))
            # Sanitize model_name for use in path/artifact_id
            safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name_from_config)
            
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            artifact_id_str = f"{safe_model_name}_{version}"

            # Create a unique temporary directory for saving the model artifact components
            # This is important because RLlib's save() creates a directory itself.
            temp_model_save_dir = Path(f"./temp_artifact_{artifact_id_str}").resolve()
            temp_model_save_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created temporary directory for model artifact: {temp_model_save_dir}")

            artifact_path_to_save: Path # This will be the path to the directory or main file saved by the model

            if hasattr(self.trained_model, 'save') and callable(self.trained_model.save):
                # Handles RLlib Algorithm.save(checkpoint_dir)
                # The save method itself creates a subdirectory within temp_model_save_dir
                rllib_checkpoint_path = self.trained_model.save(str(temp_model_save_dir))
                artifact_path_to_save = Path(rllib_checkpoint_path)
                self.logger.info(f"RLlib model saved to checkpoint directory: {artifact_path_to_save}")
            elif hasattr(self.trained_model, 'save_model') and callable(self.trained_model.save_model):
                # For custom models with a specific save_model(path_or_dir) method
                # Assume it saves to a file or dir named 'model_content' inside temp_model_save_dir
                custom_save_path = temp_model_save_dir / "model_content"
                self.trained_model.save_model(custom_save_path)
                artifact_path_to_save = custom_save_path
                self.logger.info(f"Custom model saved to: {artifact_path_to_save}")
            else:
                # Fallback to pickling the model object itself into the temp dir
                model_file_path = temp_model_save_dir / "model.pkl"
                with open(model_file_path, "wb") as f:
                    pickle.dump(self.trained_model, f)
                artifact_path_to_save = model_file_path # Save the .pkl file
                self.logger.info(f"Model (unknown type/no save method) pickled to: {artifact_path_to_save}")
            
            self.logger.info(f"Temporary model content saved at: {artifact_path_to_save}")
            self.logger.info(f"Attempting to save artifact with generated ID: {artifact_id_str}")

            # Now, save the entire directory or the specific file using artifact_store
            artifact_metadata_obj = self.artifact_store.save_artifact(
                artifact_id=artifact_id_str, # Use the generated ID
                artifact_path=artifact_path_to_save, # Path to the saved model (file or dir)
                artifact_type=ArtifactType.MODEL,
                # version parameter is omitted to let artifact_store handle it, or pass `version`
                metadata={
                    "model_type": self.global_model_config.get("model_type", "unknown"),
                    "training_config_snapshot": self.global_training_config, # Snapshot of training params
                    "data_config_snapshot": self.global_data_config, # Snapshot of data params
                    "source_data_metadata": context.get("data_metadata", {}) # From DataIngestion
                },
                tags=["trained_rl_model", str(self.global_model_config.get("model_type"))]
            )
            
            self.trained_model_artifact_id = artifact_metadata_obj.artifact_id # Assign to class attribute
            context.set("trained_model_artifact_id", self.trained_model_artifact_id)
            context.set("trained_model_version", artifact_metadata_obj.version) # Store version from artifact store
            self.logger.info(f"Trained RL model saved as artifact. Artifact ID: {self.trained_model_artifact_id}, Version: {artifact_metadata_obj.version}")
            self.logger.info(f"Successfully set 'trained_model_artifact_id' in PipelineContext: {self.trained_model_artifact_id}")

        except Exception as e:
            self.logger.error(f"Failed to save RL model artifact: {e}", exc_info=True)
        finally:
            if temp_model_save_dir and temp_model_save_dir.exists():
                import shutil
                try:
                    shutil.rmtree(temp_model_save_dir)
                    self.logger.debug(f"Cleaned up temporary model artifact directory: {temp_model_save_dir}")
                except Exception as e_rm:
                    self.logger.error(f"Error cleaning up temp model dir {temp_model_save_dir}: {e_rm}")

    # Remove old placeholder methods: _initialize_model, _split_data, _train_model, _should_stop_early, _get_final_metrics
    # These are now effectively replaced by the TrainingEngine's logic and _get_final_metrics_from_history.