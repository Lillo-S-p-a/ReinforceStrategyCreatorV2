"""Training Engine for managing model training workflows."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..models.base import ModelBase
from ..models.factory import ModelFactory, get_factory
from ..models.registry import ModelRegistry
from ..data.manager import DataManager
from ..artifact_store.base import ArtifactStore, ArtifactType
from .callbacks import (
    CallbackBase, CallbackList, LoggingCallback, 
    ModelCheckpointCallback, EarlyStoppingCallback
)


class TrainingEngine:
    """Main engine for managing model training.
    
    This engine handles the training loop, callbacks, checkpointing,
    and metrics collection for any model that implements ModelBase.
    """
    
    def __init__(
        self,
        model_factory: Optional[ModelFactory] = None,
        model_registry: Optional[ModelRegistry] = None,
        artifact_store: Optional[ArtifactStore] = None,
        data_manager: Optional[DataManager] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the training engine.
        
        Args:
            model_factory: ModelFactory instance (uses global if not provided)
            model_registry: Optional ModelRegistry for model versioning
            artifact_store: Optional ArtifactStore for saving artifacts
            data_manager: Optional DataManager for data loading
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model_factory = model_factory or get_factory()
        self.model_registry = model_registry
        self.artifact_store = artifact_store
        self.data_manager = data_manager
        
        # Set up checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger("TrainingEngine")
        
        # Training state
        self.model = None
        self.training_history = {}
        self.current_epoch = 0
        self.is_training = False
        self._stop_training = False
    
    def train(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[CallbackBase]] = None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Train a model with the specified configuration.
        
        Args:
            model_config: Model configuration including type and hyperparameters
            data_config: Data configuration for loading training data
            training_config: Training configuration (epochs, batch_size, etc.)
            callbacks: List of callbacks to use during training
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary containing training results and history
        """
        try:
            # Set training flag
            self.is_training = True
            self._stop_training = False
            
            # Default training config
            if training_config is None:
                training_config = {}
            
            # Extract training parameters
            epochs = training_config.get("epochs", 10)
            batch_size = training_config.get("batch_size", 32)
            validation_split = training_config.get("validation_split", 0.2)
            shuffle = training_config.get("shuffle", True)
            
            # Initialize or load model
            if resume_from_checkpoint:
                self.model, start_epoch = self._load_checkpoint(resume_from_checkpoint)
                self.logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
            else:
                # Create model from config
                self.model = self.model_factory.create_from_config(model_config)
                start_epoch = 0
            
            # Load data
            train_data, val_data = self._load_data(data_config, validation_split)
            
            # Build model if needed
            if not self.model.is_trained and hasattr(self.model, 'build'):
                # Infer shapes from data
                input_shape = self._get_data_shape(train_data, "input")
                output_shape = self._get_data_shape(train_data, "output")
                self.model.build(input_shape, output_shape)
            
            # Set up callbacks
            callback_list = self._setup_callbacks(
                callbacks, training_config, model_config
            )
            
            # Initialize training history
            self.training_history = {
                "loss": [],
                "val_loss": [],
                "epochs": [],
                "metrics": {}
            }
            
            # Training loop
            self.logger.info(f"Starting training for {epochs} epochs")
            callback_list.on_train_begin({"epochs": epochs, "batch_size": batch_size})
            
            for epoch in range(start_epoch, epochs):
                if self._stop_training:
                    self.logger.info("Training stopped by callback")
                    break
                
                self.current_epoch = epoch
                epoch_logs = {"epoch": epoch}
                
                # Epoch begin
                callback_list.on_epoch_begin(epoch, epoch_logs)
                
                # Training phase
                train_metrics = self._train_epoch(
                    train_data, batch_size, shuffle, callback_list
                )
                epoch_logs.update(train_metrics)
                
                # Validation phase
                if val_data is not None:
                    val_metrics = self._validate_epoch(val_data, batch_size)
                    epoch_logs.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Update history
                self._update_history(epoch_logs)
                
                # Epoch end
                callback_list.on_epoch_end(epoch, epoch_logs)
            
            # Training end
            final_logs = {
                "final_epoch": self.current_epoch,
                "history": self.training_history
            }
            callback_list.on_train_end(final_logs)
            
            # Mark model as trained
            self.model.is_trained = True
            self.model.update_metadata({
                "training_completed": datetime.now().isoformat(),
                "training_epochs": self.current_epoch + 1,
                "training_config": training_config
            })
            
            # Save final model if registry is available
            if self.model_registry and self.artifact_store:
                model_id = self._save_final_model(model_config, training_config)
                final_logs["model_id"] = model_id
            
            return {
                "success": True,
                "model": self.model,
                "history": self.training_history,
                "final_metrics": epoch_logs,
                "epochs_trained": self.current_epoch + 1,
                **final_logs
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "history": self.training_history,
                "epochs_trained": self.current_epoch
            }
        finally:
            self.is_training = False
    
    def _load_data(
        self,
        data_config: Dict[str, Any],
        validation_split: float
    ) -> Tuple[Any, Optional[Any]]:
        """Load training and validation data.
        
        Args:
            data_config: Data configuration
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        if self.data_manager:
            # Use data manager if available
            source_id = data_config.get("source_id")
            if not source_id:
                raise ValueError("data_config must contain 'source_id' when using DataManager")
            
            # Load data
            data = self.data_manager.load_data(source_id, **data_config.get("params", {}))
            
            # Split into train/validation
            if validation_split > 0:
                split_idx = int(len(data) * (1 - validation_split))
                train_data = data[:split_idx]
                val_data = data[split_idx:]
            else:
                train_data = data
                val_data = None
                
        else:
            # Direct data loading (simplified for now)
            # In practice, this would handle various data formats
            train_data = data_config.get("train_data")
            val_data = data_config.get("val_data")
            
            if train_data is None:
                raise ValueError("No training data provided")
        
        return train_data, val_data
    
    def _get_data_shape(self, data: Any, data_type: str) -> Tuple[int, ...]:
        """Infer data shape from the dataset.
        
        Args:
            data: Dataset
            data_type: "input" or "output"
            
        Returns:
            Shape tuple
        """
        # This is a simplified implementation
        # In practice, this would handle various data formats
        if hasattr(data, 'shape'):
            return data.shape[1:]  # Remove batch dimension
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            sample = data[0]
            if data_type == "input" and isinstance(sample, (list, tuple)):
                return (len(sample[0]),) if len(sample) > 0 else (1,)
            elif data_type == "output" and isinstance(sample, (list, tuple)):
                return (len(sample[1]),) if len(sample) > 1 else (1,)
        
        # Default shape
        return (1,)
    
    def _setup_callbacks(
        self,
        callbacks: Optional[List[CallbackBase]],
        training_config: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> CallbackList:
        """Set up callbacks for training.
        
        Args:
            callbacks: User-provided callbacks
            training_config: Training configuration
            model_config: Model configuration
            
        Returns:
            CallbackList instance
        """
        callback_list = CallbackList(callbacks or [])
        
        # Add default logging callback if not present
        has_logging = any(isinstance(cb, LoggingCallback) for cb in callback_list.callbacks)
        if not has_logging:
            log_callback = LoggingCallback(
                log_frequency=training_config.get("log_frequency", "epoch"),
                verbose=training_config.get("verbose", 1)
            )
            callback_list.append(log_callback)
        
        # Add checkpoint callback if not present and we have storage
        has_checkpoint = any(isinstance(cb, ModelCheckpointCallback) for cb in callback_list.callbacks)
        if not has_checkpoint and (self.model_registry or training_config.get("save_checkpoints", True)):
            checkpoint_callback = ModelCheckpointCallback(
                checkpoint_dir=self.checkpoint_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_registry=self.model_registry,
                artifact_store=self.artifact_store,
                save_frequency=training_config.get("checkpoint_frequency", "epoch"),
                save_best_only=training_config.get("save_best_only", False),
                monitor=training_config.get("monitor", "loss"),
                mode=training_config.get("monitor_mode", "min")
            )
            callback_list.append(checkpoint_callback)
        
        # Set model reference for callbacks
        callback_list.set_model(self.model)
        callback_list.set_training_config(training_config)
        
        return callback_list
    
    def _train_epoch(
        self,
        train_data: Any,
        batch_size: int,
        shuffle: bool,
        callback_list: CallbackList
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_data: Training data
            batch_size: Batch size
            shuffle: Whether to shuffle data
            callback_list: Callbacks to use
            
        Returns:
            Dictionary of training metrics
        """
        # This is a simplified implementation
        # The actual implementation would depend on the model's train method
        
        # For now, delegate to the model's train method
        if hasattr(self.model, 'train'):
            epoch_metrics = self.model.train(
                train_data,
                batch_size=batch_size,
                epochs=1,  # Train for one epoch at a time
                shuffle=shuffle,
                verbose=0  # We handle logging via callbacks
            )
            
            # Extract metrics
            if isinstance(epoch_metrics, dict):
                return epoch_metrics
            else:
                return {"loss": float(epoch_metrics) if epoch_metrics is not None else 0.0}
        else:
            # Fallback for models without train method
            return {"loss": np.random.random()}  # Placeholder
    
    def _validate_epoch(
        self,
        val_data: Any,
        batch_size: int
    ) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            val_data: Validation data
            batch_size: Batch size
            
        Returns:
            Dictionary of validation metrics
        """
        # Delegate to model's evaluate method
        if hasattr(self.model, 'evaluate'):
            val_metrics = self.model.evaluate(
                val_data,
                batch_size=batch_size,
                verbose=0
            )
            
            if isinstance(val_metrics, dict):
                return val_metrics
            else:
                return {"loss": float(val_metrics) if val_metrics is not None else 0.0}
        else:
            # Fallback
            return {"loss": np.random.random()}  # Placeholder
    
    def _update_history(self, epoch_logs: Dict[str, Any]) -> None:
        """Update training history with epoch results.
        
        Args:
            epoch_logs: Logs from the current epoch
        """
        # Update standard metrics
        if "loss" in epoch_logs:
            self.training_history["loss"].append(epoch_logs["loss"])
        if "val_loss" in epoch_logs:
            self.training_history["val_loss"].append(epoch_logs["val_loss"])
        
        self.training_history["epochs"].append(epoch_logs.get("epoch", self.current_epoch))
        
        # Update other metrics
        for key, value in epoch_logs.items():
            if key not in ["epoch", "loss", "val_loss"] and isinstance(value, (int, float)):
                if key not in self.training_history["metrics"]:
                    self.training_history["metrics"][key] = []
                self.training_history["metrics"][key].append(value)
    
    def _save_final_model(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """Save the final trained model to the registry.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            
        Returns:
            Model ID from the registry
        """
        model_name = model_config.get("name", f"{self.model.model_type}_model")
        
        # Prepare metrics
        final_metrics = {}
        if self.training_history["loss"]:
            final_metrics["final_loss"] = self.training_history["loss"][-1]
        if self.training_history["val_loss"]:
            final_metrics["final_val_loss"] = self.training_history["val_loss"][-1]
        
        # Add other final metrics
        for key, values in self.training_history["metrics"].items():
            if values:
                final_metrics[f"final_{key}"] = values[-1]
        
        # Register model
        model_id = self.model_registry.register_model(
            model=self.model,
            model_name=model_name,
            tags=["trained", f"epochs_{self.current_epoch + 1}"],
            description=f"Model trained for {self.current_epoch + 1} epochs",
            metrics=final_metrics,
            dataset_info=training_config.get("data_info", {}),
            additional_metadata={
                "model_config": model_config,
                "training_config": training_config,
                "training_history": self.training_history
            }
        )
        
        self.logger.info(f"Saved final model with ID: {model_id}")
        return model_id
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[ModelBase, int]:
        """Load model and training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple of (model, start_epoch)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "r") as f:
            training_state = json.load(f)
        
        # Load model configuration
        config_path = checkpoint_path / "config.json"
        with open(config_path, "r") as f:
            model_config = json.load(f)
        
        # Create and load model
        model = self.model_factory.create_from_config(model_config)
        model.load(checkpoint_path)
        
        # Get start epoch (next epoch after checkpoint)
        start_epoch = training_state["epoch"] + 1
        
        # Restore training history if available
        if "training_history" in training_state:
            self.training_history = training_state["training_history"]
        
        return model, start_epoch
    
    def save_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Manually save a checkpoint of the current training state.
        
        Args:
            checkpoint_name: Name for the checkpoint (auto-generated if not provided)
            additional_metadata: Additional metadata to save
            
        Returns:
            Path to the saved checkpoint
        """
        if not self.model:
            raise ValueError("No model to checkpoint")
        
        # Generate checkpoint name
        if not checkpoint_name:
            checkpoint_name = f"manual_checkpoint_epoch_{self.current_epoch}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "training_history": self.training_history,
            "checkpoint_time": datetime.now().isoformat(),
            "is_training": self.is_training
        }
        
        if additional_metadata:
            training_state.update(additional_metadata)
        
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def stop_training(self) -> None:
        """Stop the current training process gracefully."""
        self._stop_training = True
        if hasattr(self.model, '_stop_training'):
            self.model._stop_training = True
        self.logger.info("Training stop requested")