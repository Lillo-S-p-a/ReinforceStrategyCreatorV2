"""Training callbacks for the training engine."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging


class CallbackBase(ABC):
    """Abstract base class for training callbacks.
    
    Callbacks allow custom actions at various points in the training loop.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the callback.
        
        Args:
            name: Optional name for the callback
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"callback.{self.name}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training.
        
        Args:
            logs: Dictionary of logs/metrics
        """
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training.
        
        Args:
            logs: Dictionary of logs/metrics
        """
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of an epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary of logs/metrics
        """
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of a batch.
        
        Args:
            batch: Current batch number
            logs: Dictionary of logs/metrics
        """
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a batch.
        
        Args:
            batch: Current batch number
            logs: Dictionary of logs/metrics
        """
        pass


class LoggingCallback(CallbackBase):
    """Callback for logging training progress."""
    
    def __init__(
        self,
        log_frequency: str = "epoch",
        log_file: Optional[Union[str, Path]] = None,
        verbose: int = 1
    ):
        """Initialize the logging callback.
        
        Args:
            log_frequency: When to log ("epoch" or "batch")
            log_file: Optional file to write logs to
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        super().__init__("LoggingCallback")
        self.log_frequency = log_frequency
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose
        self.start_time = None
        self.epoch_start_time = None
        
        # Set up file logging if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training start."""
        self.start_time = datetime.now()
        if self.verbose > 0:
            self.logger.info(f"Training started at {self.start_time}")
            if logs:
                self.logger.info(f"Initial logs: {logs}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training end."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            if self.verbose > 0:
                self.logger.info(f"Training completed in {duration}")
                if logs:
                    self.logger.info(f"Final logs: {logs}")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        if self.verbose > 1:
            self.logger.info(f"Epoch {epoch} started")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch end."""
        if self.log_frequency == "epoch" and self.verbose > 0:
            epoch_duration = datetime.now() - self.epoch_start_time if self.epoch_start_time else None
            
            log_msg = f"Epoch {epoch} completed"
            if epoch_duration:
                log_msg += f" in {epoch_duration.total_seconds():.2f}s"
            
            if logs:
                metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                       for k, v in logs.items()])
                log_msg += f" - {metrics_str}"
            
            self.logger.info(log_msg)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log batch end."""
        if self.log_frequency == "batch" and self.verbose > 1:
            log_msg = f"Batch {batch} completed"
            if logs:
                metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                       for k, v in logs.items()])
                log_msg += f" - {metrics_str}"
            self.logger.info(log_msg)


class ModelCheckpointCallback(CallbackBase):
    """Callback for saving model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_registry: Optional[Any] = None,
        artifact_store: Optional[Any] = None,
        save_frequency: str = "epoch",
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
        verbose: int = 1
    ):
        """Initialize the checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_registry: Optional ModelRegistry instance for versioning
            artifact_store: Optional ArtifactStore instance
            save_frequency: When to save ("epoch" or "best")
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for best model
            mode: "min" or "max" for the monitored metric
            verbose: Verbosity level
        """
        super().__init__("ModelCheckpointCallback")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_registry = model_registry
        self.artifact_store = artifact_store
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.model = None
        self.training_config = None
    
    def set_model(self, model: Any) -> None:
        """Set the model reference.
        
        Args:
            model: Model instance
        """
        self.model = model
    
    def set_training_config(self, config: Dict[str, Any]) -> None:
        """Set the training configuration.
        
        Args:
            config: Training configuration
        """
        self.training_config = config
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint at epoch end if configured."""
        if not self.model:
            self.logger.warning("No model set for checkpointing")
            return
        
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        # Determine if we should save
        should_save = False
        
        if self.save_best_only and current_value is not None:
            if self.mode == "min" and current_value < self.best_value:
                self.best_value = current_value
                should_save = True
            elif self.mode == "max" and current_value > self.best_value:
                self.best_value = current_value
                should_save = True
        elif self.save_frequency == "epoch":
            should_save = True
        
        if should_save:
            self._save_checkpoint(epoch, logs)
    
    def _save_checkpoint(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch
            logs: Current metrics
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model
        save_info = self.model.save(checkpoint_path)
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "metrics": logs,
            "training_config": self.training_config,
            "checkpoint_time": datetime.now().isoformat(),
            "best_value": self.best_value,
            "monitor": self.monitor
        }
        
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        if self.verbose > 0:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Register with model registry if available
        if self.model_registry and self.artifact_store:
            try:
                model_id = self.model_registry.register_model(
                    model=self.model,
                    model_name=f"{self.model.model_type}_training",
                    version=f"epoch_{epoch}",
                    tags=["checkpoint", f"epoch_{epoch}"],
                    description=f"Training checkpoint at epoch {epoch}",
                    metrics=logs,
                    additional_metadata={"training_state": training_state}
                )
                if self.verbose > 0:
                    self.logger.info(f"Registered checkpoint with model registry: {model_id}")
            except Exception as e:
                self.logger.error(f"Failed to register checkpoint: {e}")


class EarlyStoppingCallback(CallbackBase):
    """Callback for early stopping based on metric monitoring."""
    
    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0001,
        restore_best_weights: bool = True,
        verbose: int = 1
    ):
        """Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: "min" or "max" for the monitored metric
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            verbose: Verbosity level
        """
        super().__init__("EarlyStoppingCallback")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.model = None
    
    def set_model(self, model: Any) -> None:
        """Set the model reference.
        
        Args:
            model: Model instance
        """
        self.model = model
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for improvement and potentially stop training."""
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            self.logger.warning(f"Early stopping: monitored metric '{self.monitor}' not found in logs")
            return
        
        # Check for improvement
        improved = False
        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                improved = True
        else:
            if current_value > self.best_value + self.min_delta:
                improved = True
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights if configured
            if self.restore_best_weights and self.model:
                self.best_weights = self.model.get_model_state()
            
            if self.verbose > 1:
                self.logger.info(f"Improvement detected: {self.monitor}={current_value:.4f}")
        else:
            self.wait += 1
            if self.verbose > 1:
                self.logger.info(f"No improvement for {self.wait} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                
                # Restore best weights if configured
                if self.restore_best_weights and self.best_weights and self.model:
                    self.model.set_model_state(self.best_weights)
                    if self.verbose > 0:
                        self.logger.info(f"Restoring model weights from epoch {self.best_epoch}")
                
                # Signal to stop training
                if hasattr(self.model, '_stop_training'):
                    self.model._stop_training = True
                
                if self.verbose > 0:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log early stopping info."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
            self.logger.info(f"Best epoch was {self.best_epoch} with {self.monitor}={self.best_value:.4f}")


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[CallbackBase]] = None):
        """Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: CallbackBase) -> None:
        """Add a callback to the list.
        
        Args:
            callback: Callback instance to add
        """
        self.callbacks.append(callback)
    
    def set_model(self, model: Any) -> None:
        """Set model reference for all callbacks that need it.
        
        Args:
            model: Model instance
        """
        for callback in self.callbacks:
            if hasattr(callback, 'set_model'):
                callback.set_model(model)
    
    def set_training_config(self, config: Dict[str, Any]) -> None:
        """Set training config for all callbacks that need it.
        
        Args:
            config: Training configuration
        """
        for callback in self.callbacks:
            if hasattr(callback, 'set_training_config'):
                callback.set_training_config(config)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)