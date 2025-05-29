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
        self.model_type = self.config.get("model_type", "default")
        self.model_config = self.config.get("model_config", {})
        self.training_config = self.config.get("training_config", {})
        self.validation_split = self.config.get("validation_split", 0.2)
        self.early_stopping = self.config.get("early_stopping", {})
        self.checkpoint_config = self.config.get("checkpoint_config", {})
        self.model = None
        self.training_history = None
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the training stage."""
        self.logger.info(f"Setting up {self.name} stage")
        
        # Validate required data is available
        processed_features = context.get("processed_features")
        # Check if it's a DataFrame and if it's not empty, or if it's not a DataFrame but still considered "present" (e.g. non-DataFrame data)
        # For this stage, we expect a DataFrame.
        if not isinstance(processed_features, pd.DataFrame) or processed_features.empty:
            raise ValueError("No processed features (non-empty DataFrame) found in context. Run feature engineering stage first.")
            
        # Get artifact store from context
        self.artifact_store = context.get("artifact_store")
        
        # Initialize model based on type
        self._initialize_model()
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the training stage.
        
        Args:
            context: Pipeline context containing processed features
            
        Returns:
            Updated pipeline context with trained model
        """
        self.logger.info(f"Running {self.name} stage")
        
        try:
            # Get training data from context
            features = context.get("processed_features")
            labels = context.get("labels")
            
            if labels is None:
                raise ValueError("No labels found in context")
            
            # Split data for training and validation
            train_data, val_data = self._split_data(features, labels)
            
            # Log training configuration
            self.logger.info(f"Training {self.model_type} model with config: {self.model_config}")
            self.logger.info(f"Training samples: {len(train_data[0])}, Validation samples: {len(val_data[0])}")
            
            # Train the model
            training_start = datetime.now()
            self.training_history = self._train_model(train_data, val_data)
            training_duration = (datetime.now() - training_start).total_seconds()
            
            # Store model and training results in context
            context.set("trained_model", self.model)
            context.set("training_history", self.training_history)
            context.set("model_type", self.model_type)
            context.set("model_config", self.model_config)
            
            # Store training metadata
            metadata = {
                "model_type": self.model_type,
                "training_duration_seconds": training_duration,
                "training_samples": len(train_data[0]),
                "validation_samples": len(val_data[0]),
                "final_metrics": self._get_final_metrics(),
                "hyperparameters": self.training_config
            }
            context.set("training_metadata", metadata)
            
            # Save model artifact if artifact store is available
            if self.artifact_store:
                self._save_model_artifact(context)
            
            self.logger.info(f"Training completed successfully. Duration: {training_duration:.2f}s")
            self.logger.info(f"Final metrics: {metadata['final_metrics']}")
            
            print(f"DEBUG_TRAIN_STAGE: Returning context from TrainingStage.run. Processed features type: {type(context.get('processed_features'))}")
            return context
            
        except Exception as e:
            self.logger.error(f"Error in training stage: {str(e)}")
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after training."""
        self.logger.info(f"Tearing down {self.name} stage")
        
        # Clean up any temporary files or resources
        if hasattr(self, 'temp_checkpoint_dir') and self.temp_checkpoint_dir.exists():
            import shutil
            shutil.rmtree(self.temp_checkpoint_dir)
            
    def _initialize_model(self) -> None:
        """Initialize the model based on configuration."""
        self.logger.info(f"Initializing {self.model_type} model")
        
        # This is a placeholder implementation
        # In a real implementation, this would initialize actual ML models
        # based on the model_type (e.g., sklearn, tensorflow, pytorch models)
        
        if self.model_type == "random_forest":
            # Placeholder for RandomForest initialization
            self.model = {"type": "random_forest", "params": self.model_config}
        elif self.model_type == "neural_network":
            # Placeholder for Neural Network initialization
            self.model = {"type": "neural_network", "params": self.model_config}
        elif self.model_type == "gradient_boosting":
            # Placeholder for Gradient Boosting initialization
            self.model = {"type": "gradient_boosting", "params": self.model_config}
        else:
            # Default model
            self.model = {"type": "default", "params": self.model_config}
            
        self.logger.info(f"Model initialized: {self.model}")
        
    def _split_data(self, features: Any, labels: Any) -> tuple:
        """
        Split data into training and validation sets.
        
        Args:
            features: Feature data
            labels: Label data
            
        Returns:
            Tuple of (train_data, val_data) where each is (features, labels)
        """
        # This is a simplified implementation
        # In practice, you'd use sklearn.model_selection.train_test_split or similar
        
        total_samples = len(features) if hasattr(features, '__len__') else 1000
        split_idx = int(total_samples * (1 - self.validation_split))
        
        # Placeholder split - in reality, this would properly split the data
        train_features = features  # Would be features[:split_idx]
        train_labels = labels      # Would be labels[:split_idx]
        val_features = features    # Would be features[split_idx:]
        val_labels = labels        # Would be labels[split_idx:]
        
        return (train_features, train_labels), (val_features, val_labels)
        
    def _train_model(self, train_data: tuple, val_data: tuple) -> Dict[str, List[float]]:
        """
        Train the model and return training history.
        
        Args:
            train_data: Tuple of (features, labels) for training
            val_data: Tuple of (features, labels) for validation
            
        Returns:
            Dictionary containing training history metrics
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
        
        # Get training parameters
        epochs = self.training_config.get("epochs", 10)
        batch_size = self.training_config.get("batch_size", 32)
        learning_rate = self.training_config.get("learning_rate", 0.001)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Simulate training loop
        for epoch in range(epochs):
            # In a real implementation, this would perform actual training
            # For now, we'll simulate metrics
            train_loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
            val_loss = 1.1 / (epoch + 1)
            train_acc = min(0.5 + epoch * 0.05, 0.95)  # Simulated increasing accuracy
            val_acc = min(0.45 + epoch * 0.05, 0.90)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            
            # Log progress
            if epoch % max(1, epochs // 10) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
                )
                
            # Check early stopping
            if self._should_stop_early(history):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
                
        return history
        
    def _should_stop_early(self, history: Dict[str, List[float]]) -> bool:
        """Check if early stopping criteria are met."""
        if not self.early_stopping.get("enabled", False):
            return False
            
        patience = self.early_stopping.get("patience", 5)
        min_delta = self.early_stopping.get("min_delta", 0.001)
        monitor = self.early_stopping.get("monitor", "val_loss")
        
        if monitor not in history or len(history[monitor]) < patience:
            return False
            
        # Check if metric hasn't improved for 'patience' epochs
        recent_values = history[monitor][-patience:]
        best_recent = min(recent_values) if "loss" in monitor else max(recent_values)
        
        for i in range(1, patience):
            current = recent_values[-i]
            if "loss" in monitor:
                if best_recent < current - min_delta:
                    return False
            else:
                if best_recent > current + min_delta:
                    return False
                    
        return True
        
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        if not self.training_history:
            return {}
            
        final_metrics = {}
        for metric, values in self.training_history.items():
            if values:
                final_metrics[f"final_{metric}"] = values[-1]
                
        return final_metrics
        
    def _save_model_artifact(self, context: PipelineContext) -> None:
        """Save the trained model as an artifact."""
        try:
            # Create temporary file for model
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
                temp_path = tmp.name
                
            # Serialize model (in practice, use appropriate serialization for model type)
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    "model": self.model,
                    "model_type": self.model_type,
                    "model_config": self.model_config,
                    "training_config": self.training_config,
                    "training_history": self.training_history
                }, f)
                
            # Save to artifact store
            run_id = context.get_metadata("run_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_metadata = self.artifact_store.save_artifact(
                artifact_id=f"model_{self.model_type}_{run_id}_{timestamp}",
                artifact_path=temp_path,
                artifact_type=ArtifactType.MODEL,
                description=f"Trained {self.model_type} model",
                tags=["trained", self.model_type, self.name]
            )
            
            # Store artifact reference in context
            context.set("model_artifact", artifact_metadata.artifact_id)
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            self.logger.info(f"Saved model artifact: {artifact_metadata.artifact_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save model artifact: {str(e)}")