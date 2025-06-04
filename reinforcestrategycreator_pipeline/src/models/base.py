"""Base interface for all models in the pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import pickle
from datetime import datetime


class ModelBase(ABC):
    """Abstract base class for all models.
    
    All models in the pipeline must inherit from this class and implement
    the required methods.
    """
    MODEL_FILENAME = "model.pkl"
    CONFIG_FILENAME = "config.json"
    METADATA_FILENAME = "metadata.json"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_type = config.get("model_type", self.__class__.__name__)
        self.hyperparameters = config.get("hyperparameters", {})
        self.is_trained = False
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters
        }
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        """Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
        """
        pass
    
    @abstractmethod
    def train(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """Make predictions using the model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction arguments
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save(self, path: Union[str, Path]) -> str:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            String path to the saved checkpoint directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights/parameters
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.get_model_state(), f)
        
        # Save configuration
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save metadata
        self.metadata["saved_at"] = datetime.now().isoformat()
        self.metadata["is_trained"] = self.is_trained
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Return the checkpoint directory path as a string
        return str(path)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        
        # Load configuration
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
            self.hyperparameters = self.config.get("hyperparameters", {})
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.is_trained = self.metadata.get("is_trained", False)
        
        # Load model weights/parameters
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model_state = pickle.load(f)
            self.set_model_state(model_state)
    
    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current state of the model for serialization.
        
        Returns:
            Dictionary containing model state
        """
        pass
    
    @abstractmethod
    def set_model_state(self, state: Dict[str, Any]) -> None:
        """Set the model state from a dictionary.
        
        Args:
            state: Dictionary containing model state
        """
        pass
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.hyperparameters.copy()
    
    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Update model hyperparameters.
        
        Args:
            hyperparameters: New hyperparameters
        """
        self.hyperparameters.update(hyperparameters)
        self.config["hyperparameters"] = self.hyperparameters
        self.metadata["hyperparameters"] = self.hyperparameters
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata.copy()
    
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update model metadata.
        
        Args:
            metadata: Additional metadata to add/update
        """
        self.metadata.update(metadata)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_type={self.model_type}, is_trained={self.is_trained})"