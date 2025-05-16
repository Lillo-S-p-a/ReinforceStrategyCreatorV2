# Model Training Module: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Model Training Module of the Trading Model Optimization Pipeline. This module is responsible for defining, training, validating, and saving machine learning models that can be used for trading strategy optimization.

## 2. Component Responsibilities

The Model Training Module is responsible for:

- Defining model architectures for trading strategy optimization
- Training models using data provided by the Data Management Module
- Implementing various training algorithms (supervised learning, reinforcement learning)
- Tracking training metrics and model performance
- Saving trained models and checkpoints
- Facilitating model versioning and reproducibility
- Providing abstractions for different model types and training paradigms
- Supporting distributed training via Ray

## 3. Architecture

### 3.1 Overall Architecture

The Model Training Module follows a layered architecture with clear separation of concerns:

```
┌───────────────────────────────────┐
│          Model Manager            │  High-level API for model operations
├───────────────────────────────────┤
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │    Models   │ │  Trainers   │  │  Core components with implementations
│  └─────────────┘ └─────────────┘  │
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │ Optimizers  │ │ Callbacks   │  │
│  └─────────────┘ └─────────────┘  │
│                                   │
├───────────────────────────────────┤
│          Ray Integration          │  Support for distributed execution
└───────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── models/
    ├── __init__.py
    ├── manager.py           # High-level model management interface
    ├── base.py              # Abstract base model class
    ├── architectures/
    │   ├── __init__.py
    │   ├── mlp.py           # Multi-layer Perceptron models
    │   ├── lstm.py          # LSTM models for time series
    │   ├── transformer.py   # Transformer-based models
    │   ├── cnn.py           # CNN models
    │   └── factory.py       # Model factory pattern implementation
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py       # Base trainer class
    │   ├── supervised.py    # Supervised learning trainer
    │   ├── reinforcement.py # Reinforcement learning trainer
    │   └── factory.py       # Trainer factory pattern implementation
    ├── optimization/
    │   ├── __init__.py
    │   ├── optimizers.py    # Custom optimizers
    │   └── loss.py          # Custom loss functions
    ├── callbacks/
    │   ├── __init__.py
    │   ├── base.py          # Base callback interface
    │   ├── logging.py       # Logging callbacks
    │   ├── checkpointing.py # Model checkpointing
    │   ├── early_stopping.py # Early stopping implementation
    │   └── metrics.py       # Metrics tracking callbacks
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py       # Evaluation metrics
    │   └── visualizer.py    # Visualization utilities
    ├── serialization/
    │   ├── __init__.py
    │   ├── save.py          # Model saving utilities
    │   └── load.py          # Model loading utilities
    └── utils/
        ├── __init__.py
        └── wrappers.py      # Utility wrappers and helpers
```

## 4. Core Components Design

### 4.1 Model Manager

The high-level interface for model training and management:

```python
# manager.py
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import uuid
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from trading_optimization.models.architectures.factory import ModelFactory
from trading_optimization.models.training.factory import TrainerFactory
from trading_optimization.models.serialization.save import save_model
from trading_optimization.models.serialization.load import load_model
from trading_optimization.config import ConfigManager
from trading_optimization.db.repository import ModelRepository

class ModelManager:
    """
    High-level interface for model management.
    Acts as a facade for all model operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Model Manager with configuration settings.
        
        Args:
            config: Configuration dictionary with model management settings
        """
        self.config = config
        self.model_factory = ModelFactory()
        self.trainer_factory = TrainerFactory()
        
        # Create database connection for model tracking
        try:
            from trading_optimization.db.connectors import DatabaseConnector
            db_connector = DatabaseConnector.instance()
            with db_connector.session() as session:
                self.model_repo = ModelRepository(session)
        except Exception as e:
            print(f"Warning: Could not connect to database: {str(e)}")
            self.model_repo = None
    
    def create_model(
        self,
        model_type: str,
        model_name: str = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create (e.g., 'mlp', 'lstm')
            model_name: Optional name for the model
            **kwargs: Additional parameters for model creation
            
        Returns:
            Tuple of (model_id, model_instance)
        """
        model_instance = self.model_factory.create_model(model_type, **kwargs)
        
        # Generate model ID and metadata
        model_id = str(uuid.uuid4())
        model_name = model_name or f"{model_type}_{model_id[:8]}"
        
        # Store model metadata
        model_meta = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'created_at': datetime.now().isoformat(),
            'parameters': kwargs
        }
        
        # Persist model metadata if database is available
        if self.model_repo:
            try:
                self.model_repo.create(
                    id=model_id,
                    name=model_name,
                    model_type=model_type,
                    metadata=model_meta
                )
            except Exception as e:
                print(f"Warning: Failed to store model metadata: {str(e)}")
        
        return model_id, model_instance
    
    def train_model(
        self,
        model_instance: Any,
        model_id: str,
        data_loaders: Dict[str, Any],
        trainer_type: str = 'supervised',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model using the specified trainer.
        
        Args:
            model_instance: Model to train
            model_id: ID of the model
            data_loaders: Dictionary with 'train', 'val', 'test' data loaders
            trainer_type: Type of trainer to use
            **kwargs: Additional parameters for trainer
            
        Returns:
            Dictionary with training results and metrics
        """
        trainer = self.trainer_factory.create_trainer(
            trainer_type=trainer_type,
            **kwargs
        )
        
        # Train the model
        training_results = trainer.train(
            model=model_instance,
            train_loader=data_loaders['train'],
            val_loader=data_loaders.get('val'),
            **kwargs
        )
        
        # Update model metadata with training information
        if self.model_repo:
            try:
                training_meta = {
                    'completed_at': datetime.now().isoformat(),
                    'epochs': trainer.current_epoch,
                    'training_metrics': training_results
                }
                
                self.model_repo.update(
                    model_id,
                    metadata=training_meta
                )
            except Exception as e:
                print(f"Warning: Failed to update model training info: {str(e)}")
        
        return training_results
    
    def evaluate_model(
        self,
        model_instance: Any,
        data_loader: Any,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_instance: Trained model to evaluate
            data_loader: Data loader with evaluation data
            metrics: List of metric names to compute
            
        Returns:
            Dictionary with evaluation metrics
        """
        from trading_optimization.models.evaluation.metrics import compute_metrics
        
        # Set model to evaluation mode
        if hasattr(model_instance, 'eval'):
            model_instance.eval()
        
        # Default metrics if none specified
        metrics = metrics or ['rmse', 'mae', 'r2']
        
        # Collect predictions and targets
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, target = batch[0], batch[1]
                else:
                    continue
                
                # Move to device if model is on GPU
                if hasattr(model_instance, 'device'):
                    inputs = inputs.to(model_instance.device)
                
                # Get predictions
                outputs = model_instance(inputs)
                
                # Move back to CPU for evaluation
                if torch.is_tensor(outputs):
                    outputs = outputs.cpu().numpy()
                if torch.is_tensor(target):
                    target = target.cpu().numpy()
                
                predictions.append(outputs)
                targets.append(target)
        
        # Concatenate batches
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Compute metrics
        eval_results = compute_metrics(predictions, targets, metrics)
        
        return eval_results
    
    def save_model(
        self,
        model_instance: Any,
        model_id: str,
        version: str = None,
        include_optimizer: bool = True
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_instance: Model to save
            model_id: ID of the model
            version: Optional version tag
            include_optimizer: Whether to save optimizer state
            
        Returns:
            Path to the saved model
        """
        # Generate version if not provided
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine save path from config
        base_path = self.config.get('models', {}).get(
            'save_path',
            os.path.join('artifacts', 'models')
        )
        
        # Create model directory
        model_dir = os.path.join(base_path, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        save_path = save_model(
            model=model_instance,
            save_dir=model_dir,
            version=version,
            include_optimizer=include_optimizer
        )
        
        # Update metadata in database
        if self.model_repo:
            try:
                # Create artifact record
                from trading_optimization.db.utils import ArtifactStorage
                artifact_storage = ArtifactStorage()
                
                # Register the saved model in the database
                with open(save_path, 'rb') as f:
                    storage_path, content_hash, size_bytes = artifact_storage.save(
                        f, 'model_weights', str(model_id)
                    )
                
                # Create version record in database
                from trading_optimization.db.repository import ModelVersionRepository, ModelArtifactRepository
                with self.model_repo.session as session:
                    version_repo = ModelVersionRepository(session)
                    artifact_repo = ModelArtifactRepository(session)
                    
                    version_record = version_repo.create(
                        model_id=model_id,
                        version=version,
                        status='active',
                        metrics={}
                    )
                    
                    artifact_repo.create(
                        model_version_id=version_record.id,
                        artifact_type='weights',
                        storage_path=storage_path,
                        size_bytes=size_bytes,
                        file_format='pt',  # PyTorch format
                        content_hash=content_hash
                    )
            except Exception as e:
                print(f"Warning: Failed to register model artifact: {str(e)}")
        
        return save_path
    
    def load_model(
        self,
        model_id: str,
        version: str = None,
        model_type: str = None
    ) -> Any:
        """
        Load a saved model.
        
        Args:
            model_id: ID of the model to load
            version: Optional specific version to load (latest if None)
            model_type: Model type (required if loading without metadata)
            
        Returns:
            Loaded model instance
        """
        # Determine load path from config
        base_path = self.config.get('models', {}).get(
            'save_path', 
            os.path.join('artifacts', 'models')
        )
        
        # Get model directory
        model_dir = os.path.join(base_path, model_id)
        
        # If version not specified, find latest
        if not version:
            versions = [fname.split('_')[0] for fname in os.listdir(model_dir) 
                      if fname.endswith('.pt') and '_' in fname]
            if not versions:
                raise ValueError(f"No model versions found for ID {model_id}")
            
            # Sort by version timestamp
            versions.sort()
            version = versions[-1]
        
        # If model type not provided, try to get from database
        if not model_type and self.model_repo:
            try:
                model_info = self.model_repo.get_by_id(model_id)
                if model_info:
                    model_type = model_info.model_type
            except Exception:
                pass
        
        if not model_type:
            raise ValueError("Model type must be provided if metadata not available")
        
        # Load model
        model_instance = load_model(
            model_dir=model_dir,
            version=version,
            model_type=model_type,
            model_factory=self.model_factory
        )
        
        return model_instance
```

### 4.2 Base Model Interface

The abstract base model class that all models must implement:

```python
# base.py
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration parameters.
        
        Args:
            config: Model configuration parameters
        """
        super().__init__()
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor or batch
            
        Returns:
            Model output
        """
        pass
    
    def to_device(self, device=None):
        """
        Move model to specified device.
        
        Args:
            device: Device to move to (uses config default if None)
            
        Returns:
            Self for chaining
        """
        if device is None:
            device = self.device
        else:
            self.device = device
            
        return self.to(device)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer to include in checkpoint
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
        return {'path': path, 'includes_optimizer': optimizer is not None}
    
    @classmethod
    def load_checkpoint(cls, path: str, device=None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            device: Device to load the model to
            
        Returns:
            Dictionary with loaded model and optimizer states
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create a new model instance using saved config
        model_config = checkpoint.get('model_config', {})
        model = cls(model_config)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        result = {'model': model}
        
        # Return optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        
        return result
```

### 4.3 Model Implementations

Example implementation of a neural network model for time series prediction:

```python
# architectures/lstm.py
from typing import Dict, Any, List
import torch
import torch.nn as nn

from trading_optimization.models.base import BaseModel

class LSTMModel(BaseModel):
    """
    LSTM model for time series prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM model.
        
        Args:
            config: Model configuration with parameters:
                - input_size: Number of input features
                - hidden_size: Size of LSTM hidden state
                - num_layers: Number of LSTM layers
                - output_size: Number of output features
                - dropout: Dropout probability
                - sequence_length: Length of input sequences
                - bidirectional: Whether to use bidirectional LSTM
                - fc_layers: List of fully connected layer sizes after LSTM
        """
        super().__init__(config)
        
        # Extract parameters from config
        self.input_size = config['input_size']
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.2)
        self.sequence_length = config.get('sequence_length', 60)
        self.bidirectional = config.get('bidirectional', False)
        self.fc_layers = config.get('fc_layers', [64])
        
        # Calculate direction factor for bidirectional
        self.d_factor = 2 if self.bidirectional else 1
        
        # Define LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # Define fully connected layers
        fc_layers = []
        
        # Input to first FC layer is the LSTM output
        prev_size = self.hidden_size * self.d_factor
        
        # Add FC layers based on config
        for fc_size in self.fc_layers:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_size = fc_size
        
        # Final output layer
        fc_layers.append(nn.Linear(prev_size, self.output_size))
        
        # Create sequential model of FC layers
        self.fc = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to_device()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Check input shape
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output from each sequence
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc(lstm_out)
        
        return output
```

Model factory pattern implementation:

```python
# architectures/factory.py
from typing import Dict, Any

from trading_optimization.models.architectures.mlp import MLPModel
from trading_optimization.models.architectures.lstm import LSTMModel
from trading_optimization.models.architectures.transformer import TransformerModel
from trading_optimization.models.architectures.cnn import CNNModel

class ModelFactory:
    """
    Factory for creating model instances.
    """
    
    _models = {
        'mlp': MLPModel,
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'cnn': CNNModel
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_class):
        """
        Register a new model type.
        
        Args:
            model_type: Type identifier for the model
            model_class: Class that implements the model
        """
        cls._models[model_type] = model_class
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Parameters for the model constructor
            
        Returns:
            Instance of a model
        """
        if model_type not in self._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create config dictionary from kwargs
        config = kwargs
        
        # Create and return the model instance
        return self._models[model_type](config)
```

### 4.4 Training Components

Base trainer class:

```python
# training/trainer.py
from typing import Dict, List, Any, Optional, Callable, Union
import os
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trading_optimization.models.callbacks.base import Callback

class Trainer(ABC):
    """
    Abstract base class for model trainers.
    """
    
    def __init__(
        self,
        epochs: int = 100,
        optimizer_type: str = 'adam',
        optimizer_params: Dict[str, Any] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Dict[str, Any] = None,
        callbacks: List[Callback] = None,
        device: str = None,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            epochs: Maximum number of epochs to train for
            optimizer_type: Type of optimizer to use
            optimizer_params: Parameters for the optimizer
            lr_scheduler: Type of learning rate scheduler
            lr_scheduler_params: Parameters for the scheduler
            callbacks: List of callbacks to use during training
            device: Device to train on (defaults to CUDA if available)
            **kwargs: Additional trainer parameters
        """
        self.max_epochs = epochs
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        self.callbacks = callbacks or []
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.current_epoch = 0
        self.kwargs = kwargs
        self.history = {'train': [], 'val': []}
        
        # Early stopping flag
        self.stop_training = False
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimizer instance
        """
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamw': optim.AdamW
        }
        
        if self.optimizer_type not in optimizer_map:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Get optimizer class
        optimizer_class = optimizer_map[self.optimizer_type]
        
        # Create optimizer instance
        return optimizer_class(model.parameters(), **self.optimizer_params)
    
    def _create_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Scheduler instance or None
        """
        if not self.lr_scheduler:
            return None
        
        scheduler_map = {
            'step': optim.lr_scheduler.StepLR,
            'multistep': optim.lr_scheduler.MultiStepLR,
            'exponential': optim.lr_scheduler.ExponentialLR,
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'plateau': optim.lr_scheduler.ReduceLROnPlateau
        }
        
        if self.lr_scheduler not in scheduler_map:
            raise ValueError(f"Unknown scheduler type: {self.lr_scheduler}")
        
        # Get scheduler class
        scheduler_class = scheduler_map[self.lr_scheduler]
        
        # Create scheduler instance
        return scheduler_class(optimizer, **self.lr_scheduler_params)
    
    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            model: Model being trained
            batch: Batch of data
            optimizer: Optimizer instance
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def validation_step(self, model: nn.Module, batch: Any) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            model: Model being validated
            batch: Batch of data
            
        Returns:
            Dictionary with validation metrics
        """
        pass
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for a single epoch.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            optimizer: Optimizer instance
            epoch: Current epoch number
            
        Returns:
            Dictionary with averaged training metrics
        """
        model.train()
        total_metrics = {}
        num_batches = len(train_loader)
        
        # Notify callbacks of epoch start
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=None)
        
        # Iterate through batches
        for batch_idx, batch in enumerate(train_loader):
            # Notify callbacks of batch start
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, logs=None)
            
            # Perform training step
            step_metrics = self.train_step(model, batch, optimizer)
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            
            # Notify callbacks of batch end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, logs=step_metrics)
        
        # Calculate average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Add to history
        self.history['train'].append(avg_metrics)
        
        # Notify callbacks of epoch end
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=avg_metrics)
        
        return avg_metrics
    
    def validate(self, model: nn.Module, val_loader: Optional[DataLoader], epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            model: Model to validate
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Dictionary with averaged validation metrics
        """
        if val_loader is None:
            return {}
        
        model.eval()
        total_metrics = {}
        num_batches = len(val_loader)
        
        # Iterate through batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Perform validation step
                step_metrics = self.validation_step(model, batch)
                
                # Accumulate metrics
                for k, v in step_metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0.0
                    total_metrics[k] += v
        
        # Calculate average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Add to history
        self.history['val'].append(avg_metrics)
        
        return avg_metrics
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            checkpoint_dir: Optional directory for saving checkpoints
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history and results
        """
        # Set model to device
        if hasattr(model, 'to_device'):
            model.to_device(self.device)
        else:
            model.to(self.device)
        
        # Create optimizer
        optimizer = self._create_optimizer(model)
        
        # Create LR scheduler
        scheduler = self._create_lr_scheduler(optimizer)
        
        # Create checkpoint directory if needed
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Notify callbacks of training start
        for callback in self.callbacks:
            callback.on_train_begin(logs={'model': model, 'optimizer': optimizer})
        
        # Main training loop
        start_time = time.time()
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validate if validation data is provided
            val_metrics = self.validate(model, val_loader, epoch)
            
            # Update learning rate if scheduler exists
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_metrics:
                    # ReduceLROnPlateau needs a metric value
                    scheduler.step(val_metrics.get('loss', train_metrics.get('loss', 0)))
                else:
                    scheduler.step()
            
            # Print progress
            log_msg = f"Epoch {epoch+1}/{self.max_epochs} - "
            log_msg += " - ".join([f"train_{k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                log_msg += " - " + " - ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(log_msg)
            
            # Save checkpoint if directory provided
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                if hasattr(model, 'save_checkpoint'):
                    model.save_checkpoint(checkpoint_path, optimizer)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }, checkpoint_path)
            
            # Check for early stopping
            if self.stop_training:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Notify callbacks of training end
        end_logs = {
            'training_time': training_time,
            'epochs_completed': self.current_epoch + 1,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }
        for callback in self.callbacks:
            callback.on_train_end(logs=end_logs)
        
        # Return training results
        results = {
            'history': self.history,
            'training_time': training_time,
            'epochs_completed': self.current_epoch + 1,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }
        
        return results
```

Supervised learning trainer implementation:

```python
# training/supervised.py
from typing import Dict, Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trading_optimization.models.training.trainer import Trainer
from trading_optimization.models.optimization.loss import get_loss_function

class SupervisedTrainer(Trainer):
    """
    Trainer for supervised learning models.
    """
    
    def __init__(
        self,
        loss_fn: Union[str, nn.Module] = 'mse',
        loss_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize supervised trainer.
        
        Args:
            loss_fn: Loss function name or instance
            loss_params: Parameters for the loss function
            **kwargs: Additional trainer parameters
        """
        super().__init__(**kwargs)
        
        # Initialize loss function
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_function(loss_fn, **(loss_params or {}))
        else:
            self.loss_fn = loss_fn
    
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform a single supervised training step.
        
        Args:
            model: Model being trained
            batch: Batch of data (X, y)
            optimizer: Optimizer instance
            
        Returns:
            Dictionary with training metrics
        """
        # Reset gradients
        optimizer.zero_grad()
        
        # Unpack batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X, y = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to contain inputs and targets")
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate additional metrics if needed
        metrics = {'loss': loss.item()}
        
        return metrics
    
    def validation_step(
        self,
        model: nn.Module,
        batch: Any
    ) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            model: Model being validated
            batch: Batch of data (X, y)
            
        Returns:
            Dictionary with validation metrics
        """
        # Unpack batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X, y = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to contain inputs and targets")
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Calculate additional metrics
        metrics = {'loss': loss.item()}
        
        # Add more metrics if needed
        if y_pred.size() == y.size():
            # MSE
            mse = F.mse_loss(y_pred, y).item()
            metrics['mse'] = mse
            
            # MAE
            mae = F.l1_loss(y_pred, y).item()
            metrics['mae'] = mae
        
        return metrics
```

Trainer factory implementation:

```python
# training/factory.py
from typing import Dict, Any

from trading_optimization.models.training.trainer import Trainer
from trading_optimization.models.training.supervised import SupervisedTrainer
from trading_optimization.models.training.reinforcement import ReinforcementTrainer

class TrainerFactory:
    """
    Factory for creating trainer instances.
    """
    
    _trainers = {
        'supervised': SupervisedTrainer,
        'reinforcement': ReinforcementTrainer
    }
    
    @classmethod
    def register_trainer(cls, trainer_type: str, trainer_class):
        """
        Register a new trainer type.
        
        Args:
            trainer_type: Type identifier for the trainer
            trainer_class: Class that implements the trainer
        """
        cls._trainers[trainer_type] = trainer_class
    
    def create_trainer(self, trainer_type: str, **kwargs) -> Trainer:
        """
        Create a trainer instance.
        
        Args:
            trainer_type: Type of trainer to create
            **kwargs: Parameters for the trainer constructor
            
        Returns:
            Instance of a Trainer subclass
        """
        if trainer_type not in self._trainers:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        
        # Create and return the trainer instance
        return self._trainers[trainer_type](**kwargs)
```

### 4.5 Callbacks System

Base callback interface:

```python
# callbacks/base.py
from typing import Dict, Any, Optional

class Callback:
    """
    Base class for all callbacks.
    """
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a batch."""
        pass
```

Early stopping implementation:

```python
# callbacks/early_stopping.py
from typing import Dict, Any, Optional

import numpy as np
from trading_optimization.models.callbacks.base import Callback

class EarlyStopping(Callback):
    """
    Early stopping callback to halt training when a metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0,
        patience: int = 10,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement before stopping
            mode: 'min' or 'max' for whether lower or higher is better
            restore_best_weights: Whether to restore model to best weights
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Initialize tracking variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Set initial best value
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Reset internal variables at the start of training."""
        self.wait = 0
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Check for improvement at the end of each epoch."""
        logs = logs or {}
        
        # Extract the monitored metric
        current = logs.get(self.monitor)
        if current is None:
            # Skip if metric is not available
            return
        
        # Check for improvement
        improved = False
        if self.mode == 'min':
            # Lower is better (e.g., loss)
            improved = current < (self.best - self.min_delta)
        else:
            # Higher is better (e.g., accuracy)
            improved = current > (self.best + self.min_delta)
        
        if improved:
            # Update best value and reset waiting counter
            self.best = current
            self.wait = 0
            
            # Store weights if configured
            if self.restore_best_weights and hasattr(logs, 'model'):
                self.best_weights = logs['model'].state_dict().copy()
        else:
            # Increment waiting counter
            self.wait += 1
            if self.wait >= self.patience:
                # Patience exceeded, stop training
                self.stopped_epoch = epoch
                logs['model'].trainer.stop_training = True
                
                # Restore best weights if configured
                if self.restore_best_weights and self.best_weights is not None:
                    logs['model'].load_state_dict(self.best_weights)
                    print(f"Restoring model weights from epoch with best {self.monitor}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Print message if stopped early."""
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch+1}")
```

### 4.6 Model Serialization

Save and load utilities:

```python
# serialization/save.py
from typing import Any, Dict, Optional
import os
import json
import torch
import datetime

def save_model(
    model: Any,
    save_dir: str,
    version: str = None,
    include_optimizer: bool = True
) -> str:
    """
    Save a model to disk.
    
    Args:
        model: Model to save
        save_dir: Directory to save the model in
        version: Version tag (defaults to timestamp)
        include_optimizer: Whether to save optimizer state
        
    Returns:
        Path to the saved model
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate version if not provided
    if not version:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model metadata
    metadata = {
        'model_type': model.__class__.__name__,
        'version': version,
        'saved_at': datetime.datetime.now().isoformat(),
        'model_config': getattr(model, 'config', {})
    }
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"{version}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create checkpoint with model weights
    checkpoint = {'model_state_dict': model.state_dict()}
    
    # Add optimizer state if requested and available
    if include_optimizer:
        optimizer = getattr(model, 'optimizer', None)
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save checkpoint
    model_path = os.path.join(save_dir, f"{version}_model.pt")
    torch.save(checkpoint, model_path)
    
    return model_path
```

```python
# serialization/load.py
from typing import Any, Dict, Optional
import os
import json
import torch

def load_model(
    model_dir: str,
    version: str,
    model_type: str = None,
    model_factory = None,
    device = None
) -> Any:
    """
    Load a saved model.
    
    Args:
        model_dir: Directory where model is saved
        version: Version to load
        model_type: Type of model (required if no metadata or factory)
        model_factory: Optional factory for creating model instance
        device: Device to load model on
        
    Returns:
        Loaded model instance
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for metadata file
    metadata_path = os.path.join(model_dir, f"{version}_metadata.json")
    metadata = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Get model path
    model_path = os.path.join(model_dir, f"{version}_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    if model_factory and metadata:
        # Use factory with metadata config
        model_type = metadata.get('model_type', model_type)
        model_config = metadata.get('model_config', {})
        model = model_factory.create_model(model_type, **model_config)
    elif hasattr(model_type, 'load_checkpoint'):
        # Use class method if available
        return model_type.load_checkpoint(model_path, device)
    else:
        raise ValueError("Cannot create model instance: need model_factory or model_type with load_checkpoint")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    if hasattr(model, 'to_device'):
        model.to_device(device)
    else:
        model.to(device)
    
    return model
```

## 5. Ray Integration for Distributed Training

The Model Training Module integrates with Ray for distributed training:

```python
# training/ray_trainer.py
from typing import Dict, List, Any, Optional, Callable, Union
import os
import time

import torch
import torch.nn as nn

import ray
from ray import train
from ray.train import Trainer as RayTrainer
from ray.train.torch import TorchTrainer

from trading_optimization.models.training.trainer import Trainer

class RayDistributedTrainer(Trainer):
    """
    Trainer implementation using Ray for distributed training.
    """
    
    def __init__(
        self,
        num_workers: int = 2,
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize Ray distributed trainer.
        
        Args:
            num_workers: Number of workers for distributed training
            use_gpu: Whether to use GPUs for training
            **kwargs: Additional trainer parameters
        """
        super().__init__(**kwargs)
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()
    
    def _training_func(self, config: Dict[str, Any]):
        """
        Training function executed on each worker.
        
        Args:
            config: Configuration dictionary with model and data parameters
        """
        # Extract parameters from config
        model = config['model']
        train_loader = config['train_loader']
        val_loader = config.get('val_loader')
        epochs = config.get('epochs', self.max_epochs)
        
        # Set model to device
        device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        if hasattr(model, 'to_device'):
            model.to_device(device)
        else:
            model.to(device)
        
        # Create optimizer
        optimizer = self._create_optimizer(model)
        
        # Create LR scheduler
        scheduler = self._create_lr_scheduler(optimizer)
        
        # Main training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validate if validation data is provided
            val_metrics = self.validate(model, val_loader, epoch)
            
            # Update learning rate if scheduler exists
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_metrics:
                    scheduler.step(val_metrics.get('loss', train_metrics.get('loss', 0)))
                else:
                    scheduler.step()
            
            # Report metrics to Ray
            metrics = {'train_loss': train_metrics.get('loss', 0.0)}
            if val_metrics:
                metrics['val_loss'] = val_metrics.get('loss', 0.0)
            
            train.report(metrics)
            
            # Check for early stopping
            if self.stop_training:
                break
    
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model using Ray distributed training.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            checkpoint_dir: Optional directory for saving checkpoints
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history and results
        """
        # Create config for training function
        train_config = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'epochs': self.max_epochs
        }
        
        # Create Ray trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._training_func,
            train_loop_config=train_config,
            scaling_config={"num_workers": self.num_workers, "use_gpu": self.use_gpu},
            run_config=ray.train.RunConfig(
                storage_path=checkpoint_dir,
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute="val_loss",
                    checkpoint_score_order="min"
                )
            )
        )
        
        # Start distributed training
        start_time = time.time()
        result = trainer.fit()
        training_time = time.time() - start_time
        
        # Get best model
        if checkpoint_dir:
            best_checkpoint = result.best_checkpoint
            # Load best model
            with best_checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return training results
        results = {
            'training_time': training_time,
            'epochs_completed': self.max_epochs,
            'metrics': result.metrics
        }
        
        return results
```

## 6. Configuration

### 6.1 Model Configuration Schema

```yaml
# Example model configuration
models:
  defaults:
    save_path: "artifacts/models"
    device: "cuda"
  
  lstm:
    input_size: 10
    hidden_size: 128
    num_layers: 2
    output_size: 1
    dropout: 0.2
    bidirectional: false
    fc_layers: [64]

  mlp:
    input_size: 10
    hidden_layers: [128, 64, 32]
    output_size: 1
    activation: "relu"
    dropout: 0.2

  transformer:
    input_size: 10
    n_head: 8
    d_model: 128
    n_layers: 3
    dropout: 0.1
    output_size: 1
```

### 6.2 Training Configuration Schema

```yaml
# Example training configuration
training:
  defaults:
    epochs: 100
    batch_size: 32
    early_stopping: true
    patience: 10
    monitor: "val_loss"
    checkpoint_frequency: 10
  
  optimizers:
    adam:
      lr: 0.001
      weight_decay: 0.0001
      betas: [0.9, 0.999]
    
    sgd:
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0001
  
  schedulers:
    step:
      step_size: 30
      gamma: 0.1
    
    plateau:
      factor: 0.5
      patience: 5
      threshold: 0.0001
```

## 7. Usage Examples

### 7.1 Basic Model Creation and Training

```python
# Example usage of the Model Training Module

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager

# Load configuration
config = ConfigManager.instance()
model_config = config.get('models', {})
training_config = config.get('training', {})

# Create managers
model_manager = ModelManager(model_config)
data_manager = DataManager(config.get('data', {}))

# Prepare data
data_pipeline_id = data_manager.create_pipeline("btc_analysis", pipeline_config)
df = data_manager.execute_pipeline(data_pipeline_id)
data_splits = data_manager.split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
data_loaders = data_manager.get_data_loaders(data_splits, batch_size=32)

# Create model
lstm_config = {
    'input_size': len(df.columns) - 1,  # Exclude target column
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'sequence_length': 60
}
model_id, model = model_manager.create_model('lstm', model_name="btc_predictor", **lstm_config)

# Training configuration
train_config = {
    'epochs': 100,
    'optimizer_type': 'adam',
    'optimizer_params': {'lr': 0.001, 'weight_decay': 1e-5},
    'lr_scheduler': 'plateau',
    'lr_scheduler_params': {'patience': 10, 'factor': 0.5},
    'loss_fn': 'mse',
    'callbacks': [
        EarlyStopping(patience=15, restore_best_weights=True)
    ]
}

# Train model
results = model_manager.train_model(
    model, 
    model_id,
    data_loaders, 
    trainer_type='supervised',
    **train_config
)

# Evaluate model
eval_metrics = model_manager.evaluate_model(
    model,
    data_loaders['test'],
    metrics=['rmse', 'mae', 'r2']
)
print(f"Evaluation metrics: {eval_metrics}")

# Save model
save_path = model_manager.save_model(
    model,
    model_id,
    include_optimizer=True
)
print(f"Model saved to {save_path}")
```

### 7.2 Distributed Training with Ray

```python
# Example of distributed training with Ray

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.models.training.ray_trainer import RayDistributedTrainer

# Load configuration
config = ConfigManager.instance()

# Create managers
model_manager = ModelManager(config.get('models', {}))
data_manager = DataManager(config.get('data', {}))

# Prepare data
# ... (same as previous example)

# Create model
model_id, model = model_manager.create_model('transformer', **transformer_config)

# Create Ray distributed trainer
ray_trainer = RayDistributedTrainer(
    num_workers=4,
    use_gpu=True,
    epochs=50,
    optimizer_type='adam',
    optimizer_params={'lr': 0.001},
    loss_fn='mse'
)

# Train model using Ray
results = ray_trainer.train(
    model,
    data_loaders['train'],
    data_loaders['val'],
    checkpoint_dir='./ray_checkpoints'
)

# Save final model
save_path = model_manager.save_model(
    model,
    model_id,
    include_optimizer=True
)
```

### 7.3 Creating a Custom Model

```python
# Example of creating a custom model

import torch
import torch.nn as nn
from trading_optimization.models.base import BaseModel
from trading_optimization.models.architectures.factory import ModelFactory

class GRUAttentionModel(BaseModel):
    """
    Custom GRU model with attention mechanism.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Extract parameters from config
        self.input_size = config['input_size']
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.2)
        
        # Define GRU layer
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Define attention mechanism
        self.attention = nn.Linear(self.hidden_size, 1)
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to_device()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'attention' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # GRU output: all hidden states for each time step
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        
        # Compute attention weights
        attn_weights = self.attention(gru_out)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention to GRU outputs
        context = torch.bmm(gru_out.transpose(1, 2), attn_weights)  # (batch_size, hidden_size, 1)
        context = context.squeeze(2)  # (batch_size, hidden_size)
        
        # Final output
        output = self.fc(context)  # (batch_size, output_size)
        
        return output

# Register the custom model
ModelFactory.register_model('gru_attention', GRUAttentionModel)

# Now the custom model can be used like any other model
model_id, model = model_manager.create_model('gru_attention', **custom_config)
```

## 8. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. Configuration management system is implemented
3. Data management module is implemented
4. Database integration is available (for model versioning)
5. Required libraries are installed:
   - torch (PyTorch)
   - numpy
   - ray (for distributed training)

## 9. Implementation Sequence

1. Set up the directory structure
2. Implement base model class
3. Create model architectures (MLP, LSTM, etc.)
4. Develop the training framework
5. Implement the serialization utilities
6. Create the model manager interface
7. Integrate with database for model tracking
8. Add Ray support for distributed training
9. Implement callbacks system
10. Create evaluation metrics
11. Add comprehensive unit tests

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# Example unit tests for model architectures

import unittest
import torch
import numpy as np

from trading_optimization.models.architectures.lstm import LSTMModel
from trading_optimization.models.architectures.mlp import MLPModel

class TestModels(unittest.TestCase):
    
    def test_lstm_model(self):
        """Test LSTM model architecture."""
        # Create model config
        config = {
            'input_size': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.2,
            'bidirectional': False,
            'fc_layers': [32],
            'device': 'cpu'  # Use CPU for testing
        }
        
        # Create model
        model = LSTMModel(config)
        
        # Create sample input
        batch_size = 16
        seq_length = 60
        input_tensor = torch.randn(batch_size, seq_length, config['input_size'])
        
        # Perform forward pass
        output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, config['output_size']))
        
        # Check parameter count
        param_count = model.get_parameter_count()
        self.assertGreater(param_count['total'], 0)
        self.assertEqual(param_count['total'], param_count['trainable'])
    
    def test_mlp_model(self):
        """Test MLP model architecture."""
        # Create model config
        config = {
            'input_size': 15,
            'hidden_layers': [64, 32],
            'output_size': 2,
            'dropout': 0.2,
            'activation': 'relu',
            'device': 'cpu'  # Use CPU for testing
        }
        
        # Create model
        model = MLPModel(config)
        
        # Create sample input
        batch_size = 16
        input_tensor = torch.randn(batch_size, config['input_size'])
        
        # Perform forward pass
        output = model(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, config['output_size']))
```

### 10.2 Integration Tests

```python
# Example integration tests for model training

import unittest
import torch
import torch.nn as nn
import tempfile
import os
import shutil

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.models.architectures.mlp import MLPModel
from trading_optimization.models.training.supervised import SupervisedTrainer
from trading_optimization.models.callbacks.early_stopping import EarlyStopping

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal configuration
        self.config = {
            'models': {
                'save_path': self.temp_dir
            }
        }
        
        # Create a simple dataset
        x = torch.linspace(-5, 5, 1000).view(-1, 1)
        y = x.pow(2) + 0.1 * torch.randn_like(x)
        
        # Split into train/val sets
        train_size = int(0.8 * len(x))
        self.x_train, self.x_val = x[:train_size], x[train_size:]
        self.y_train, self.y_val = y[:train_size], y[train_size:]
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(self.x_train, self.y_train)
        val_ds = TensorDataset(self.x_val, self.y_val)
        
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        
        self.data_loaders = {
            'train': self.train_loader,
            'val': self.val_loader
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_model_training_workflow(self):
        """Test the full model training workflow."""
        # Create model manager
        model_manager = ModelManager(self.config['models'])
        
        # Create model
        model_config = {
            'input_size': 1,
            'hidden_layers': [32, 16],
            'output_size': 1
        }
        
        model_id, model = model_manager.create_model('mlp', **model_config)
        
        # Train model
        results = model_manager.train_model(
            model,
            model_id,
            self.data_loaders,
            trainer_type='supervised',
            epochs=10,
            optimizer_type='adam',
            optimizer_params={'lr': 0.01},
            loss_fn='mse'
        )
        
        # Check training results
        self.assertIn('history', results)
        self.assertIn('training_time', results)
        self.assertIn('epochs_completed', results)
        self.assertEqual(results['epochs_completed'], 10)
        
        # Check that loss decreased during training
        train_losses = [epoch_data.get('loss', float('inf')) 
                       for epoch_data in results['history']['train']]
        self.assertLess(train_losses[-1], train_losses[0])
        
        # Save and load model
        save_path = model_manager.save_model(model, model_id)
        self.assertTrue(os.path.exists(save_path))
        
        loaded_model = model_manager.load_model(model_id, model_type='mlp')
        
        # Validate loaded model
        with torch.no_grad():
            test_input = torch.tensor([[2.0]])
            orig_output = model(test_input)
            loaded_output = loaded_model(test_input)
            
            # Outputs should be identical
            self.assertTrue(torch.allclose(orig_output, loaded_output, rtol=1e-4))
```

## 11. Performance Considerations

The Model Training Module is designed with performance in mind:

1. **GPU Acceleration**:
   - All models support both CPU and GPU execution
   - Automatic device detection and management

2. **Memory Efficiency**:
   - Proper memory management during training
   - Support for gradient accumulation for large models

3. **Distributed Training**:
   - Ray integration for multi-node, multi-GPU training
   - Data parallelism for improved scalability

## 12. Extension Points

The module is designed to be easily extended:

1. **New Model Architectures**:
   - Create new model classes that inherit from BaseModel
   - Register them with the ModelFactory

2. **Custom Trainers**:
   - Create new trainer classes that inherit from Trainer
   - Register them with the TrainerFactory

3. **Custom Callbacks**:
   - Create new callback classes that inherit from Callback
   - Add to the trainer's callbacks list

4. **Custom Loss Functions**:
   - Define custom loss functions
   - Register with the loss function registry