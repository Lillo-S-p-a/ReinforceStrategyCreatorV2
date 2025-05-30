"""Example usage of the Training Engine."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import required components
from src.training import (
    TrainingEngine,
    LoggingCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback
)
from src.models.factory import get_factory
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalFileSystemStore as LocalArtifactStore
from src.data.manager import DataManager
from src.config.manager import ConfigManager


def create_sample_data(n_samples=1000, n_features=10):
    """Create sample training data."""
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some pattern
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1
    
    # Convert to DataFrame for compatibility with data manager
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    return df


def example_basic_training():
    """Example of basic training without persistence."""
    print("\n=== Basic Training Example ===")
    
    # Initialize training engine
    engine = TrainingEngine()
    
    # Generate sample data
    data = create_sample_data(1000, 10)
    train_data = data.iloc[:800]
    val_data = data.iloc[800:]
    
    # Model configuration
    model_config = {
        "model_type": "DQN",  # Assuming this is registered
        "name": "example_model",
        "hyperparameters": {
            "learning_rate": 0.001,
            "hidden_layers": [64, 32],
            "activation": "relu"
        }
    }
    
    # Data configuration
    data_config = {
        "train_data": train_data,
        "val_data": val_data
    }
    
    # Training configuration
    training_config = {
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.0,  # We're providing validation data directly
        "verbose": 1,
        "log_frequency": "epoch"
    }
    
    # Train the model
    result = engine.train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )
    
    if result["success"]:
        print(f"\nTraining completed successfully!")
        print(f"Epochs trained: {result['epochs_trained']}")
        print(f"Final metrics: {result.get('final_metrics', {})}")
    else:
        print(f"\nTraining failed: {result.get('error', 'Unknown error')}")
    
    return result


def example_training_with_callbacks():
    """Example of training with custom callbacks."""
    print("\n=== Training with Callbacks Example ===")
    
    # Initialize training engine
    engine = TrainingEngine(checkpoint_dir="./checkpoints/example_run")
    
    # Generate sample data
    data = create_sample_data(1000, 10)
    
    # Set up callbacks
    callbacks = [
        # Detailed logging
        LoggingCallback(
            log_frequency="epoch",
            log_file="./logs/training.log",
            verbose=2
        ),
        
        # Model checkpointing - save best model only
        ModelCheckpointCallback(
            checkpoint_dir="./checkpoints/example_run",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        
        # Early stopping
        EarlyStoppingCallback(
            monitor="val_loss",
            patience=3,
            mode="min",
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Model configuration
    model_config = {
        "model_type": "PPO",
        "name": "example_ppo_model",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2
        }
    }
    
    # Data configuration
    data_config = {
        "train_data": data
    }
    
    # Training configuration
    training_config = {
        "epochs": 20,
        "batch_size": 64,
        "validation_split": 0.2,
        "shuffle": True,
        "verbose": 1
    }
    
    # Train the model
    result = engine.train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        callbacks=callbacks
    )
    
    if result["success"]:
        print(f"\nTraining completed!")
        print(f"Training history: {result['history']}")
    
    return result


def example_training_with_persistence():
    """Example of training with full persistence using registry and artifact store."""
    print("\n=== Training with Persistence Example ===")
    
    # Set up artifact store and model registry
    artifact_store = LocalArtifactStore(root_path="./artifacts")
    model_registry = ModelRegistry(artifact_store)
    
    # Set up data manager
    config_manager = ConfigManager(
        config_dir="./configs",
        environment="development"
    )
    config_manager.load_config()  # Load the configuration
    data_manager = DataManager(
        config_manager=config_manager,
        artifact_store=artifact_store
    )
    
    # Register a data source
    data_manager.register_source(
        source_id="training_data",
        source_type="csv",
        config={
            "file_path": "./data/training_data.csv",  # This would need to exist
            "parse_dates": False
        }
    )
    
    # Initialize training engine with all components
    engine = TrainingEngine(
        model_registry=model_registry,
        artifact_store=artifact_store,
        data_manager=data_manager,
        checkpoint_dir="./checkpoints/persistent_run"
    )
    
    # Model configuration
    model_config = {
        "model_type": "A2C",
        "name": "production_model",
        "hyperparameters": {
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "vf_coef": 0.25,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5
        }
    }
    
    # Data configuration using data manager
    data_config = {
        "source_id": "training_data",
        "params": {
            "columns": ["feature_0", "feature_1", "target"]
        }
    }
    
    # Training configuration
    training_config = {
        "epochs": 15,
        "batch_size": 128,
        "validation_split": 0.2,
        "save_checkpoints": True,
        "checkpoint_frequency": "epoch",
        "save_best_only": True,
        "monitor": "val_loss",
        "data_info": {
            "dataset_name": "example_dataset",
            "version": "1.0"
        }
    }
    
    # Train the model
    result = engine.train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )
    
    if result["success"]:
        print(f"\nTraining completed!")
        print(f"Model saved with ID: {result.get('model_id', 'N/A')}")
        
        # List all models in registry
        models = model_registry.list_models(model_name="production_model")
        print(f"\nModels in registry: {len(models)}")
        for model in models:
            print(f"  - {model['model_id']} (v{model['version']}): {model.get('description', 'No description')}")
    
    return result


def example_resume_training():
    """Example of resuming training from a checkpoint."""
    print("\n=== Resume Training Example ===")
    
    # Path to existing checkpoint (this would need to exist)
    checkpoint_path = "./checkpoints/example_run/checkpoint_epoch_5"
    
    # Initialize training engine
    engine = TrainingEngine()
    
    # Generate new data for continued training
    data = create_sample_data(1000, 10)
    
    # Model config (must match checkpoint)
    model_config = {
        "model_type": "PPO",
        "name": "example_ppo_model"
    }
    
    # Data configuration
    data_config = {
        "train_data": data
    }
    
    # Training configuration - train for 10 more epochs
    training_config = {
        "epochs": 15,  # Total epochs (will continue from checkpoint)
        "batch_size": 64,
        "validation_split": 0.2
    }
    
    # Resume training
    result = engine.train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        resume_from_checkpoint=checkpoint_path
    )
    
    if result["success"]:
        print(f"\nTraining resumed and completed!")
        print(f"Total epochs after resume: {len(result['history']['epochs'])}")
    
    return result


def example_custom_callback():
    """Example of creating and using a custom callback."""
    print("\n=== Custom Callback Example ===")
    
    from src.training.callbacks import CallbackBase
    
    class MetricThresholdCallback(CallbackBase):
        """Stop training when a metric reaches a threshold."""
        
        def __init__(self, metric, threshold, mode="max"):
            super().__init__("MetricThresholdCallback")
            self.metric = metric
            self.threshold = threshold
            self.mode = mode
            
        def on_epoch_end(self, epoch, logs=None):
            if logs and self.metric in logs:
                value = logs[self.metric]
                
                if self.mode == "max" and value >= self.threshold:
                    self.logger.info(f"{self.metric} reached threshold: {value} >= {self.threshold}")
                    if hasattr(self.model, '_stop_training'):
                        self.model._stop_training = True
                elif self.mode == "min" and value <= self.threshold:
                    self.logger.info(f"{self.metric} reached threshold: {value} <= {self.threshold}")
                    if hasattr(self.model, '_stop_training'):
                        self.model._stop_training = True
    
    # Initialize training engine
    engine = TrainingEngine()
    
    # Use custom callback
    callbacks = [
        LoggingCallback(verbose=1),
        MetricThresholdCallback(
            metric="loss",
            threshold=0.1,
            mode="min"
        )
    ]
    
    # Train with custom callback
    result = engine.train(
        model_config={"model_type": "DQN", "name": "threshold_model"},
        data_config={"train_data": create_sample_data(500, 5)},
        training_config={"epochs": 50, "batch_size": 32},
        callbacks=callbacks
    )
    
    print(f"\nTraining stopped at epoch: {result['epochs_trained']}")
    
    return result


if __name__ == "__main__":
    # Run examples
    print("Training Engine Examples")
    print("=" * 50)
    
    # Note: These examples assume that model types like "DQN", "PPO", etc.
    # are registered with the model factory. In a real scenario, you would need
    # to ensure these models are implemented and registered.
    
    try:
        # Basic training
        example_basic_training()
    except Exception as e:
        import traceback
        print(f"Basic training example failed: {e}")
        traceback.print_exc()
    
    try:
        # Training with callbacks
        example_training_with_callbacks()
    except Exception as e:
        import traceback
        print(f"Callbacks example failed: {e}")
        traceback.print_exc()
    
    try:
        # Custom callback
        example_custom_callback()
    except Exception as e:
        import traceback
        print(f"Custom callback example failed: {e}")
        traceback.print_exc()
    
    try:
        # Training with persistence
        example_training_with_persistence()
    except Exception as e:
        print(f"Persistence example failed: {e}")
    
    # Note: The resume example would require additional setup
    # (existing checkpoints) to run successfully
    
    print("\n" + "=" * 50)
    print("Examples completed!")