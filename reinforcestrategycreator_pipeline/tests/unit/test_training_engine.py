"""Unit tests for the training engine."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import pandas as pd
import pytest

from src.training.engine import TrainingEngine
from src.training.callbacks import CallbackBase, LoggingCallback, ModelCheckpointCallback
from src.models.base import ModelBase


class MockModel(ModelBase):
    """Mock model for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self._stop_training = False
        self.train_called = False
        self.evaluate_called = False
        self.build_called = False
    
    def build(self, input_shape, output_shape):
        self.build_called = True
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def train(self, train_data, validation_data=None, **kwargs):
        self.train_called = True
        # Simulate decreasing loss
        return {"loss": np.random.random() * 0.5}
    
    def predict(self, data, **kwargs):
        return np.random.random((len(data), 1))
    
    def evaluate(self, test_data, **kwargs):
        self.evaluate_called = True
        return {"loss": np.random.random() * 0.3}
    
    def get_model_state(self):
        return {"weights": "mock_weights"}
    
    def set_model_state(self, state):
        self.state = state


class TestTrainingEngine:
    """Test the training engine."""
    
    def test_training_engine_initialization(self):
        """Test TrainingEngine initialization."""
        engine = TrainingEngine()
        assert engine.model_factory is not None
        assert engine.checkpoint_dir.exists()
        assert not engine.is_training
        
        # Test with custom components
        mock_factory = Mock()
        mock_registry = Mock()
        mock_store = Mock()
        mock_data_manager = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(
                model_factory=mock_factory,
                model_registry=mock_registry,
                artifact_store=mock_store,
                data_manager=mock_data_manager,
                checkpoint_dir=tmpdir
            )
            assert engine.model_factory == mock_factory
            assert engine.model_registry == mock_registry
            assert engine.artifact_store == mock_store
            assert engine.data_manager == mock_data_manager
            assert engine.checkpoint_dir == Path(tmpdir)
    
    def test_train_basic_workflow(self):
        """Test basic training workflow."""
        # Set up mocks
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        mock_factory.create_from_config.return_value = mock_model
        
        engine = TrainingEngine(model_factory=mock_factory)
        
        # Training configuration
        model_config = {"model_type": "test", "name": "test_model"}
        data_config = {
            "train_data": np.random.random((100, 10)),
            "val_data": np.random.random((20, 10))
        }
        training_config = {
            "epochs": 2,
            "batch_size": 32,
            "validation_split": 0.0  # Using provided val_data
        }
        
        # Train
        result = engine.train(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config
        )
        
        # Verify results
        assert result["success"]
        assert result["epochs_trained"] == 2
        assert "history" in result
        assert mock_model.train_called
        assert mock_model.evaluate_called
        assert mock_model.is_trained
    
    def test_train_with_callbacks(self):
        """Test training with custom callbacks."""
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        mock_factory.create_from_config.return_value = mock_model
        
        engine = TrainingEngine(model_factory=mock_factory)
        
        # Custom callback
        mock_callback = Mock(spec=CallbackBase)
        
        # Train with callback
        result = engine.train(
            model_config={"model_type": "test"},
            data_config={"train_data": np.random.random((50, 5))},
            training_config={"epochs": 1},
            callbacks=[mock_callback]
        )
        
        # Verify callback was used
        assert mock_callback.on_train_begin.called
        assert mock_callback.on_epoch_begin.called
        assert mock_callback.on_epoch_end.called
        assert mock_callback.on_train_end.called
    
    def test_train_with_data_manager(self):
        """Test training with data manager."""
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        mock_factory.create_from_config.return_value = mock_model
        
        # Mock data manager
        mock_data_manager = Mock()
        mock_data = pd.DataFrame(np.random.random((100, 10)))
        mock_data_manager.load_data.return_value = mock_data
        
        engine = TrainingEngine(
            model_factory=mock_factory,
            data_manager=mock_data_manager
        )
        
        # Train with data manager
        data_config = {
            "source_id": "test_source",
            "params": {"param1": "value1"}
        }
        
        result = engine.train(
            model_config={"model_type": "test"},
            data_config=data_config,
            training_config={"epochs": 1, "validation_split": 0.2}
        )
        
        # Verify data manager was used
        mock_data_manager.load_data.assert_called_with(
            "test_source", param1="value1"
        )
        assert result["success"]
    
    def test_train_with_model_registry(self):
        """Test training with model registry."""
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        mock_factory.create_from_config.return_value = mock_model
        
        # Mock registry and store
        mock_registry = Mock()
        mock_registry.register_model.return_value = "model_123"
        mock_store = Mock()
        
        engine = TrainingEngine(
            model_factory=mock_factory,
            model_registry=mock_registry,
            artifact_store=mock_store
        )
        
        # Train
        result = engine.train(
            model_config={"model_type": "test", "name": "test_model"},
            data_config={"train_data": np.random.random((50, 5))},
            training_config={"epochs": 1}
        )
        
        # Verify model was registered
        assert mock_registry.register_model.called
        assert result["model_id"] == "model_123"
    
    def test_train_with_checkpoint_resume(self):
        """Test resuming training from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()
            
            # Create mock checkpoint files
            config = {"model_type": "test", "hyperparameters": {}}
            with open(checkpoint_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            training_state = {
                "epoch": 2,
                "training_history": {
                    "loss": [0.5, 0.4],
                    "epochs": [0, 1],
                    "metrics": {}
                }
            }
            with open(checkpoint_dir / "training_state.json", "w") as f:
                json.dump(training_state, f)
            
            # Create other required files
            (checkpoint_dir / "model.pkl").touch()
            metadata = {
                "created_at": "2025-01-01T00:00:00",
                "model_type": "test",
                "hyperparameters": {},
                "is_trained": True
            }
            with open(checkpoint_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # Set up engine
            mock_factory = Mock()
            mock_model = MockModel(config)
            mock_factory.create_from_config.return_value = mock_model
            
            engine = TrainingEngine(model_factory=mock_factory)
            
            # Train with resume
            result = engine.train(
                model_config={"model_type": "test"},
                data_config={"train_data": np.random.random((50, 5))},
                training_config={"epochs": 5},  # Total 5 epochs
                resume_from_checkpoint=checkpoint_dir
            )
            
            # Should train for 3 more epochs (5 total - 2 already done)
            assert result["epochs_trained"] == 3
            assert len(engine.training_history["loss"]) == 5  # 2 from checkpoint + 3 new
    
    def test_train_early_stopping(self):
        """Test that training stops when requested."""
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        mock_factory.create_from_config.return_value = mock_model
        
        engine = TrainingEngine(model_factory=mock_factory)
        
        # Callback that stops training after 1 epoch
        class StopCallback(CallbackBase):
            def on_epoch_end(self, epoch, logs=None):
                if epoch == 0:
                    engine.stop_training()
        
        # Train
        result = engine.train(
            model_config={"model_type": "test"},
            data_config={"train_data": np.random.random((50, 5))},
            training_config={"epochs": 10},
            callbacks=[StopCallback()]
        )
        
        # Should stop after 1 epoch
        assert result["epochs_trained"] == 1
    
    def test_train_error_handling(self):
        """Test error handling during training."""
        mock_factory = Mock()
        mock_factory.create_from_config.side_effect = ValueError("Model creation failed")
        
        engine = TrainingEngine(model_factory=mock_factory)
        
        # Train should handle error gracefully
        result = engine.train(
            model_config={"model_type": "invalid"},
            data_config={"train_data": np.random.random((50, 5))},
            training_config={"epochs": 1}
        )
        
        assert not result["success"]
        assert "error" in result
        assert "Model creation failed" in result["error"]
    
    def test_save_checkpoint_manual(self):
        """Test manual checkpoint saving."""
        mock_factory = Mock()
        mock_model = MockModel({"model_type": "test"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(
                model_factory=mock_factory,
                checkpoint_dir=tmpdir
            )
            engine.model = mock_model
            engine.current_epoch = 5
            engine.training_history = {"loss": [0.5, 0.4, 0.3]}
            
            # Save checkpoint
            checkpoint_path = engine.save_checkpoint(
                checkpoint_name="manual_save",
                additional_metadata={"custom": "data"}
            )
            
            # Verify checkpoint was saved
            assert checkpoint_path.exists()
            assert (checkpoint_path / "training_state.json").exists()
            
            # Check saved state
            with open(checkpoint_path / "training_state.json", "r") as f:
                state = json.load(f)
            
            assert state["epoch"] == 5
            assert state["custom"] == "data"
            assert state["training_history"] == {"loss": [0.5, 0.4, 0.3]}
    
    def test_get_data_shape(self):
        """Test data shape inference."""
        engine = TrainingEngine()
        
        # NumPy array
        data = np.random.random((100, 10, 5))
        shape = engine._get_data_shape(data, "input")
        assert shape == (10, 5)
        
        # List of tuples (input, output)
        data = [(np.zeros(8), np.zeros(2)) for _ in range(50)]
        input_shape = engine._get_data_shape(data, "input")
        output_shape = engine._get_data_shape(data, "output")
        assert input_shape == (8,)
        assert output_shape == (2,)
        
        # Default case
        shape = engine._get_data_shape("unknown_format", "input")
        assert shape == (1,)
    
    def test_setup_callbacks_defaults(self):
        """Test default callback setup."""
        mock_model = MockModel({"model_type": "test"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TrainingEngine(checkpoint_dir=tmpdir)
            engine.model = mock_model
            
            # No callbacks provided
            callback_list = engine._setup_callbacks(
                callbacks=None,
                training_config={},
                model_config={}
            )
            
            # Should have default logging callback
            assert any(isinstance(cb, LoggingCallback) 
                      for cb in callback_list.callbacks)
            
            # Should have checkpoint callback
            assert any(isinstance(cb, ModelCheckpointCallback) 
                      for cb in callback_list.callbacks)
    
    def test_update_history(self):
        """Test history update logic."""
        engine = TrainingEngine()
        engine.training_history = {
            "loss": [],
            "val_loss": [],
            "epochs": [],
            "metrics": {}
        }
        
        # Update with epoch logs
        epoch_logs = {
            "epoch": 0,
            "loss": 0.5,
            "val_loss": 0.4,
            "accuracy": 0.8,
            "custom_metric": 1.2
        }
        
        engine._update_history(epoch_logs)
        
        assert engine.training_history["loss"] == [0.5]
        assert engine.training_history["val_loss"] == [0.4]
        assert engine.training_history["epochs"] == [0]
        assert engine.training_history["metrics"]["accuracy"] == [0.8]
        assert engine.training_history["metrics"]["custom_metric"] == [1.2]