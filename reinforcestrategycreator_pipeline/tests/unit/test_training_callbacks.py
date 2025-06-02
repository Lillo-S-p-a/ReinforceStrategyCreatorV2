"""Unit tests for training callbacks."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.training.callbacks import (
    CallbackBase,
    CallbackList,
    LoggingCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback
)


class TestCallbackBase:
    """Test the base callback class."""
    
    def test_callback_base_initialization(self):
        """Test CallbackBase initialization."""
        callback = CallbackBase()
        assert callback.name == "CallbackBase"
        
        # Test with custom name
        callback = CallbackBase(name="CustomCallback")
        assert callback.name == "CustomCallback"
    
    def test_callback_base_methods(self):
        """Test that base methods can be called without error."""
        callback = CallbackBase()
        
        # All methods should accept logs and do nothing
        callback.on_train_begin({"test": "value"})
        callback.on_train_end({"test": "value"})
        callback.on_epoch_begin(0, {"test": "value"})
        callback.on_epoch_end(0, {"test": "value"})
        callback.on_batch_begin(0, {"test": "value"})
        callback.on_batch_end(0, {"test": "value"})


class TestLoggingCallback:
    """Test the logging callback."""
    
    def test_logging_callback_initialization(self):
        """Test LoggingCallback initialization."""
        callback = LoggingCallback()
        assert callback.log_frequency == "epoch"
        assert callback.verbose == 1
        assert callback.log_file is None
        
        # Test with custom parameters
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            callback = LoggingCallback(
                log_frequency="batch",
                log_file=log_file,
                verbose=2
            )
            assert callback.log_frequency == "batch"
            assert callback.verbose == 2
            assert callback.log_file == log_file
    
    @patch('logging.Logger.info')
    def test_logging_callback_epoch_logging(self, mock_log):
        """Test epoch-level logging."""
        callback = LoggingCallback(log_frequency="epoch", verbose=1)
        
        # Start training
        callback.on_train_begin({"epochs": 10})
        assert mock_log.called
        
        # Epoch logging
        callback.on_epoch_begin(0, {})
        callback.on_epoch_end(0, {"loss": 0.5, "accuracy": 0.9})
        
        # Check that epoch end was logged
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert any("Epoch 0 completed" in call for call in log_calls)
        assert any("loss: 0.5000" in call for call in log_calls)
    
    @patch('logging.Logger.info')
    def test_logging_callback_batch_logging(self, mock_log):
        """Test batch-level logging."""
        callback = LoggingCallback(log_frequency="batch", verbose=2)
        
        # Batch logging
        callback.on_batch_end(0, {"loss": 0.5})
        
        # Check that batch was logged
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        assert any("Batch 0 completed" in call for call in log_calls)


class TestModelCheckpointCallback:
    """Test the model checkpoint callback."""
    
    def test_checkpoint_callback_initialization(self):
        """Test ModelCheckpointCallback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(checkpoint_dir=tmpdir)
            assert callback.checkpoint_dir == Path(tmpdir)
            assert callback.save_frequency == "epoch"
            assert not callback.save_best_only
            assert callback.monitor == "loss"
            assert callback.mode == "min"
    
    def test_checkpoint_callback_save_logic(self):
        """Test checkpoint saving logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test save best only with min mode
            callback = ModelCheckpointCallback(
                checkpoint_dir=tmpdir,
                save_best_only=True,
                monitor="loss",
                mode="min"
            )
            
            # Mock model
            mock_model = Mock()
            mock_model.save.return_value = {"model_path": "test_path"}
            mock_model.get_hyperparameters.return_value = {}
            mock_model.get_metadata.return_value = {}
            callback.set_model(mock_model)
            
            # First epoch - should save (best so far)
            callback.on_epoch_end(0, {"loss": 0.5})
            assert mock_model.save.called
            mock_model.save.reset_mock()
            
            # Second epoch - worse loss, should not save
            callback.on_epoch_end(1, {"loss": 0.6})
            assert not mock_model.save.called
            
            # Third epoch - better loss, should save
            callback.on_epoch_end(2, {"loss": 0.4})
            assert mock_model.save.called
    
    def test_checkpoint_callback_max_mode(self):
        """Test checkpoint with max mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpointCallback(
                checkpoint_dir=tmpdir,
                save_best_only=True,
                monitor="accuracy",
                mode="max"
            )
            
            # Mock model
            mock_model = Mock()
            mock_model.save.return_value = {"model_path": "test_path"}
            callback.set_model(mock_model)
            
            # First epoch
            callback.on_epoch_end(0, {"accuracy": 0.8})
            assert mock_model.save.called
            mock_model.save.reset_mock()
            
            # Second epoch - worse accuracy
            callback.on_epoch_end(1, {"accuracy": 0.7})
            assert not mock_model.save.called
            
            # Third epoch - better accuracy
            callback.on_epoch_end(2, {"accuracy": 0.9})
            assert mock_model.save.called


class TestEarlyStoppingCallback:
    """Test the early stopping callback."""
    
    def test_early_stopping_initialization(self):
        """Test EarlyStoppingCallback initialization."""
        callback = EarlyStoppingCallback()
        assert callback.monitor == "loss"
        assert callback.patience == 5
        assert callback.mode == "min"
        assert callback.min_delta == 0.0001
        assert callback.restore_best_weights
    
    def test_early_stopping_min_mode(self):
        """Test early stopping with min mode."""
        callback = EarlyStoppingCallback(
            monitor="loss",
            patience=2,
            mode="min",
            min_delta=0.01
        )
        
        # Mock model
        mock_model = Mock()
        mock_model.get_model_state.return_value = {"weights": "test"}
        callback.set_model(mock_model)
        
        # Reset state
        callback.on_train_begin()
        
        # Improving epochs
        callback.on_epoch_end(0, {"loss": 0.5})
        assert callback.wait == 0
        
        callback.on_epoch_end(1, {"loss": 0.4})
        assert callback.wait == 0
        
        # No improvement (within min_delta)
        callback.on_epoch_end(2, {"loss": 0.395})
        assert callback.wait == 1
        
        # No improvement
        callback.on_epoch_end(3, {"loss": 0.41})
        assert callback.wait == 2
        
        # Should trigger early stopping
        callback.on_epoch_end(4, {"loss": 0.42})
        assert hasattr(mock_model, '_stop_training')
        assert mock_model._stop_training == True
    
    def test_early_stopping_max_mode(self):
        """Test early stopping with max mode."""
        callback = EarlyStoppingCallback(
            monitor="accuracy",
            patience=1,
            mode="max"
        )
        
        mock_model = Mock()
        callback.set_model(mock_model)
        callback.on_train_begin()
        
        # Improving
        callback.on_epoch_end(0, {"accuracy": 0.8})
        assert callback.wait == 0
        
        # No improvement
        callback.on_epoch_end(1, {"accuracy": 0.79})
        assert callback.wait == 1
        
        # Should stop
        callback.on_epoch_end(2, {"accuracy": 0.78})
        assert hasattr(mock_model, '_stop_training')
    
    def test_early_stopping_restore_weights(self):
        """Test weight restoration."""
        callback = EarlyStoppingCallback(
            patience=1,
            restore_best_weights=True
        )
        
        mock_model = Mock()
        best_weights = {"layer1": [1, 2, 3]}
        mock_model.get_model_state.return_value = best_weights
        callback.set_model(mock_model)
        
        callback.on_train_begin()
        
        # Best epoch
        callback.on_epoch_end(0, {"loss": 0.3})
        
        # Worse epochs
        callback.on_epoch_end(1, {"loss": 0.4})
        callback.on_epoch_end(2, {"loss": 0.5})
        
        # Should restore best weights
        mock_model.set_model_state.assert_called_with(best_weights)


class TestCallbackList:
    """Test the callback list container."""
    
    def test_callback_list_initialization(self):
        """Test CallbackList initialization."""
        # Empty list
        callback_list = CallbackList()
        assert len(callback_list.callbacks) == 0
        
        # With callbacks
        cb1 = LoggingCallback()
        cb2 = EarlyStoppingCallback()
        callback_list = CallbackList([cb1, cb2])
        assert len(callback_list.callbacks) == 2
    
    def test_callback_list_append(self):
        """Test appending callbacks."""
        callback_list = CallbackList()
        callback = LoggingCallback()
        callback_list.append(callback)
        assert len(callback_list.callbacks) == 1
        assert callback_list.callbacks[0] == callback
    
    def test_callback_list_propagation(self):
        """Test that callbacks are called properly."""
        # Create mock callbacks
        cb1 = Mock(spec=CallbackBase)
        cb2 = Mock(spec=CallbackBase)
        
        callback_list = CallbackList([cb1, cb2])
        
        # Test all callback methods
        callback_list.on_train_begin({"test": 1})
        cb1.on_train_begin.assert_called_with({"test": 1})
        cb2.on_train_begin.assert_called_with({"test": 1})
        
        callback_list.on_epoch_end(0, {"loss": 0.5})
        cb1.on_epoch_end.assert_called_with(0, {"loss": 0.5})
        cb2.on_epoch_end.assert_called_with(0, {"loss": 0.5})
    
    def test_callback_list_set_model(self):
        """Test setting model on callbacks."""
        # Create callbacks with set_model method
        cb1 = Mock(spec=ModelCheckpointCallback)
        cb2 = Mock(spec=EarlyStoppingCallback)
        cb3 = Mock(spec=LoggingCallback)  # No set_model
        
        callback_list = CallbackList([cb1, cb2, cb3])
        
        mock_model = Mock()
        callback_list.set_model(mock_model)
        
        # Only callbacks with set_model should be called
        cb1.set_model.assert_called_with(mock_model)
        cb2.set_model.assert_called_with(mock_model)
        assert not hasattr(cb3, 'set_model') or not cb3.set_model.called