"""Unit tests for the Model Factory."""

import pytest
from typing import Dict, Any, Tuple
import numpy as np

from reinforcestrategycreator_pipeline.src.models.base import ModelBase
from reinforcestrategycreator_pipeline.src.models.factory import ModelFactory, get_factory, create_model, register_model, list_available_models
from reinforcestrategycreator_pipeline.src.models.implementations import DQN, PPO, A2C


class MockModel(ModelBase):
    """Mock model for testing."""
    
    model_type = "MockModel"
    
    def build(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def train(self, train_data: Any, validation_data=None, **kwargs) -> Dict[str, Any]:
        return {"loss": 0.1}
    
    def predict(self, data: Any, **kwargs) -> Any:
        return np.array([0.5])
    
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        return {"accuracy": 0.95}
    
    def get_model_state(self) -> Dict[str, Any]:
        return {"mock_state": True}
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        pass


class TestModelFactory:
    """Test cases for ModelFactory."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = ModelFactory()
        assert isinstance(factory, ModelFactory)
        assert hasattr(factory, '_registry')
    
    def test_builtin_models_registered(self):
        """Test that built-in models are automatically registered."""
        factory = ModelFactory()
        
        # Check that built-in models are registered
        assert factory.is_model_registered("DQN")
        assert factory.is_model_registered("PPO")
        assert factory.is_model_registered("A2C")
    
    def test_register_model(self):
        """Test registering a custom model."""
        factory = ModelFactory()
        
        # Register mock model
        factory.register_model("MockModel", MockModel)
        
        assert factory.is_model_registered("MockModel")
        assert factory.get_model_class("MockModel") == MockModel
    
    def test_register_invalid_model(self):
        """Test registering an invalid model raises error."""
        factory = ModelFactory()
        
        class InvalidModel:
            pass
        
        with pytest.raises(ValueError, match="must inherit from ModelBase"):
            factory.register_model("Invalid", InvalidModel)
    
    def test_create_model(self):
        """Test creating model instances."""
        factory = ModelFactory()
        
        # Create DQN model
        config = {
            "hyperparameters": {
                "hidden_layers": [128, 64],
                "memory_size": 5000
            }
        }
        model = factory.create_model("DQN", config)
        
        assert isinstance(model, DQN)
        assert model.model_type == "DQN"
        assert model.hyperparameters["hidden_layers"] == [128, 64]
        assert model.hyperparameters["memory_size"] == 5000
    
    def test_create_model_without_config(self):
        """Test creating model without config."""
        factory = ModelFactory()
        
        model = factory.create_model("PPO")
        assert isinstance(model, PPO)
        assert model.model_type == "PPO"
    
    def test_create_unknown_model(self):
        """Test creating unknown model raises error."""
        factory = ModelFactory()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            factory.create_model("UnknownModel")
    
    def test_list_available_models(self):
        """Test listing available models."""
        factory = ModelFactory()
        
        models = factory.list_available_models()
        assert isinstance(models, list)
        assert "DQN" in models
        assert "PPO" in models
        assert "A2C" in models
        assert models == sorted(models)  # Check sorted
    
    def test_unregister_model(self):
        """Test unregistering a model."""
        factory = ModelFactory()
        
        # Register and then unregister
        factory.register_model("TempModel", MockModel)
        assert factory.is_model_registered("TempModel")
        
        factory.unregister_model("TempModel")
        assert not factory.is_model_registered("TempModel")
    
    def test_unregister_unknown_model(self):
        """Test unregistering unknown model raises error."""
        factory = ModelFactory()
        
        with pytest.raises(ValueError, match="is not registered"):
            factory.unregister_model("UnknownModel")
    
    def test_create_from_config(self):
        """Test creating model from config dictionary."""
        factory = ModelFactory()
        
        config = {
            "model_type": "A2C",
            "hyperparameters": {
                "shared_layers": [512, 256],
                "n_steps": 10
            }
        }
        
        model = factory.create_from_config(config)
        assert isinstance(model, A2C)
        assert model.hyperparameters["shared_layers"] == [512, 256]
        assert model.hyperparameters["n_steps"] == 10
    
    def test_create_from_config_missing_type(self):
        """Test creating from config without model_type raises error."""
        factory = ModelFactory()
        
        config = {"hyperparameters": {}}
        
        with pytest.raises(ValueError, match="must contain 'model_type'"):
            factory.create_from_config(config)
    
    def test_global_factory_functions(self):
        """Test global factory functions."""
        # Test get_factory
        factory = get_factory()
        assert isinstance(factory, ModelFactory)
        
        # Test create_model
        model = create_model("DQN", {"hyperparameters": {"memory_size": 1000}})
        assert isinstance(model, DQN)
        assert model.hyperparameters["memory_size"] == 1000
        
        # Test register_model
        register_model("GlobalMock", MockModel)
        assert "GlobalMock" in list_available_models()
        
        # Clean up
        factory.unregister_model("GlobalMock")
    
    def test_model_type_in_config(self):
        """Test that model_type is added to config when creating model."""
        factory = ModelFactory()
        
        config = {"hyperparameters": {"learning_rate": 0.001}}
        model = factory.create_model("PPO", config)
        
        assert model.config["model_type"] == "PPO"
    
    def test_overwrite_warning(self, capsys):
        """Test that overwriting a model type shows warning."""
        factory = ModelFactory()
        
        # Register once
        factory.register_model("TestModel", MockModel)
        
        # Register again with same name
        factory.register_model("TestModel", MockModel)
        
        captured = capsys.readouterr()
        assert "Warning: Overwriting existing model type 'TestModel'" in captured.out
        
        # Clean up
        factory.unregister_model("TestModel")


class TestModelCreation:
    """Test creating and using different model types."""
    
    def test_create_dqn(self):
        """Test creating and using DQN model."""
        config = {
            "hyperparameters": {
                "hidden_layers": [256, 128],
                "double_dqn": True,
                "memory_size": 10000
            }
        }
        
        model = create_model("DQN", config)
        assert isinstance(model, DQN)
        
        # Test building
        model.build(input_shape=(4,), output_shape=(2,))
        assert model.n_actions == 2
        
        # Test prediction
        state = np.random.randn(4)
        q_values = model.predict(state)
        assert q_values.shape == (2,)
    
    def test_create_ppo(self):
        """Test creating and using PPO model."""
        config = {
            "hyperparameters": {
                "policy_layers": [128, 64],
                "value_layers": [128, 64],
                "clip_range": 0.2
            }
        }
        
        model = create_model("PPO", config)
        assert isinstance(model, PPO)
        
        # Test building
        model.build(input_shape=(8,), output_shape=(4,))
        assert model.n_actions == 4
        
        # Test prediction
        state = np.random.randn(8)
        prediction = model.predict(state)
        assert "actions" in prediction
        assert "values" in prediction
        assert "action_probs" in prediction
    
    def test_create_a2c(self):
        """Test creating and using A2C model."""
        config = {
            "hyperparameters": {
                "shared_layers": [256],
                "policy_head_layers": [64],
                "value_head_layers": [64]
            }
        }
        
        model = create_model("A2C", config)
        assert isinstance(model, A2C)
        
        # Test building
        model.build(input_shape=(10,), output_shape=(3,))
        assert model.n_actions == 3
        
        # Test prediction
        state = np.random.randn(10)
        prediction = model.predict(state)
        assert "actions" in prediction
        assert "values" in prediction
        assert "action_probs" in prediction


if __name__ == "__main__":
    pytest.main([__file__])