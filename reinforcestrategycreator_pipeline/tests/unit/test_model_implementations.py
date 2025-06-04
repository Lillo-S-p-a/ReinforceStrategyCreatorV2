"""Unit tests for model implementations."""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path

from reinforcestrategycreator_pipeline.src.models.implementations import DQN, PPO, A2C
from reinforcestrategycreator_pipeline.src.models.base import ModelBase


class TestModelBase:
    """Test base model functionality using implementations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization with config."""
        config = {
            "model_type": "DQN",
            "hyperparameters": {
                "hidden_layers": [128, 64],
                "learning_rate": 0.001
            }
        }
        
        model = DQN(config)
        assert model.model_type == "DQN"
        assert model.hyperparameters["hidden_layers"] == [128, 64]
        assert model.hyperparameters["learning_rate"] == 0.001
        assert model.is_trained == False
        assert "created_at" in model.metadata
    
    def test_save_load_model(self, temp_dir):
        """Test saving and loading a model."""
        # Create and build model
        config = {"hyperparameters": {"hidden_layers": [64, 32]}}
        model = DQN(config)
        model.build(input_shape=(4,), output_shape=(2,))
        model.is_trained = True
        
        # Save model
        save_path = Path(temp_dir) / "test_model"
        save_info = model.save(save_path)
        
        # Verify files were created
        assert (save_path / "model.pkl").exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "metadata.json").exists()
        
        # Create new model and load
        new_model = DQN({})
        new_model.load(save_path)
        
        # Verify loaded state
        assert new_model.hyperparameters["hidden_layers"] == [64, 32]
        assert new_model.is_trained == True
        assert new_model.input_shape == (4,)
        assert new_model.output_shape == (2,)
    
    def test_hyperparameter_management(self):
        """Test getting and setting hyperparameters."""
        model = PPO({"hyperparameters": {"clip_range": 0.2}})
        
        # Get hyperparameters
        hyperparams = model.get_hyperparameters()
        assert hyperparams["clip_range"] == 0.2
        
        # Set new hyperparameters
        model.set_hyperparameters({"clip_range": 0.3, "n_epochs": 20})
        assert model.hyperparameters["clip_range"] == 0.3
        assert model.hyperparameters["n_epochs"] == 20
    
    def test_metadata_management(self):
        """Test metadata operations."""
        model = A2C({})
        
        # Get initial metadata
        metadata = model.get_metadata()
        assert "created_at" in metadata
        assert metadata["model_type"] == "A2C"
        
        # Update metadata
        model.update_metadata({
            "experiment_id": "exp_001",
            "dataset_version": "v1.2"
        })
        
        updated_metadata = model.get_metadata()
        assert updated_metadata["experiment_id"] == "exp_001"
        assert updated_metadata["dataset_version"] == "v1.2"


class TestDQN:
    """Test DQN implementation."""
    
    def test_dqn_build(self):
        """Test building DQN network."""
        model = DQN({
            "hyperparameters": {
                "hidden_layers": [256, 128],
                "memory_size": 5000
            }
        })
        
        model.build(input_shape=(10,), output_shape=(4,))
        
        assert model.input_shape == (10,)
        assert model.output_shape == (4,)
        assert model.n_actions == 4
        assert model.q_network is not None
        assert model.target_network is not None
        assert len(model.replay_buffer) == 0
    
    def test_dqn_predict(self):
        """Test DQN prediction."""
        model = DQN({})
        model.build(input_shape=(8,), output_shape=(3,))
        
        # Single state prediction
        state = np.random.randn(8)
        q_values = model.predict(state)
        assert q_values.shape == (3,)
        
        # Batch prediction
        batch_states = np.random.randn(5, 8)
        batch_q_values = model.predict(batch_states)
        assert batch_q_values.shape == (5, 3)
    
    def test_dqn_action_selection(self):
        """Test epsilon-greedy action selection."""
        model = DQN({})
        model.build(input_shape=(4,), output_shape=(2,))
        
        state = np.random.randn(4)
        
        # Test greedy selection (epsilon=0)
        actions = [model.select_action(state, epsilon=0.0) for _ in range(10)]
        assert all(a == actions[0] for a in actions)  # Should be deterministic
        
        # Test exploration (epsilon=1)
        actions = [model.select_action(state, epsilon=1.0) for _ in range(100)]
        assert len(set(actions)) == 2  # Should explore both actions
    
    def test_dqn_replay_buffer(self):
        """Test replay buffer functionality."""
        model = DQN({"hyperparameters": {"memory_size": 100}})
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Add experiences
        for i in range(150):
            state = np.random.randn(4)
            action = np.random.randint(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            model.replay_buffer.push(state, action, reward, next_state, done)
        
        # Buffer should be capped at max size
        assert len(model.replay_buffer) == 100
        
        # Test sampling
        batch = model.replay_buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (32, 4)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 4)
        assert dones.shape == (32,)
    
    def test_dqn_training(self):
        """Test DQN training process."""
        model = DQN({
            "hyperparameters": {
                "hidden_layers": [64],
                "memory_size": 1000,
                "update_frequency": 2,
                "target_update_frequency": 10
            }
        })
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Train for a few episodes
        history = model.train(
            train_data=None,  # Simulated in the implementation
            episodes=5,
            batch_size=16,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.9
        )
        
        assert model.is_trained == True
        assert len(history["episode_rewards"]) == 5
        assert len(history["episode_lengths"]) == 5
        assert len(history["epsilon_values"]) == 5
        assert history["epsilon_values"][-1] < history["epsilon_values"][0]
    
    def test_dqn_evaluation(self):
        """Test DQN evaluation."""
        model = DQN({})
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Train first
        model.train(episodes=2)
        
        # Evaluate
        metrics = model.evaluate(test_data=None, n_episodes=5)
        
        assert "mean_episode_reward" in metrics
        assert "std_episode_reward" in metrics
        assert "mean_episode_length" in metrics
        assert "min_episode_reward" in metrics
        assert "max_episode_reward" in metrics


class TestPPO:
    """Test PPO implementation."""
    
    def test_ppo_build(self):
        """Test building PPO networks."""
        model = PPO({
            "hyperparameters": {
                "policy_layers": [128, 64],
                "value_layers": [128, 64],
                "clip_range": 0.2
            }
        })
        
        model.build(input_shape=(8,), output_shape=(4,))
        
        assert model.input_shape == (8,)
        assert model.output_shape == (4,)
        assert model.n_actions == 4
        assert model.policy_network is not None
        assert model.value_network is not None
    
    def test_ppo_predict(self):
        """Test PPO prediction."""
        model = PPO({})
        model.build(input_shape=(6,), output_shape=(3,))
        
        # Single state prediction
        state = np.random.randn(6)
        prediction = model.predict(state)
        
        assert "actions" in prediction
        assert "values" in prediction
        assert "log_probs" in prediction
        assert "action_probs" in prediction
        assert prediction["action_probs"].shape == (3,)
        assert np.allclose(np.sum(prediction["action_probs"]), 1.0)
    
    def test_ppo_experience_collection(self):
        """Test PPO experience collection."""
        model = PPO({
            "hyperparameters": {
                "n_steps": 10
            }
        })
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Collect experience
        experience = model.collect_experience(env=None, n_steps=10)
        
        assert experience["states"].shape == (10, 4)
        assert experience["actions"].shape == (10,)
        assert experience["rewards"].shape == (10,)
        assert experience["values"].shape == (10,)
        assert experience["log_probs"].shape == (10,)
        assert experience["dones"].shape == (10,)
    
    def test_ppo_advantage_computation(self):
        """Test GAE computation."""
        model = PPO({"hyperparameters": {"gae_lambda": 0.95}})
        
        # Create sample data
        rewards = np.array([1.0, 0.5, -0.5, 1.0, 0.0])
        values = np.array([0.8, 0.6, 0.4, 0.9, 0.1])
        dones = np.array([False, False, True, False, False])
        
        advantages, returns = model.compute_advantages(rewards, values, dones, gamma=0.99)
        
        assert advantages.shape == (5,)
        assert returns.shape == (5,)
        # Advantages should be normalized
        assert np.abs(np.mean(advantages)) < 0.1
        assert np.abs(np.std(advantages) - 1.0) < 0.1
    
    def test_ppo_training(self):
        """Test PPO training process."""
        model = PPO({
            "hyperparameters": {
                "policy_layers": [64],
                "value_layers": [64],
                "n_steps": 64,
                "n_epochs": 2
            }
        })
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Train
        history = model.train(
            train_data=None,
            total_timesteps=256,
            batch_size=32,
            learning_rate=0.001
        )
        
        assert model.is_trained == True
        assert len(history["policy_losses"]) > 0
        assert len(history["value_losses"]) > 0
        assert len(history["entropy_losses"]) > 0
        assert len(history["explained_variance"]) > 0


class TestA2C:
    """Test A2C implementation."""
    
    def test_a2c_build(self):
        """Test building A2C network."""
        model = A2C({
            "hyperparameters": {
                "shared_layers": [128],
                "policy_head_layers": [32],
                "value_head_layers": [32]
            }
        })
        
        model.build(input_shape=(10,), output_shape=(5,))
        
        assert model.input_shape == (10,)
        assert model.output_shape == (5,)
        assert model.n_actions == 5
        assert model.network is not None
        assert "shared_weights" in model.network
        assert "policy_head_weights" in model.network
        assert "value_head_weights" in model.network
    
    def test_a2c_predict(self):
        """Test A2C prediction."""
        model = A2C({})
        model.build(input_shape=(8,), output_shape=(4,))
        
        # Single state
        state = np.random.randn(8)
        prediction = model.predict(state)
        
        assert "actions" in prediction
        assert "values" in prediction
        assert "action_probs" in prediction
        assert prediction["action_probs"].shape == (4,)
        
        # Batch prediction
        batch_states = np.random.randn(3, 8)
        batch_prediction = model.predict(batch_states)
        
        assert batch_prediction["actions"].shape == (3,)
        assert batch_prediction["values"].shape == (3,)
        assert batch_prediction["action_probs"].shape == (3, 4)
    
    def test_a2c_shared_representation(self):
        """Test that A2C uses shared representation."""
        model = A2C({
            "hyperparameters": {
                "shared_layers": [64, 32]
            }
        })
        model.build(input_shape=(6,), output_shape=(3,))
        
        # Get shared features
        state = np.random.randn(6)
        shared_features = model._forward_shared(state)
        
        # Should match last shared layer size
        assert shared_features.shape == (32,)
    
    def test_a2c_training(self):
        """Test A2C training process."""
        model = A2C({
            "hyperparameters": {
                "shared_layers": [64],
                "n_steps": 5,
                "use_rms_prop": True
            }
        })
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Train
        history = model.train(
            train_data=None,
            total_timesteps=100,
            learning_rate=0.001
        )
        
        assert model.is_trained == True
        assert len(history["actor_losses"]) > 0
        assert len(history["critic_losses"]) > 0
        assert len(history["entropy_losses"]) > 0
        assert model.optimizer_state is not None
    
    def test_a2c_optimizer_state(self):
        """Test RMSprop optimizer state management."""
        model = A2C({
            "hyperparameters": {
                "use_rms_prop": True,
                "rms_prop_eps": 1e-5
            }
        })
        model.build(input_shape=(4,), output_shape=(2,))
        
        # Check optimizer state initialization
        assert model.optimizer_state is not None
        
        # Verify state for each parameter
        for key in model.network["shared_weights"]:
            opt_key = f"shared_{key}"
            assert opt_key in model.optimizer_state
            assert "square_avg" in model.optimizer_state[opt_key]


class TestModelComparison:
    """Test comparing different model implementations."""
    
    def test_model_interfaces(self):
        """Test that all models implement the same interface."""
        models = [
            DQN({"hyperparameters": {}}),
            PPO({"hyperparameters": {}}),
            A2C({"hyperparameters": {}})
        ]
        
        for model in models:
            # Check required methods
            assert hasattr(model, 'build')
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'evaluate')
            assert hasattr(model, 'save')
            assert hasattr(model, 'load')
            assert hasattr(model, 'get_model_state')
            assert hasattr(model, 'set_model_state')
            
            # Check they're all ModelBase instances
            assert isinstance(model, ModelBase)
    
    def test_model_state_serialization(self):
        """Test that all models can serialize/deserialize state."""
        models = [
            (DQN({}), (4,), (2,)),
            (PPO({}), (6,), (3,)),
            (A2C({}), (8,), (4,))
        ]
        
        for model, input_shape, output_shape in models:
            # Build model
            model.build(input_shape, output_shape)
            
            # Get state
            state = model.get_model_state()
            assert isinstance(state, dict)
            
            # Create new model and set state
            new_model = model.__class__({})
            new_model.set_model_state(state)
            
            # Verify shapes match
            if hasattr(new_model, 'input_shape'):
                assert new_model.input_shape == input_shape
            if hasattr(new_model, 'output_shape'):
                assert new_model.output_shape == output_shape


if __name__ == "__main__":
    pytest.main([__file__])