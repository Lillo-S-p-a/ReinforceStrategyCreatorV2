"""Test script to verify the shape error fix."""

import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import required components
from src.training import TrainingEngine
from src.models.factory import get_factory

def create_sample_data(n_samples=100, n_features=5):
    """Create sample training data."""
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1
    
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    return df

def test_basic_training():
    """Test basic training with DQN model."""
    print("\n=== Testing DQN Training (2 epochs) ===")
    
    engine = TrainingEngine()
    data = create_sample_data(100, 5)
    
    model_config = {
        "model_type": "DQN",
        "name": "test_dqn",
        "hyperparameters": {
            "learning_rate": 0.001,
            "hidden_layers": [32, 16],
            "memory_size": 100
        }
    }
    
    training_config = {
        "epochs": 2,  # Just 2 epochs for quick test
        "batch_size": 32
    }
    
    result = engine.train(
        model_config=model_config,
        data_config={"train_data": data},
        training_config=training_config
    )
    
    if result["success"]:
        print(f"✅ DQN Training completed successfully!")
        print(f"   Epochs trained: {result['epochs_trained']}")
    else:
        print(f"❌ DQN Training failed: {result.get('error', 'Unknown error')}")
    
    return result["success"]

def test_ppo_training():
    """Test PPO training."""
    print("\n=== Testing PPO Training (2 epochs) ===")
    
    engine = TrainingEngine()
    data = create_sample_data(100, 5)
    
    model_config = {
        "model_type": "PPO",
        "name": "test_ppo",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 64,
            "batch_size": 32
        }
    }
    
    training_config = {
        "epochs": 2,
        "batch_size": 32
    }
    
    result = engine.train(
        model_config=model_config,
        data_config={"train_data": data},
        training_config=training_config
    )
    
    if result["success"]:
        print(f"✅ PPO Training completed successfully!")
        print(f"   Epochs trained: {result['epochs_trained']}")
    else:
        print(f"❌ PPO Training failed: {result.get('error', 'Unknown error')}")
    
    return result["success"]

def test_a2c_training():
    """Test A2C training."""
    print("\n=== Testing A2C Training (2 epochs) ===")
    
    engine = TrainingEngine()
    data = create_sample_data(100, 5)
    
    model_config = {
        "model_type": "A2C",
        "name": "test_a2c",
        "hyperparameters": {
            "learning_rate": 0.0007,
            "n_steps": 5
        }
    }
    
    training_config = {
        "epochs": 2,
        "batch_size": 32
    }
    
    result = engine.train(
        model_config=model_config,
        data_config={"train_data": data},
        training_config=training_config
    )
    
    if result["success"]:
        print(f"✅ A2C Training completed successfully!")
        print(f"   Epochs trained: {result['epochs_trained']}")
    else:
        print(f"❌ A2C Training failed: {result.get('error', 'Unknown error')}")
    
    return result["success"]

if __name__ == "__main__":
    print("Testing Shape Error Fix")
    print("=" * 50)
    
    # Test each model type
    dqn_success = test_basic_training()
    ppo_success = test_ppo_training()
    a2c_success = test_a2c_training()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"DQN: {'✅ PASSED' if dqn_success else '❌ FAILED'}")
    print(f"PPO: {'✅ PASSED' if ppo_success else '❌ FAILED'}")
    print(f"A2C: {'✅ PASSED' if a2c_success else '❌ FAILED'}")
    
    if all([dqn_success, ppo_success, a2c_success]):
        print("\n🎉 All tests passed! The shape error has been fixed.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")