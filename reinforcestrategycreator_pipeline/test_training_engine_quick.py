"""Quick test of TrainingEngine with fixed models - minimal epochs and episodes"""
import numpy as np
import pytest
from reinforcestrategycreator_pipeline.src.training.engine import TrainingEngine
from reinforcestrategycreator_pipeline.src.models.implementations.dqn import DQN
from reinforcestrategycreator_pipeline.src.models.implementations.ppo import PPO
from reinforcestrategycreator_pipeline.src.models.implementations.a2c import A2C
from reinforcestrategycreator_pipeline.src.training.callbacks import LoggingCallback, ModelCheckpointCallback

def create_dummy_data(n_samples=100):
    """Create dummy data for testing"""
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    return X, y

@pytest.mark.parametrize("model_class, model_name", [
    (DQN, "DQN"),
    (PPO, "PPO"),
    (A2C, "A2C")
])
def test_model_with_engine(model_class, model_name):
    """Test a model with TrainingEngine"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name} with TrainingEngine")
    print('='*50)
    
    # Create dummy data
    X_train, y_train = create_dummy_data(100)
    X_val, y_val = create_dummy_data(20)
    
    # Initialize model with config
    if model_class == DQN:
        config = {
            "model_type": "DQN",
            "name": f"{model_name}_test",
            "hyperparameters": {
                "learning_rate": 0.001,
                "hidden_layers": [64, 32],
                "activation": "relu",
                "episodes_per_epoch": 10,  # Very few episodes
                "epsilon_decay": 0.99
            }
        }
    elif model_class == PPO:
        config = {
            "model_type": "PPO",
            "name": f"{model_name}_test",
            "hyperparameters": {
                "learning_rate": 0.001,
                "hidden_layers": [64, 32],
                "activation": "relu",
                "n_updates": 10  # Very few updates
            }
        }
    else:  # A2C
        config = {
            "model_type": "A2C",
            "name": f"{model_name}_test",
            "hyperparameters": {
                "learning_rate": 0.001,
                "hidden_layers": [64, 32],
                "activation": "relu",
                "n_steps": 100  # Fewer steps
            }
        }
    
    model = model_class(config)
    
    # Build the model
    model.build(input_shape=(10,), output_shape=(2,))
    
    # Create engine
    engine = TrainingEngine(
        checkpoint_dir=f"test_checkpoints/{model_name.lower()}_quick"
    )
    
    # Prepare callbacks
    callbacks = [
        LoggingCallback(),
        ModelCheckpointCallback(
            checkpoint_dir=f"test_checkpoints/{model_name.lower()}_quick",
            save_frequency=1
        )
    ]
    
    # Prepare data config
    import pandas as pd
    train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(10)])
    train_df["target"] = y_train
    val_df = pd.DataFrame(X_val, columns=[f"feature_{i}" for i in range(10)])
    val_df["target"] = y_val
    
    data_config = {
        "train_data": train_df,
        "val_data": val_df
    }
    
    # Training configuration
    training_config = {
        "epochs": 2,
        "batch_size": 32,
        "validation_split": 0.0,  # We're providing validation data directly
        "verbose": 1
    }
    
    # Train for just 2 epochs
    try:
        result = engine.train(
            model_config=config,
            data_config=data_config,
            training_config=training_config,
            callbacks=callbacks
        )
        
        if result["success"]:
            print(f"✅ {model_name} training completed successfully!")
            print(f"   Epochs trained: {result['epochs_trained']}")
            return True
        else:
            print(f"❌ {model_name} training failed: {result.get('error', 'Unknown error')}")
            return False
        print(f"✅ {model_name} training completed successfully!")
        print(f"   Final loss: {history['loss'][-1]:.4f}")
        return True
    except Exception as e:
        print(f"❌ {model_name} training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick tests for all models"""
    print("Quick TrainingEngine Test with Fixed Models")
    print("="*50)
    
    results = {}
    
    # Test each model
    for model_class, model_name in [(DQN, "DQN"), (PPO, "PPO"), (A2C, "A2C")]:
        results[model_name] = test_model_with_engine(model_class, model_name)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All models work correctly with TrainingEngine!")
    else:
        print("\n⚠️ Some models failed. Check the errors above.")

if __name__ == "__main__":
    main()