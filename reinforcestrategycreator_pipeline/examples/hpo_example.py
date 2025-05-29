"""Example script demonstrating Hyperparameter Optimization (HPO) usage."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training import TrainingEngine, HPOptimizer
from src.models.factory import ModelFactory
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalArtifactStore
from src.data.manager import DataManager
from src.config.loader import ConfigLoader


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_data():
    """Create sample data for demonstration."""
    import numpy as np
    
    # Create synthetic trading data
    n_samples = 1000
    n_features = 10
    
    # Features: price, volume, technical indicators, etc.
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 3, n_samples)  # 3 actions: buy, hold, sell
    
    X_val = np.random.randn(200, n_features)
    y_val = np.random.randint(0, 3, 200)
    
    return {
        "train_data": (X_train, y_train),
        "val_data": (X_val, y_val)
    }


def example_basic_hpo():
    """Basic HPO example with minimal configuration."""
    print("\n=== Basic HPO Example ===")
    
    # Initialize components
    model_factory = ModelFactory()
    artifact_store = LocalArtifactStore(base_path="./artifacts")
    
    training_engine = TrainingEngine(
        model_factory=model_factory,
        artifact_store=artifact_store,
        checkpoint_dir="./checkpoints"
    )
    
    hpo_optimizer = HPOptimizer(
        training_engine=training_engine,
        artifact_store=artifact_store,
        results_dir="./hpo_results"
    )
    
    # Define configurations
    model_config = {
        "type": "ppo",
        "name": "trading_ppo",
        "hyperparameters": {
            "learning_rate": 0.001,  # Will be overridden by HPO
            "n_steps": 128,
            "batch_size": 64
        }
    }
    
    data_config = create_sample_data()
    
    training_config = {
        "epochs": 5,  # Short for demo
        "batch_size": 32,
        "validation_split": 0.2
    }
    
    # Define search space
    param_space = {
        "learning_rate": {
            "type": "loguniform",
            "low": 0.00001,
            "high": 0.01
        },
        "n_steps": {
            "type": "choice",
            "values": [64, 128, 256]
        },
        "batch_size": {
            "type": "choice",
            "values": [32, 64, 128]
        }
    }
    
    # Run optimization
    results = hpo_optimizer.optimize(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        param_space=param_space,
        num_trials=3,  # Small number for demo
        search_algorithm="random",
        metric="loss",
        mode="min",
        name="basic_hpo_demo"
    )
    
    print(f"\nBest parameters: {results['best_params']}")
    print(f"Best score: {results['best_score']}")
    
    # Analyze results
    analysis = hpo_optimizer.analyze_results(results)
    print(f"\nTop trials:")
    for trial in analysis["top_k_trials"]:
        print(f"  Rank {trial['rank']}: {trial['metric']:.4f} - {trial['params']}")


def example_advanced_hpo():
    """Advanced HPO example with Optuna and ASHA scheduler."""
    print("\n=== Advanced HPO Example ===")
    
    # Initialize components with registry
    model_factory = ModelFactory()
    model_registry = ModelRegistry(storage_backend=LocalArtifactStore("./model_registry"))
    artifact_store = LocalArtifactStore(base_path="./artifacts")
    
    training_engine = TrainingEngine(
        model_factory=model_factory,
        model_registry=model_registry,
        artifact_store=artifact_store
    )
    
    hpo_optimizer = HPOptimizer(
        training_engine=training_engine,
        artifact_store=artifact_store,
        results_dir="./hpo_results"
    )
    
    # Load configuration from YAML
    config_loader = ConfigLoader()
    hpo_config = config_loader.load_config("configs/base/hpo.yaml")
    
    # Model configuration
    model_config = {
        "type": "dqn",
        "name": "trading_dqn",
        "hyperparameters": {
            # Default values - will be overridden by HPO
            "learning_rate": 0.001,
            "buffer_size": 100000,
            "batch_size": 32,
            "gamma": 0.99
        }
    }
    
    data_config = create_sample_data()
    
    training_config = {
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.2,
        "save_checkpoints": True,
        "save_best_only": True,
        "monitor": "val_loss"
    }
    
    # Use predefined search space from config
    param_space = hpo_config["search_spaces"]["dqn"]
    param_mapping = hpo_config["param_mappings"]["dqn"]
    
    # Run optimization with advanced settings
    results = hpo_optimizer.optimize(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        param_space=param_space,
        num_trials=10,
        max_concurrent_trials=2,
        search_algorithm="optuna",
        scheduler="asha",
        metric="val_loss",
        mode="min",
        param_mapping=param_mapping,
        resources_per_trial={"cpu": 2},
        name="advanced_hpo_demo"
    )
    
    # Get best model configuration
    best_model_config = hpo_optimizer.get_best_model_config(
        model_config, param_mapping
    )
    
    print(f"\nBest model configuration:")
    print(f"  Type: {best_model_config['type']}")
    print(f"  Hyperparameters: {best_model_config['hyperparameters']}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_result = training_engine.train(
        model_config=best_model_config,
        data_config=data_config,
        training_config={
            "epochs": 20,  # More epochs for final training
            "batch_size": 32,
            "validation_split": 0.2
        }
    )
    
    if final_result["success"]:
        print(f"Final model trained successfully!")
        print(f"Final validation loss: {final_result['final_metrics'].get('val_loss', 'N/A')}")


def example_experiment_preset():
    """Example using predefined experiment presets."""
    print("\n=== Experiment Preset Example ===")
    
    # Load HPO configuration
    config_loader = ConfigLoader()
    hpo_config = config_loader.load_config("configs/base/hpo.yaml")
    
    # Initialize components
    training_engine = TrainingEngine()
    hpo_optimizer = HPOptimizer(training_engine=training_engine)
    
    # Use quick test preset
    quick_test_settings = hpo_config["experiments"]["quick_test"]
    
    model_config = {
        "type": "a2c",
        "name": "trading_a2c",
        "hyperparameters": {}
    }
    
    # Run with preset
    results = hpo_optimizer.optimize(
        model_config=model_config,
        data_config=create_sample_data(),
        training_config={"epochs": 5},
        param_space=hpo_config["search_spaces"]["a2c"],
        **quick_test_settings,  # Unpack preset settings
        metric="loss",
        mode="min",
        name="preset_demo"
    )
    
    print(f"\nQuick test completed with {results['num_trials']} trials")
    print(f"Best parameters found: {results['best_params']}")


def example_resume_from_results():
    """Example of loading and continuing from previous results."""
    print("\n=== Resume from Results Example ===")
    
    # Initialize optimizer
    training_engine = TrainingEngine()
    hpo_optimizer = HPOptimizer(training_engine=training_engine)
    
    # Check if previous results exist
    results_dir = Path("./hpo_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*_results.json"))
        if result_files:
            # Load most recent results
            latest_results = sorted(result_files)[-1]
            print(f"Loading results from: {latest_results}")
            
            results = hpo_optimizer.load_results(latest_results)
            
            # Analyze loaded results
            analysis = hpo_optimizer.analyze_results(results)
            
            print(f"\nLoaded optimization results:")
            print(f"  Total trials: {analysis['total_trials']}")
            print(f"  Best score: {results['best_score']}")
            print(f"  Parameter importance:")
            for param, importance in list(analysis['parameter_importance'].items())[:5]:
                print(f"    {param}: {importance:.3f}")
        else:
            print("No previous results found")
    else:
        print("No results directory found")


def main():
    """Run all examples."""
    setup_logging()
    
    # Note: These examples use mock data and simplified models
    # In practice, you would use real trading data and properly configured models
    
    try:
        # Run basic example
        example_basic_hpo()
        
        # Run advanced example (requires Ray Tune and Optuna)
        # Uncomment to run:
        # example_advanced_hpo()
        
        # Run preset example
        # example_experiment_preset()
        
        # Run resume example
        example_resume_from_results()
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Make sure to install required dependencies:")
        print("  pip install 'ray[tune]' optuna")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()