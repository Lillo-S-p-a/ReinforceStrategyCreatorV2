"""Example usage of the CrossValidator for model evaluation."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the cross-validation components
from src.evaluation import CrossValidator, CVVisualizer
from src.models.factory import ModelFactory
from src.data.manager import DataManager
from src.data.splitter import DataSplitter


def generate_sample_data(n_samples=1000, n_features=10, noise=0.1):
    """Generate sample regression data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some linear relationship
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise * np.random.randn(n_samples)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add timestamp for time series CV
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    return df


def example_basic_cv():
    """Example: Basic cross-validation with a single model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Cross-Validation")
    print("="*60)
    
    # Generate data
    data = generate_sample_data()
    
    # Initialize CrossValidator
    cv = CrossValidator(
        checkpoint_dir="./cv_results/basic_example"
    )
    
    # Define model configuration
    model_config = {
        "type": "dqn",  # Using DQN as example
        "name": "example_model",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "gamma": 0.99
        }
    }
    
    # Define data configuration
    data_config = {
        "data": data
    }
    
    # Define CV configuration
    cv_config = {
        "method": "kfold",
        "n_folds": 5,
        "shuffle": True,
        "random_seed": 42
    }
    
    # Define training configuration
    training_config = {
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.0,  # We're using CV folds for validation
        "verbose": 0  # Suppress training output
    }
    
    # Run cross-validation
    results = cv.cross_validate(
        model_config=model_config,
        data_config=data_config,
        cv_config=cv_config,
        training_config=training_config,
        metrics=["loss", "mae"],  # Metrics to track
        scoring_metric="loss",    # Metric for model selection
        scoring_mode="min",       # Minimize the scoring metric
        save_models=False         # Don't save intermediate models
    )
    
    # Print results summary
    print(f"\nCross-validation completed in {results.total_time:.2f} seconds")
    print(f"Best fold: {results.best_fold_idx + 1}")
    
    print("\nAggregated Metrics:")
    for metric, stats in results.aggregated_metrics.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return results


def example_time_series_cv():
    """Example: Time series cross-validation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Time Series Cross-Validation")
    print("="*60)
    
    # Generate time series data
    data = generate_sample_data(n_samples=2000)
    
    # Initialize CrossValidator
    cv = CrossValidator(
        checkpoint_dir="./cv_results/time_series_example"
    )
    
    # Model configuration
    model_config = {
        "type": "ppo",
        "name": "time_series_model",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64
        }
    }
    
    # CV configuration for time series
    cv_config = {
        "method": "time_series",
        "n_folds": 4  # Creates expanding window folds
    }
    
    # Run cross-validation
    results = cv.cross_validate(
        model_config=model_config,
        data_config={"data": data},
        cv_config=cv_config,
        training_config={"epochs": 5},
        metrics=["loss"],
        scoring_metric="loss"
    )
    
    print(f"\nTime series CV completed with {len(results.fold_results)} folds")
    
    # Show fold sizes (demonstrating expanding window)
    for i, fold_result in enumerate(results.fold_results):
        print(f"  Fold {i+1} training time: {fold_result.training_time:.2f}s")
    
    return results


def example_model_comparison():
    """Example: Comparing multiple models using CV."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Comparison")
    print("="*60)
    
    # Generate data
    data = generate_sample_data()
    
    # Initialize CrossValidator with parallel execution
    cv = CrossValidator(
        checkpoint_dir="./cv_results/comparison_example",
        n_jobs=2,  # Use 2 parallel jobs
        use_multiprocessing=False  # Use threading
    )
    
    # Define multiple model configurations to compare
    model_configs = [
        {
            "type": "dqn",
            "name": "DQN_baseline",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        },
        {
            "type": "ppo",
            "name": "PPO_baseline",
            "hyperparameters": {
                "learning_rate": 0.0003,
                "n_steps": 2048
            }
        },
        {
            "type": "a2c",
            "name": "A2C_baseline",
            "hyperparameters": {
                "learning_rate": 0.0007,
                "n_steps": 5
            }
        }
    ]
    
    # Compare models
    comparison_results = cv.compare_models(
        model_configs=model_configs,
        data_config={"data": data},
        cv_config={"n_folds": 3},
        training_config={"epochs": 5},
        metrics=["loss"],
        scoring_metric="loss",
        scoring_mode="min"
    )
    
    # Print comparison summary
    print("\nModel Comparison Results:")
    for model_name, cv_results in comparison_results.items():
        val_loss = cv_results.aggregated_metrics.get("val_loss", {})
        print(f"  {model_name}: {val_loss.get('mean', 'N/A'):.4f} ± {val_loss.get('std', 'N/A'):.4f}")
    
    return comparison_results


def example_visualization():
    """Example: Visualizing cross-validation results."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Visualization")
    print("="*60)
    
    # First, run a basic CV to get results
    data = generate_sample_data()
    cv = CrossValidator(checkpoint_dir="./cv_results/visualization_example")
    
    results = cv.cross_validate(
        model_config={"type": "dqn", "name": "viz_example"},
        data_config={"data": data},
        cv_config={"n_folds": 5},
        training_config={"epochs": 10},
        metrics=["loss", "mae"]
    )
    
    # Initialize visualizer
    viz = CVVisualizer(figsize=(12, 8))
    
    # Create visualization directory
    viz_dir = Path("./cv_results/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate various plots
    print("\nGenerating visualizations...")
    
    # 1. Fold metrics bar chart
    viz.plot_fold_metrics(
        results,
        metrics=["train_loss", "val_loss"],
        save_path=viz_dir / "fold_metrics.png"
    )
    print("  ✓ Fold metrics plot saved")
    
    # 2. Metric distribution box plot
    viz.plot_metric_distribution(
        results,
        save_path=viz_dir / "metric_distribution.png"
    )
    print("  ✓ Metric distribution plot saved")
    
    # 3. Train vs validation comparison
    viz.plot_train_val_comparison(
        results,
        metric="loss",
        save_path=viz_dir / "train_val_comparison.png"
    )
    print("  ✓ Train/validation comparison plot saved")
    
    # 4. Create comprehensive report
    report_dir = viz.create_cv_report(
        results,
        output_dir=viz_dir / "full_report"
    )
    print(f"  ✓ Full report saved to {report_dir}")
    
    return results


def example_advanced_features():
    """Example: Advanced cross-validation features."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Advanced Features")
    print("="*60)
    
    # Generate stratified data
    data = generate_sample_data(n_samples=1000)
    # Add categorical target for stratification
    data['category'] = pd.qcut(data['target'], q=3, labels=['low', 'medium', 'high'])
    
    # Initialize CrossValidator
    cv = CrossValidator(
        checkpoint_dir="./cv_results/advanced_example",
        n_jobs=-1  # Use all available CPUs
    )
    
    # 1. Stratified CV
    print("\n1. Running stratified cross-validation...")
    stratified_results = cv.cross_validate(
        model_config={"type": "dqn", "name": "stratified_model"},
        data_config={"data": data},
        cv_config={
            "method": "stratified",
            "n_folds": 4,
            "target_column": "category"
        },
        training_config={"epochs": 5},
        save_models=True  # Save trained models
    )
    print(f"  Stratified CV completed with balanced folds")
    
    # 2. Multi-metric evaluation
    print("\n2. Multi-metric evaluation...")
    multi_metric_results = cv.cross_validate(
        model_config={"type": "ppo", "name": "multi_metric_model"},
        data_config={"data": data},
        cv_config={"n_folds": 3},
        training_config={"epochs": 5},
        metrics=["loss", "mae", "mse", "r2"],  # Multiple metrics
        scoring_metric="r2",  # Use R² for selection
        scoring_mode="max"    # Maximize R²
    )
    
    print("  Metrics tracked:")
    for metric in multi_metric_results.aggregated_metrics.keys():
        print(f"    - {metric}")
    
    return stratified_results, multi_metric_results


def main():
    """Run all examples."""
    print("Cross-Validation Examples")
    print("========================\n")
    
    # Note: These examples use mock models for demonstration
    # In practice, you would need actual model implementations
    
    try:
        # Run examples
        example_basic_cv()
        example_time_series_cv()
        example_model_comparison()
        example_visualization()
        example_advanced_features()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the ./cv_results directory for outputs")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Note: These examples require the model implementations to be available.")
        print("The mock models in the tests can be used for demonstration purposes.")


if __name__ == "__main__":
    main()