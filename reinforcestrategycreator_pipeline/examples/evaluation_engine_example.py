"""Example script demonstrating the Evaluation Engine usage.

This script shows how to:
1. Set up the evaluation engine with required dependencies
2. Evaluate a trained model on test data
3. Compare performance against benchmark strategies
4. Generate reports in multiple formats
5. Save and retrieve evaluation results
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.manager import ConfigManager
from src.models.registry import ModelRegistry
from src.data.manager import DataManager
from src.artifact_store.local_adapter import LocalFileSystemStore as LocalArtifactStore
from src.evaluation import EvaluationEngine
from src.models.base import ModelBase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample price data for demonstration."""
    # Generate synthetic price data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates) - 1)
    prices = [100]  # Starting price
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
    })
    
    return df


def main():
    """Main function demonstrating evaluation engine usage."""
    
    print("=== Evaluation Engine Example ===\n")
    
    # 1. Initialize components
    print("1. Initializing components...")
    
    # Initialize config manager
    config_manager = ConfigManager(
        config_dir="configs",
        environment="development"
    )
    config_manager.load_config()
    
    # Initialize artifact store
    artifact_store = LocalArtifactStore(
        root_path="./artifacts"
    )
    
    # Initialize data manager
    data_manager = DataManager(
        config_manager=config_manager,
        artifact_store=artifact_store
    )
    
    # Initialize model registry
    model_registry = ModelRegistry(artifact_store=artifact_store)
    
    # 2. Register sample data source
    print("\n2. Registering data source...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to CSV for data source
    data_path = Path("./temp_data/sample_prices.csv")
    data_path.parent.mkdir(exist_ok=True)
    sample_data.to_csv(data_path, index=False)
    
    # Register CSV data source
    data_manager.register_source(
        source_id="sample_prices",
        source_type="csv",
        config={
            "file_path": str(data_path),
            "parse_dates": ["date"]
        }
    )
    
    # 3. Create and register a mock trained model
    print("\n3. Creating mock trained model...")
    
    # For demonstration, we'll create a simple model
    # In practice, this would be a trained RL model
    from src.models.factory import get_factory
    
    factory = get_factory()
    
    # Register a mock model type for demonstration
    class MockModel(ModelBase):
        def __init__(self, config):
            super().__init__(config)
            self.is_trained = True
            self.model_state = {"weights": np.random.randn(10, 10)}
        
        def build(self, input_shape, output_shape):
            """Mock build method."""
            pass
        
        def train(self, train_data, validation_data=None, **kwargs):
            """Mock train method."""
            return {"loss": 0.1, "accuracy": 0.95}
        
        def predict(self, data, **kwargs):
            """Mock predict method."""
            # Return random actions for demonstration
            return np.random.randint(0, 4, size=len(data))
        
        def evaluate(self, test_data, **kwargs):
            """Mock evaluate method."""
            return {"test_loss": 0.15, "test_accuracy": 0.92}
        
        def get_model_state(self):
            """Get model state for serialization."""
            return self.model_state
        
        def set_model_state(self, state):
            """Set model state from deserialization."""
            self.model_state = state
    
    # Register the mock model type
    factory.register_model("mock_model", MockModel)
    
    # Create and register model
    mock_model = MockModel({"name": "test_model", "model_type": "mock_model"})
    model_id = model_registry.register_model(
        model=mock_model,
        model_name="demo_model",
        description="Demo model for evaluation example",
        metrics={"training_sharpe": 1.5},
        tags=["demo", "example"]
    )
    
    print(f"Registered model with ID: {model_id}")
    
    # 4. Initialize evaluation engine
    print("\n4. Initializing evaluation engine...")
    
    evaluation_engine = EvaluationEngine(
        model_registry=model_registry,
        data_manager=data_manager,
        artifact_store=artifact_store,
        metrics_config={
            "risk_free_rate": 0.02,
            "trading_days_per_year": 252,
            "sharpe_window_size": 60
        },
        benchmark_config={
            "initial_balance": 10000,
            "transaction_fee": 0.001,
            "sma_short_window": 20,
            "sma_long_window": 50,
            "random_trade_probability": 0.05,
            "random_seed": 42
        }
    )
    
    # 5. Run evaluation
    print("\n5. Running model evaluation...")
    
    try:
        results = evaluation_engine.evaluate(
            model_id=model_id,
            data_source_id="sample_prices",
            metrics=[
                "pnl", "pnl_percentage", "sharpe_ratio", "sortino_ratio",
                "max_drawdown", "calmar_ratio", "volatility"
            ],
            compare_benchmarks=True,
            save_results=True,
            report_formats=["markdown", "html"],
            evaluation_name="Demo Model Evaluation",
            tags=["demo", "example", "benchmark_comparison"]
        )
        
        print("\nEvaluation completed successfully!")
        
        # 6. Display results
        print("\n6. Evaluation Results:")
        print("-" * 50)
        
        # Model metrics
        print("\nModel Performance Metrics:")
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Benchmark comparison
        if "benchmarks" in results:
            print("\n\nBenchmark Performance:")
            print("-" * 50)
            
            for strategy, metrics in results["benchmarks"].items():
                print(f"\n{strategy.replace('_', ' ').title()}:")
                print(f"  PnL: ${metrics.get('pnl', 0):.2f}")
                print(f"  PnL %: {metrics.get('pnl_percentage', 0):.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
                print(f"  Trades: {metrics.get('trades', 0)}")
            
            print("\n\nRelative Performance vs Benchmarks:")
            print("-" * 50)
            
            for strategy, perf in results["relative_performance"].items():
                print(f"\nvs {strategy.replace('_', ' ').title()}:")
                print(f"  PnL Difference: ${perf['absolute_difference']:.2f}")
                print(f"  PnL % Difference: {perf['percentage_difference']:+.2f}%")
                print(f"  Sharpe Difference: {perf['sharpe_ratio_difference']:+.4f}")
        
        # 7. Save reports
        print("\n\n7. Saving evaluation reports...")
        
        reports_dir = Path("./evaluation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        for format_type, content in results["reports"].items():
            ext = "json" if format_type == "json" else format_type
            report_path = reports_dir / f"demo_evaluation.{ext}"
            
            with open(report_path, "w") as f:
                f.write(content)
            
            print(f"  Saved {format_type} report to: {report_path}")
        
        # 8. List and load evaluations
        print("\n\n8. Listing saved evaluations...")
        
        evaluations = evaluation_engine.list_evaluations(
            model_id=model_id,
            tags=["demo"]
        )
        
        print(f"\nFound {len(evaluations)} evaluation(s):")
        for eval_info in evaluations:
            print(f"  - {eval_info['evaluation_id']}")
            print(f"    Created: {eval_info['created_at']}")
            print(f"    Model: {eval_info['model_id']} (v{eval_info['model_version']})")
            print(f"    PnL: ${eval_info['metrics'].get('pnl', 0):.2f}")
        
        # Load the evaluation
        if evaluations:
            print("\n\n9. Loading evaluation results...")
            loaded_results = evaluation_engine.load_evaluation(
                evaluations[0]['evaluation_id']
            )
            print(f"  Successfully loaded evaluation: {loaded_results['evaluation_id']}")
            print(f"  Timestamp: {loaded_results['timestamp']}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
    
    finally:
        # Cleanup
        print("\n\n10. Cleaning up temporary files...")
        
        # Remove temporary data file
        if data_path.exists():
            data_path.unlink()
        if data_path.parent.exists():
            data_path.parent.rmdir()
        
        print("\nExample completed!")


if __name__ == "__main__":
    main()