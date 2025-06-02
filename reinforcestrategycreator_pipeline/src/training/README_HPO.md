# Hyperparameter Optimization (HPO) Module

This module provides hyperparameter optimization capabilities for the ML pipeline using Ray Tune as the primary optimization engine.

## Features

- **Multiple Search Algorithms**: Random search, Optuna integration, and extensible to other algorithms
- **Advanced Schedulers**: ASHA (Asynchronous Successive Halving) and PBT (Population Based Training)
- **Parallel Trial Execution**: Efficient resource utilization with configurable concurrency
- **Comprehensive Results Analysis**: Built-in analysis tools and visualization support
- **Integration with Training Engine**: Seamless integration with the pipeline's TrainingEngine
- **Configurable Search Spaces**: YAML-based configuration for different model types

## Installation

```bash
# Core HPO dependencies
pip install 'ray[tune]>=2.9.0' optuna>=3.5.0

# Optional visualization dependencies
pip install matplotlib>=3.5.0 seaborn>=0.12.0
```

## Quick Start

```python
from src.training import TrainingEngine, HPOptimizer

# Initialize components
training_engine = TrainingEngine()
hpo_optimizer = HPOptimizer(training_engine=training_engine)

# Define search space
param_space = {
    "learning_rate": {
        "type": "loguniform",
        "low": 0.00001,
        "high": 0.01
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
    num_trials=20,
    metric="val_loss",
    mode="min"
)

# Get best configuration
best_config = hpo_optimizer.get_best_model_config(model_config)
```

## Configuration

### Search Space Definition

Search spaces can be defined programmatically or loaded from YAML configuration:

```yaml
# configs/base/hpo.yaml
search_spaces:
  ppo:
    learning_rate:
      type: "loguniform"
      low: 0.00001
      high: 0.01
    n_steps:
      type: "choice"
      values: [128, 256, 512, 1024]
```

### Supported Parameter Types

- `uniform`: Uniform distribution between low and high
- `loguniform`: Log-uniform distribution (good for learning rates)
- `choice`: Discrete choice from a list of values
- `randint`: Random integer between low and high
- `quniform`: Quantized uniform distribution

### Search Algorithms

1. **Random Search** (default)
   - Simple and robust
   - Good for initial exploration

2. **Optuna**
   - Bayesian optimization
   - More efficient for complex search spaces
   - Requires: `pip install optuna`

3. **Custom Algorithms**
   - Extensible to support other algorithms

### Schedulers

1. **ASHA (Asynchronous Successive Halving)**
   - Early stopping of poor trials
   - Efficient resource allocation
   - Good for most use cases

2. **PBT (Population Based Training)**
   - Dynamic hyperparameter adjustment
   - Good for long training runs
   - Requires careful configuration

## Advanced Usage

### Parameter Mapping

For complex model configurations, use parameter mapping:

```python
param_mapping = {
    "lr": "hyperparameters.learning_rate",
    "hidden": "model.layers.hidden_units"
}

results = hpo_optimizer.optimize(
    model_config=model_config,
    param_space=param_space,
    param_mapping=param_mapping,
    # ... other parameters
)
```

### Resource Allocation

Configure resources per trial:

```python
results = hpo_optimizer.optimize(
    # ... other parameters
    resources_per_trial={
        "cpu": 2,
        "gpu": 0.5  # Fractional GPU allocation
    },
    max_concurrent_trials=4
)
```

### Results Analysis

```python
# Analyze results
analysis = hpo_optimizer.analyze_results(results)

print(f"Best score: {analysis['best_trial']['score']}")
print(f"Parameter importance: {analysis['parameter_importance']}")
print(f"Top trials: {analysis['top_k_trials']}")
```

### Visualization

```python
from src.training import HPOVisualizer

visualizer = HPOVisualizer()

# Plot optimization history
visualizer.plot_optimization_history(results)

# Plot parameter importance
visualizer.plot_parameter_importance(analysis)

# Create comprehensive report
visualizer.create_summary_report(
    results, 
    analysis, 
    output_dir="./hpo_reports/run_001"
)
```

## Best Practices

1. **Start Small**: Begin with a small number of trials to validate your setup
2. **Use Appropriate Scales**: Use log-scale for parameters like learning rate
3. **Set Resource Limits**: Configure max_concurrent_trials based on available resources
4. **Monitor Progress**: Use Ray Dashboard or logs to monitor optimization progress
5. **Save Results**: Always save HPO results for reproducibility

## Experiment Presets

The module includes predefined experiment configurations:

```python
# Load HPO configuration
from src.config.loader import ConfigLoader
config_loader = ConfigLoader()
hpo_config = config_loader.load_config("configs/base/hpo.yaml")

# Use a preset
quick_test = hpo_config["experiments"]["quick_test"]
results = hpo_optimizer.optimize(
    model_config=model_config,
    data_config=data_config,
    training_config=training_config,
    param_space=hpo_config["search_spaces"]["ppo"],
    **quick_test  # Unpack preset settings
)
```

Available presets:
- `quick_test`: 5 trials for testing
- `standard`: 50 trials with Optuna and ASHA
- `extensive`: 200 trials with PBT
- `production`: 100 trials optimized for production

## Troubleshooting

### Common Issues

1. **Import Error for Ray**
   ```
   ImportError: Ray Tune is required for HPOptimizer
   ```
   Solution: Install Ray Tune with `pip install 'ray[tune]'`

2. **Resource Allocation Errors**
   - Ensure you're not requesting more resources than available
   - Check Ray initialization parameters

3. **Memory Issues**
   - Reduce max_concurrent_trials
   - Use ASHA scheduler for early stopping
   - Monitor system resources

### Debugging Tips

1. Enable verbose logging:
   ```python
   import logging
   logging.getLogger("HPOptimizer").setLevel(logging.DEBUG)
   ```

2. Use Ray Dashboard:
   ```python
   # Initialize Ray with dashboard
   ray_config = {"include_dashboard": True}
   hpo_optimizer = HPOptimizer(ray_config=ray_config)
   ```

3. Start with simple search spaces and gradually increase complexity

## API Reference

### HPOptimizer

Main class for hyperparameter optimization.

#### Methods

- `optimize()`: Run hyperparameter optimization
- `define_search_space()`: Define and validate search space
- `analyze_results()`: Analyze optimization results
- `get_best_model_config()`: Get configuration with best parameters
- `load_results()`: Load results from file

### HPOVisualizer

Visualization utilities for HPO results.

#### Methods

- `plot_optimization_history()`: Plot metric improvement over trials
- `plot_parameter_importance()`: Plot parameter importance scores
- `plot_parallel_coordinates()`: Plot parallel coordinates for top trials
- `plot_parameter_distributions()`: Plot parameter value distributions
- `plot_metric_vs_parameter()`: Plot metric vs specific parameter
- `create_summary_report()`: Generate comprehensive report with plots

## Examples

See `examples/hpo_example.py` for complete working examples including:
- Basic HPO with random search
- Advanced HPO with Optuna and ASHA
- Using experiment presets
- Loading and analyzing previous results
- Creating visualization reports