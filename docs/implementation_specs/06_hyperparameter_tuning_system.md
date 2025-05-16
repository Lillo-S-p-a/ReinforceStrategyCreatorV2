# Hyperparameter Tuning System: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Hyperparameter Tuning System of the Trading Model Optimization Pipeline. This component is responsible for systematically searching optimal hyperparameters for trading models through various optimization techniques, including Bayesian optimization, grid search, random search, and evolutionary algorithms.

## 2. Component Responsibilities

The Hyperparameter Tuning System is responsible for:

- Defining hyperparameter search spaces for models
- Implementing various hyperparameter search strategies
- Coordinating parallel/distributed hyperparameter trials
- Tracking and comparing trial results
- Implementing early stopping for unpromising trials
- Providing visualizations of the hyperparameter optimization process
- Suggesting optimal hyperparameter configurations
- Integrating with Ray Tune for distributed optimization

## 3. Architecture

### 3.1 Overall Architecture

The Hyperparameter Tuning System follows a layered architecture with clear separation of concerns:

```
┌───────────────────────────────────┐
│        Tuning Manager             │  High-level API for tuning operations
├───────────────────────────────────┤
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │   Search    │ │   Trial     │  │  Core components for search strategies
│  │  Strategies │ │  Execution  │  │  and trial management
│  └─────────────┘ └─────────────┘  │
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │   Search    │ │  Analysis   │  │  Components for space definition
│  │   Spaces    │ │     &       │  │  and results analysis
│  │             │ │ Visualization│  │
│  └─────────────┘ └─────────────┘  │
│                                   │
├───────────────────────────────────┤
│          Ray Tune Integration     │  Support for distributed execution
└───────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── hyperparameter/
    ├── __init__.py
    ├── manager.py            # High-level tuning management interface
    ├── search/
    │   ├── __init__.py
    │   ├── base.py           # Abstract base search strategy
    │   ├── grid.py           # Grid search implementation
    │   ├── random.py         # Random search implementation
    │   ├── bayesian.py       # Bayesian optimization
    │   ├── hyperband.py      # HyperBand algorithm implementation
    │   ├── evolutionary.py   # Evolutionary algorithms
    │   ├── factory.py        # Search strategy factory
    │   └── ray_strategies.py # Ray Tune integration
    ├── space/
    │   ├── __init__.py
    │   ├── parameter.py      # Parameter definition classes
    │   ├── distributions.py  # Probability distributions for sampling
    │   ├── constraints.py    # Parameter constraints implementation
    │   ├── transformers.py   # Parameter transformation utilities
    │   └── factory.py        # Search space factory
    ├── trial/
    │   ├── __init__.py
    │   ├── executor.py       # Trial execution management
    │   ├── scheduler.py      # Scheduling for parallel trials
    │   ├── pruner.py         # Early stopping mechanisms
    │   └── reporter.py       # Trial progress reporting
    ├── analysis/
    │   ├── __init__.py
    │   ├── evaluator.py      # Trial evaluation utilities
    │   ├── comparator.py     # Trial comparison tools
    │   ├── visualizer.py     # Visualization utilities
    │   └── reporter.py       # Results reporting
    ├── utils/
    │   ├── __init__.py
    │   └── serialization.py  # Serialization for search spaces and results
    └── ray_integration.py    # Ray Tune integration utilities
```

## 4. Core Components Design

### 4.1 Tuning Manager

The high-level interface for hyperparameter tuning management:

```python
# manager.py
from typing import Dict, List, Any, Optional, Union, Callable
import os
import json
import uuid
from datetime import datetime

import ray
from ray import tune
import pandas as pd
import numpy as np

from trading_optimization.hyperparameter.search.factory import SearchStrategyFactory
from trading_optimization.hyperparameter.space.parameter import SearchSpace
from trading_optimization.hyperparameter.trial.executor import TrialExecutor
from trading_optimization.hyperparameter.analysis.evaluator import TrialEvaluator
from trading_optimization.hyperparameter.utils.serialization import save_tuning_results
from trading_optimization.config import ConfigManager

class TuningManager:
    """
    High-level interface for hyperparameter tuning management.
    Acts as a facade for all tuning operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Tuning Manager with configuration settings.
        
        Args:
            config: Configuration dictionary with tuning settings
        """
        self.config = config
        self.search_strategy_factory = SearchStrategyFactory()
        
        # Set up database connection if available
        try:
            from trading_optimization.db.connectors import DatabaseConnector
            db_connector = DatabaseConnector.instance()
            with db_connector.session() as session:
                from trading_optimization.db.repository import HyperparameterResultRepository
                self.result_repo = HyperparameterResultRepository(session)
        except Exception as e:
            print(f"Warning: Could not connect to database: {str(e)}")
            self.result_repo = None
    
    def create_search_space(
        self, 
        parameters: Dict[str, Any]
    ) -> SearchSpace:
        """
        Create a hyperparameter search space.
        
        Args:
            parameters: Dictionary defining parameters and their search spaces
            
        Returns:
            SearchSpace object
        """
        from trading_optimization.hyperparameter.space.factory import create_search_space
        
        search_space = create_search_space(parameters)
        return search_space
    
    def tune(
        self,
        model_type: str,
        search_space: SearchSpace,
        objective: Union[str, Callable],
        strategy: str = "bayesian",
        strategy_config: Optional[Dict[str, Any]] = None,
        max_trials: int = 10,
        timeout: Optional[int] = None,
        n_concurrent: int = 1,
        use_ray: bool = True,
        checkpoint_dir: Optional[str] = None,
        data_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            model_type: Type of model to tune
            search_space: Search space definition
            objective: Objective function name or callable
            strategy: Search strategy name
            strategy_config: Configuration for search strategy
            max_trials: Maximum number of trials
            timeout: Maximum runtime in seconds
            n_concurrent: Number of concurrent trials
            use_ray: Whether to use Ray for distributed execution
            checkpoint_dir: Directory for checkpoints
            data_config: Configuration for data loading
            training_config: Configuration for training
            
        Returns:
            Dictionary with tuning results
        """
        # Generate unique ID for this tuning session
        tuning_id = str(uuid.uuid4())
        
        # Create checkpoint directory if needed
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create search strategy
        strategy_config = strategy_config or {}
        search_strategy = self.search_strategy_factory.create_strategy(
            strategy, **strategy_config
        )
        
        # Prepare configs
        max_trials = max_trials or self.config.get('max_trials', 10)
        n_concurrent = n_concurrent or self.config.get('n_concurrent', 1)
        
        # Prepare the objective function
        if isinstance(objective, str):
            objective_fn = self._create_objective_function(
                objective, 
                model_type, 
                data_config or {}, 
                training_config or {}
            )
        else:
            objective_fn = objective
        
        # Determine whether to use Ray
        use_ray = use_ray and self._is_ray_available()
        
        if use_ray:
            # Use Ray Tune for distributed hyperparameter tuning
            tuning_results = self._tune_with_ray(
                tuning_id,
                search_strategy,
                search_space,
                objective_fn,
                max_trials,
                timeout,
                n_concurrent,
                checkpoint_dir
            )
        else:
            # Use custom implementation for local hyperparameter tuning
            tuning_results = self._tune_locally(
                tuning_id,
                search_strategy,
                search_space,
                objective_fn,
                max_trials,
                timeout,
                n_concurrent,
                checkpoint_dir
            )
        
        # Store results in database if available
        if self.result_repo:
            try:
                # Log the best parameters and results
                best_params = tuning_results.get('best_params', {})
                best_result = tuning_results.get('best_result', {})
                
                # Create a database entry
                self.result_repo.create(
                    tuning_id=tuning_id,
                    model_type=model_type,
                    strategy=strategy,
                    max_trials=max_trials,
                    best_params=best_params,
                    best_value=best_result.get('objective_value', 0.0),
                    metrics=best_result
                )
            except Exception as e:
                print(f"Warning: Failed to store tuning results in database: {str(e)}")
        
        # Save results to file
        if checkpoint_dir:
            results_file = os.path.join(checkpoint_dir, f"tuning_results_{tuning_id}.json")
            save_tuning_results(tuning_results, results_file)
            
        return tuning_results
    
    def _create_objective_function(
        self,
        objective_name: str,
        model_type: str,
        data_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Callable:
        """
        Create an objective function from configuration.
        
        Args:
            objective_name: Name of the objective (e.g., 'val_loss', 'sharpe_ratio')
            model_type: Type of model
            data_config: Configuration for data loading
            training_config: Configuration for training
            
        Returns:
            Callable objective function
        """
        from trading_optimization.models.manager import ModelManager
        from trading_optimization.data.interface import DataManager
        
        # Get model and data managers
        model_config = self.config.get('models', {})
        data_manager = DataManager(self.config.get('data', {}))
        model_manager = ModelManager(model_config)
        
        def objective_function(trial_params):
            """Objective function to be minimized/maximized."""
            try:
                # Prepare data
                if 'pipeline_id' in data_config:
                    # Use existing pipeline
                    pipeline_id = data_config['pipeline_id']
                    df = data_manager.execute_pipeline(pipeline_id)
                else:
                    # Create new pipeline
                    pipeline_config = data_config.get('pipeline_config', {})
                    pipeline_id = data_manager.create_pipeline(
                        f"pipeline_for_trial_{trial_params.get('trial_id', 'unknown')}",
                        pipeline_config
                    )
                    df = data_manager.execute_pipeline(pipeline_id)
                
                # Split data
                split_config = data_config.get('split_config', {})
                data_splits = data_manager.split_data(df, **split_config)
                
                # Create data loaders
                loader_config = data_config.get('loader_config', {})
                data_loaders = data_manager.get_data_loaders(data_splits, **loader_config)
                
                # Create model with trial parameters
                model_id, model = model_manager.create_model(
                    model_type, 
                    model_name=f"trial_{trial_params.get('trial_id', 'unknown')}",
                    **trial_params
                )
                
                # Train model
                train_result = model_manager.train_model(
                    model,
                    model_id,
                    data_loaders,
                    **training_config
                )
                
                # Get objective value
                if objective_name.startswith('val_') and 'final_val_metrics' in train_result:
                    # Get from validation metrics
                    metric_name = objective_name[4:]  # Remove 'val_' prefix
                    value = train_result['final_val_metrics'].get(metric_name)
                elif objective_name.startswith('train_') and 'final_train_metrics' in train_result:
                    # Get from training metrics
                    metric_name = objective_name[6:]  # Remove 'train_' prefix
                    value = train_result['final_train_metrics'].get(metric_name)
                else:
                    # Try to find in either validation or training metrics
                    value = (
                        train_result.get('final_val_metrics', {}).get(objective_name) or
                        train_result.get('final_train_metrics', {}).get(objective_name)
                    )
                
                # Check if we're minimizing or maximizing
                if value is not None:
                    # Default to minimizing
                    minimize = training_config.get('minimize', True)
                    if not minimize:
                        # Convert to minimization problem by negating
                        value = -value
                        
                    return value
                else:
                    # If metric not found, return a large value for minimization
                    print(f"Warning: Metric '{objective_name}' not found in training results")
                    return float('inf')
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
                import traceback
                traceback.print_exc()
                return float('inf')
        
        return objective_function
    
    def _is_ray_available(self) -> bool:
        """Check if Ray is available for distributed execution."""
        try:
            import ray
            return True
        except ImportError:
            return False
    
    def _tune_with_ray(
        self,
        tuning_id: str,
        search_strategy: Any,
        search_space: SearchSpace,
        objective_fn: Callable,
        max_trials: int,
        timeout: Optional[int],
        n_concurrent: int,
        checkpoint_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Run tuning using Ray Tune.
        
        Args:
            tuning_id: Unique ID for this tuning session
            search_strategy: Search strategy object
            search_space: Search space definition
            objective_fn: Objective function to minimize
            max_trials: Maximum number of trials
            timeout: Maximum runtime in seconds
            n_concurrent: Number of concurrent trials
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Dictionary with tuning results
        """
        import ray
        from ray import tune
        from ray.tune.search import ConcurrencyLimiter
        from trading_optimization.hyperparameter.ray_integration import (
            convert_search_space_to_ray, 
            convert_search_strategy_to_ray
        )
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()
        
        # Convert search space to Ray format
        ray_search_space = convert_search_space_to_ray(search_space)
        
        # Convert search strategy to Ray format
        ray_search_alg = convert_search_strategy_to_ray(search_strategy)
        
        # Limit concurrent trials if needed
        if n_concurrent and n_concurrent > 0:
            ray_search_alg = ConcurrencyLimiter(
                ray_search_alg, max_concurrent=n_concurrent
            )
        
        # Wrap objective function for Ray
        def ray_objective(config):
            # Add trial_id to params
            params = dict(config)
            params['trial_id'] = tune.get_trial_id()
            
            # Call original objective
            result = objective_fn(params)
            
            # Report result to Ray
            tune.report(objective=result)
        
        # Set up tuning run
        tune_config = tune.TuneConfig(
            mode="min",
            num_samples=max_trials,
            search_alg=ray_search_alg,
            time_budget_s=timeout,
        )
        
        # Define run configuration
        run_config = ray.train.RunConfig(
            name=f"tuning_{tuning_id}",
            storage_path=checkpoint_dir,
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="objective",
                checkpoint_score_order="min",
            ),
            verbose=2
        )
        
        # Run the tuning
        tuner = tune.Tuner(
            ray_objective,
            param_space=ray_search_space,
            tune_config=tune_config,
            run_config=run_config
        )
        
        results = tuner.fit()
        
        # Extract best trial and parameters
        best_trial = results.get_best_trial("objective", "min")
        best_params = best_trial.config if best_trial else {}
        best_result = {
            "objective_value": best_trial.last_result.get("objective") if best_trial else None,
            "training_time": best_trial.time_total_s if best_trial else None
        }
        
        # Convert results to our format
        all_trials = []
        for trial in results.trials:
            trial_data = {
                "trial_id": trial.trial_id,
                "params": trial.config,
                "result": {
                    "objective_value": trial.last_result.get("objective") if trial.last_result else None,
                    "training_time": trial.time_total_s,
                    "status": trial.status,
                }
            }
            all_trials.append(trial_data)
        
        # Return compiled results
        return {
            "tuning_id": tuning_id,
            "best_params": best_params,
            "best_result": best_result,
            "all_trials": all_trials,
            "completed_trials": len(results.trials),
            "total_time": sum(trial.time_total_s for trial in results.trials if trial.time_total_s),
        }
    
    def _tune_locally(
        self,
        tuning_id: str,
        search_strategy: Any,
        search_space: SearchSpace,
        objective_fn: Callable,
        max_trials: int,
        timeout: Optional[int],
        n_concurrent: int,
        checkpoint_dir: Optional[str]
    ) -> Dict[str, Any]:
        """
        Run tuning using local implementation.
        
        Args:
            tuning_id: Unique ID for this tuning session
            search_strategy: Search strategy object
            search_space: Search space definition
            objective_fn: Objective function to minimize
            max_trials: Maximum number of trials
            timeout: Maximum runtime in seconds
            n_concurrent: Number of concurrent trials
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Dictionary with tuning results
        """
        trial_executor = TrialExecutor(
            search_strategy=search_strategy,
            search_space=search_space,
            objective_fn=objective_fn,
            max_trials=max_trials,
            timeout=timeout,
            n_concurrent=n_concurrent,
            checkpoint_dir=checkpoint_dir
        )
        
        # Run the tuning
        results = trial_executor.run()
        
        # Add tuning ID to results
        results["tuning_id"] = tuning_id
        
        return results
    
    def analyze_results(
        self,
        results: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperparameter tuning results.
        
        Args:
            results: Tuning results from tune()
            output_dir: Optional directory to save analysis artifacts
            
        Returns:
            Dictionary with analysis results
        """
        from trading_optimization.hyperparameter.analysis.evaluator import TrialEvaluator
        
        evaluator = TrialEvaluator()
        analysis_results = evaluator.analyze(results)
        
        if output_dir:
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the analysis results
            analysis_file = os.path.join(
                output_dir, 
                f"analysis_{results.get('tuning_id', 'unknown')}.json"
            )
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Generate and save visualizations
            from trading_optimization.hyperparameter.analysis.visualizer import TuningVisualizer
            visualizer = TuningVisualizer()
            
            # Importance plot
            importance_plot_path = os.path.join(output_dir, "parameter_importance.png")
            visualizer.plot_parameter_importance(results, save_path=importance_plot_path)
            
            # Parallel coordinates plot
            parallel_plot_path = os.path.join(output_dir, "parallel_coordinates.png")
            visualizer.plot_parallel_coordinates(results, save_path=parallel_plot_path)
            
            # Add visualization paths to analysis results
            analysis_results['visualizations'] = {
                'importance_plot': importance_plot_path,
                'parallel_coordinates': parallel_plot_path
            }
        
        return analysis_results
    
    def get_best_params(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract best parameters from tuning results.
        
        Args:
            results: Tuning results from tune()
            
        Returns:
            Dictionary with best parameters
        """
        return results.get('best_params', {})
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load tuning results from file.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            Loaded tuning results
        """
        from trading_optimization.hyperparameter.utils.serialization import load_tuning_results
        return load_tuning_results(file_path)
```

### 4.2 Parameter Space Definitions

Classes to define and manage hyperparameter search spaces:

```python
# space/parameter.py
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np

class Parameter:
    """
    Base class for hyperparameter definition.
    """
    
    def __init__(self, name: str):
        """
        Initialize parameter with name.
        
        Args:
            name: Parameter name
        """
        self.name = name
    
    def sample(self) -> Any:
        """
        Sample a value from the parameter space.
        
        Returns:
            Sampled parameter value
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary.
        
        Returns:
            Dictionary representation of parameter
        """
        return {"name": self.name, "type": self.__class__.__name__}


class CategoricalParameter(Parameter):
    """
    Parameter with categorical (discrete) values.
    """
    
    def __init__(self, name: str, values: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize categorical parameter.
        
        Args:
            name: Parameter name
            values: List of possible values
            weights: Optional probability weights for values
        """
        super().__init__(name)
        self.values = values
        self.weights = weights
        
        # Normalize weights if provided
        if weights is not None:
            if len(weights) != len(values):
                raise ValueError("Weights list must have same length as values list")
            self.weights = np.array(weights) / sum(weights)
    
    def sample(self) -> Any:
        """
        Sample a value from the parameter space.
        
        Returns:
            Sampled parameter value
        """
        return np.random.choice(self.values, p=self.weights)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary.
        
        Returns:
            Dictionary representation of parameter
        """
        result = super().to_dict()
        result.update({
            "values": self.values,
            "weights": self.weights.tolist() if self.weights is not None else None
        })
        return result


class RangeParameter(Parameter):
    """
    Parameter with a numeric range.
    """
    
    def __init__(
        self, 
        name: str, 
        low: float, 
        high: float, 
        distribution: str = "uniform",
        q: Optional[float] = None,
        log: bool = False
    ):
        """
        Initialize range parameter.
        
        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            distribution: Distribution to sample from ('uniform' or 'normal')
            q: Quantization factor for discrete values
            log: Whether to sample in log space
        """
        super().__init__(name)
        self.low = low
        self.high = high
        self.distribution = distribution
        self.q = q
        self.log = log
    
    def sample(self) -> float:
        """
        Sample a value from the parameter space.
        
        Returns:
            Sampled parameter value
        """
        if self.log:
            low, high = np.log(self.low), np.log(self.high)
            value = np.exp(np.random.uniform(low, high))
        else:
            if self.distribution == "uniform":
                value = np.random.uniform(self.low, self.high)
            elif self.distribution == "normal":
                # Use a normal distribution centered between low and high
                mean = (self.low + self.high) / 2
                # Standard deviation as 1/4 of the range to keep most values within bounds
                std = (self.high - self.low) / 4
                value = np.random.normal(mean, std)
                # Clip to bounds
                value = np.clip(value, self.low, self.high)
            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")
        
        # Quantize if needed
        if self.q is not None:
            value = round(value / self.q) * self.q
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary.
        
        Returns:
            Dictionary representation of parameter
        """
        result = super().to_dict()
        result.update({
            "low": self.low,
            "high": self.high,
            "distribution": self.distribution,
            "q": self.q,
            "log": self.log
        })
        return result


class IntegerParameter(RangeParameter):
    """
    Parameter with integer values.
    """
    
    def __init__(
        self, 
        name: str, 
        low: int, 
        high: int, 
        distribution: str = "uniform",
        log: bool = False
    ):
        """
        Initialize integer parameter.
        
        Args:
            name: Parameter name
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            distribution: Distribution to sample from ('uniform' or 'normal')
            log: Whether to sample in log space
        """
        super().__init__(name, low, high, distribution, q=1, log=log)
    
    def sample(self) -> int:
        """
        Sample an integer value from the parameter space.
        
        Returns:
            Sampled integer parameter value
        """
        if self.log:
            low, high = np.log(self.low), np.log(self.high)
            value = int(np.exp(np.random.uniform(low, high)))
        else:
            if self.distribution == "uniform":
                value = np.random.randint(self.low, self.high + 1)
            elif self.distribution == "normal":
                # Use a normal distribution centered between low and high
                mean = (self.low + self.high) / 2
                # Standard deviation as 1/4 of the range to keep most values within bounds
                std = (self.high - self.low) / 4
                value = int(round(np.random.normal(mean, std)))
                # Clip to bounds
                value = np.clip(value, self.low, self.high)
            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")
        
        return value


class SearchSpace:
    """
    Container for multiple hyperparameters defining a search space.
    """
    
    def __init__(self, parameters: Optional[List[Parameter]] = None):
        """
        Initialize search space with parameters.
        
        Args:
            parameters: List of Parameter objects
        """
        self.parameters = parameters or []
        self._parameter_map = {p.name: p for p in self.parameters}
    
    def add_parameter(self, parameter: Parameter):
        """
        Add a parameter to the search space.
        
        Args:
            parameter: Parameter object
        """
        self.parameters.append(parameter)
        self._parameter_map[parameter.name] = parameter
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """
        Get a parameter by name.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter object or None if not found
        """
        return self._parameter_map.get(name)
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample all parameters in the search space.
        
        Returns:
            Dictionary of parameter names to sampled values
        """
        return {p.name: p.sample() for p in self.parameters}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search space to dictionary.
        
        Returns:
            Dictionary representation of search space
        """
        return {
            "parameters": [p.to_dict() for p in self.parameters]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchSpace':
        """
        Create search space from dictionary.
        
        Args:
            data: Dictionary representation of search space
            
        Returns:
            SearchSpace object
        """
        from trading_optimization.hyperparameter.space.factory import parameter_from_dict
        
        space = cls()
        for param_data in data.get("parameters", []):
            param = parameter_from_dict(param_data)
            if param:
                space.add_parameter(param)
        
        return space
    
    def to_json(self) -> str:
        """
        Convert search space to JSON string.
        
        Returns:
            JSON string representation of search space
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SearchSpace':
        """
        Create search space from JSON string.
        
        Args:
            json_str: JSON string representation of search space
            
        Returns:
            SearchSpace object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
```

Search space factory:

```python
# space/factory.py
from typing import Dict, Any, Optional, List
from trading_optimization.hyperparameter.space.parameter import (
    Parameter, 
    CategoricalParameter, 
    RangeParameter, 
    IntegerParameter,
    SearchSpace
)

def parameter_from_dict(data: Dict[str, Any]) -> Optional[Parameter]:
    """
    Create a parameter from dictionary representation.
    
    Args:
        data: Dictionary representation of parameter
        
    Returns:
        Parameter object or None if invalid
    """
    param_type = data.get("type")
    name = data.get("name")
    
    if not name:
        return None
    
    if param_type == "CategoricalParameter":
        values = data.get("values", [])
        weights = data.get("weights")
        return CategoricalParameter(name, values, weights)
    
    elif param_type == "RangeParameter":
        low = data.get("low", 0)
        high = data.get("high", 1)
        distribution = data.get("distribution", "uniform")
        q = data.get("q")
        log = data.get("log", False)
        return RangeParameter(name, low, high, distribution, q, log)
    
    elif param_type == "IntegerParameter":
        low = data.get("low", 0)
        high = data.get("high", 1)
        distribution = data.get("distribution", "uniform")
        log = data.get("log", False)
        return IntegerParameter(name, low, high, distribution, log)
    
    return None

def create_search_space(config: Dict[str, Any]) -> SearchSpace:
    """
    Create a search space from configuration dictionary.
    
    Args:
        config: Dictionary with parameter definitions
        
    Returns:
        SearchSpace object
    """
    space = SearchSpace()
    
    for param_name, param_config in config.items():
        # Skip internal keys starting with underscore
        if param_name.startswith('_'):
            continue
            
        param_type = param_config.get("type", "range")
        
        if param_type == "categorical":
            values = param_config.get("values", [])
            weights = param_config.get("weights")
            param = CategoricalParameter(param_name, values, weights)
            
        elif param_type == "range":
            low = param_config.get("low", 0)
            high = param_config.get("high", 1)
            distribution = param_config.get("distribution", "uniform")
            q = param_config.get("q")
            log = param_config.get("log", False)
            param = RangeParameter(param_name, low, high, distribution, q, log)
            
        elif param_type == "integer":
            low = param_config.get("low", 0)
            high = param_config.get("high", 1)
            distribution = param_config.get("distribution", "uniform")
            log = param_config.get("log", False)
            param = IntegerParameter(param_name, low, high, distribution, log)
            
        else:
            # Unknown parameter type
            continue
        
        space.add_parameter(param)
    
    return space
```

### 4.3 Search Strategies

Base search strategy:

```python
# search/base.py
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod

from trading_optimization.hyperparameter.space.parameter import SearchSpace

class SearchStrategy(ABC):
    """
    Abstract base class for hyperparameter search strategies.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize search strategy with configuration.
        
        Args:
            **kwargs: Strategy-specific configuration
        """
        self.config = kwargs
        self.trials = []
    
    @abstractmethod
    def suggest(
        self, 
        trial_id: str,
        search_space: SearchSpace
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for next trial.
        
        Args:
            trial_id: Unique identifier for the trial
            search_space: Search space to sample from
            
        Returns:
            Dictionary of parameter values for the trial
        """
        pass
    
    def update(
        self, 
        trial_id: str,
        params: Dict[str, Any],
        result: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update strategy with trial results.
        
        Args:
            trial_id: Unique identifier for the trial
            params: Dictionary of trial parameters
            result: Objective function value
            metadata: Additional trial metadata
        """
        # Store trial result
        trial_data = {
            "trial_id": trial_id,
            "params": params,
            "result": result,
            "metadata": metadata or {}
        }
        self.trials.append(trial_data)
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get the best trial so far.
        
        Returns:
            Trial data dictionary or None if no trials yet
        """
        if not self.trials:
            return None
        
        # Find trial with minimum result (assuming minimization)
        return min(self.trials, key=lambda t: t["result"])
```

Random search strategy:

```python
# search/random.py
from typing import Dict, List, Any, Optional
import random

from trading_optimization.hyperparameter.search.base import SearchStrategy
from trading_optimization.hyperparameter.space.parameter import SearchSpace

class RandomSearch(SearchStrategy):
    """
    Random search strategy for hyperparameter optimization.
    """
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """
        Initialize random search strategy.
        
        Args:
            seed: Optional random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)
    
    def suggest(
        self, 
        trial_id: str,
        search_space: SearchSpace
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters by random sampling.
        
        Args:
            trial_id: Unique identifier for the trial
            search_space: Search space to sample from
            
        Returns:
            Dictionary of parameter values for the trial
        """
        # Simply sample randomly from the search space
        return search_space.sample()
```

Bayesian optimization strategy:

```python
# search/bayesian.py
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

from trading_optimization.hyperparameter.search.base import SearchStrategy
from trading_optimization.hyperparameter.space.parameter import (
    SearchSpace,
    Parameter,
    CategoricalParameter,
    RangeParameter,
    IntegerParameter
)

class BayesianOptimization(SearchStrategy):
    """
    Bayesian optimization strategy for hyperparameter search.
    """
    
    def __init__(
        self,
        initial_points: int = 5,
        n_ei_candidates: int = 100,
        xi: float = 0.01,
        kappa: float = 2.0,
        kernel: Any = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Bayesian optimization strategy.
        
        Args:
            initial_points: Number of random points to sample initially
            n_ei_candidates: Number of candidates for expected improvement
            xi: Exploration-exploitation trade-off parameter
            kappa: Parameter controlling exploitation vs exploration in acquisition function
            kernel: Gaussian Process kernel (defaults to Matern)
            random_state: Random seed
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.initial_points = initial_points
        self.n_ei_candidates = n_ei_candidates
        self.xi = xi
        self.kappa = kappa
        self.random_state = random_state
        
        # Set default kernel if not provided
        self.kernel = kernel or Matern(nu=2.5)
        
        # Initialize GP model (will be fit later)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,  # Small regularization
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        
        # Initialize parameter space info
        self.param_types = {}  # Maps param name to type ('cat', 'real', 'int')
        self.param_categories = {}  # For categorical params: maps name to value index mapping
        self.scaler = StandardScaler()
        self.fitted = False
    
    def _analyze_search_space(self, search_space: SearchSpace):
        """
        Analyze search space and prepare parameter transformations.
        
        Args:
            search_space: Search space to analyze
        """
        self.param_types = {}
        self.param_categories = {}
        
        for param in search_space.parameters:
            name = param.name
            
            if isinstance(param, CategoricalParameter):
                self.param_types[name] = 'cat'
                # Create mapping from values to indices
                self.param_categories[name] = {
                    value: i for i, value in enumerate(param.values)
                }
            elif isinstance(param, IntegerParameter):
                self.param_types[name] = 'int'
            else:
                self.param_types[name] = 'real'
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert parameter dictionary to vector representation.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Numpy array with numeric representation of parameters
        """
        result = []
        
        for name, value in params.items():
            if name in self.param_types:
                if self.param_types[name] == 'cat':
                    # Convert categorical to one-hot
                    if name in self.param_categories and value in self.param_categories[name]:
                        result.append(float(self.param_categories[name][value]))
                    else:
                        # Fallback
                        result.append(0.0)
                else:
                    # Numeric params
                    result.append(float(value))
        
        return np.array(result).reshape(1, -1)
    
    def _vector_to_params(
        self, 
        vector: np.ndarray, 
        search_space: SearchSpace
    ) -> Dict[str, Any]:
        """
        Convert vector representation back to parameter dictionary.
        
        Args:
            vector: Numeric representation of parameters
            search_space: Original search space
            
        Returns:
            Parameter dictionary
        """
        params = {}
        idx = 0
        
        for param in search_space.parameters:
            name = param.name
            
            if name in self.param_types:
                if self.param_types[name] == 'cat':
                    # Convert index back to categorical value
                    cat_idx = int(round(vector[idx]))
                    cat_idx = max(0, min(cat_idx, len(param.values) - 1))
                    params[name] = param.values[cat_idx]
                elif self.param_types[name] == 'int':
                    # Convert to integer
                    params[name] = int(round(vector[idx]))
                else:
                    # Real parameter
                    params[name] = float(vector[idx])
                
                idx += 1
        
        return params
    
    def _fit_gp(self):
        """Fit Gaussian Process model with existing trials."""
        if len(self.trials) < 2:
            # Not enough data to fit
            self.fitted = False
            return
        
        # Extract parameters and results from trials
        X = np.array([
            self._params_to_vector(t["params"]).flatten() for t in self.trials
        ])
        y = np.array([t["result"] for t in self.trials])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GP model
        try:
            self.gp.fit(X_scaled, y)
            self.fitted = True
        except Exception as e:
            print(f"Error fitting GP model: {str(e)}")
            self.fitted = False
    
    def _expected_improvement(
        self,
        X: np.ndarray,
        best_y: float
    ) -> np.ndarray:
        """
        Compute expected improvement acquisition function.
        
        Args:
            X: Points to evaluate
            best_y: Best observed value
            
        Returns:
            Expected improvement values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Compute Expected Improvement
        with np.errstate(divide='ignore'):
            # Calculate improvement
            improvement = best_y - mu
            
            # Calculate Z-score
            z = improvement / sigma
            
            # Calculate EI
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Set NaN/Inf to 0
            ei[np.isnan(ei)] = 0.0
        
        return ei
    
    def _upper_confidence_bound(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute upper confidence bound acquisition function.
        
        Args:
            X: Points to evaluate
            
        Returns:
            UCB values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        # For minimization problems, we use the negative UCB
        return -(mu - self.kappa * sigma)
    
    def suggest(
        self, 
        trial_id: str,
        search_space: SearchSpace
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Bayesian optimization.
        
        Args:
            trial_id: Unique identifier for the trial
            search_space: Search space to sample from
            
        Returns:
            Dictionary of parameter values for the trial
        """
        # Analyze search space on first call
        if not self.param_types:
            self._analyze_search_space(search_space)
        
        # If we don't have enough trials, use random sampling
        if len(self.trials) < self.initial_points:
            return search_space.sample()
        
        # Fit GP model
        self._fit_gp()
        
        if not self.fitted:
            # Fallback to random search if fitting failed
            return search_space.sample()
        
        # Find best result so far
        best_y = min(t["result"] for t in self.trials)
        
        # Generate random candidates
        candidate_params = []
        candidate_vectors = []
        
        for _ in range(self.n_ei_candidates):
            params = search_space.sample()
            vector = self._params_to_vector(params).flatten()
            candidate_params.append(params)
            candidate_vectors.append(vector)
        
        # Scale candidates
        X = np.vstack(candidate_vectors)
        X_scaled = self.scaler.transform(X)
        
        # Compute acquisition function
        ei_values = self._expected_improvement(X_scaled, best_y)
        
        # Find best candidate
        best_idx = np.argmax(ei_values)
        
        return candidate_params[best_idx]
    
    def update(
        self, 
        trial_id: str,
        params: Dict[str, Any],
        result: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update strategy with trial results.
        
        Args:
            trial_id: Unique identifier for the trial
            params: Dictionary of trial parameters
            result: Objective function value
            metadata: Additional trial metadata
        """
        super().update(trial_id, params, result, metadata)
        
        # Refit the model when we have new data
        if len(self.trials) >= self.initial_points:
            self._fit_gp()
```

Search strategy factory:

```python
# search/factory.py
from typing import Dict, Any

from trading_optimization.hyperparameter.search.base import SearchStrategy
from trading_optimization.hyperparameter.search.random import RandomSearch
from trading_optimization.hyperparameter.search.grid import GridSearch
from trading_optimization.hyperparameter.search.bayesian import BayesianOptimization
from trading_optimization.hyperparameter.search.hyperband import HyperBand
from trading_optimization.hyperparameter.search.evolutionary import EvolutionarySearch

class SearchStrategyFactory:
    """
    Factory for creating search strategy instances.
    """
    
    _strategies = {
        'random': RandomSearch,
        'grid': GridSearch,
        'bayesian': BayesianOptimization,
        'hyperband': HyperBand,
        'evolutionary': EvolutionarySearch
    }
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a new search strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class
    
    def create_strategy(self, name: str, **kwargs) -> SearchStrategy:
        """
        Create a search strategy instance.
        
        Args:
            name: Strategy name
            **kwargs: Strategy configuration parameters
            
        Returns:
            SearchStrategy instance
        """
        if name not in self._strategies:
            raise ValueError(f"Unknown search strategy: {name}")
        
        # Create and return the strategy
        return self._strategies[name](**kwargs)
```

### 4.4 Trial Execution

Trial executor to manage the execution of hyperparameter trials:

```python
# trial/executor.py
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import time
import uuid
import concurrent.futures
from datetime import datetime

from trading_optimization.hyperparameter.search.base import SearchStrategy
from trading_optimization.hyperparameter.space.parameter import SearchSpace
from trading_optimization.hyperparameter.trial.scheduler import TrialScheduler
from trading_optimization.hyperparameter.trial.pruner import TrialPruner

class TrialExecutor:
    """
    Manages execution of hyperparameter tuning trials.
    """
    
    def __init__(
        self,
        search_strategy: SearchStrategy,
        search_space: SearchSpace,
        objective_fn: Callable,
        max_trials: int = 10,
        timeout: Optional[int] = None,
        n_concurrent: int = 1,
        checkpoint_dir: Optional[str] = None,
        scheduler: Optional[TrialScheduler] = None,
        pruner: Optional[TrialPruner] = None
    ):
        """
        Initialize trial executor.
        
        Args:
            search_strategy: Search strategy to use
            search_space: Search space definition
            objective_fn: Objective function to minimize
            max_trials: Maximum number of trials
            timeout: Maximum runtime in seconds
            n_concurrent: Number of concurrent trials
            checkpoint_dir: Directory for checkpoints
            scheduler: Optional trial scheduler
            pruner: Optional trial pruner
        """
        self.search_strategy = search_strategy
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.max_trials = max_trials
        self.timeout = timeout
        self.n_concurrent = n_concurrent
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.pruner = pruner
        
        # Trial tracking
        self.trials = []
        self.trial_results = {}
        self.completed_trials = 0
        self.best_trial = None
        self.best_result = float('inf')  # We're minimizing
    
    def _execute_trial(self, trial_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single trial.
        
        Args:
            trial_id: Unique identifier for the trial
            params: Parameters for the trial
            
        Returns:
            Trial results
        """
        start_time = time.time()
        
        # Add trial_id to params
        trial_params = dict(params)
        trial_params['trial_id'] = trial_id
        
        # Run the objective function
        try:
            result = self.objective_fn(trial_params)
        except Exception as e:
            print(f"Error in trial {trial_id}: {str(e)}")
            # Return failure result
            return {
                "status": "error",
                "error": str(e),
                "training_time": time.time() - start_time,
                "objective_value": float('inf')
            }
        
        # Return success result
        return {
            "status": "completed",
            "objective_value": result if result is not None else float('inf'),
            "training_time": time.time() - start_time
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the hyperparameter tuning process.
        
        Returns:
            Dictionary with tuning results
        """
        start_time = time.time()
        
        # Set up for parallel execution if requested
        if self.n_concurrent > 1:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_concurrent)
            futures = {}
        else:
            executor = None
        
        try:
            # Main loop for trials
            for trial_num in range(self.max_trials):
                # Check timeout
                if self.timeout and (time.time() - start_time) > self.timeout:
                    print(f"Timeout after {self.timeout} seconds")
                    break
                
                # Generate trial ID
                trial_id = f"trial_{uuid.uuid4().hex[:8]}"
                
                # Get parameters for this trial
                params = self.search_strategy.suggest(trial_id, self.search_space)
                
                # Track the trial
                self.trials.append({
                    "trial_id": trial_id,
                    "trial_num": trial_num,
                    "params": params,
                    "status": "running"
                })
                
                # Check if we're doing parallel execution
                if executor:
                    # Submit trial to executor
                    future = executor.submit(self._execute_trial, trial_id, params)
                    futures[future] = (trial_id, params)
                    
                    # Check for completed futures
                    done_futures = [f for f in futures if f.done()]
                    for future in done_futures:
                        trial_id, params = futures[future]
                        result = future.result()
                        
                        # Process the result
                        self._process_result(trial_id, params, result)
                        
                        # Remove from futures map
                        del futures[future]
                else:
                    # Execute trial synchronously
                    result = self._execute_trial(trial_id, params)
                    
                    # Process the result
                    self._process_result(trial_id, params, result)
            
            # Wait for any remaining futures if using parallel execution
            if executor:
                for future in concurrent.futures.as_completed(futures):
                    trial_id, params = futures[future]
                    result = future.result()
                    
                    # Process the result
                    self._process_result(trial_id, params, result)
        finally:
            # Clean up executor if used
            if executor:
                executor.shutdown()
        
        # Compile final results
        total_time = time.time() - start_time
        
        results = {
            "best_params": self.best_trial["params"] if self.best_trial else None,
            "best_result": {
                "objective_value": self.best_result,
                "trial_id": self.best_trial["trial_id"] if self.best_trial else None
            },
            "all_trials": self.trials,
            "completed_trials": self.completed_trials,
            "total_time": total_time
        }
        
        return results
    
    def _process_result(
        self,
        trial_id: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Process a trial result.
        
        Args:
            trial_id: Unique identifier for the trial
            params: Parameters used for the trial
            result: Trial result
        """
        # Update trial status
        for trial in self.trials:
            if trial["trial_id"] == trial_id:
                trial["status"] = result["status"]
                trial["result"] = result
                break
        
        # Increment completed count
        self.completed_trials += 1
        
        # Store result
        self.trial_results[trial_id] = result
        
        # Update search strategy
        objective_value = result.get("objective_value", float('inf'))
        self.search_strategy.update(
            trial_id,
            params,
            objective_value,
            metadata=result
        )
        
        # Check if this is the best result so far
        if objective_value < self.best_result:
            self.best_result = objective_value
            for trial in self.trials:
                if trial["trial_id"] == trial_id:
                    self.best_trial = trial
                    break
        
        # Print progress
        print(f"Completed trial {self.completed_trials}/{self.max_trials}: "
              f"objective={objective_value:.6f}, "
              f"time={result.get('training_time', 0):.2f}s")
```

### 4.5 Analysis and Visualization

Trial analysis and visualization utilities:

```python
# analysis/evaluator.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TrialEvaluator:
    """
    Analyzes hyperparameter tuning results.
    """
    
    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze tuning results.
        
        Args:
            results: Tuning results from TuningManager.tune()
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Extract trial data
        all_trials = results.get("all_trials", [])
        if not all_trials:
            return {"error": "No trial data available"}
        
        # Convert to DataFrame for easier analysis
        df = self._trials_to_dataframe(all_trials)
        analysis["trials_summary"] = self._get_trials_summary(df)
        
        # Parameter importance analysis
        importance = self._parameter_importance(df)
        if importance is not None:
            analysis["parameter_importance"] = importance
        
        # Correlation analysis
        correlations = self._parameter_correlations(df)
        if correlations is not None:
            analysis["parameter_correlations"] = correlations
        
        # Best regions analysis
        best_regions = self._best_parameter_regions(df)
        if best_regions is not None:
            analysis["best_parameter_regions"] = best_regions
        
        return analysis
    
    def _trials_to_dataframe(self, trials: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert trial data to DataFrame.
        
        Args:
            trials: List of trial dictionaries
            
        Returns:
            DataFrame with trial data
        """
        # Extract parameter values and results
        data = []
        
        for trial in trials:
            if "params" not in trial or "result" not in trial:
                continue
                
            row = {}
            # Add parameters
            row.update(trial["params"])
            
            # Add result
            if isinstance(trial["result"], dict) and "objective_value" in trial["result"]:
                row["objective"] = trial["result"]["objective_value"]
            else:
                continue  # Skip trials without objective value
                
            # Add metadata
            row["trial_id"] = trial["trial_id"]
            if "status" in trial:
                row["status"] = trial["status"]
            if "trial_num" in trial:
                row["trial_num"] = trial["trial_num"]
            if isinstance(trial["result"], dict) and "training_time" in trial["result"]:
                row["training_time"] = trial["result"]["training_time"]
                
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    
    def _get_trials_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for trials.
        
        Args:
            df: DataFrame with trial data
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Basic stats about objective
        if "objective" in df.columns:
            summary["objective"] = {
                "min": df["objective"].min(),
                "max": df["objective"].max(),
                "mean": df["objective"].mean(),
                "median": df["objective"].median(),
                "std": df["objective"].std()
            }
        
        # Training time stats
        if "training_time" in df.columns:
            summary["training_time"] = {
                "min": df["training_time"].min(),
                "max": df["training_time"].max(),
                "mean": df["training_time"].mean(),
                "total": df["training_time"].sum()
            }
        
        # Status counts
        if "status" in df.columns:
            summary["status_counts"] = df["status"].value_counts().to_dict()
        
        return summary
    
    def _parameter_importance(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculate parameter importance using Random Forest.
        
        Args:
            df: DataFrame with trial data
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            # Check if we have enough trials and an objective column
            if len(df) < 5 or "objective" not in df.columns:
                return None
            
            # Get parameter columns (exclude metadata)
            exclude_cols = ["objective", "trial_id", "status", 
                           "trial_num", "training_time"]
            param_cols = [col for col in df.columns 
                        if col not in exclude_cols]
            
            if not param_cols:
                return None
            
            # Prepare data
            X = df[param_cols].copy()
            
            # Convert categorical parameters to numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Simple label encoding
                    X[col] = pd.Categorical(X[col]).codes
            
            y = df["objective"].values
            
            # Train a Random Forest to get feature importances
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Get importances
            importances = rf.feature_importances_
            
            # Create dictionary mapping parameters to importance
            importance_dict = {param: float(imp) for param, imp in zip(param_cols, importances)}
            
            # Sort by importance
            importance_dict = {k: v for k, v in sorted(
                importance_dict.items(), key=lambda item: item[1], reverse=True
            )}
            
            return importance_dict
        except Exception as e:
            print(f"Error calculating parameter importance: {str(e)}")
            return None
    
    def _parameter_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calculate correlations between parameters and objective.
        
        Args:
            df: DataFrame with trial data
            
        Returns:
            Dictionary with correlations
        """
        try:
            # Check if we have enough trials and an objective column
            if len(df) < 5 or "objective" not in df.columns:
                return None
            
            # Get parameter columns (exclude metadata)
            exclude_cols = ["trial_id", "status", "trial_num", "training_time"]
            columns = [col for col in df.columns if col not in exclude_cols]
            
            if len(columns) < 2:  # Need at least objective and one parameter
                return None
            
            # Calculate correlations
            corr_df = df[columns].copy()
            
            # Convert categorical variables
            for col in corr_df.columns:
                if corr_df[col].dtype == 'object':
                    # Skip categorical variables
                    corr_df.drop(columns=[col], inplace=True)
            
            # Calculate correlation matrix
            corr_matrix = corr_df.corr()
            
            # Create nested dictionary
            result = {}
            for col in corr_matrix.columns:
                result[col] = {
                    other_col: float(corr_matrix.loc[col, other_col])
                    for other_col in corr_matrix.columns
                    if other_col != col
                }
                
            return result
        except Exception as e:
            print(f"Error calculating parameter correlations: {str(e)}")
            return None
    
    def _best_parameter_regions(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Identify best regions in parameter space.
        
        Args:
            df: DataFrame with trial data
            
        Returns:
            Dictionary with best parameter regions
        """
        try:
            # Check if we have enough trials and an objective column
            if len(df) < 5 or "objective" not in df.columns:
                return None
            
            # Get parameter columns (exclude metadata)
            exclude_cols = ["objective", "trial_id", "status", 
                           "trial_num", "training_time"]
            param_cols = [col for col in df.columns if col not in exclude_cols]
            
            if not param_cols:
                return None
            
            # Get the top 20% of trials by objective value
            top_frac = 0.2
            n_top = max(1, int(len(df) * top_frac))
            top_df = df.nsmallest(n_top, "objective")
            
            # Analyze each parameter
            parameter_regions = {}
            
            for param in param_cols:
                if df[param].dtype in ['int64', 'float64']:
                    # Numeric parameter
                    param_min = float(top_df[param].min())
                    param_max = float(top_df[param].max())
                    param_mean = float(top_df[param].mean())
                    
                    parameter_regions[param] = {
                        "type": "numeric",
                        "min": param_min,
                        "max": param_max,
                        "mean": param_mean
                    }
                else:
                    # Categorical parameter
                    value_counts = top_df[param].value_counts()
                    total = value_counts.sum()
                    proportions = {str(k): float(v / total) 
                                 for k, v in value_counts.items()}
                    
                    parameter_regions[param] = {
                        "type": "categorical",
                        "value_proportions": proportions,
                        "most_common": str(value_counts.index[0])
                    }
            
            return parameter_regions
        except Exception as e:
            print(f"Error identifying best parameter regions: {str(e)}")
            return None
```

Visualization utilities:

```python
# analysis/visualizer.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class TuningVisualizer:
    """
    Visualizes hyperparameter tuning results.
    """
    
    def plot_parameter_importance(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        top_n: int = 10
    ) -> plt.Figure:
        """
        Plot parameter importance.
        
        Args:
            results: Tuning results from TuningManager.tune()
            save_path: Path to save the plot
            top_n: Number of top parameters to include
            
        Returns:
            Matplotlib figure object
        """
        # Extract trial data
        all_trials = results.get("all_trials", [])
        if not all_trials:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center')
            return fig
        
        # Convert to DataFrame
        df = self._trials_to_dataframe(all_trials)
        
        # Calculate parameter importance
        from trading_optimization.hyperparameter.analysis.evaluator import TrialEvaluator
        evaluator = TrialEvaluator()
        analysis = evaluator.analyze(results)
        importance = analysis.get("parameter_importance")
        
        if not importance:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Could not calculate parameter importance", 
                   ha='center', va='center')
            return fig
        
        # Take top N parameters
        top_params = list(importance.keys())[:top_n]
        top_values = [importance[p] for p in top_params]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_params))
        
        ax.barh(y_pos, top_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_params)
        ax.invert_yaxis()  # Highest values at the top
        ax.set_xlabel('Importance')
        ax.set_title('Parameter Importance')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_parallel_coordinates(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        top_n_params: int = 7
    ) -> plt.Figure:
        """
        Plot parallel coordinates of parameters.
        
        Args:
            results: Tuning results from TuningManager.tune()
            save_path: Path to save the plot
            top_n_params: Number of top parameters to include
            
        Returns:
            Matplotlib figure object
        """
        # Extract trial data
        all_trials = results.get("all_trials", [])
        if not all_trials:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trial data available", 
                   ha='center', va='center')
            return fig
        
        # Convert to DataFrame
        df = self._trials_to_dataframe(all_trials)
        
        # Calculate parameter importance to select top parameters
        from trading_optimization.hyperparameter.analysis.evaluator import TrialEvaluator
        evaluator = TrialEvaluator()
        analysis = evaluator.analyze(results)
        importance = analysis.get("parameter_importance", {})
        
        # If importance calculation failed, use all numeric parameters
        if not importance:
            # Use numeric columns
            numeric_params = df.select_dtypes(include=['number']).columns.tolist()
            # Exclude metadata columns
            exclude_cols = ["objective", "trial_num", "training_time"]
            numeric_params = [p for p in numeric_params if p not in exclude_cols]
            # Take top N
            top_params = numeric_params[:top_n_params]
        else:
            # Take top N parameters that are numeric
            top_params = []
            for param in importance.keys():
                if param in df.columns and pd.api.types.is_numeric_dtype(df[param]):
                    top_params.append(param)
                    if len(top_params) >= top_n_params:
                        break
        
        if not top_params:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric parameters available", 
                   ha='center', va='center')
            return fig
        
        # Add objective column
        plot_cols = top_params + ["objective"]
        
        # Filter only required columns and complete rows
        plot_df = df[plot_cols].dropna()
        
        if len(plot_df) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data for parallel coordinates plot", 
                   ha='center', va='center')
            return fig
        
        # Scale the data for better visualization
        scaler = MinMaxScaler()
        plot_df_scaled = pd.DataFrame(
            scaler.fit_transform(plot_df), 
            columns=plot_df.columns
        )
        
        # Color by objective value (lower is better)
        cmap = plt.get_cmap('viridis_r')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each trial as a line
        for i, (_, row) in enumerate(plot_df_scaled.iterrows()):
            color = cmap(row["objective"])
            ax.plot(range(len(plot_cols)), row, color=color, alpha=0.6)
        
        # Set up the axes
        ax.set_xticks(range(len(plot_cols)))
        ax.set_xticklabels(plot_cols, rotation=45)
        ax.grid(True)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(plot_df_scaled["objective"])
        cbar = plt.colorbar(sm)
        cbar.set_label("Objective Value (scaled)")
        
        plt.title("Parallel Coordinates Plot of Hyperparameters")
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def _trials_to_dataframe(self, trials: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert trial data to DataFrame.
        
        Args:
            trials: List of trial dictionaries
            
        Returns:
            DataFrame with trial data
        """
        # Extract parameter values and results
        data = []
        
        for trial in trials:
            if "params" not in trial or "result" not in trial:
                continue
                
            row = {}
            # Add parameters
            row.update(trial["params"])
            
            # Add result
            if isinstance(trial["result"], dict) and "objective_value" in trial["result"]:
                row["objective"] = trial["result"]["objective_value"]
            else:
                continue  # Skip trials without objective value
                
            # Add metadata
            row["trial_id"] = trial["trial_id"]
            if "status" in trial:
                row["status"] = trial["status"]
            if "trial_num" in trial:
                row["trial_num"] = trial["trial_num"]
            if isinstance(trial["result"], dict) and "training_time" in trial["result"]:
                row["training_time"] = trial["result"]["training_time"]
                
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
```

### 4.6 Ray Integration

Ray Tune integration utilities:

```python
# ray_integration.py
from typing import Dict, Any, Optional
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

from trading_optimization.hyperparameter.space.parameter import (
    SearchSpace, 
    Parameter,
    CategoricalParameter,
    RangeParameter,
    IntegerParameter
)
from trading_optimization.hyperparameter.search.base import SearchStrategy

def convert_search_space_to_ray(search_space: SearchSpace) -> Dict[str, Any]:
    """
    Convert internal search space to Ray Tune format.
    
    Args:
        search_space: Internal SearchSpace object
        
    Returns:
        Ray Tune compatible search space dictionary
    """
    ray_space = {}
    
    for param in search_space.parameters:
        name = param.name
        
        if isinstance(param, CategoricalParameter):
            ray_space[name] = tune.choice(param.values, weights=param.weights)
            
        elif isinstance(param, IntegerParameter):
            if param.log:
                ray_space[name] = tune.loguniform(
                    param.low, param.high, base=10
                )
            else:
                ray_space[name] = tune.randint(
                    param.low, param.high + 1  # Ray's randint is exclusive of upper bound
                )
                
        elif isinstance(param, RangeParameter):
            if param.log:
                ray_space[name] = tune.loguniform(
                    param.low, param.high, base=10
                )
            else:
                ray_space[name] = tune.uniform(
                    param.low, param.high
                )
                
                if param.q is not None:
                    ray_space[name] = tune.sample_from(
                        lambda spec: round(spec[name] / param.q) * param.q
                    )
    
    return ray_space

def convert_search_strategy_to_ray(search_strategy: SearchStrategy) -> Any:
    """
    Convert internal search strategy to Ray Tune search algorithm.
    
    Args:
        search_strategy: Internal SearchStrategy object
        
    Returns:
        Ray Tune search algorithm
    """
    strategy_type = search_strategy.__class__.__name__
    
    if strategy_type == "RandomSearch":
        # Ray's default is already random search
        return None
        
    elif strategy_type == "BayesianOptimization":
        # Use HyperOpt (TPE) as it's generally more stable than BayesOpt
        return HyperOptSearch(
            metric="objective",
            mode="min",
            random_state_seed=search_strategy.config.get("random_state", None)
        )
        
    elif strategy_type == "GridSearch":
        # Grid search isn't directly supported as a search algorithm in Ray,
        # but can be done using grid_search() in the parameter space
        # We'll use a basic algorithm for now
        return None
        
    elif strategy_type == "HyperBand":
        # Use Ray's HyperBand scheduler instead of a search algorithm
        from ray.tune.schedulers import HyperBandScheduler
        return HyperBandScheduler(
            metric="objective",
            mode="min"
        )
        
    elif strategy_type == "EvolutionarySearch":
        # Use Ray's Population-Based Training (PBT)
        from ray.tune.schedulers import PopulationBasedTraining
        return PopulationBasedTraining(
            metric="objective",
            mode="min"
        )
    
    # Default to None, which uses Ray's default (random) search
    return None
```

## 5. Configuration

### 5.1 Hyperparameter Space Configuration Schema

```yaml
# Example hyperparameter space configuration for LSTM model
hyperparameter:
  spaces:
    lstm:
      hidden_size:
        type: "integer"
        low: 32
        high: 256
        log: true
        
      num_layers:
        type: "integer"
        low: 1
        high: 4
        
      dropout:
        type: "range"
        low: 0.0
        high: 0.5
        
      bidirectional:
        type: "categorical"
        values: [true, false]
        
      fc_layers:
        type: "categorical"
        values: [
          [64],
          [128, 64],
          [256, 128],
          [512, 256, 128]
        ]
```

### 5.2 Tuning Strategy Configuration Schema

```yaml
# Example tuning strategy configuration
hyperparameter:
  strategies:
    random:
      max_trials: 50
      seed: 42
      
    bayesian:
      initial_points: 10
      max_trials: 30
      n_ei_candidates: 100
      
    hyperband:
      max_trials: 50
      brackets: 3
      
    evolutionary:
      population_size: 10
      generations: 5
      mutation_rate: 0.1
  
  defaults:
    strategy: "bayesian"
    max_trials: 30
    n_concurrent: 4
    use_ray: true
```

## 6. Usage Examples

### 6.1 Basic Hyperparameter Tuning

```python
# Example usage of the Hyperparameter Tuning System

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.hyperparameter.manager import TuningManager

# Load configuration
config = ConfigManager.instance()
hyperparameter_config = config.get('hyperparameter', {})

# Create managers
tuning_manager = TuningManager(hyperparameter_config)
data_manager = DataManager(config.get('data', {}))

# Prepare data pipeline
pipeline_id = data_manager.create_pipeline("btc_analysis", pipeline_config)
df = data_manager.execute_pipeline(pipeline_id)
data_splits = data_manager.split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
data_loaders = data_manager.get_data_loaders(data_splits, batch_size=32)

# Configure the search space for LSTM model
lstm_search_space = {
    "hidden_size": {"type": "integer", "low": 32, "high": 256, "log": True},
    "num_layers": {"type": "integer", "low": 1, "high": 4},
    "dropout": {"type": "range", "low": 0.0, "high": 0.5},
    "bidirectional": {"type": "categorical", "values": [True, False]},
    "fc_layers": {
        "type": "categorical", 
        "values": [[64], [128, 64], [256, 128], [512, 256, 128]]
    }
}

# Create search space
search_space = tuning_manager.create_search_space(lstm_search_space)

# Configure data and training
data_config = {
    "pipeline_id": pipeline_id,
    "split_config": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "loader_config": {
        "batch_size": 32,
        "dataset_type": "timeseries",
        "sequence_length": 60,
        "forecast_horizon": 1
    }
}

training_config = {
    "trainer_type": "supervised",
    "epochs": 30,
    "optimizer_type": "adam",
    "optimizer_params": {"lr": 0.001},
    "loss_fn": "mse",
    "early_stopping": True,
    "patience": 10
}

# Set up fixed parameters
fixed_params = {
    "input_size": len(df.columns) - 1,  # Exclude target column
    "output_size": 1,
    "sequence_length": 60
}

# Run hyperparameter optimization
tuning_results = tuning_manager.tune(
    model_type="lstm",
    search_space=search_space,
    objective="val_loss",  # Minimize validation loss
    strategy="bayesian",
    max_trials=30,
    n_concurrent=4,
    use_ray=True,  # Use Ray for distributed execution
    checkpoint_dir="./checkpoints/hyperopt",
    data_config=data_config,
    training_config=training_config
)

# Analyze the results
analysis = tuning_manager.analyze_results(
    tuning_results,
    output_dir="./checkpoints/hyperopt/analysis"
)

# Get the best hyperparameters
best_params = tuning_manager.get_best_params(tuning_results)
print(f"Best hyperparameters: {best_params}")
```

### 6.2 Evolving Parameters with Ray Evolution Strategies

```python
# Example of using evolutionary strategies with Ray

from trading_optimization.hyperparameter.manager import TuningManager

# Create tuning manager
tuning_manager = TuningManager(config.get('hyperparameter', {}))

# Define search space for a neural network model
nn_search_space = {
    "learning_rate": {"type": "range", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "values": [16, 32, 64, 128]},
    "optimizer": {"type": "categorical", "values": ["adam", "rmsprop", "sgd"]},
    "activation": {"type": "categorical", "values": ["relu", "elu", "leaky_relu"]},
    "hidden_layers": {
        "type": "categorical",
        "values": [
            [64], 
            [128, 64], 
            [256, 128, 64], 
            [512, 256, 128, 64]
        ]
    },
    "dropout": {"type": "range", "low": 0.0, "high": 0.5},
    "weight_decay": {"type": "range", "low": 0.0, "high": 0.1}
}

# Create search space
search_space = tuning_manager.create_search_space(nn_search_space)

# Run hyperparameter optimization with evolutionary strategy
tuning_results = tuning_manager.tune(
    model_type="mlp",
    search_space=search_space,
    objective="val_loss",
    strategy="evolutionary",
    strategy_config={
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8
    },
    max_trials=50,  # Maximum total trials
    n_concurrent=8,  # Number of concurrent trials
    use_ray=True,
    checkpoint_dir="./checkpoints/evolutionary"
)

# Visualize how parameters evolved over generations
from trading_optimization.hyperparameter.analysis.visualizer import TuningVisualizer
visualizer = TuningVisualizer()
visualizer.plot_evolution_history(tuning_results, save_path="./plots/evolution.png")
```

### 6.3 Multi-Stage Hyperparameter Optimization

```python
# Example of multi-stage hyperparameter optimization

from trading_optimization.hyperparameter.manager import TuningManager
import numpy as np

# Create tuning manager
tuning_manager = TuningManager(config.get('hyperparameter', {}))

# First stage: Coarse search
coarse_search_space = {
    "hidden_size": {"type": "integer", "low": 32, "high": 512, "log": True},
    "num_layers": {"type": "integer", "low": 1, "high": 5},
    "dropout": {"type": "range", "low": 0.0, "high": 0.5},
    "learning_rate": {"type": "range", "low": 1e-5, "high": 1e-1, "log": True}
}

coarse_space = tuning_manager.create_search_space(coarse_search_space)

# Run first stage
coarse_results = tuning_manager.tune(
    model_type="lstm",
    search_space=coarse_space,
    objective="val_loss",
    strategy="bayesian",
    max_trials=20,
    checkpoint_dir="./checkpoints/stage1"
)

# Get best parameters from first stage
best_params = tuning_manager.get_best_params(coarse_results)

# Create refined search space around best parameters
refined_search_space = {}

# For each parameter, create a narrower search space around the best value
for param, value in best_params.items():
    if param == "hidden_size":
        # Integer parameter, create range around best value
        low = max(32, int(value * 0.5))
        high = min(512, int(value * 1.5))
        refined_search_space[param] = {"type": "integer", "low": low, "high": high}
    elif param == "num_layers":
        # Integer parameter, create small range around best value
        low = max(1, value - 1)
        high = min(5, value + 1)
        refined_search_space[param] = {"type": "integer", "low": low, "high": high}
    elif param == "dropout":
        # Real parameter, create narrower range around best value
        low = max(0.0, value - 0.1)
        high = min(0.5, value + 0.1)
        refined_search_space[param] = {"type": "range", "low": low, "high": high}
    elif param == "learning_rate":
        # Log-scale parameter, create narrower range around best value
        low = max(1e-5, value / 3)
        high = min(1e-1, value * 3)
        refined_search_space[param] = {"type": "range", "low": low, "high": high, "log": True}

# Create refined search space
refined_space = tuning_manager.create_search_space(refined_search_space)

# Run second stage with finer search
refined_results = tuning_manager.tune(
    model_type="lstm",
    search_space=refined_space,
    objective="val_loss",
    strategy="bayesian",
    max_trials=30,
    checkpoint_dir="./checkpoints/stage2"
)

# Get final best parameters
final_best_params = tuning_manager.get_best_params(refined_results)
print(f"Final best hyperparameters: {final_best_params}")
```

## 7. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. Configuration management system is implemented
3. Data management module is implemented
4. Model training module is implemented
5. Database infrastructure is implemented (optional but recommended)
6. Required libraries are installed:
   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
   - ray[tune] (for distributed execution)

## 8. Implementation Sequence

1. Set up the directory structure
2. Implement the parameter space definitions
3. Create basic search strategies (random, grid)
4. Develop the trial execution framework
5. Implement the advanced search strategies (Bayesian, HyperBand)
6. Create visualization and analysis utilities
7. Integrate with Ray for distributed execution
8. Implement the high-level tuning manager
9. Add comprehensive unit tests
10. Create integration tests with data and model components

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# Example unit tests for parameter space

import unittest
import numpy as np

from trading_optimization.hyperparameter.space.parameter import (
    CategoricalParameter, 
    RangeParameter, 
    IntegerParameter,
    SearchSpace
)

class TestParameterSpace(unittest.TestCase):
    
    def test_categorical_parameter(self):
        """Test categorical parameter behavior."""
        values = ['relu', 'tanh', 'sigmoid']
        weights = [0.6, 0.3, 0.1]
        
        # Create parameter
        param = CategoricalParameter('activation', values, weights)
        
        # Check properties
        self.assertEqual(param.name, 'activation')
        self.assertEqual(param.values, values)
        np.testing.assert_allclose(param.weights, np.array([0.6, 0.3, 0.1]))
        
        # Test sampling
        np.random.seed(42)  # For reproducibility
        samples = [param.sample() for _ in range(100)]
        
        # Check that all samples are valid values
        for sample in samples:
            self.assertIn(sample, values)
        
        # Check that weights are approximately respected
        counts = {v: samples.count(v) for v in values}
        for value, weight in zip(values, weights):
            expected = weight * 100
            actual = counts[value]
            # Allow for some random variation
            self.assertLess(abs(expected - actual), 20)
    
    def test_range_parameter(self):
        """Test range parameter behavior."""
        # Create parameter
        param = RangeParameter('learning_rate', 0.001, 0.1, log=True)
        
        # Check properties
        self.assertEqual(param.name, 'learning_rate')
        self.assertEqual(param.low, 0.001)
        self.assertEqual(param.high, 0.1)
        self.assertTrue(param.log)
        
        # Test sampling
        np.random.seed(42)  # For reproducibility
        samples = [param.sample() for _ in range(100)]
        
        # Check that all samples are within range
        for sample in samples:
            self.assertGreaterEqual(sample, param.low)
            self.assertLessEqual(sample, param.high)
        
        # For log scale, check that we have distribution of values
        log_samples = np.log10(samples)
        # Standard deviation should be significant
        self.assertGreater(np.std(log_samples), 0.1)
    
    def test_search_space(self):
        """Test search space behavior."""
        # Create parameters
        p1 = CategoricalParameter('activation', ['relu', 'tanh'])
        p2 = RangeParameter('learning_rate', 0.001, 0.1)
        p3 = IntegerParameter('hidden_size', 32, 256)
        
        # Create search space
        space = SearchSpace([p1, p2, p3])
        
        # Check parameter retrieval
        self.assertEqual(space.get_parameter('activation'), p1)
        self.assertEqual(space.get_parameter('learning_rate'), p2)
        self.assertEqual(space.get_parameter('hidden_size'), p3)
        
        # Test sampling
        np.random.seed(42)  # For reproducibility
        sample = space.sample()
        
        # Check that sample contains all parameters
        self.assertIn('activation', sample)
        self.assertIn('learning_rate', sample)
        self.assertIn('hidden_size', sample)
        
        # Check value types
        self.assertIsInstance(sample['activation'], str)
        self.assertIsInstance(sample['learning_rate'], float)
        self.assertIsInstance(sample['hidden_size'], int)
        
        # Test serialization/deserialization
        json_str = space.to_json()
        loaded_space = SearchSpace.from_json(json_str)
        
        # Check that loaded space has same parameters
        self.assertEqual(len(loaded_space.parameters), 3)
        self.assertIsNotNone(loaded_space.get_parameter('activation'))
        self.assertIsNotNone(loaded_space.get_parameter('learning_rate'))
        self.assertIsNotNone(loaded_space.get_parameter('hidden_size'))
```

### 9.2 Integration Tests

```python
# Example integration tests for hyperparameter tuning

import unittest
import tempfile
import shutil
import os

from trading_optimization.config import ConfigManager
from trading_optimization.hyperparameter.manager import TuningManager

class TestHyperparameterTuning(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal configuration
        self.config = {
            'hyperparameter': {
                'max_trials': 5,  # Use small number for testing
                'n_concurrent': 1
            }
        }
        
        # Create tuning manager with test config
        self.tuning_manager = TuningManager(self.config['hyperparameter'])
        
        # Define a simple objective function for testing
        def test_objective(params):
            """Simple quadratic objective function."""
            x = params.get('x', 0)
            y = params.get('y', 0)
            return (x - 3) ** 2 + (y - 4) ** 2
        
        self.objective_fn = test_objective
        
        # Create a simple search space
        self.search_space_config = {
            'x': {'type': 'range', 'low': -5, 'high': 10},
            'y': {'type': 'range', 'low': -5, 'high': 10}
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_random_search(self):
        """Test random search strategy."""
        search_space = self.tuning_manager.create_search_space(self.search_space_config)
        
        # Run random search
        results = self.tuning_manager.tune(
            model_type="test_model",
            search_space=search_space,
            objective=self.objective_fn,
            strategy="random",
            max_trials=5,
            checkpoint_dir=os.path.join(self.temp_dir, "random"),
            use_ray=False  # Disable Ray for unit testing
        )
        
        # Check basic results structure
        self.assertIn('best_params', results)
        self.assertIn('best_result', results)
        self.assertIn('all_trials', results)
        self.assertEqual(results['completed_trials'], 5)
        
        # Check trial results
        all_trials = results['all_trials']
        self.assertEqual(len(all_trials), 5)
        
        # Check best result is reasonable
        best_value = results['best_result']['objective_value']
        self.assertGreaterEqual(best_value, 0)  # Minimum is 0 at x=3, y=4
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization strategy."""
        search_space = self.tuning_manager.create_search_space(self.search_space_config)
        
        # Run Bayesian optimization
        results = self.tuning_manager.tune(
            model_type="test_model",
            search_space=search_space,
            objective=self.objective_fn,
            strategy="bayesian",
            max_trials=5,
            checkpoint_dir=os.path.join(self.temp_dir, "bayesian"),
            use_ray=False  # Disable Ray for unit testing
        )
        
        # Check basic results structure
        self.assertIn('best_params', results)
        self.assertIn('best_result', results)
        self.assertIn('all_trials', results)
        self.assertEqual(results['completed_trials'], 5)
        
        # Check trial results
        all_trials = results['all_trials']
        self.assertEqual(len(all_trials), 5)
        
        # Check best result is reasonable
        best_value = results['best_result']['objective_value']
        self.assertGreaterEqual(best_value, 0)  # Minimum is 0 at x=3, y=4
    
    def test_results_analysis(self):
        """Test results analysis functionality."""
        search_space = self.tuning_manager.create_search_space(self.search_space_config)
        
        # Run optimization
        results = self.tuning_manager.tune(
            model_type="test_model",
            search_space=search_space,
            objective=self.objective_fn,
            strategy="random",
            max_trials=10,  # Need more trials for meaningful analysis
            checkpoint_dir=os.path.join(self.temp_dir, "analysis_test"),
            use_ray=False
        )
        
        # Run analysis
        analysis = self.tuning_manager.analyze_results(
            results,
            output_dir=os.path.join(self.temp_dir, "analysis_output")
        )
        
        # Check analysis results
        self.assertIn('trials_summary', analysis)
        self.assertIn('parameter_importance', analysis)
        
        # Check that output files were created
        viz_files = analysis.get('visualizations', {})
        for path in viz_files.values():
            self.assertTrue(os.path.exists(path))
```

## 10. Integration with Other Components

The Hyperparameter Tuning System integrates with:

1. **Data Management Module**: To access and prepare data for model training.
2. **Model Training Module**: To create and train models with specific hyperparameters.
3. **Results Database Infrastructure**: To store and retrieve tuning results and metadata.

Integration is primarily through the TuningManager, which encapsulates these dependencies and provides a clean interface for hyperparameter optimization.

## 11. Extension Points

The module is designed to be easily extended:

1. **New Search Strategies**:
   - Create new classes that inherit from SearchStrategy
   - Register them with the SearchStrategyFactory

2. **Custom Parameter Types**:
   - Create new classes that inherit from Parameter
   - Implement the required sampling methods

3. **Advanced Visualizations**:
   - Add new visualization methods to TuningVisualizer

4. **Integration with Additional Frameworks**:
   - Create adapters for other optimization frameworks beyond Ray Tune