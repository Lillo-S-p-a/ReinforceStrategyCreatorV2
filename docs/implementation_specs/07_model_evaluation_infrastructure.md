# Model Evaluation Infrastructure: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Model Evaluation Infrastructure of the Trading Model Optimization Pipeline. This component is responsible for comprehensive evaluation of trained models using various metrics relevant to trading, statistical analysis of model outputs, backtesting against historical data, and generating detailed evaluation reports.

## 2. Component Responsibilities

The Model Evaluation Infrastructure is responsible for:

- Evaluating model performance with standard regression/classification metrics
- Calculating trading-specific performance metrics (e.g., Sharpe ratio, Calmar ratio)
- Providing backtesting functionality against historical data
- Handling model comparison and ranking
- Generating detailed evaluation reports and visualizations
- Supporting ensemble model evaluation
- Facilitating A/B testing of different models
- Tracking evaluation metrics in the results database

## 3. Architecture

### 3.1 Overall Architecture

The Model Evaluation Infrastructure is organized with the following components:

```
┌────────────────────────────────┐
│      Evaluation Manager        │  High-level API and facade
├────────────────────────────────┤
│                                │
│  ┌─────────────┐ ┌───────────┐ │
│  │   Metric    │ │  Model    │ │  Core evaluation capabilities
│  │  Calculator │ │ Comparer  │ │
│  └─────────────┘ └───────────┘ │
│                                │
│  ┌─────────────┐ ┌───────────┐ │
│  │ BackTesting │ │ Reporting │ │  Specialized functionality
│  │    Engine   │ │  Engine   │ │
│  └─────────────┘ └───────────┘ │
│                                │
├────────────────────────────────┤
│    Statistical Analysis Tool   │  Support tooling
└────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── evaluation/
    ├── __init__.py
    ├── manager.py               # High-level evaluation manager
    ├── metrics/
    │   ├── __init__.py
    │   ├── base.py              # Base metric classes
    │   ├── standard.py          # Standard ML metrics
    │   ├── trading.py           # Trading-specific metrics
    │   ├── custom.py            # Custom/user-defined metrics
    │   └── registry.py          # Metric registry
    ├── backtesting/
    │   ├── __init__.py
    │   ├── engine.py            # Backtesting core logic
    │   ├── data_handler.py      # Historical data handlers
    │   ├── strategy.py          # Strategy implementation
    │   ├── simulator.py         # Market simulator
    │   ├── position.py          # Position tracking
    │   └── risk_manager.py      # Risk management
    ├── comparison/
    │   ├── __init__.py
    │   ├── comparer.py          # Model comparison
    │   ├── ranking.py           # Model ranking
    │   ├── ensemble.py          # Ensemble evaluation
    │   └── significance.py      # Statistical significance tests
    ├── reporting/
    │   ├── __init__.py
    │   ├── report_generator.py  # Report generation
    │   ├── visualizer.py        # Visualization utilities
    │   ├── templates/           # Report templates
    │   │   ├── standard.html
    │   │   ├── trading.html
    │   │   └── comparison.html
    │   └── exporters/           # Report export formats
    │       ├── __init__.py
    │       ├── html.py
    │       ├── pdf.py
    │       └── json.py
    ├── statistical/
    │   ├── __init__.py
    │   ├── hypothesis.py        # Hypothesis tests
    │   ├── distribution.py      # Distribution analysis
    │   ├── correlation.py       # Correlation analysis
    │   └── residual.py          # Residual analysis
    └── utils/
        ├── __init__.py
        ├── data_loader.py       # Evaluation data loading utilities
        └── persistence.py       # Evaluation result persistence
```

## 4. Core Components Design

### 4.1 Evaluation Manager

The high-level interface for model evaluation:

```python
# manager.py
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np

from trading_optimization.models.interface import Model
from trading_optimization.evaluation.metrics.registry import MetricRegistry
from trading_optimization.evaluation.backtesting.engine import BacktestEngine
from trading_optimization.evaluation.comparison.comparer import ModelComparer
from trading_optimization.evaluation.reporting.report_generator import ReportGenerator
from trading_optimization.db.repository import EvaluationResultRepository
from trading_optimization.config import ConfigManager

class EvaluationManager:
    """
    High-level interface for model evaluation.
    Acts as a facade for model evaluation functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Evaluation Manager.
        
        Args:
            config: Configuration dictionary with evaluation settings
        """
        self.config = config or ConfigManager.instance().get('evaluation', {})
        self.metric_registry = MetricRegistry()
        
        # Initialize database connection if available
        try:
            from trading_optimization.db.connectors import DatabaseConnector
            db_connector = DatabaseConnector.instance()
            with db_connector.session() as session:
                self.result_repo = EvaluationResultRepository(session)
        except Exception as e:
            print(f"Warning: Could not connect to database: {str(e)}")
            self.result_repo = None
    
    def evaluate_model(
        self, 
        model: Model,
        model_id: str,
        data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model: Model instance to evaluate
            model_id: Unique identifier for the model
            data: Dictionary with evaluation data
                (must include at least X_test and y_test)
            metrics: List of metric names to calculate
                (if None, use default metrics from config)
            output_dir: Directory to save evaluation results
            save_results: Whether to save results to database
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate unique ID for this evaluation
        evaluation_id = str(uuid.uuid4())
        
        # Use default metrics if not specified
        if metrics is None:
            metrics = self.config.get('default_metrics', 
                ['rmse', 'mae', 'r2', 'sharpe_ratio'])
        
        # Calculate all specified metrics
        metric_results = self._calculate_metrics(
            model=model, 
            data=data, 
            metric_names=metrics
        )
        
        # Create full evaluation result
        timestamp = datetime.now().isoformat()
        evaluation_result = {
            'evaluation_id': evaluation_id,
            'model_id': model_id,
            'timestamp': timestamp,
            'metrics': metric_results,
            'model_type': model.__class__.__name__
        }
        
        # Add feature importance if possible
        feature_importance = self._extract_feature_importance(model)
        if feature_importance:
            evaluation_result['feature_importance'] = feature_importance
        
        # Add prediction samples
        try:
            # Get a sample of predictions for visualization
            X_test = data.get('X_test')
            if X_test is not None:
                if isinstance(X_test, pd.DataFrame) or isinstance(X_test, np.ndarray):
                    sample_size = min(100, len(X_test))
                    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
                    X_sample = X_test[sample_indices]
                    
                    # Get predictions
                    y_pred_sample = model.predict(X_sample).tolist()
                    
                    # Get actual values if available
                    y_test = data.get('y_test')
                    if y_test is not None:
                        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, np.ndarray):
                            y_true_sample = y_test[sample_indices].tolist()
                            evaluation_result['prediction_samples'] = {
                                'indices': sample_indices.tolist(),
                                'y_pred': y_pred_sample,
                                'y_true': y_true_sample
                            }
                        else:
                            evaluation_result['prediction_samples'] = {
                                'indices': sample_indices.tolist(),
                                'y_pred': y_pred_sample
                            }
        except Exception as e:
            print(f"Warning: Failed to extract prediction samples: {str(e)}")
        
        # Save to database if requested
        if save_results and self.result_repo:
            try:
                self.result_repo.create(
                    evaluation_id=evaluation_id,
                    model_id=model_id,
                    metrics=metric_results,
                    timestamp=timestamp,
                    model_type=model.__class__.__name__,
                    metadata=evaluation_result
                )
                evaluation_result['saved_to_db'] = True
            except Exception as e:
                print(f"Warning: Failed to save evaluation results to database: {str(e)}")
                evaluation_result['saved_to_db'] = False
        
        # Save to file if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"evaluation_{evaluation_id}.json")
            try:
                with open(output_file, 'w') as f:
                    json.dump(evaluation_result, f, indent=2)
                evaluation_result['saved_to_file'] = output_file
            except Exception as e:
                print(f"Warning: Failed to save evaluation results to file: {str(e)}")
        
        return evaluation_result
        
    def evaluate_multiple_models(
        self, 
        models: Dict[str, Model],
        data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models with the same data.
        
        Args:
            models: Dictionary mapping model IDs to Model instances
            data: Dictionary with evaluation data
            metrics: List of metric names to calculate
            output_dir: Directory to save evaluation results
            save_results: Whether to save results to database
            
        Returns:
            Dictionary mapping model IDs to evaluation results
        """
        results = {}
        
        # Evaluate each model individually
        for model_id, model in models.items():
            model_result = self.evaluate_model(
                model=model,
                model_id=model_id,
                data=data,
                metrics=metrics,
                output_dir=output_dir,
                save_results=save_results
            )
            results[model_id] = model_result
        
        # Create a comparison report if there are multiple models
        if len(models) > 1:
            comparison = self.compare_models(
                evaluation_results=results,
                output_dir=output_dir
            )
            # Include comparison summary in results
            for model_id in results:
                results[model_id]['comparison_rank'] = comparison['rankings'].get(model_id)
            
            # Save full comparison report
            if output_dir:
                comparison_file = os.path.join(output_dir, "model_comparison.json")
                try:
                    with open(comparison_file, 'w') as f:
                        json.dump(comparison, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save comparison results: {str(e)}")
        
        return results
    
    def compare_models(
        self, 
        evaluation_results: Dict[str, Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple evaluated models.
        
        Args:
            evaluation_results: Dictionary mapping model IDs to evaluation results
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary with comparison results
        """
        # Create model comparer
        comparer = ModelComparer(self.config.get('comparison', {}))
        
        # Extract metrics from evaluation results
        model_metrics = {}
        for model_id, result in evaluation_results.items():
            metrics = result.get('metrics', {})
            model_metrics[model_id] = metrics
        
        # Run comparison
        comparison_result = comparer.compare(model_metrics)
        
        # Generate comparison report if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create report generator
            report_generator = ReportGenerator()
            
            # Generate HTML comparison report
            report_path = os.path.join(output_dir, "model_comparison_report.html")
            report_generator.generate_comparison_report(
                comparison_result, 
                evaluation_results,
                output_path=report_path
            )
            comparison_result['report_path'] = report_path
        
        return comparison_result
    
    def backtest_model(
        self,
        model: Model,
        model_id: str,
        data: Dict[str, Any],
        strategy_config: Dict[str, Any],
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtesting for a trading model.
        
        Args:
            model: Model instance to backtest
            model_id: Unique identifier for the model
            data: Dictionary with historical data
            strategy_config: Trading strategy configuration
            output_dir: Directory to save backtesting results
            save_results: Whether to save results to database
            
        Returns:
            Dictionary with backtesting results
        """
        # Generate unique ID for this backtest
        backtest_id = str(uuid.uuid4())
        
        # Create backtest engine
        engine = BacktestEngine(
            config=strategy_config,
            model=model
        )
        
        # Prepare data for backtesting
        historical_data = data.get('historical_data')
        if historical_data is None:
            raise ValueError("Historical data must be provided for backtesting")
        
        # Run backtest
        backtest_result = engine.run(historical_data)
        
        # Add metadata
        timestamp = datetime.now().isoformat()
        backtest_result.update({
            'backtest_id': backtest_id,
            'model_id': model_id,
            'timestamp': timestamp,
            'model_type': model.__class__.__name__,
            'strategy_config': strategy_config
        })
        
        # Save to database if requested
        if save_results and self.result_repo:
            try:
                self.result_repo.create_backtest_result(
                    backtest_id=backtest_id,
                    model_id=model_id,
                    timestamp=timestamp,
                    strategy_name=strategy_config.get('strategy_name', 'default'),
                    metrics=backtest_result.get('metrics', {}),
                    trades=backtest_result.get('trades', []),
                    equity_curve=backtest_result.get('equity_curve', []).tolist(),
                    metadata=backtest_result
                )
                backtest_result['saved_to_db'] = True
            except Exception as e:
                print(f"Warning: Failed to save backtest results to database: {str(e)}")
                backtest_result['saved_to_db'] = False
        
        # Save to file if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"backtest_{backtest_id}.json")
            try:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = self._make_serializable(backtest_result)
                with open(output_file, 'w') as f:
                    json.dump(serializable_result, f, indent=2)
                backtest_result['saved_to_file'] = output_file
            except Exception as e:
                print(f"Warning: Failed to save backtest results to file: {str(e)}")
        
        # Generate report
        if output_dir:
            report_generator = ReportGenerator()
            report_path = os.path.join(output_dir, f"backtest_report_{backtest_id}.html")
            report_generator.generate_backtest_report(
                backtest_result,
                output_path=report_path
            )
            backtest_result['report_path'] = report_path
        
        return backtest_result
    
    def generate_evaluation_report(
        self,
        evaluation_result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate a detailed evaluation report.
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Create report generator
        report_generator = ReportGenerator()
        
        # Generate the report
        report_path = report_generator.generate_evaluation_report(
            evaluation_result,
            output_path=output_path
        )
        
        return report_path
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an evaluation result from the database.
        
        Args:
            evaluation_id: ID of the evaluation to retrieve
            
        Returns:
            Evaluation result dictionary or None if not found
        """
        if not self.result_repo:
            print("Warning: Database connection not available")
            return None
        
        try:
            result = self.result_repo.get_by_id(evaluation_id)
            return result
        except Exception as e:
            print(f"Error retrieving evaluation result: {str(e)}")
            return None
    
    def get_evaluations_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all evaluation results for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of evaluation result dictionaries
        """
        if not self.result_repo:
            print("Warning: Database connection not available")
            return []
        
        try:
            results = self.result_repo.get_by_model_id(model_id)
            return results
        except Exception as e:
            print(f"Error retrieving evaluation results: {str(e)}")
            return []
    
    def _calculate_metrics(
        self, 
        model: Model, 
        data: Dict[str, Any], 
        metric_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate multiple metrics for model evaluation.
        
        Args:
            model: Model instance
            data: Dictionary with evaluation data
            metric_names: List of metric names to calculate
            
        Returns:
            Dictionary mapping metric names to values
        """
        results = {}
        
        # Check for required data
        X_test = data.get('X_test')
        y_test = data.get('y_test')
        
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test are required for metric calculation")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate each requested metric
        for metric_name in metric_names:
            # Get metric function from registry
            metric_func = self.metric_registry.get_metric(metric_name)
            if metric_func is None:
                print(f"Warning: Metric '{metric_name}' not found in registry")
                continue
                
            try:
                # Check if this is a trading metric that needs additional data
                if metric_name in self.metric_registry.get_trading_metrics():
                    # Trading metrics may need additional data like price series
                    metric_value = metric_func(
                        y_true=y_test,
                        y_pred=y_pred,
                        **{k: v for k, v in data.items() if k not in ['X_test', 'y_test']}
                    )
                else:
                    # Standard metrics typically just need y_true and y_pred
                    metric_value = metric_func(y_true=y_test, y_pred=y_pred)
                
                results[metric_name] = float(metric_value)
            except Exception as e:
                print(f"Error calculating metric '{metric_name}': {str(e)}")
                results[metric_name] = None
        
        return results
    
    def _extract_feature_importance(self, model: Model) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model if possible.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary mapping feature names to importance values,
            or None if feature importance is not available
        """
        try:
            # Try to get feature importance
            if hasattr(model, 'feature_importance') and callable(getattr(model, 'feature_importance')):
                # If model has feature_importance method, call it
                return model.feature_importance()
            elif hasattr(model.model, 'feature_importances_'):
                # For sklearn-like models
                importances = model.model.feature_importances_
                feature_names = getattr(model, 'feature_names_', None)
                
                if feature_names is None:
                    # If no feature names available, use indices
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            elif hasattr(model.model, 'coef_'):
                # For linear models
                coefficients = model.model.coef_
                feature_names = getattr(model, 'feature_names_', None)
                
                if feature_names is None:
                    # If no feature names available, use indices
                    if len(coefficients.shape) > 1:
                        feature_names = [f'feature_{i}' for i in range(coefficients.shape[1])]
                        coefficients = coefficients[0]
                    else:
                        feature_names = [f'feature_{i}' for i in range(len(coefficients))]
                
                # Take absolute values for importance
                importances = np.abs(coefficients)
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
        except Exception as e:
            print(f"Warning: Failed to extract feature importance: {str(e)}")
        
        return None
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert numpy arrays and other non-serializable objects
        to serializable Python types.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'to_dict'):
            try:
                # For pandas objects and others with to_dict method
                return self._make_serializable(obj.to_dict())
            except:
                return str(obj)
        else:
            # Default fallback, convert to string
            return str(obj)
```

### 4.2 Metric System

Base metric classes and standard metric implementations:

```python
# metrics/base.py
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    
    def __init__(self, name: str, higher_is_better: bool = True):
        """
        Initialize metric.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
                (affects ranking in model comparison)
        """
        self.name = name
        self.higher_is_better = higher_is_better
    
    @abstractmethod
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate metric value.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            **kwargs: Additional data that may be needed
            
        Returns:
            Calculated metric value
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get metric configuration.
        
        Returns:
            Dictionary with metric configuration
        """
        return {
            'name': self.name,
            'higher_is_better': self.higher_is_better,
            'class': self.__class__.__name__
        }


class StandardMetric(Metric):
    """
    Standard metric that wraps a function from a library.
    """
    
    def __init__(
        self, 
        name: str, 
        func: Callable,
        higher_is_better: bool = True,
        **kwargs
    ):
        """
        Initialize standard metric.
        
        Args:
            name: Metric name
            func: Function that calculates the metric
            higher_is_better: Whether higher values are better
            **kwargs: Additional arguments to pass to the function
        """
        super().__init__(name, higher_is_better)
        self.func = func
        self.kwargs = kwargs
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate metric value.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            **kwargs: Additional data that may be needed
            
        Returns:
            Calculated metric value
        """
        # Combine instance kwargs with call kwargs (call kwargs take precedence)
        all_kwargs = {**self.kwargs, **kwargs}
        return self.func(y_true, y_pred, **all_kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get metric configuration.
        
        Returns:
            Dictionary with metric configuration
        """
        config = super().get_config()
        config.update({
            'func': self.func.__name__,
            'kwargs': self.kwargs
        })
        return config


class CompositeMetric(Metric):
    """
    Metric composed of multiple metrics combined with a function.
    """
    
    def __init__(
        self, 
        name: str, 
        metrics: List[Metric],
        combiner: Callable[[List[float]], float],
        higher_is_better: bool = True
    ):
        """
        Initialize composite metric.
        
        Args:
            name: Metric name
            metrics: List of metrics to combine
            combiner: Function to combine metric values
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
        self.metrics = metrics
        self.combiner = combiner
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate metric value.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            **kwargs: Additional data that may be needed
            
        Returns:
            Calculated metric value
        """
        # Calculate individual metrics
        values = [metric(y_true, y_pred, **kwargs) for metric in self.metrics]
        # Combine values
        return self.combiner(values)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get metric configuration.
        
        Returns:
            Dictionary with metric configuration
        """
        config = super().get_config()
        config.update({
            'metrics': [metric.get_config() for metric in self.metrics],
            'combiner': self.combiner.__name__
        })
        return config
```

Standard metrics implementation:

```python
# metrics/standard.py
from typing import Any, Callable, Dict, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    balanced_accuracy_score
)

from trading_optimization.evaluation.metrics.base import StandardMetric, Metric

def rmse(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Root mean squared error.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def wmape(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """
    Weighted Mean Absolute Percentage Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        WMAPE value
    """
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))

def r2_adj(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_features: Optional[int] = None,
    **kwargs
) -> float:
    """
    Adjusted R-squared.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        n_features: Number of features in the model
        
    Returns:
        Adjusted R-squared value
    """
    n = len(y_true)
    if n_features is None:
        n_features = kwargs.get('X_test', np.empty((n, 1))).shape[1]
    
    r2 = r2_score(y_true, y_pred)
    
    # Adjust for number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return float(adj_r2)

# Standard Regression Metrics
regression_metrics = {
    'rmse': StandardMetric('rmse', rmse, higher_is_better=False),
    'mae': StandardMetric('mae', mean_absolute_error, higher_is_better=False),
    'mape': StandardMetric('mape', mean_absolute_percentage_error, higher_is_better=False),
    'wmape': StandardMetric('wmape', wmape, higher_is_better=False),
    'r2': StandardMetric('r2', r2_score, higher_is_better=True),
    'r2_adj': StandardMetric('r2_adj', r2_adj, higher_is_better=True),
    'explained_variance': StandardMetric('explained_variance', explained_variance_score, higher_is_better=True),
    'max_error': StandardMetric('max_error', max_error, higher_is_better=False)
}

# Standard Classification Metrics
classification_metrics = {
    'accuracy': StandardMetric('accuracy', accuracy_score, higher_is_better=True),
    'balanced_accuracy': StandardMetric('balanced_accuracy', balanced_accuracy_score, higher_is_better=True),
    'precision': StandardMetric('precision', precision_score, higher_is_better=True),
    'recall': StandardMetric('recall', recall_score, higher_is_better=True),
    'f1': StandardMetric('f1', f1_score, higher_is_better=True),
    'roc_auc': StandardMetric('roc_auc', roc_auc_score, higher_is_better=True),
    'log_loss': StandardMetric('log_loss', log_loss, higher_is_better=False)
}

# All standard metrics
standard_metrics = {**regression_metrics, **classification_metrics}
```

Trading-specific metrics:

```python
# metrics/trading.py
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from trading_optimization.evaluation.metrics.base import Metric

class SharpeRatio(Metric):
    """
    Sharpe ratio metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'sharpe_ratio',
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
        higher_is_better: bool = True
    ):
        """
        Initialize Sharpe ratio metric.
        
        Args:
            name: Metric name
            risk_free_rate: Risk-free rate (annual)
            annualization_factor: Factor to annualize returns
                (252 for daily returns, 12 for monthly, etc.)
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            y_true: True price changes/returns (not used directly)
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional actual returns series
                  (if provided, used instead of constructing from y_true/y_pred)
            
        Returns:
            Sharpe ratio
        """
        if 'returns' in kwargs:
            returns = kwargs['returns']
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-dimensional predictions, use the first column as signal
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                # Multi-dimensional targets, use the first column as price changes
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            # (signals from previous period applied to current price change)
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
        
        # Calculate mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0  # Avoid division by zero
        
        # Daily Sharpe (assuming returns are daily)
        daily_sharpe = (mean_return - self.risk_free_rate / self.annualization_factor) / std_return
        
        # Annualize
        sharpe = daily_sharpe * np.sqrt(self.annualization_factor)
        
        return float(sharpe)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get metric configuration.
        
        Returns:
            Dictionary with metric configuration
        """
        config = super().get_config()
        config.update({
            'risk_free_rate': self.risk_free_rate,
            'annualization_factor': self.annualization_factor
        })
        return config


class SortinoRatio(Metric):
    """
    Sortino ratio metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'sortino_ratio',
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
        higher_is_better: bool = True
    ):
        """
        Initialize Sortino ratio metric.
        
        Args:
            name: Metric name
            risk_free_rate: Risk-free rate (annual)
            annualization_factor: Factor to annualize returns
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate Sortino ratio from returns.
        
        Args:
            y_true: True price changes/returns (not used directly)
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional actual returns series
            
        Returns:
            Sortino ratio
        """
        if 'returns' in kwargs:
            returns = kwargs['returns']
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
        
        # Calculate mean return
        mean_return = np.mean(returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            # No negative returns, use a small value to avoid division by zero
            downside_deviation = 0.0001
        else:
            # Calculate downside deviation
            downside_deviation = np.std(downside_returns, ddof=1)
        
        # Daily Sortino (assuming returns are daily)
        daily_sortino = (mean_return - self.risk_free_rate / self.annualization_factor) / downside_deviation
        
        # Annualize
        sortino = daily_sortino * np.sqrt(self.annualization_factor)
        
        return float(sortino)


class MaxDrawdown(Metric):
    """
    Maximum drawdown metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'max_drawdown',
        higher_is_better: bool = False
    ):
        """
        Initialize maximum drawdown metric.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
                (typically False for drawdown)
        """
        super().__init__(name, higher_is_better)
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate maximum drawdown from returns or equity curve.
        
        Args:
            y_true: True price changes/returns (not used directly)
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional returns series
                - equity_curve: Optional equity curve
            
        Returns:
            Maximum drawdown as a positive fraction (e.g., 0.25 for 25% drawdown)
        """
        # Use equity curve if provided, otherwise calculate from returns
        if 'equity_curve' in kwargs:
            equity = kwargs['equity_curve']
        elif 'returns' in kwargs:
            returns = kwargs['returns']
            # Calculate equity curve from returns (starting with 1.0)
            equity = np.cumprod(1 + returns)
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
            
            # Calculate equity curve from returns
            equity = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown
        drawdown = (running_max - equity) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return float(max_drawdown)


class CalmarRatio(Metric):
    """
    Calmar ratio metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'calmar_ratio',
        lookback_years: float = 3.0,
        higher_is_better: bool = True
    ):
        """
        Initialize Calmar ratio metric.
        
        Args:
            name: Metric name
            lookback_years: Lookback period in years
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
        self.lookback_years = lookback_years
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            y_true: True price changes/returns
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional returns series
                - equity_curve: Optional equity curve
                - period_per_year: Number of periods per year (e.g., 252 for daily data)
            
        Returns:
            Calmar ratio
        """
        period_per_year = kwargs.get('period_per_year', 252)
        
        # Calculate returns if not provided
        if 'returns' in kwargs:
            returns = kwargs['returns']
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
        
        # Calculate equity curve if not provided
        if 'equity_curve' in kwargs:
            equity = kwargs['equity_curve']
        else:
            equity = np.cumprod(1 + returns)
        
        # Calculate annualized return
        total_return = equity[-1] / equity[0] - 1
        years = len(returns) / period_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate maximum drawdown
        max_drawdown = MaxDrawdown()(y_true, y_pred, equity_curve=equity)
        
        if max_drawdown == 0:
            return float('inf')  # Avoid division by zero
        
        # Calculate Calmar ratio
        calmar = annualized_return / max_drawdown
        
        return float(calmar)


class WinRate(Metric):
    """
    Win rate metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'win_rate',
        higher_is_better: bool = True
    ):
        """
        Initialize win rate metric.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate win rate.
        
        Args:
            y_true: True price changes/returns
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional returns series
                - trades: Optional list of trades with profit/loss info
            
        Returns:
            Win rate as a fraction (e.g., 0.6 for 60%)
        """
        if 'trades' in kwargs:
            trades = kwargs['trades']
            # Calculate win rate from trades
            wins = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            total = len(trades)
            
            if total == 0:
                return 0.0
                
            return float(wins / total)
            
        elif 'returns' in kwargs:
            returns = kwargs['returns']
            # Calculate win rate from returns series
            wins = np.sum(returns > 0)
            total = len(returns)
            
            if total == 0:
                return 0.0
                
            return float(wins / total)
            
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
            
            # Calculate win rate from returns
            wins = np.sum(returns > 0)
            total = np.sum(np.abs(signals[:-1]) > 0)  # Only count actual trades
            
            if total == 0:
                return 0.0
                
            return float(wins / total)


class ProfitFactor(Metric):
    """
    Profit factor metric for trading models.
    """
    
    def __init__(
        self, 
        name: str = 'profit_factor',
        higher_is_better: bool = True
    ):
        """
        Initialize profit factor metric.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
        """
        super().__init__(name, higher_is_better)
    
    def __call__(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        **kwargs
    ) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Args:
            y_true: True price changes/returns
            y_pred: Predicted returns or signals
            **kwargs: 
                - returns: Optional returns series
                - trades: Optional list of trades with profit/loss info
            
        Returns:
            Profit factor (>1 means profitable)
        """
        if 'trades' in kwargs:
            trades = kwargs['trades']
            # Calculate from trades
            gross_profits = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
            gross_losses = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
            
        elif 'returns' in kwargs:
            returns = kwargs['returns']
            # Calculate from returns series
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
            
        else:
            # Assume y_pred are signals (-1, 0, 1) and y_true are price changes
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                signals = y_pred[:, 0]
            else:
                signals = y_pred.flatten()
            
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                price_changes = y_true[:, 0]
            else:
                price_changes = y_true.flatten()
                
            # Generate returns from signals and price changes
            returns = np.zeros_like(price_changes)
            returns[1:] = signals[:-1] * price_changes[1:]
            
            # Calculate from returns
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
        
        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0
            
        return float(gross_profits / gross_losses)


# Dictionary of trading metrics
trading_metrics = {
    'sharpe_ratio': SharpeRatio(),
    'sortino_ratio': SortinoRatio(),
    'max_drawdown': MaxDrawdown(),
    'calmar_ratio': CalmarRatio(),
    'win_rate': WinRate(),
    'profit_factor': ProfitFactor()
}
```

Metric registry to provide centralized access to all metrics:

```python
# metrics/registry.py
from typing import Dict, List, Any, Optional, Union, Callable
from trading_optimization.evaluation.metrics.base import Metric
from trading_optimization.evaluation.metrics.standard import standard_metrics
from trading_optimization.evaluation.metrics.trading import trading_metrics

class MetricRegistry:
    """
    Registry for evaluation metrics.
    """
    
    def __init__(self):
        """
        Initialize registry with standard metrics.
        """
        self._metrics = {}
        
        # Register standard metrics
        for name, metric in standard_metrics.items():
            self.register_metric(name, metric)
        
        # Register trading metrics
        for name, metric in trading_metrics.items():
            self.register_metric(name, metric)
    
    def register_metric(
        self, 
        name: str, 
        metric: Union[Metric, Callable]
    ):
        """
        Register a metric.
        
        Args:
            name: Name to register the metric under
            metric: Metric instance or callable function
        """
        self._metrics[name] = metric
    
    def get_metric(
        self, 
        name: str
    ) -> Optional[Union[Metric, Callable]]:
        """
        Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric instance or callable, or None if not found
        """
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Union[Metric, Callable]]:
        """
        Get all registered metrics.
        
        Returns:
            Dictionary mapping metric names to instances/callables
        """
        return dict(self._metrics)
    
    def get_trading_metrics(self) -> List[str]:
        """
        Get names of all trading-specific metrics.
        
        Returns:
            List of trading metric names
        """
        return list(trading_metrics.keys())
    
    def get_standard_metrics(self) -> List[str]:
        """
        Get names of all standard metrics.
        
        Returns:
            List of standard metric names
        """
        return list(standard_metrics.keys())
```

### 4.3 Backtesting Engine

Core backtesting implementation:

```python
# backtesting/engine.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from trading_optimization.models.interface import Model
from trading_optimization.evaluation.backtesting.data_handler import DataHandler
from trading_optimization.evaluation.backtesting.strategy import TradingStrategy
from trading_optimization.evaluation.backtesting.simulator import MarketSimulator
from trading_optimization.evaluation.backtesting.position import PositionTracker
from trading_optimization.evaluation.backtesting.risk_manager import RiskManager

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[Model] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dictionary
            model: Optional ML model to use for predictions
        """
        self.config = config
        self.model = model
        
        # Default configuration values
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.commission = config.get('commission', 0.001)  # 0.1% by default
        self.slippage = config.get('slippage', 0.0005)    # 0.05% by default
        self.position_size_pct = config.get('position_size_pct', 0.1)  # 10% by default
        
        # Create components
        self.data_handler = DataHandler(config.get('data_handler', {}))
        
        # Create strategy (with or without model)
        strategy_config = config.get('strategy', {})
        if model is not None:
            strategy_config['model'] = model
        self.strategy = self._create_strategy(strategy_config)
        
        # Create market simulator
        simulator_config = config.get('simulator', {})
        simulator_config.update({
            'commission': self.commission,
            'slippage': self.slippage
        })
        self.simulator = MarketSimulator(simulator_config)
        
        # Create position tracker
        position_config = config.get('position', {})
        position_config.update({
            'initial_capital': self.initial_capital,
            'position_size_pct': self.position_size_pct
        })
        self.position_tracker = PositionTracker(position_config)
        
        # Create risk manager
        risk_config = config.get('risk', {})
        self.risk_manager = RiskManager(risk_config)
    
    def _create_strategy(self, strategy_config: Dict[str, Any]) -> TradingStrategy:
        """
        Create a trading strategy based on configuration.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Strategy instance
        """
        strategy_type = strategy_config.get('type', 'model_based')
        
        if strategy_type == 'model_based' and self.model is not None:
            from trading_optimization.evaluation.backtesting.strategy import ModelBasedStrategy
            return ModelBasedStrategy(strategy_config, model=self.model)
        elif strategy_type == 'moving_average':
            from trading_optimization.evaluation.backtesting.strategy import MovingAverageStrategy
            return MovingAverageStrategy(strategy_config)
        elif strategy_type == 'rsi':
            from trading_optimization.evaluation.backtesting.strategy import RSIStrategy
            return RSIStrategy(strategy_config)
        elif strategy_type == 'custom':
            # Attempt to load custom strategy class
            strategy_class = strategy_config.get('class')
            if strategy_class is not None:
                # Try to import the class dynamically
                try:
                    module_path, class_name = strategy_class.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    strategy_class = getattr(module, class_name)
                    return strategy_class(strategy_config)
                except Exception as e:
                    print(f"Error loading custom strategy class: {str(e)}")
        
        # Default to base strategy if all else fails
        print(f"Warning: Using default base strategy. Strategy type '{strategy_type}' not recognized.")
        return TradingStrategy(strategy_config)
    
    def run(
        self, 
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            data: Historical data for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        # Prepare data
        if isinstance(data, pd.DataFrame):
            # Single dataframe
            prepared_data = self.data_handler.prepare_data(data)
            asset_names = ['default']
        else:
            # Dictionary of dataframes (multi-asset)
            prepared_data = {
                asset: self.data_handler.prepare_data(df) 
                for asset, df in data.items()
            }
            asset_names = list(data.keys())
        
        # Initialize tracking variables
        start_time = datetime.now()
        all_positions = []
        all_trades = []
        equity_curve = []
        
        # Initialize position tracker
        self.position_tracker.reset(self.initial_capital)
        
        # For single asset
        if isinstance(prepared_data, pd.DataFrame):
            # Process each time step
            for i in range(1, len(prepared_data)):  # Start from 1 to have at least one lookback point
                current_data = prepared_data.iloc[:i+1]  # All data up to and including current point
                current_timestamp = current_data.index[-1]
                
                # Get current prices
                current_price = current_data.iloc[-1]['close']
                
                # Get strategy decision
                signal = self.strategy.generate_signal(current_data)
                
                # Apply risk management
                signal = self.risk_manager.adjust_signal(
                    signal, 
                    current_data, 
                    self.position_tracker.get_positions()
                )
                
                # Process the signal
                if signal != 0:
                    # Calculate position size
                    position_size = self.position_tracker.calculate_position_size(
                        asset='default',
                        price=current_price,
                        signal=signal
                    )
                    
                    if position_size > 0:
                        # Execute trade
                        trade_result = self.simulator.execute_trade(
                            asset='default',
                            price=current_price,
                            size=position_size,
                            direction=signal,
                            timestamp=current_timestamp
                        )
                        
                        # Update positions
                        self.position_tracker.update_position(
                            asset='default',
                            trade_result=trade_result
                        )
                        
                        # Record trade
                        all_trades.append(trade_result)
                
                # Record positions and equity at this step
                current_positions = self.position_tracker.get_positions()
                all_positions.append(current_positions)
                equity_curve.append(self.position_tracker.get_total_equity(current_data))
        else:
            # For multi-asset, use common timestamp index
            # Get a common index from all dataframes
            all_indices = [df.index for df in prepared_data.values()]
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            # Sort the common index
            common_index = sorted(common_index)
            
            # Process each time step
            for i, timestamp in enumerate(common_index):
                if i < 1:  # Skip first point to have some history
                    continue
                
                # Get current data for each asset
                current_data = {}
                current_prices = {}
                
                for asset, df in prepared_data.items():
                    # Get data up to current timestamp
                    current_data[asset] = df.loc[:timestamp]
                    current_prices[asset] = current_data[asset].iloc[-1]['close']
                
                # Get strategy decision for each asset
                signals = {}
                for asset, asset_data in current_data.items():
                    signals[asset] = self.strategy.generate_signal(
                        asset_data, 
                        asset=asset
                    )
                
                # Apply risk management
                signals = self.risk_manager.adjust_signals(
                    signals, 
                    current_data, 
                    self.position_tracker.get_positions()
                )
                
                # Process signals
                for asset, signal in signals.items():
                    if signal != 0:
                        # Calculate position size
                        position_size = self.position_tracker.calculate_position_size(
                            asset=asset,
                            price=current_prices[asset],
                            signal=signal
                        )
                        
                        if position_size > 0:
                            # Execute trade
                            trade_result = self.simulator.execute_trade(
                                asset=asset,
                                price=current_prices[asset],
                                size=position_size,
                                direction=signal,
                                timestamp=timestamp
                            )
                            
                            # Update positions
                            self.position_tracker.update_position(
                                asset=asset,
                                trade_result=trade_result
                            )
                            
                            # Record trade
                            all_trades.append(trade_result)
                
                # Record positions and equity at this step
                current_positions = self.position_tracker.get_positions()
                all_positions.append(current_positions)
                equity_curve.append(self.position_tracker.get_total_equity(
                    {asset: data.loc[timestamp:timestamp] for asset, data in prepared_data.items()}
                ))
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, all_trades)
        
        # Compile results
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': equity_curve[-1] if equity_curve else self.initial_capital,
            'total_return_pct': (
                (equity_curve[-1] / self.initial_capital - 1) * 100 
                if equity_curve else 0
            ),
            'metrics': metrics,
            'equity_curve': np.array(equity_curve),
            'trades': all_trades,
            'positions': all_positions,
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'backtest_config': self.config,
            'asset_names': asset_names
        }
        
        return results
    
    def _calculate_metrics(
        self, 
        equity_curve: List[float],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from equity curve and trades.
        
        Args:
            equity_curve: List of equity values
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        if not equity_curve or len(equity_curve) < 2:
            return {}
        
        equity_array = np.array(equity_curve)
        
        # Calculate returns
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Import metrics
        from trading_optimization.evaluation.metrics.registry import MetricRegistry
        registry = MetricRegistry()
        
        # Calculate trading metrics
        metrics = {}
        
        # Calculate Sharpe and Sortino ratios assuming daily data
        sharpe_ratio = registry.get_metric('sharpe_ratio')
        if sharpe_ratio:
            metrics['sharpe_ratio'] = sharpe_ratio(
                y_true=None, 
                y_pred=None, 
                returns=returns
            )
        
        sortino_ratio = registry.get_metric('sortino_ratio')
        if sortino_ratio:
            metrics['sortino_ratio'] = sortino_ratio(
                y_true=None, 
                y_pred=None, 
                returns=returns
            )
        
        # Calculate maximum drawdown
        max_drawdown = registry.get_metric('max_drawdown')
        if max_drawdown:
            metrics['max_drawdown'] = max_drawdown(
                y_true=None, 
                y_pred=None, 
                equity_curve=equity_array
            )
        
        # Calculate Calmar ratio
        calmar_ratio = registry.get_metric('calmar_ratio')
        if calmar_ratio:
            metrics['calmar_ratio'] = calmar_ratio(
                y_true=None, 
                y_pred=None, 
                returns=returns, 
                equity_curve=equity_array
            )
        
        # Calculate win rate
        win_rate = registry.get_metric('win_rate')
        if win_rate and trades:
            metrics['win_rate'] = win_rate(
                y_true=None, 
                y_pred=None, 
                trades=trades
            )
        
        # Calculate profit factor
        profit_factor = registry.get_metric('profit_factor')
        if profit_factor:
            metrics['profit_factor'] = profit_factor(
                y_true=None, 
                y_pred=None, 
                trades=trades
            )
        
        # Add basic statistics
        metrics['total_trades'] = len(trades)
        metrics['profit_trades'] = sum(1 for t in trades if t.get('profit', 0) > 0)
        metrics['loss_trades'] = sum(1 for t in trades if t.get('profit', 0) < 0)
        
        # Calculate total profit
        metrics['total_profit'] = sum(t.get('profit', 0) for t in trades)
        
        # Calculate annualized return (assuming 252 trading days per year)
        if len(equity_array) > 1:
            total_return = equity_array[-1] / equity_array[0] - 1
            years = len(equity_array) / 252  # Approximate number of years
            metrics['annualized_return'] = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        
        return metrics
```

Trading strategy base class and implementations:

```python
# backtesting/strategy.py
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

from trading_optimization.models.interface import Model

class TradingStrategy:
    """
    Base class for trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        
        # Initialize any shared parameters
        self.name = config.get('name', self.__class__.__name__)
    
    def generate_signal(
        self, 
        data: pd.DataFrame,
        asset: Optional[str] = None
    ) -> int:
        """
        Generate trading signal from data.
        
        Args:
            data: DataFrame with historical data
            asset: Optional asset identifier for multi-asset strategies
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for no action
        """
        # Base implementation returns no signal
        # Should be overridden by subclasses
        return 0


class ModelBasedStrategy(TradingStrategy):
    """
    Strategy that uses a machine learning model for signals.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        model: Model
    ):
        """
        Initialize model-based strategy.
        
        Args:
            config: Strategy configuration
            model: Model for predictions
        """
        super().__init__(config)
        self.model = model
        
        # Get strategy parameters
        self.prediction_threshold = config.get('prediction_threshold', 0.0)
        self.prediction_column = config.get('prediction_column', 0)  # Index if multi-output model
        self.lookback = config.get('lookback', 1)  # How many bars to look at
        
        # Features to use for prediction (if None, use all except target)
        self.features = config.get('features', None)
        
        # Threshold parameters for signal generation
        self.long_threshold = config.get('long_threshold', 0.001)  # Minimum prediction for long
        self.short_threshold = config.get('short_threshold', -0.001)  # Maximum prediction for short
        
    def generate_signal(
        self, 
        data: pd.DataFrame,
        asset: Optional[str] = None
    ) -> int:
        """
        Generate trading signal using model prediction.
        
        Args:
            data: DataFrame with historical data
            asset: Optional asset identifier for multi-asset strategies
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for no action
        """
        # Check if we have enough data
        if len(data) < self.lookback + 1:
            return 0
        
        try:
            # Prepare features for prediction
            latest_data = data.iloc[-self.lookback:]
            
            # Check if we need specific features
            if self.features:
                features = latest_data[self.features]
            else:
                # Use all columns except potential target columns
                exclude_cols = ['target', 'return', 'next_return']
                features = latest_data.drop(
                    columns=[col for col in exclude_cols if col in latest_data.columns],
                    errors='ignore'
                )
            
            # Reshape if needed (e.g., for LSTM models expecting 3D input)
            model_input = features
            
            # Get prediction
            prediction = self.model.predict(model_input)
            
            # Extract specific prediction if multi-output
            if isinstance(prediction, np.ndarray) and len(prediction.shape) > 1:
                if prediction.shape[1] > 1:
                    # Multi-output model, get specific column
                    prediction_value = prediction[0, self.prediction_column]
                else:
                    # Single output in 2D array
                    prediction_value = prediction[0, 0]
            else:
                # Single scalar output
                prediction_value = prediction[0]
            
            # Generate signal
            if prediction_value > self.long_threshold:
                return 1  # Buy signal
            elif prediction_value < self.short_threshold:
                return -1  # Sell signal
            else:
                return 0  # No action
                
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return 0


class MovingAverageStrategy(TradingStrategy):
    """
    Strategy based on moving average crossovers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize moving average strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Get strategy parameters
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 50)
        self.price_column = config.get('price_column', 'close')
    
    def generate_signal(
        self, 
        data: pd.DataFrame,
        asset: Optional[str] = None
    ) -> int:
        """
        Generate signal based on moving average crossover.
        
        Args:
            data: DataFrame with historical data
            asset: Optional asset identifier for multi-asset strategies
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for no action
        """
        # Check if we have enough data
        if len(data) < self.slow_period + 1:
            return 0
        
        try:
            # Calculate moving averages
            fast_ma = data[self.price_column].rolling(window=self.fast_period).mean()
            slow_ma = data[self.price_column].rolling(window=self.slow_period).mean()
            
            # Check for crossover
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            
            # Generate signal
            if prev_fast <= prev_slow and current_fast > current_slow:
                return 1  # Buy signal (fast MA crossed above slow MA)
            elif prev_fast >= prev_slow and current_fast < current_slow:
                return -1  # Sell signal (fast MA crossed below slow MA)
            else:
                return 0  # No crossover
                
        except Exception as e:
            print(f"Error generating MA signal: {str(e)}")
            return 0


class RSIStrategy(TradingStrategy):
    """
    Strategy based on Relative Strength Index (RSI).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RSI strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Get strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.price_column = config.get('price_column', 'close')
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)
    
    def generate_signal(
        self, 
        data: pd.DataFrame,
        asset: Optional[str] = None
    ) -> int:
        """
        Generate signal based on RSI indicator.
        
        Args:
            data: DataFrame with historical data
            asset: Optional asset identifier for multi-asset strategies
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for no action
        """
        # Check if we have enough data
        if len(data) < self.rsi_period + 10:  # Need some extra data for reliable RSI
            return 0
        
        try:
            # Calculate price changes
            delta = data[self.price_column].diff()
            
            # Calculate RSI
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get current and previous RSI values
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            # Generate signal
            if prev_rsi <= self.oversold and current_rsi > self.oversold:
                return 1  # Buy signal (RSI crosses above oversold level)
            elif prev_rsi >= self.overbought and current_rsi < self.overbought:
                return -1  # Sell signal (RSI crosses below overbought level)
            else:
                return 0  # No action
                
        except Exception as e:
            print(f"Error generating RSI signal: {str(e)}")
            return 0
```

### 4.4 Model Comparison

Model comparison and ranking system:

```python
# comparison/comparer.py
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

class ModelComparer:
    """
    Compares and ranks multiple models based on their metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model comparer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get configuration for metric weights
        self.metric_weights = self.config.get('metric_weights', {})
        
        # Default weights for commonly used metrics
        self.default_weights = {
            'rmse': 1.0,
            'mae': 1.0,
            'r2': 1.0,
            'sharpe_ratio': 1.0,
            'sortino_ratio': 1.0,
            'max_drawdown': 1.0,
            'win_rate': 0.5,
            'profit_factor': 0.5
        }
    
    def compare(
        self, 
        model_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_metrics: Dictionary mapping model IDs to metric dictionaries
            
        Returns:
            Dictionary with comparison results
        """
        if not model_metrics:
            return {'error': 'No models to compare'}
        
        # Find all unique metrics across models
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Create a DataFrame with all metrics
        metrics_df = pd.DataFrame(index=model_metrics.keys(), columns=list(all_metrics))
        
        # Fill in values
        for model_id, metrics in model_metrics.items():
            for metric_name, value in metrics.items():
                metrics_df.loc[model_id, metric_name] = value
                
        # Check if we need to standardize metrics
        standardize = self.config.get('standardize_metrics', True)
        if standardize:
            metrics_df = self._standardize_metrics(metrics_df)
        
        # Calculate weighted scores
        scores = self._calculate_scores(metrics_df)
        
        # Rank models
        ranks = scores.rank(ascending=False)
        
        # Generate pairwise comparison
        pairwise = self._pairwise_comparison(model_metrics)
        
        # Compile results
        results = {
            'scores': scores.to_dict(),
            'rankings': ranks.to_dict(),
            'metrics': metrics_df.to_dict(),
            'pairwise_comparison': pairwise
        }
        
        return results
    
    def _standardize_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize metrics so they are on comparable scales.
        
        Args:
            metrics_df: DataFrame with metrics
            
        Returns:
            Standardized DataFrame
        """
        # Create a copy of the DataFrame
        std_df = metrics_df.copy()
        
        # For each metric
        for metric in std_df.columns:
            values = std_df[metric].dropna()
            
            # Skip if not enough non-NA values
            if len(values) < 2:
                continue
                
            # Check if this is a higher-is-better or lower-is-better metric
            # based on common naming patterns
            lower_is_better = any(m in metric.lower() for m in 
                                ['error', 'loss', 'drawdown'])
            
            # Standardize
            mean = values.mean()
            std = values.std()
            
            if std > 0:
                # Z-score standardization
                std_values = (values - mean) / std
                
                # Flip sign for lower-is-better metrics
                if lower_is_better:
                    std_values = -std_values
                    
                std_df[metric] = std_values
        
        return std_df
    
    def _calculate_scores(self, metrics_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted scores for each model.
        
        Args:
            metrics_df: DataFrame with standardized metrics
            
        Returns:
            Series with model scores
        """
        # Get weights for each metric
        weights = {}
        
        for metric in metrics_df.columns:
            # Use configured weight if available
            weight = self.metric_weights.get(metric)
            
            # Otherwise use default weight if available
            if weight is None:
                weight = self.default_weights.get(metric, 1.0)
                
            weights[metric] = weight
        
        # Calculate weighted sum for each model
        scores = pd.Series(0.0, index=metrics_df.index)
        
        for metric, weight in weights.items():
            if metric in metrics_df.columns:
                # Multiply by weight and add to scores
                # Skip NaN values
                scores += metrics_df[metric].fillna(0) * weight
        
        return scores
    
    def _pairwise_comparison(
        self, 
        model_metrics: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Generate pairwise comparisons between models.
        
        Args:
            model_metrics: Dictionary mapping model IDs to metric dictionaries
            
        Returns:
            List of pairwise comparison results
        """
        results = []
        
        # Get model IDs
        model_ids = list(model_metrics.keys())
        
        # Compare each pair of models
        for i in range(len(model_ids)):
            for j in range(i+1, len(model_ids)):
                model_id1 = model_ids[i]
                model_id2 = model_ids[j]
                
                metrics1 = model_metrics[model_id1]
                metrics2 = model_metrics[model_id2]
                
                # Find common metrics
                common_metrics = set(metrics1.keys()).intersection(set(metrics2.keys()))
                
                # Compare each common metric
                comparison = {
                    'model1': model_id1,
                    'model2': model_id2,
                    'metrics': {}
                }
                
                for metric in common_metrics:
                    value1 = metrics1[metric]
                    value2 = metrics2[metric]
                    
                    # Check if higher is better based on common naming patterns
                    lower_is_better = any(m in metric.lower() for m in 
                                     ['error', 'loss', 'drawdown'])
                    
                    # Determine which model is better
                    if lower_is_better:
                        better_model = model_id1 if value1 < value2 else model_id2
                        pct_diff = (max(value1, value2) / min(value1, value2) - 1) * 100
                    else:
                        better_model = model_id1 if value1 > value2 else model_id2
                        pct_diff = (max(value1, value2) / min(value1, value2) - 1) * 100
                    
                    comparison['metrics'][metric] = {
                        'value1': value1,
                        'value2': value2,
                        'better_model': better_model,
                        'pct_diff': pct_diff
                    }
                
                # Add to results
                results.append(comparison)
        
        return results
```

### 4.5 Report Generator

Report generation for model evaluation and backtesting results:

```python
# reporting/report_generator.py
from typing import Dict, List, Any, Optional, Union
import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    """
    Generates HTML reports for model evaluation and backtesting results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Set up Jinja2 environment for templates
        template_dir = self.config.get(
            'template_dir', 
            os.path.join(os.path.dirname(__file__), 'templates')
        )
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_evaluation_report(
        self, 
        evaluation_result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate evaluation report for a single model.
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Prepare report data
        report_data = {
            'title': f"Model Evaluation Report: {evaluation_result.get('model_id', 'Unknown')}",
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_id': evaluation_result.get('evaluation_id', 'Unknown'),
            'model_id': evaluation_result.get('model_id', 'Unknown'),
            'model_type': evaluation_result.get('model_type', 'Unknown'),
            'metrics': evaluation_result.get('metrics', {}),
            'feature_importance': evaluation_result.get('feature_importance', {})
        }
        
        # Create charts
        charts = {}
        
        # Create residual plot if prediction samples available
        prediction_samples = evaluation_result.get('prediction_samples', {})
        if prediction_samples and 'y_pred' in prediction_samples and 'y_true' in prediction_samples:
            residual_path = f"{output_path.rsplit('.', 1)[0]}_residual.png"
            self._create_residual_plot(
                y_true=prediction_samples['y_true'],
                y_pred=prediction_samples['y_pred'],
                output_path=residual_path
            )
            charts['residual_plot'] = os.path.basename(residual_path)
        
        # Create feature importance plot if available
        feature_importance = evaluation_result.get('feature_importance')
        if feature_importance:
            importance_path = f"{output_path.rsplit('.', 1)[0]}_importance.png"
            self._create_feature_importance_plot(
                feature_importance,
                output_path=importance_path
            )
            charts['feature_importance_plot'] = os.path.basename(importance_path)
        
        # Add charts to report data
        report_data['charts'] = charts
        
        # Render template
        template = self.env.get_template('standard.html')
        html = template.render(**report_data)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def generate_comparison_report(
        self, 
        comparison_result: Dict[str, Any],
        evaluation_results: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Generate comparison report for multiple models.
        
        Args:
            comparison_result: Comparison result dictionary
            evaluation_results: Dictionary mapping model IDs to evaluation results
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Prepare report data
        report_data = {
            'title': "Model Comparison Report",
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': comparison_result.get('metrics', {}),
            'scores': comparison_result.get('scores', {}),
            'rankings': comparison_result.get('rankings', {}),
            'pairwise_comparison': comparison_result.get('pairwise_comparison', []),
            'models': evaluation_results
        }
        
        # Create charts
        charts = {}
        
        # Create ranking chart
        if comparison_result.get('scores'):
            ranking_path = f"{output_path.rsplit('.', 1)[0]}_ranking.png"
            self._create_ranking_chart(
                scores=comparison_result['scores'],
                output_path=ranking_path
            )
            charts['ranking_chart'] = os.path.basename(ranking_path)
        
        # Create metric comparison chart
        if comparison_result.get('metrics'):
            metrics_path = f"{output_path.rsplit('.', 1)[0]}_metrics.png"
            self._create_metrics_comparison_chart(
                metrics=comparison_result['metrics'],
                output_path=metrics_path
            )
            charts['metrics_chart'] = os.path.basename(metrics_path)
        
        # Add charts to report data
        report_data['charts'] = charts
        
        # Render template
        template = self.env.get_template('comparison.html')
        html = template.render(**report_data)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def generate_backtest_report(
        self, 
        backtest_result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate backtest report.
        
        Args:
            backtest_result: Backtest result dictionary
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Prepare report data
        report_data = {
            'title': f"Backtest Report: {backtest_result.get('model_id', 'Unknown')}",
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backtest_id': backtest_result.get('backtest_id', 'Unknown'),
            'model_id': backtest_result.get('model_id', 'Unknown'),
            'model_type': backtest_result.get('model_type', 'Unknown'),
            'metrics': backtest_result.get('metrics', {}),
            'initial_capital': backtest_result.get('initial_capital', 0),
            'final_equity': backtest_result.get('final_equity', 0),
            'total_return_pct': backtest_result.get('total_return_pct', 0),
            'trades': backtest_result.get('trades', [])[:100], # Limit to first 100 trades
            'trade_count': len(backtest_result.get('trades', [])),
            'strategy_config': backtest_result.get('strategy_config', {})
        }
        
        # Create charts
        charts = {}
        
        # Create equity curve chart
        equity_curve = backtest_result.get('equity_curve')
        if equity_curve is not None:
            equity_path = f"{output_path.rsplit('.', 1)[0]}_equity.png"
            self._create_equity_curve_chart(
                equity_curve=equity_curve,
                output_path=equity_path
            )
            charts['equity_curve_chart'] = os.path.basename(equity_path)
        
        # Create drawdown chart
        if equity_curve is not None:
            drawdown_path = f"{output_path.rsplit('.', 1)[0]}_drawdown.png"
            self._create_drawdown_chart(
                equity_curve=equity_curve,
                output_path=drawdown_path
            )
            charts['drawdown_chart'] = os.path.basename(drawdown_path)
        
        # Create trade distribution chart
        trades = backtest_result.get('trades', [])
        if trades:
            trades_path = f"{output_path.rsplit('.', 1)[0]}_trades.png"
            self._create_trade_distribution_chart(
                trades=trades,
                output_path=trades_path
            )
            charts['trades_chart'] = os.path.basename(trades_path)
        
        # Calculate some additional statistics
        if trades:
            report_data['avg_trade_profit'] = sum(t.get('profit', 0) for t in trades) / len(trades)
            report_data['avg_win'] = sum(max(0, t.get('profit', 0)) for t in trades) / max(1, sum(1 for t in trades if t.get('profit', 0) > 0))
            report_data['avg_loss'] = sum(min(0, t.get('profit', 0)) for t in trades) / max(1, sum(1 for t in trades if t.get('profit', 0) < 0))
            report_data['largest_win'] = max((t.get('profit', 0) for t in trades), default=0)
            report_data['largest_loss'] = min((t.get('profit', 0) for t in trades), default=0)
        
        # Add charts to report data
        report_data['charts'] = charts
        
        # Render template
        template = self.env.get_template('trading.html')
        html = template.render(**report_data)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _create_residual_plot(
        self, 
        y_true: List[float],
        y_pred: List[float],
        output_path: str
    ):
        """
        Create residual plot for model evaluation.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot of predictions vs actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs True Values')
        
        # Residual plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted Values')
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Histogram of Residuals')
        
        # Q-Q plot
        import scipy.stats as stats
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        # Add key statistics as text
        mean_error = np.mean(residuals)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        r2 = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2))
        
        fig.text(0.5, 0.01, f'Mean Error: {mean_error:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}', 
                 ha='center', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_feature_importance_plot(
        self, 
        feature_importance: Dict[str, float],
        output_path: str
    ):
        """
        Create feature importance plot.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            output_path: Path to save the plot
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Limit to top 20 features
        if len(sorted_features) > 20:
            sorted_features = sorted_features[:20]
        
        # Extract names and values
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Create horizontal bar chart
        y_pos = range(len(feature_names))
        plt.barh(y_pos, importance_values, align='center')
        plt.yticks(y_pos, feature_names)
        
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_ranking_chart(
        self, 
        scores: Dict[str, float],
        output_path: str
    ):
        """
        Create model ranking chart.
        
        Args:
            scores: Dictionary mapping model IDs to scores
            output_path: Path to save the plot
        """
        # Sort by score
        sorted_scores = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Extract model IDs and scores
        model_ids = [s[0] for s in sorted_scores]
        score_values = [s[1] for s in sorted_scores]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        plt.bar(model_ids, score_values)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Ranking by Score')
        
        # Rotate x-axis labels if many models
        if len(model_ids) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_metrics_comparison_chart(
        self, 
        metrics: Dict[str, Dict[str, float]],
        output_path: str
    ):
        """
        Create metric comparison chart for multiple models.
        
        Args:
            metrics: Dictionary mapping model IDs to metric dictionaries
            output_path: Path to save the plot
        """
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Normalize metrics for better visualization
        normalized_df = pd.DataFrame()
        
        for col in metrics_df.columns:
            # Skip columns with all NaN
            if metrics_df[col].isna().all():
                continue
                
            # Check if this is a higher-is-better or lower-is-better metric
            # based on common naming patterns
            lower_is_better = any(m in col.lower() for m in 
                               ['error', 'loss', 'drawdown'])
            
            # Min-max scaling
            values = metrics_df[col].fillna(metrics_df[col].mean())
            min_val = values.min()
            max_val = values.max()
            
            if min_val == max_val:
                # All values are the same
                normalized_df[col] = 0.5
            else:
                # Normalize to [0, 1]
                normalized = (values - min_val) / (max_val - min_val)
                
                # Invert if lower is better
                if lower_is_better:
                    normalized = 1 - normalized
                    
                normalized_df[col] = normalized
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(normalized_df.T, annot=True, cmap='viridis', 
                    vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'})
        
        plt.xlabel('Model')
        plt.ylabel('Metric')
        plt.title('Model Comparison by Metric (Normalized Scores)')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_equity_curve_chart(
        self, 
        equity_curve: Union[List[float], np.ndarray],
        output_path: str
    ):
        """
        Create equity curve chart for backtest results.
        
        Args:
            equity_curve: List or array of equity values
            output_path: Path to save the plot
        """
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create line chart
        plt.plot(equity_curve)
        
        plt.xlabel('Time Step')
        plt.ylabel('Equity')
        plt.title('Equity Curve')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_drawdown_chart(
        self, 
        equity_curve: Union[List[float], np.ndarray],
        output_path: str
    ):
        """
        Create drawdown chart for backtest results.
        
        Args:
            equity_curve: List or array of equity values
            output_path: Path to save the plot
        """
        # Calculate drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create line chart
        plt.plot(drawdown)
        plt.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red')
        
        plt.xlabel('Time Step')
        plt.ylabel('Drawdown')
        plt.title('Drawdown')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Invert y-axis for better visualization (0 at top)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    def _create_trade_distribution_chart(
        self, 
        trades: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Create trade distribution chart for backtest results.
        
        Args:
            trades: List of trade dictionaries
            output_path: Path to save the plot
        """
        # Extract profits
        profits = [trade.get('profit', 0) for trade in trades]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of profits
        ax1.hist(profits, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_xlabel('Profit')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trade Profits')
        
        # Cumulative profit
        cumulative = np.cumsum(profits)
        ax2.plot(cumulative)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative Profit')
        ax2.set_title('Cumulative Profit')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
```

## 5. Configuration

### 5.1 Evaluation Configuration Schema

```yaml
# Example evaluation configuration
evaluation:
  # Default metrics to use if not specified
  default_metrics: 
    - rmse
    - mae
    - r2
    - sharpe_ratio
    - max_drawdown
    - calmar_ratio
  
  # Configuration for model comparison
  comparison:
    # Whether to standardize metrics before comparison
    standardize_metrics: true
    
    # Weights for different metrics in comparing models
    metric_weights:
      rmse: 1.5
      mae: 1.0
      r2: 1.0
      sharpe_ratio: 2.0
      sortino_ratio: 1.5
      max_drawdown: 1.0
      calmar_ratio: 2.0
      win_rate: 0.5
      profit_factor: 1.0
  
  # Backtesting defaults
  backtesting:
    initial_capital: 100000.0
    commission: 0.001  # 0.1%
    slippage: 0.0005   # 0.05%
    position_size_pct: 0.1  # 10% of capital per trade
    
    # Risk management configuration
    risk:
      max_drawdown: 0.2  # 20%
      max_position_pct: 0.2  # 20%
      stop_loss_pct: 0.05  # 5%
```

### 5.2 Reporting Template Schema

HTML templates for reports are stored in the `templates` directory:

- `standard.html` - Standard evaluation report template
- `comparison.html` - Model comparison report template
- `trading.html` - Backtesting report template

Example template structure for the trading report:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2980b9;
        }
        .chart-container {
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>Generated on: {{ timestamp }}</p>
        
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div>Initial Capital</div>
                <div class="metric-value">{{ initial_capital | format_currency }}</div>
            </div>
            <div class="metric-card">
                <div>Final Equity</div>
                <div class="metric-value">{{ final_equity | format_currency }}</div>
            </div>
            <div class="metric-card">
                <div>Total Return</div>
                <div class="metric-value {% if total_return_pct >= 0 %}positive{% else %}negative{% endif %}">
                    {{ total_return_pct | format_percent }}
                </div>
            </div>
            <div class="metric-card">
                <div>Total Trades</div>
                <div class="metric-value">{{ trade_count }}</div>
            </div>
        </div>
        
        <h2>Performance Metrics</h2>
        <div class="metrics">
            {% for name, value in metrics.items() %}
            <div class="metric-card">
                <div>{{ name | title | replace('_', ' ') }}</div>
                <div class="metric-value">{{ value | format_number }}</div>
            </div>
            {% endfor %}
        </div>
        
        <h2>Equity Curve</h2>
        <div class="chart-container">
            {% if charts.equity_curve_chart %}
            <img src="{{ charts.equity_curve_chart }}" alt="Equity Curve" style="width: 100%;">
            {% else %}
            <p>No equity curve chart available</p>
            {% endif %}
        </div>
        
        <h2>Drawdown</h2>
        <div class="chart-container">
            {% if charts.drawdown_chart %}
            <img src="{{ charts.drawdown_chart }}" alt="Drawdown" style="width: 100%;">
            {% else %}
            <p>No drawdown chart available</p>
            {% endif %}
        </div>
        
        <h2>Trade Analysis</h2>
        <div class="chart-container">
            {% if charts.trades_chart %}
            <img src="{{ charts.trades_chart }}" alt="Trade Distribution" style="width: 100%;">
            {% else %}
            <p>No trade distribution chart available</p>
            {% endif %}
        </div>
        
        {% if trades %}
        <h2>Trade Details</h2>
        <p>Showing the first {{ trades | length }} trades{% if trade_count > trades | length %} (out of {{ trade_count }} total){% endif %}</p>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Time</th>
                    <th>Asset</th>
                    <th>Direction</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Size</th>
                    <th>Profit</th>
                    <th>Return</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trades %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ trade.timestamp }}</td>
                    <td>{{ trade.asset }}</td>
                    <td>{% if trade.direction == 1 %}Long{% else %}Short{% endif %}</td>
                    <td>{{ trade.entry_price | format_number }}</td>
                    <td>{{ trade.exit_price | format_number }}</td>
                    <td>{{ trade.size | format_number }}</td>
                    <td class="{% if trade.profit >= 0 %}positive{% else %}negative{% endif %}">
                        {{ trade.profit | format_currency }}
                    </td>
                    <td class="{% if trade.profit >= 0 %}positive{% else %}negative{% endif %}">
                        {{ (trade.profit / (trade.entry_price * trade.size) * 100) | format_percent }}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        <h2>Strategy Configuration</h2>
        <pre>{{ strategy_config | json_pretty }}</pre>
    </div>
</body>
</html>
```

## 6. Usage Examples

### 6.1 Basic Model Evaluation

```python
# Example of basic model evaluation

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.evaluation.manager import EvaluationManager

# Load configuration
config = ConfigManager.instance()
model_config = config.get('models', {})
data_config = config.get('data', {})
eval_config = config.get('evaluation', {})

# Create managers
model_manager = ModelManager(model_config)
data_manager = DataManager(data_config)
eval_manager = EvaluationManager(eval_config)

# Load data
data_pipeline_id = "daily_btc_features"
data = data_manager.execute_pipeline(data_pipeline_id)

# Split data
train_data, val_data, test_data = data_manager.split_data(
    data, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15
)

# Create and train a model
model_id, model = model_manager.create_model("lstm", model_name="btc_price_model")
training_result = model_manager.train_model(
    model,
    model_id,
    train_data=train_data,
    val_data=val_data
)

# Evaluate model on test data
evaluation_result = eval_manager.evaluate_model(
    model=model,
    model_id=model_id,
    data={
        'X_test': test_data['X'],
        'y_test': test_data['y'],
        # Additional data for trading metrics
        'price_series': test_data['close_price']
    },
    metrics=[
        'rmse', 'mae', 'r2',  # Standard metrics
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown'  # Trading metrics
    ],
    output_dir="./results/evaluations"
)

# Generate evaluation report
report_path = eval_manager.generate_evaluation_report(
    evaluation_result,
    output_path="./results/reports/lstm_model_evaluation.html"
)

print(f"Evaluation complete. Report generated at: {report_path}")
print(f"Key metrics: RMSE={evaluation_result['metrics']['rmse']:.4f}, "
      f"Sharpe Ratio={evaluation_result['metrics']['sharpe_ratio']:.4f}")
```

### 6.2 Model Comparison

```python
# Example of comparing multiple models

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.evaluation.manager import EvaluationManager

# Load configuration
config = ConfigManager.instance()
eval_manager = EvaluationManager(config.get('evaluation', {}))
model_manager = ModelManager(config.get('models', {}))
data_manager = DataManager(config.get('data', {}))

# Load data
data = data_manager.execute_pipeline('daily_btc_features')
train_data, val_data, test_data = data_manager.split_data(data)

# Create and train multiple models
models = {}

# LSTM model
lstm_id, lstm_model = model_manager.create_model(
    "lstm", 
    model_name="lstm_model",
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)
model_manager.train_model(lstm_model, lstm_id, train_data, val_data)
models[lstm_id] = lstm_model

# GRU model
gru_id, gru_model = model_manager.create_model(
    "gru", 
    model_name="gru_model",
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)
model_manager.train_model(gru_model, gru_id, train_data, val_data)
models[gru_id] = gru_model

# Random Forest model
rf_id, rf_model = model_manager.create_model(
    "random_forest", 
    model_name="rf_model",
    n_estimators=100,
    max_depth=10
)
model_manager.train_model(rf_model, rf_id, train_data, val_data)
models[rf_id] = rf_model

# Evaluate all models
eval_results = eval_manager.evaluate_multiple_models(
    models=models,
    data={
        'X_test': test_data['X'],
        'y_test': test_data['y'],
        'price_series': test_data['close_price']
    },
    metrics=[
        'rmse', 'mae', 'r2',  # Standard metrics
        'sharpe_ratio', 'max_drawdown'  # Trading metrics
    ],
    output_dir="./results/evaluations"
)

# Compare models explicitly
comparison_result = eval_manager.compare_models(
    evaluation_results=eval_results,
    output_dir="./results/comparisons"
)

# Print comparison summary
print("Model Comparison Summary:")
print("-------------------------")
for model_id, rank in comparison_result['rankings'].items():
    print(f"Model: {model_id}, Rank: {rank}")

print("\nBest Model:")
best_model_id = min(comparison_result['rankings'].items(), key=lambda x: x[1])[0]
print(f"- ID: {best_model_id}")
print(f"- Score: {comparison_result['scores'][best_model_id]:.4f}")

print("\nKey Metrics for Best Model:")
for metric, value in eval_results[best_model_id]['metrics'].items():
    print(f"- {metric}: {value:.4f}")
```

### 6.3 Backtesting

```python
# Example of backtesting a model

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.evaluation.manager import EvaluationManager

# Load configuration
config = ConfigManager.instance()
eval_manager = EvaluationManager(config.get('evaluation', {}))
model_manager = ModelManager(config.get('models', {}))
data_manager = DataManager(config.get('data', {}))

# Load historical data with price information
historical_data = data_manager.execute_pipeline('daily_crypto_ohlcv')

# Split data for training
train_data, val_data, test_data = data_manager.split_temporal_data(
    historical_data,
    train_end_date='2022-01-01',
    val_end_date='2022-06-01'
)

# Create and train a model
model_id, model = model_manager.create_model("lstm", model_name="price_predictor")
model_manager.train_model(model, model_id, train_data, val_data)

# Configure trading strategy
strategy_config = {
    'type': 'model_based',
    'prediction_threshold': 0.0,
    'long_threshold': 0.01,   # 1% predicted increase for long
    'short_threshold': -0.01, # 1% predicted decrease for short
    'lookback': 20            # Use 20 days of data for prediction
}

# Run backtest
backtest_result = eval_manager.backtest_model(
    model=model,
    model_id=model_id,
    data={'historical_data': test_data},
    strategy_config=strategy_config,
    output_dir="./results/backtests"
)

# Print backtest summary
print("Backtest Summary:")
print("-----------------")
print(f"Initial Capital: ${backtest_result['initial_capital']:.2f}")
print(f"Final Equity: ${backtest_result['final_equity']:.2f}")
print(f"Total Return: {backtest_result['total_return_pct']:.2f}%")
print(f"Total Trades: {len(backtest_result['trades'])}")

print("\nPerformance Metrics:")
for metric, value in backtest_result['metrics'].items():
    print(f"- {metric}: {value:.4f}")

# Generate detailed report
report_path = eval_manager.generate_backtest_report(
    backtest_result,
    output_path="./results/reports/lstm_backtest_report.html"
)
print(f"\nBacktest report generated at: {report_path}")
```

## 7. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is already set up
2. Configuration management system is implemented
3. Data management module is implemented
4. Model training module is implemented
5. Results database infrastructure is implemented
6. Required libraries are installed:
   - numpy
   - pandas
   - matplotlib
   - seaborn
   - scipy
   - scikit-learn
   - jinja2

## 8. Implementation Sequence

1. Set up directory structure for evaluation components
2. Implement base metric classes and standard metrics
3. Create trading-specific metrics
4. Develop the metric registry
5. Implement backtesting engine components
6. Create model comparison tools
7. Develop reporting and visualization utilities
8. Integrate with results database
9. Create high-level evaluation manager
10. Add comprehensive documentation and usage examples

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# Example unit tests for metrics

import unittest
import numpy as np

from trading_optimization.evaluation.metrics.registry import MetricRegistry
from trading_optimization.evaluation.metrics.trading import SharpeRatio, MaxDrawdown

class TestStandardMetrics(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Sample data
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.2, 2.3, 2.8, 3.9, 5.2])
        
        # Registry
        self.registry = MetricRegistry()
    
    def test_rmse(self):
        """Test RMSE calculation."""
        rmse_func = self.registry.get_metric('rmse')
        self.assertIsNotNone(rmse_func)
        
        # Calculate RMSE
        rmse = rmse_func(self.y_true, self.y_pred)
        
        # Check result is close to expected
        expected = np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        self.assertAlmostEqual(rmse, expected, places=5)
    
    def test_mae(self):
        """Test MAE calculation."""
        mae_func = self.registry.get_metric('mae')
        self.assertIsNotNone(mae_func)
        
        # Calculate MAE
        mae = mae_func(self.y_true, self.y_pred)
        
        # Check result is close to expected
        expected = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertAlmostEqual(mae, expected, places=5)
    
    def test_r2(self):
        """Test R² calculation."""
        r2_func = self.registry.get_metric('r2')
        self.assertIsNotNone(r2_func)
        
        # Calculate R²
        r2 = r2_func(self.y_true, self.y_pred)
        
        # Check result is reasonable (should be close to 1 for good fit)
        self.assertGreater(r2, 0.9)  # Strong correlation

class TestTradingMetrics(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Sample returns
        self.returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012])
        
        # Sample equity curve
        self.equity_curve = np.cumprod(1 + self.returns)
        
        # Sample trades
        self.trades = [
            {'profit': 100.0, 'entry_price': 1000, 'exit_price': 1100},
            {'profit': -50.0, 'entry_price': 1100, 'exit_price': 1050},
            {'profit': 80.0, 'entry_price': 1050, 'exit_price': 1130},
            {'profit': -30.0, 'entry_price': 1130, 'exit_price': 1100}
        ]
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = SharpeRatio(risk_free_rate=0.0, annualization_factor=252)
        
        # Calculate Sharpe ratio from returns
        result = sharpe(y_true=None, y_pred=None, returns=self.returns)
        
        # Check result is positive (decent performance)
        self.assertGreater(result, 0)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd = MaxDrawdown()
        
        # Calculate max drawdown from equity curve
        result = max_dd(y_true=None, y_pred=None, equity_curve=self.equity_curve)
        
        # Check result is positive and reasonable
        self.assertGreater(result, 0)
        self.assertLess(result, 1.0)  # Max drawdown should be less than 100%
    
    def test_win_rate(self):
        """Test win rate calculation."""
        win_rate_func = self.registry.get_metric('win_rate')
        self.assertIsNotNone(win_rate_func)
        
        # Calculate win rate from trades
        result = win_rate_func(y_true=None, y_pred=None, trades=self.trades)
        
        # Expected win rate: 2 out of 4 trades are profitable
        self.assertAlmostEqual(result, 0.5)
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        profit_factor_func = self.registry.get_metric('profit_factor')
        self.assertIsNotNone(profit_factor_func)
        
        # Calculate profit factor from trades
        result = profit_factor_func(y_true=None, y_pred=None, trades=self.trades)
        
        # Expected profit factor: (100 + 80) / (50 + 30) = 180 / 80 = 2.25
        self.assertAlmostEqual(result, 2.25)
```

### 9.2 Integration Tests

```python
# Example integration tests for evaluation manager

import unittest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd

from trading_optimization.evaluation.manager import EvaluationManager
from trading_optimization.models.interface import Model

class SimpleModel(Model):
    """Simple model for testing."""
    
    def __init__(self, prediction_offset=0.1):
        """Initialize model with fixed offset."""
        self.prediction_offset = prediction_offset
    
    def predict(self, X):
        """Predict with fixed offset."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Add offset to first column
        if X.ndim > 1:
            return X[:, 0] + self.prediction_offset
        else:
            return X + self.prediction_offset
    
    def feature_importance(self):
        """Return dummy feature importance."""
        return {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2}

class TestEvaluationManager(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temp directory for artifacts
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal configuration
        cls.config = {
            'default_metrics': ['rmse', 'mae', 'r2', 'sharpe_ratio']
        }
        
        # Create evaluation manager with test config
        cls.eval_manager = EvaluationManager(cls.config)
        
        # Create simple test models
        cls.model1 = SimpleModel(prediction_offset=0.1)
        cls.model2 = SimpleModel(prediction_offset=0.2)
        
        # Create test data
        X_test = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y_test = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Create test returns
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        
        cls.test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'returns': returns
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove the temp directory
        shutil.rmtree(cls.temp_dir)
    
    def test_evaluate_model(self):
        """Test evaluating a single model."""
        # Evaluate model
        result = self.eval_manager.evaluate_model(
            model=self.model1,
            model_id='test_model1',
            data=self.test_data,
            output_dir=os.path.join(self.temp_dir, 'evaluations'),
            save_results=False
        )
        
        # Check result structure
        self.assertIn('evaluation_id', result)
        self.assertIn('metrics', result)
        self.assertIn('rmse', result['metrics'])
        self.assertIn('mae', result['metrics'])
        self.assertIn('r2', result['metrics'])
        
        # Check metrics are reasonable
        self.assertLess(result['metrics']['rmse'], 0.2)  # Small error
        self.assertLess(result['metrics']['mae'], 0.2)   # Small error
        self.assertGreater(result['metrics']['r2'], 0.9) # Good fit
    
    def test_evaluate_multiple_models(self):
        """Test evaluating multiple models."""
        # Define models
        models = {
            'test_model1': self.model1,
            'test_model2': self.model2
        }
        
        # Evaluate models
        results = self.eval_manager.evaluate_multiple_models(
            models=models,
            data=self.test_data,
            output_dir=os.path.join(self.temp_dir, 'evaluations'),
            save_results=False
        )
        
        # Check results for both models
        self.assertIn('test_model1', results)
        self.assertIn('test_model2', results)
        
        # Check comparison was done
        self.assertIn('comparison_rank', results['test_model1'])
        self.assertIn('comparison_rank', results['test_model2'])
        
        # Model1 should be better (smaller offset)
        self.assertLess(
            results['test_model1']['metrics']['rmse'],
            results['test_model2']['metrics']['rmse']
        )
    
    def test_generate_report(self):
        """Test generating evaluation report."""
        # Evaluate a model
        result = self.eval_manager.evaluate_model(
            model=self.model1,
            model_id='test_model1',
            data=self.test_data,
            save_results=False
        )
        
        # Generate report
        report_path = os.path.join(self.temp_dir, 'reports', 'test_report.html')
        actual_path = self.eval_manager.generate_evaluation_report(
            result, 
            report_path
        )
        
        # Check report was created
        self.assertTrue(os.path.exists(actual_path))
        self.assertTrue(os.path.getsize(actual_path) > 0)
```

## 10. Integration with Other Components

The Model Evaluation Infrastructure integrates with:

1. **Data Management Module**: To access data for evaluation and backtesting.
2. **Model Training Module**: To access trained models for evaluation.
3. **Results Database Infrastructure**: To store and retrieve evaluation results.

Integration is primarily through the EvaluationManager, which provides a clean interface for model evaluation, comparison, and backtesting.

## 11. Extension Points

The module is designed to be easily extended:

1. **New Metrics**:
   - Create new classes that inherit from Metric
   - Register them with the MetricRegistry

2. **New Trading Strategies**:
   - Implement new strategies by inheriting from TradingStrategy
   - Register them in the BacktestEngine._create_strategy method

3. **Advanced Analysis**:
   - Add new statistical analysis methods to the statistical module
   - Implement new visualizations in the reporting module

4. **Report Templates**:
   - Create new HTML templates in the templates directory
   - Update the ReportGenerator to use these templates