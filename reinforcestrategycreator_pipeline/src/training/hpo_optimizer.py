"""Hyperparameter Optimization (HPO) module using Ray Tune."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray Tune not available. Install with: pip install 'ray[tune]'")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..models.base import ModelBase
from ..models.factory import ModelFactory
from ..artifact_store.base import ArtifactStore, ArtifactType
from .engine import TrainingEngine


class HPOptimizer:
    """Hyperparameter Optimizer using Ray Tune.
    
    This class provides an interface for hyperparameter optimization
    of models using Ray Tune. It supports various search algorithms
    and schedulers for efficient hyperparameter search.
    """
    
    def __init__(
        self,
        training_engine: TrainingEngine,
        artifact_store: Optional[ArtifactStore] = None,
        results_dir: Optional[Union[str, Path]] = None,
        ray_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the HPOptimizer.
        
        Args:
            training_engine: TrainingEngine instance for model training
            artifact_store: Optional ArtifactStore for saving results
            results_dir: Directory for saving HPO results
            ray_config: Configuration for Ray initialization
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is required for HPOptimizer. Install with: pip install 'ray[tune]'")
        
        self.training_engine = training_engine
        self.artifact_store = artifact_store
        
        # Set up results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path("./hpo_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Ray configuration
        self.ray_config = ray_config or {}
        
        # Logger
        self.logger = logging.getLogger("HPOptimizer")
        
        # HPO state
        self.current_study = None
        self.best_params = None
        self.best_score = None
        self.all_trials = []
    
    def define_search_space(
        self,
        param_space: Dict[str, Any],
        search_algorithm: str = "random",
        metric: str = "loss",
        mode: str = "min"
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """Define the hyperparameter search space.
        
        Args:
            param_space: Dictionary defining the search space
            search_algorithm: Algorithm to use ("random", "optuna", "bayesopt")
            metric: Metric to optimize
            mode: Optimization mode ("min" or "max")
            
        Returns:
            Tuple of (processed param space, search algorithm instance)
        """
        # Process parameter space for Ray Tune
        processed_space = {}
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "uniform")
                
                if param_type == "uniform":
                    processed_space[param_name] = tune.uniform(
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "loguniform":
                    processed_space[param_name] = tune.loguniform(
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "choice":
                    processed_space[param_name] = tune.choice(param_config["values"])
                elif param_type == "randint":
                    processed_space[param_name] = tune.randint(
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "quniform":
                    processed_space[param_name] = tune.quniform(
                        param_config["low"], param_config["high"], param_config.get("q", 1)
                    )
                else:
                    # Direct value
                    processed_space[param_name] = param_config
            else:
                # Direct value
                processed_space[param_name] = param_config
        
        # Set up search algorithm
        search_alg = None
        if search_algorithm == "optuna" and OPTUNA_AVAILABLE:
            search_alg = OptunaSearch(metric=metric, mode=mode)
        elif search_algorithm == "bayesopt":
            # Could add other search algorithms here
            self.logger.warning(f"Search algorithm '{search_algorithm}' not implemented, using random search")
        
        return processed_space, search_alg
    
    def _create_trainable(
        self,
        model_config_template: Dict[str, Any],
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        param_mapping: Optional[Dict[str, str]] = None
    ) -> Callable:
        """Create a trainable function for Ray Tune.
        
        Args:
            model_config_template: Base model configuration
            data_config: Data configuration
            training_config: Training configuration
            param_mapping: Mapping from HPO params to config paths
            
        Returns:
            Trainable function for Ray Tune
        """
        def trainable(config: Dict[str, Any]):
            """Trainable function that Ray Tune will call."""
            # Create a copy of the model config
            model_config = model_config_template.copy()
            
            # Apply hyperparameters to model config
            if param_mapping:
                for hpo_param, config_path in param_mapping.items():
                    if hpo_param in config:
                        # Navigate the config path and set value
                        self._set_nested_config(model_config, config_path, config[hpo_param])
            else:
                # Direct mapping - assume HPO params match model hyperparameters
                if "hyperparameters" not in model_config:
                    model_config["hyperparameters"] = {}
                model_config["hyperparameters"].update(config)
            
            # Train the model
            result = self.training_engine.train(
                model_config=model_config,
                data_config=data_config,
                training_config=training_config,
                callbacks=None  # Use default callbacks
            )
            
            # Extract metrics for Ray Tune
            if result["success"]:
                metrics = result.get("final_metrics", {})
                # Add training history summary
                if "history" in result:
                    history = result["history"]
                    if history.get("loss"):
                        metrics["final_loss"] = history["loss"][-1]
                        metrics["min_loss"] = min(history["loss"])
                    if history.get("val_loss"):
                        metrics["final_val_loss"] = history["val_loss"][-1]
                        metrics["min_val_loss"] = min(history["val_loss"])
                
                # Report to Ray Tune
                for epoch_idx, epoch in enumerate(history.get("epochs", [])):
                    epoch_metrics = {
                        "loss": history["loss"][epoch_idx] if epoch_idx < len(history["loss"]) else None,
                        "val_loss": history["val_loss"][epoch_idx] if epoch_idx < len(history.get("val_loss", [])) else None,
                        "epoch": epoch
                    }
                    # Remove None values
                    epoch_metrics = {k: v for k, v in epoch_metrics.items() if v is not None}
                    tune.report(metrics=epoch_metrics)
            else:
                # Report failure - Ray Tune expects metrics in a specific format
                tune.report(metrics={"loss": float('inf'), "error": result.get("error", "Unknown error")})
        
        return trainable
    
    def _set_nested_config(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a value in a nested configuration dictionary.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "hyperparameters.learning_rate")
            value: Value to set
        """
        keys = path.split(".")
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def optimize(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        param_space: Dict[str, Any],
        num_trials: int = 10,
        max_concurrent_trials: Optional[int] = None,
        search_algorithm: str = "random",
        scheduler: Optional[str] = "asha",
        metric: str = "loss",
        mode: str = "min",
        param_mapping: Optional[Dict[str, str]] = None,
        resources_per_trial: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            model_config: Base model configuration
            data_config: Data configuration
            training_config: Training configuration
            param_space: Hyperparameter search space
            num_trials: Number of trials to run
            max_concurrent_trials: Maximum concurrent trials
            search_algorithm: Search algorithm to use
            scheduler: Trial scheduler ("asha", "pbt", None)
            metric: Metric to optimize
            mode: Optimization mode ("min" or "max")
            param_mapping: Mapping from HPO params to config paths
            resources_per_trial: Resources per trial (CPUs, GPUs)
            name: Name for this HPO run
            
        Returns:
            Dictionary containing optimization results
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**self.ray_config)
        
        try:
            # Set up search space and algorithm
            processed_space, search_alg = self.define_search_space(
                param_space, search_algorithm, metric, mode
            )
            
            # Apply concurrency limit if specified
            if search_alg and max_concurrent_trials:
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)
            
            # Set up scheduler
            tune_scheduler = None
            if scheduler == "asha":
                tune_scheduler = ASHAScheduler(
                    metric=metric,
                    mode=mode,
                    max_t=training_config.get("epochs", 100),
                    grace_period=1,
                    reduction_factor=2
                )
            elif scheduler == "pbt":
                tune_scheduler = PopulationBasedTraining(
                    metric=metric,
                    mode=mode,
                    perturbation_interval=4,
                    hyperparam_mutations=processed_space
                )
            
            # Create trainable function
            trainable = self._create_trainable(
                model_config, data_config, training_config, param_mapping
            )
            
            # Set up progress reporter
            reporter = CLIReporter(
                metric_columns=[metric, "training_iteration"],
                parameter_columns=list(param_space.keys())[:5]  # Show first 5 params
            )
            
            # Run optimization
            run_name = name or f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Starting HPO run '{run_name}' with {num_trials} trials")
            
            analysis = tune.run(
                trainable,
                name=run_name,
                config=processed_space,
                num_samples=num_trials,
                scheduler=tune_scheduler,
                search_alg=search_alg,
                resources_per_trial=resources_per_trial or {"cpu": 1},
                storage_path=f"file://{self.results_dir.absolute()}",
                progress_reporter=reporter,
                verbose=1,
                stop={"training_iteration": training_config.get("epochs", 100)}
            )
            
            # Extract results
            best_trial = analysis.get_best_trial(metric, mode)
            self.best_params = best_trial.config
            self.best_score = best_trial.last_result[metric]
            
            # Collect all trials
            self.all_trials = []
            for trial in analysis.trials:
                trial_data = {
                    "trial_id": trial.trial_id,
                    "params": trial.config,
                    "metric": trial.last_result.get(metric) if trial.last_result else None,
                    "status": trial.status,
                    "iterations": trial.last_result.get("training_iteration", 0) if trial.last_result else 0
                }
                self.all_trials.append(trial_data)
            
            # Save results
            results = {
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                "num_trials": num_trials,
                "metric": metric,
                "mode": mode,
                "best_params": self.best_params,
                "best_score": self.best_score,
                "all_trials": self.all_trials,
                "search_algorithm": search_algorithm,
                "scheduler": scheduler,
                "param_space": param_space
            }
            
            # Save to file
            results_file = self.results_dir / f"{run_name}_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save to artifact store if available
            if self.artifact_store:
                artifact_metadata = self.artifact_store.save_artifact(
                    artifact_id=f"hpo_results_{run_name}",
                    artifact_path=results_file,
                    artifact_type=ArtifactType.REPORT,
                    metadata={
                        "type": "hpo_results",
                        "run_name": run_name,
                        "best_score": self.best_score,
                        "best_params": self.best_params,
                        "num_trials": num_trials
                    },
                    tags=["hpo", "optimization", "results"]
                )
                results["artifact_id"] = artifact_metadata.artifact_id
            
            self.logger.info(f"HPO completed. Best {metric}: {self.best_score}")
            self.logger.info(f"Best parameters: {self.best_params}")
            
            return results
            
        finally:
            # Shutdown Ray if we initialized it
            if ray.is_initialized():
                ray.shutdown()
    
    def analyze_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Analyze HPO results.
        
        Args:
            results: Results dictionary (uses last run if not provided)
            top_k: Number of top trials to analyze
            
        Returns:
            Analysis dictionary
        """
        if results is None and self.all_trials:
            results = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "all_trials": self.all_trials
            }
        
        if not results or "all_trials" not in results:
            return {"error": "No results to analyze"}
        
        # Convert trials to DataFrame for analysis
        trials_data = []
        for trial in results["all_trials"]:
            trial_flat = {"trial_id": trial["trial_id"], "metric": trial.get("metric")}
            # Flatten parameters
            for param, value in trial.get("params", {}).items():
                trial_flat[f"param_{param}"] = value
            trials_data.append(trial_flat)
        
        df = pd.DataFrame(trials_data)
        
        # Remove failed trials
        df = df.dropna(subset=["metric"])
        
        if df.empty:
            return {"error": "No successful trials to analyze"}
        
        # Sort by metric
        df = df.sort_values("metric", ascending=(results.get("mode", "min") == "min"))
        
        # Analysis
        analysis = {
            "total_trials": len(results["all_trials"]),
            "successful_trials": len(df),
            "best_trial": {
                "params": results["best_params"],
                "score": results["best_score"]
            },
            "top_k_trials": []
        }
        
        # Get top k trials
        for idx, row in df.head(top_k).iterrows():
            trial_info = {
                "rank": len(analysis["top_k_trials"]) + 1,
                "trial_id": row["trial_id"],
                "metric": row["metric"],
                "params": {}
            }
            # Extract parameters
            for col in df.columns:
                if col.startswith("param_"):
                    param_name = col[6:]  # Remove "param_" prefix
                    trial_info["params"][param_name] = row[col]
            analysis["top_k_trials"].append(trial_info)
        
        # Parameter importance (simple variance-based)
        param_importance = {}
        for col in df.columns:
            if col.startswith("param_"):
                param_name = col[6:]
                # Calculate correlation with metric
                if df[col].dtype in [np.float64, np.int64]:
                    correlation = abs(df[col].corr(df["metric"]))
                    param_importance[param_name] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # Sort by importance
        analysis["parameter_importance"] = dict(
            sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Metric statistics
        analysis["metric_stats"] = {
            "mean": float(df["metric"].mean()),
            "std": float(df["metric"].std()),
            "min": float(df["metric"].min()),
            "max": float(df["metric"].max()),
            "median": float(df["metric"].median())
        }
        
        return analysis
    
    def get_best_model_config(
        self,
        base_config: Dict[str, Any],
        param_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get the model configuration with best hyperparameters.
        
        Args:
            base_config: Base model configuration
            param_mapping: Parameter mapping used during optimization
            
        Returns:
            Model configuration with best hyperparameters
        """
        if not self.best_params:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Create a copy of base config
        best_config = base_config.copy()
        
        # Apply best parameters
        if param_mapping:
            for hpo_param, config_path in param_mapping.items():
                if hpo_param in self.best_params:
                    self._set_nested_config(best_config, config_path, self.best_params[hpo_param])
        else:
            # Direct mapping
            if "hyperparameters" not in best_config:
                best_config["hyperparameters"] = {}
            best_config["hyperparameters"].update(self.best_params)
        
        return best_config
    
    def load_results(self, results_file: Union[str, Path]) -> Dict[str, Any]:
        """Load HPO results from file.
        
        Args:
            results_file: Path to results JSON file
            
        Returns:
            Results dictionary
        """
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Update internal state
        self.best_params = results.get("best_params")
        self.best_score = results.get("best_score")
        self.all_trials = results.get("all_trials", [])
        
        return results