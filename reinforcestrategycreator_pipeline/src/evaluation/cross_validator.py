"""Cross-Validation system for model evaluation and selection."""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import json

from ..models.base import ModelBase
from ..models.factory import ModelFactory, get_factory
from ..models.registry import ModelRegistry
from ..training.engine import TrainingEngine
from ..data.splitter import DataSplitter
from ..data.manager import DataManager
from ..artifact_store.base import ArtifactStore


@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold."""
    fold_idx: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    training_time: float
    model_id: Optional[str] = None
    model_path: Optional[Path] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVResults:
    """Aggregated cross-validation results."""
    fold_results: List[CVFoldResult]
    aggregated_metrics: Dict[str, Dict[str, float]]  # metric -> {mean, std, min, max}
    best_fold_idx: int
    total_time: float
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "fold_results": [
                {
                    "fold_idx": fr.fold_idx,
                    "train_metrics": fr.train_metrics,
                    "val_metrics": fr.val_metrics,
                    "training_time": fr.training_time,
                    "model_id": fr.model_id,
                    "model_path": str(fr.model_path) if fr.model_path else None,
                    "additional_info": fr.additional_info
                }
                for fr in self.fold_results
            ],
            "aggregated_metrics": self.aggregated_metrics,
            "best_fold_idx": self.best_fold_idx,
            "total_time": self.total_time,
            "config": self.config,
            "timestamp": self.timestamp
        }
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of fold results."""
        data = []
        for fr in self.fold_results:
            row = {"fold": fr.fold_idx}
            row.update({f"train_{k}": v for k, v in fr.train_metrics.items()})
            row.update({f"val_{k}": v for k, v in fr.val_metrics.items()})
            row["training_time"] = fr.training_time
            data.append(row)
        return pd.DataFrame(data)


class CrossValidator:
    """Enhanced Cross-Validator for model evaluation.
    
    Supports multiple splitting strategies, parallel execution,
    and multi-metric evaluation.
    """
    
    def __init__(
        self,
        model_factory: Optional[ModelFactory] = None,
        model_registry: Optional[ModelRegistry] = None,
        artifact_store: Optional[ArtifactStore] = None,
        data_manager: Optional[DataManager] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        n_jobs: int = 1,
        use_multiprocessing: bool = False
    ):
        """Initialize the CrossValidator.
        
        Args:
            model_factory: ModelFactory instance (uses global if not provided)
            model_registry: Optional ModelRegistry for model versioning
            artifact_store: Optional ArtifactStore for saving artifacts
            data_manager: Optional DataManager for data loading
            checkpoint_dir: Directory for saving checkpoints
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            use_multiprocessing: Use multiprocessing instead of threading
        """
        self.model_factory = model_factory or get_factory()
        self.model_registry = model_registry
        self.artifact_store = artifact_store
        self.data_manager = data_manager
        
        # Set up checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("./cv_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Parallel execution settings
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None uses all CPUs
        self.use_multiprocessing = use_multiprocessing
        
        # Logger
        self.logger = logging.getLogger("CrossValidator")
    
    def cross_validate(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        cv_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        scoring_metric: Optional[str] = None,
        scoring_mode: str = "min",
        save_models: bool = False,
        callbacks: Optional[List[Any]] = None
    ) -> CVResults:
        """Perform cross-validation on a model.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            cv_config: Cross-validation configuration
                - method: 'kfold', 'time_series', 'stratified'
                - n_folds: Number of folds (default: 5)
                - shuffle: Whether to shuffle (for kfold)
                - random_seed: Random seed
            training_config: Training configuration for each fold
            metrics: List of metrics to evaluate
            scoring_metric: Primary metric for model selection
            scoring_mode: 'min' or 'max' for the scoring metric
            save_models: Whether to save trained models
            callbacks: Training callbacks to use
            
        Returns:
            CVResults object with fold results and aggregated metrics
        """
        start_time = time.time()
        
        # Default configurations
        if cv_config is None:
            cv_config = {}
        if training_config is None:
            training_config = {}
        if metrics is None:
            metrics = ["loss"]
        if scoring_metric is None:
            scoring_metric = metrics[0]
            
        # Validate scoring metric
        if scoring_metric not in metrics:
            metrics.append(scoring_metric)
            
        # Load data
        self.logger.info("Loading data for cross-validation")
        data = self._load_cv_data(data_config)
        
        # Create folds
        self.logger.info("Creating cross-validation folds")
        folds = self._create_folds(data, cv_config)
        n_folds = len(folds)
        self.logger.info(f"Created {n_folds} folds")
        
        # Run cross-validation
        if self.n_jobs == 1:
            # Sequential execution
            fold_results = []
            for i, (train_data, val_data) in enumerate(folds):
                self.logger.info(f"Training fold {i+1}/{n_folds}")
                result = self._train_fold(
                    i, train_data, val_data, model_config, 
                    training_config, metrics, save_models, callbacks
                )
                fold_results.append(result)
        else:
            # Parallel execution
            fold_results = self._parallel_cv(
                folds, model_config, training_config, 
                metrics, save_models, callbacks
            )
        
        # Sort results by fold index
        fold_results.sort(key=lambda x: x.fold_idx)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(fold_results)
        
        # Find best fold
        best_fold_idx = self._find_best_fold(
            fold_results, scoring_metric, scoring_mode
        )
        
        # Create results object
        total_time = time.time() - start_time
        cv_results = CVResults(
            fold_results=fold_results,
            aggregated_metrics=aggregated_metrics,
            best_fold_idx=best_fold_idx,
            total_time=total_time,
            config={
                "model_config": model_config,
                "cv_config": cv_config,
                "training_config": training_config,
                "metrics": metrics,
                "scoring_metric": scoring_metric,
                "scoring_mode": scoring_mode,
                "n_folds": n_folds
            }
        )
        
        # Save results
        self._save_cv_results(cv_results)
        
        # Log summary
        self._log_cv_summary(cv_results)
        
        return cv_results
    
    def _load_cv_data(self, data_config: Dict[str, Any]) -> Any:
        """Load data for cross-validation."""
        if self.data_manager:
            source_id = data_config.get("source_id")
            if not source_id:
                raise ValueError("data_config must contain 'source_id' when using DataManager")
            return self.data_manager.load_data(source_id, **data_config.get("params", {}))
        else:
            # Direct data loading
            data = data_config.get("data")
            if data is None:
                raise ValueError("No data provided in data_config")
            return data
    
    def _create_folds(
        self, 
        data: Any, 
        cv_config: Dict[str, Any]
    ) -> List[Tuple[Any, Any]]:
        """Create cross-validation folds."""
        method = cv_config.get("method", "kfold")
        n_folds = cv_config.get("n_folds", 5)
        random_seed = cv_config.get("random_seed", 42)
        
        # Map CV method to DataSplitter method
        splitter_method_map = {
            "kfold": "random",
            "time_series": "time_series",
            "stratified": "stratified"
        }
        
        splitter_method = splitter_method_map.get(method, "random")
        splitter = DataSplitter(method=splitter_method, random_seed=random_seed)
        
        # Handle different data types
        if isinstance(data, pd.DataFrame):
            target_column = cv_config.get("target_column")
            return splitter.create_folds(data, n_folds, target_column)
        else:
            # For non-DataFrame data, create simple index-based folds
            n_samples = len(data) if hasattr(data, '__len__') else data.shape[0]
            indices = np.arange(n_samples)
            
            if method == "time_series":
                # Time series CV: expanding window
                folds = []
                fold_size = n_samples // (n_folds + 1)
                for i in range(n_folds):
                    train_end = (i + 1) * fold_size
                    val_start = train_end
                    val_end = val_start + fold_size
                    
                    train_indices = indices[:train_end]
                    val_indices = indices[val_start:val_end]
                    
                    train_data = self._subset_data(data, train_indices)
                    val_data = self._subset_data(data, val_indices)
                    
                    folds.append((train_data, val_data))
            else:
                # Standard k-fold
                np.random.seed(random_seed)
                if cv_config.get("shuffle", True):
                    np.random.shuffle(indices)
                
                fold_size = n_samples // n_folds
                folds = []
                
                for i in range(n_folds):
                    start = i * fold_size
                    end = start + fold_size if i < n_folds - 1 else n_samples
                    
                    val_indices = indices[start:end]
                    train_indices = np.concatenate([indices[:start], indices[end:]])
                    
                    train_data = self._subset_data(data, train_indices)
                    val_data = self._subset_data(data, val_indices)
                    
                    folds.append((train_data, val_data))
            
            return folds
    
    def _subset_data(self, data: Any, indices: np.ndarray) -> Any:
        """Subset data based on indices."""
        if hasattr(data, 'iloc'):
            return data.iloc[indices]
        elif hasattr(data, '__getitem__'):
            if isinstance(data, np.ndarray):
                return data[indices]
            else:
                # Handle list-like data
                return [data[i] for i in indices]
        else:
            # Fallback
            return data
    
    def _train_fold(
        self,
        fold_idx: int,
        train_data: Any,
        val_data: Any,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        metrics: List[str],
        save_models: bool,
        callbacks: Optional[List[Any]]
    ) -> CVFoldResult:
        """Train a single fold."""
        fold_start_time = time.time()
        
        # Create training engine for this fold
        engine = TrainingEngine(
            model_factory=self.model_factory,
            model_registry=self.model_registry if save_models else None,
            artifact_store=self.artifact_store if save_models else None,
            checkpoint_dir=self.checkpoint_dir / f"fold_{fold_idx}"
        )
        
        # Prepare data config for the engine
        fold_data_config = {
            "train_data": train_data,
            "val_data": val_data
        }
        
        # Train the model
        result = engine.train(
            model_config=model_config,
            data_config=fold_data_config,
            training_config=training_config,
            callbacks=callbacks
        )
        
        if not result["success"]:
            raise RuntimeError(f"Training failed for fold {fold_idx}: {result.get('error')}")
        
        # Extract metrics
        final_metrics = result.get("final_metrics", {})
        train_metrics = {k: v for k, v in final_metrics.items() if not k.startswith("val_")}
        val_metrics = {k.replace("val_", ""): v for k, v in final_metrics.items() if k.startswith("val_")}
        
        # Ensure requested metrics are present
        for metric in metrics:
            if metric not in train_metrics:
                train_metrics[metric] = np.nan
            if metric not in val_metrics:
                val_metrics[metric] = np.nan
        
        # Create fold result
        fold_result = CVFoldResult(
            fold_idx=fold_idx,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time=time.time() - fold_start_time,
            model_id=result.get("model_id"),
            model_path=self.checkpoint_dir / f"fold_{fold_idx}" if save_models else None,
            additional_info={
                "epochs_trained": result.get("epochs_trained", 0),
                "history": result.get("history", {})
            }
        )
        
        return fold_result
    
    def _parallel_cv(
        self,
        folds: List[Tuple[Any, Any]],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        metrics: List[str],
        save_models: bool,
        callbacks: Optional[List[Any]]
    ) -> List[CVFoldResult]:
        """Run cross-validation in parallel."""
        fold_results = []
        
        # Choose executor based on settings
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.n_jobs) as executor:
            # Submit all fold training jobs
            future_to_fold = {}
            for i, (train_data, val_data) in enumerate(folds):
                future = executor.submit(
                    self._train_fold,
                    i, train_data, val_data, model_config,
                    training_config, metrics, save_models, callbacks
                )
                future_to_fold[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    result = future.result()
                    fold_results.append(result)
                    self.logger.info(f"Completed fold {fold_idx + 1}/{len(folds)}")
                except Exception as e:
                    self.logger.error(f"Fold {fold_idx} failed: {str(e)}")
                    raise
        
        return fold_results
    
    def _aggregate_metrics(
        self, 
        fold_results: List[CVFoldResult]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across folds."""
        aggregated = {}
        
        # Collect all metrics
        all_metrics = set()
        for fr in fold_results:
            all_metrics.update(fr.train_metrics.keys())
            all_metrics.update(fr.val_metrics.keys())
        
        # Aggregate each metric
        for metric in all_metrics:
            train_values = []
            val_values = []
            
            for fr in fold_results:
                if metric in fr.train_metrics:
                    train_values.append(fr.train_metrics[metric])
                if metric in fr.val_metrics:
                    val_values.append(fr.val_metrics[metric])
            
            # Calculate statistics
            if train_values:
                aggregated[f"train_{metric}"] = {
                    "mean": np.mean(train_values),
                    "std": np.std(train_values),
                    "min": np.min(train_values),
                    "max": np.max(train_values)
                }
            
            if val_values:
                aggregated[f"val_{metric}"] = {
                    "mean": np.mean(val_values),
                    "std": np.std(val_values),
                    "min": np.min(val_values),
                    "max": np.max(val_values)
                }
        
        return aggregated
    
    def _find_best_fold(
        self,
        fold_results: List[CVFoldResult],
        scoring_metric: str,
        scoring_mode: str
    ) -> int:
        """Find the best fold based on validation metric."""
        val_scores = []
        
        for fr in fold_results:
            score = fr.val_metrics.get(scoring_metric, np.nan)
            val_scores.append(score)
        
        # Handle NaN values
        valid_scores = [(i, s) for i, s in enumerate(val_scores) if not np.isnan(s)]
        
        if not valid_scores:
            self.logger.warning(f"No valid scores found for metric '{scoring_metric}'")
            return 0
        
        # Find best based on mode
        if scoring_mode == "min":
            best_idx = min(valid_scores, key=lambda x: x[1])[0]
        else:
            best_idx = max(valid_scores, key=lambda x: x[1])[0]
        
        return best_idx
    
    def _save_cv_results(self, cv_results: CVResults) -> None:
        """Save cross-validation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.checkpoint_dir / f"cv_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(cv_results.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved CV results to {results_file}")
        
        # Also save summary DataFrame
        summary_df = cv_results.get_summary_df()
        summary_file = self.checkpoint_dir / f"cv_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Saved CV summary to {summary_file}")
    
    def _log_cv_summary(self, cv_results: CVResults) -> None:
        """Log cross-validation summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("CROSS-VALIDATION SUMMARY")
        self.logger.info("="*60)
        
        # Log aggregated metrics
        for metric, stats in cv_results.aggregated_metrics.items():
            self.logger.info(
                f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )
        
        # Log best fold
        best_fold = cv_results.fold_results[cv_results.best_fold_idx]
        scoring_metric = cv_results.config["scoring_metric"]
        self.logger.info(f"\nBest fold: {cv_results.best_fold_idx + 1}")
        self.logger.info(
            f"Best {scoring_metric}: {best_fold.val_metrics.get(scoring_metric, 'N/A')}"
        )
        
        # Log timing
        self.logger.info(f"\nTotal time: {cv_results.total_time:.2f}s")
        avg_fold_time = np.mean([fr.training_time for fr in cv_results.fold_results])
        self.logger.info(f"Average fold time: {avg_fold_time:.2f}s")
        self.logger.info("="*60 + "\n")
    
    def compare_models(
        self,
        model_configs: List[Dict[str, Any]],
        data_config: Dict[str, Any],
        cv_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        scoring_metric: Optional[str] = None,
        scoring_mode: str = "min"
    ) -> Dict[str, CVResults]:
        """Compare multiple models using cross-validation.
        
        Args:
            model_configs: List of model configurations to compare
            data_config: Data configuration
            cv_config: Cross-validation configuration
            training_config: Training configuration
            metrics: Metrics to evaluate
            scoring_metric: Primary metric for comparison
            scoring_mode: 'min' or 'max' for the scoring metric
            
        Returns:
            Dictionary mapping model names to CVResults
        """
        results = {}
        
        for i, model_config in enumerate(model_configs):
            model_name = model_config.get("name", f"model_{i}")
            self.logger.info(f"\nEvaluating model: {model_name}")
            
            cv_results = self.cross_validate(
                model_config=model_config,
                data_config=data_config,
                cv_config=cv_config,
                training_config=training_config,
                metrics=metrics,
                scoring_metric=scoring_metric,
                scoring_mode=scoring_mode,
                save_models=False  # Don't save intermediate models
            )
            
            results[model_name] = cv_results
        
        # Log comparison summary
        self._log_comparison_summary(results, scoring_metric, scoring_mode)
        
        return results
    
    def _log_comparison_summary(
        self,
        results: Dict[str, CVResults],
        scoring_metric: str,
        scoring_mode: str
    ) -> None:
        """Log model comparison summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("MODEL COMPARISON SUMMARY")
        self.logger.info("="*60)
        
        # Create comparison table
        comparison_data = []
        for model_name, cv_results in results.items():
            val_metric_key = f"val_{scoring_metric}"
            if val_metric_key in cv_results.aggregated_metrics:
                stats = cv_results.aggregated_metrics[val_metric_key]
                comparison_data.append({
                    "Model": model_name,
                    f"{scoring_metric} (mean)": stats["mean"],
                    f"{scoring_metric} (std)": stats["std"],
                    "Best Fold": cv_results.best_fold_idx + 1,
                    "Training Time": cv_results.total_time
                })
        
        # Sort by scoring metric
        comparison_data.sort(
            key=lambda x: x[f"{scoring_metric} (mean)"],
            reverse=(scoring_mode == "max")
        )
        
        # Log results
        for i, data in enumerate(comparison_data):
            self.logger.info(f"\nRank {i+1}: {data['Model']}")
            self.logger.info(
                f"  {scoring_metric}: {data[f'{scoring_metric} (mean)']:.4f} "
                f"± {data[f'{scoring_metric} (std)']:.4f}"
            )
            self.logger.info(f"  Best fold: {data['Best Fold']}")
            self.logger.info(f"  Training time: {data['Training Time']:.2f}s")
        
        self.logger.info("="*60 + "\n")