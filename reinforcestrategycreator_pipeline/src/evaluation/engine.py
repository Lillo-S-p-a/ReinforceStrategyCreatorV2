"""Evaluation Engine for comprehensive model evaluation.

This module provides the EvaluationEngine class for evaluating trained models,
calculating performance metrics, comparing against benchmarks, and generating reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

from ..models.base import ModelBase
from ..models.registry import ModelRegistry
from ..data.manager import DataManager
from ..artifact_store.base import ArtifactStore, ArtifactType
from .metrics import MetricsCalculator
from .benchmarks import BenchmarkEvaluator
from ..visualization.performance_visualizer import PerformanceVisualizer
from ..visualization.report_generator import ReportGenerator


logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Engine for comprehensive model evaluation.
    
    This class orchestrates model evaluation including:
    - Multi-metric performance calculation
    - Benchmark comparisons
    - Report generation
    - Result persistence
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        data_manager: DataManager,
        artifact_store: ArtifactStore,
        metrics_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        visualization_config: Optional[Dict[str, Any]] = None,
        report_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the evaluation engine.
        
        Args:
            model_registry: Registry for loading models
            data_manager: Manager for loading evaluation data
            artifact_store: Store for saving evaluation results
            metrics_config: Configuration for metrics calculation
            benchmark_config: Configuration for benchmark strategies
            visualization_config: Configuration for visualization
            report_config: Configuration for report generation
        """
        self.model_registry = model_registry
        self.data_manager = data_manager
        self.artifact_store = artifact_store
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            config=metrics_config or {}
        )
        
        # Initialize benchmark evaluator
        self.benchmark_evaluator = BenchmarkEvaluator(
            config=benchmark_config or {},
            metrics_calculator=self.metrics_calculator
        )
        
        # Initialize visualization and reporting
        self.performance_visualizer = PerformanceVisualizer(
            config=visualization_config or {}
        )
        self.report_generator = ReportGenerator(
            config=report_config or {}
        )
        
        # Default report formats
        self.report_formats = ["json", "markdown", "html"]
    
    def evaluate(
        self,
        model_id: str,
        data_source_id: str,
        model_version: Optional[str] = None,
        data_version: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        compare_benchmarks: bool = True,
        save_results: bool = True,
        report_formats: Optional[List[str]] = None,
        generate_visualizations: bool = True,
        evaluation_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a model on specified data.
        
        Args:
            model_id: ID of the model to evaluate
            data_source_id: ID of the data source to use
            model_version: Specific model version (latest if None)
            data_version: Specific data version (latest if None)
            metrics: List of metrics to calculate (all if None)
            compare_benchmarks: Whether to compare against benchmarks
            save_results: Whether to save evaluation results
            report_formats: Formats for reports (uses defaults if None)
            generate_visualizations: Whether to generate visualization plots
            evaluation_name: Name for this evaluation run
            tags: Tags for categorizing the evaluation
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation of model {model_id} on data {data_source_id}")
        
        # Generate evaluation ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{model_id}_{timestamp}"
        
        if evaluation_name is None:
            evaluation_name = f"Evaluation of {model_id}"
        
        try:
            # Load model
            logger.info(f"Loading model {model_id} (version: {model_version or 'latest'})")
            model = self.model_registry.load_model(model_id, version=model_version)
            model_metadata = self.model_registry.get_model_metadata(model_id, version=model_version)
            
            # Load data
            logger.info(f"Loading data from {data_source_id} (version: {data_version or 'latest'})")
            if data_version:
                data = self.data_manager.load_data(data_source_id, version=data_version)
            else:
                data = self.data_manager.load_data(data_source_id)
            
            # Run model evaluation
            logger.info("Running model evaluation")
            model_metrics, portfolio_values = self._evaluate_model(model, data, metrics, **kwargs)
            
            # Prepare results
            results = {
                "evaluation_id": eval_id,
                "evaluation_name": evaluation_name,
                "timestamp": datetime.now().isoformat(),
                "model": {
                    "id": model_id,
                    "version": model_metadata.get("version"),
                    "type": model_metadata.get("model_type"),
                    "hyperparameters": model_metadata.get("hyperparameters", {})
                },
                "data": {
                    "source_id": data_source_id,
                    "version": data_version,
                    "shape": list(data.shape),
                    "columns": list(data.columns) if hasattr(data, 'columns') else None
                },
                "metrics": model_metrics,
                "evaluation_config": {
                    "metrics_requested": metrics,
                    "compare_benchmarks": compare_benchmarks,
                    "kwargs": kwargs
                }
            }
            
            # Compare with benchmarks if requested
            if compare_benchmarks:
                logger.info("Comparing with benchmark strategies")
                benchmark_results = self.benchmark_evaluator.compare_with_benchmarks(
                    data, model_metrics
                )
                results["benchmarks"] = benchmark_results["benchmarks"]
                results["benchmarks"] = benchmark_results["benchmarks"]
                results["relative_performance"] = benchmark_results["relative_performance"]
            
            # Generate visualizations if requested
            visualization_paths = {}
            if generate_visualizations and portfolio_values is not None:
                logger.info("Generating visualizations")
                viz_dir = Path(f"/tmp/{eval_id}/visualizations")
                viz_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate various plots
                try:
                    # Cumulative returns plot
                    fig = self.performance_visualizer.plot_cumulative_returns(
                        portfolio_values,
                        title=f"Cumulative Returns - {model_id}",
                        save_path=viz_dir / "cumulative_returns.png",
                        show=False
                    )
                    visualization_paths["cumulative_returns"] = str(viz_dir / "cumulative_returns.png")
                    
                    # Drawdown plot
                    fig = self.performance_visualizer.plot_drawdown(
                        portfolio_values,
                        title=f"Drawdown - {model_id}",
                        save_path=viz_dir / "drawdown.png",
                        show=False
                    )
                    visualization_paths["drawdown"] = str(viz_dir / "drawdown.png")
                    
                    # Metrics comparison if benchmarks available
                    if compare_benchmarks and "benchmarks" in results:
                        all_metrics = {"Model": model_metrics}
                        all_metrics.update(results["benchmarks"])
                        
                        fig = self.performance_visualizer.plot_metrics_comparison(
                            all_metrics,
                            title="Model vs Benchmarks",
                            save_path=viz_dir / "metrics_comparison.png",
                            show=False
                        )
                        visualization_paths["metrics_comparison"] = str(viz_dir / "metrics_comparison.png")
                    
                    # Performance dashboard
                    fig = self.performance_visualizer.create_performance_dashboard(
                        results,
                        save_path=viz_dir / "performance_dashboard.png",
                        show=False
                    )
                    visualization_paths["performance_dashboard"] = str(viz_dir / "performance_dashboard.png")
                    
                except Exception as e:
                    logger.error(f"Error generating visualizations: {str(e)}")
            
            # Generate reports using the new ReportGenerator
            if report_formats is None:
                report_formats = self.report_formats
            
            reports = {}
            for format_type in report_formats:
                logger.info(f"Generating {format_type} report")
                
                # Use the new report generator
                report_path = Path(f"/tmp/{eval_id}/reports/report.{format_type}")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                report = self.report_generator.generate_report(
                    results,
                    format_type=format_type,
                    output_path=report_path,
                    include_visualizations=bool(visualization_paths),
                    visualization_paths=visualization_paths
                )
                reports[format_type] = report
            
            results["reports"] = reports
            
            # Save results if requested
            if save_results:
                logger.info("Saving evaluation results")
                self._save_results(eval_id, results, tags)
            
            logger.info(f"Evaluation completed successfully: {eval_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise
    
    def _evaluate_model(
        self,
        model: ModelBase,
        data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Dict[str, float], Optional[List[float]]]:
        """Evaluate model and calculate metrics.
        
        Args:
            model: Model to evaluate
            data: Data to evaluate on
            metrics: Specific metrics to calculate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Tuple of (metrics dictionary, portfolio values list)
        """
        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Create a trading environment with the data
        # 2. Run the model through episodes
        # 3. Collect performance data
        # 4. Calculate requested metrics
        
        # For now, we'll simulate some metrics
        logger.warning("Using simulated metrics - implement actual model evaluation")
        
        # Simulate running the model and getting portfolio values
        num_steps = len(data)
        initial_value = 10000.0
        
        # Simulate portfolio values with some randomness
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, 0.02, num_steps - 1)
        portfolio_values = [initial_value]
        
        for ret in returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        # Simulate trades
        num_trades = np.random.randint(50, 200)
        win_rate = np.random.uniform(0.45, 0.65)
        
        # Calculate metrics using the metrics calculator
        calculated_metrics = self.metrics_calculator.calculate_all_metrics(
            portfolio_values=portfolio_values,
            returns=returns,
            trades_count=num_trades,
            win_rate=win_rate,
            requested_metrics=metrics
        )
        
        return calculated_metrics, portfolio_values
    
    def _save_results(
        self,
        eval_id: str,
        results: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> None:
        """Save evaluation results to artifact store.
        
        Args:
            eval_id: Evaluation ID
            results: Results to save
            tags: Tags for the artifact
        """
        # Create temporary directory for results
        temp_dir = Path(f"/tmp/{eval_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save results as JSON
            results_file = temp_dir / "results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save reports
            if "reports" in results:
                reports_dir = temp_dir / "reports"
                reports_dir.mkdir(exist_ok=True)
                
                for format_type, content in results["reports"].items():
                    ext = "json" if format_type == "json" else format_type
                    report_file = reports_dir / f"report.{ext}"
                    with open(report_file, "w") as f:
                        f.write(content)
            
            # Save to artifact store
            self.artifact_store.save_artifact(
                artifact_id=eval_id,
                artifact_path=temp_dir,
                artifact_type=ArtifactType.EVALUATION,
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                metadata={
                    "model_id": results["model"]["id"],
                    "model_version": results["model"]["version"],
                    "data_source_id": results["data"]["source_id"],
                    "metrics": results["metrics"],
                    "has_benchmarks": "benchmarks" in results
                },
                tags=tags or [],
                description=results["evaluation_name"]
            )
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            # Clean up on error
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            raise e
    
    def list_evaluations(
        self,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List evaluation results.
        
        Args:
            model_id: Filter by model ID
            tags: Filter by tags
            
        Returns:
            List of evaluation metadata
        """
        # Get all evaluation artifacts
        evaluations = self.artifact_store.list_artifacts(
            artifact_type=ArtifactType.EVALUATION,
            tags=tags
        )
        
        # Filter by model_id if provided
        if model_id:
            evaluations = [
                e for e in evaluations
                if e.properties.get("model_id") == model_id
            ]
        
        # Convert to list of dicts
        return [
            {
                "evaluation_id": e.artifact_id,
                "version": e.version,
                "created_at": e.created_at.isoformat(),
                "model_id": e.properties.get("model_id"),
                "model_version": e.properties.get("model_version"),
                "data_source_id": e.properties.get("data_source_id"),
                "metrics": e.properties.get("metrics", {}),
                "has_benchmarks": e.properties.get("has_benchmarks", False),
                "tags": e.tags,
                "description": e.description
            }
            for e in evaluations
        ]
    
    def load_evaluation(
        self,
        evaluation_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load evaluation results.
        
        Args:
            evaluation_id: ID of the evaluation
            version: Specific version to load
            
        Returns:
            Evaluation results dictionary
        """
        # Load from artifact store
        eval_path = self.artifact_store.load_artifact(
            artifact_id=evaluation_id,
            version=version
        )
        
        # Load results
        results_file = eval_path / "results.json"
        with open(results_file, "r") as f:
            results = json.load(f)
        
        return results