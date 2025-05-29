"""Evaluation stage implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd # Added import
import json
import numpy as np
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactType


class EvaluationStage(PipelineStage):
    """
    Stage responsible for evaluating trained models.
    
    This stage handles:
    - Loading trained models and test data
    - Computing evaluation metrics
    - Generating performance reports
    - Comparing against baseline models
    - Saving evaluation results as artifacts
    """
    
    def __init__(self, name: str = "evaluation", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation stage.
        
        Args:
            name: Stage name
            config: Stage configuration containing:
                - metrics: List of metrics to compute
                - test_data_key: Key to retrieve test data from context
                - baseline_model: Optional baseline model for comparison
                - threshold_config: Performance thresholds for pass/fail
                - report_format: Format for evaluation report (json, html, etc.)
        """
        super().__init__(name, config or {})
        self.metrics = self.config.get("metrics", ["accuracy", "precision", "recall", "f1"])
        self.test_data_key = self.config.get("test_data_key", "test_data")
        self.baseline_model = self.config.get("baseline_model")
        self.threshold_config = self.config.get("threshold_config", {})
        self.report_format = self.config.get("report_format", "json")
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the evaluation stage."""
        self.logger.info(f"Setting up {self.name} stage")
        
        # Validate required data is available
        if not context.get("trained_model"):
            raise ValueError("No trained model found in context. Run training stage first.")
            
        # Get artifact store from context
        self.artifact_store = context.get("artifact_store")
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the evaluation stage.
        
        Args:
            context: Pipeline context containing trained model and test data
            
        Returns:
            Updated pipeline context with evaluation results
        """
        self.logger.info(f"Running {self.name} stage")
        
        try:
            # Get model and test data from context
            model = context.get("trained_model")
            
            # Handle DataFrame boolean ambiguity for features
            raw_test_features = context.get("test_features")
            if isinstance(raw_test_features, pd.DataFrame) and not raw_test_features.empty:
                test_features = raw_test_features
            else: # Handles None or empty DataFrame
                raw_processed_features = context.get("processed_features")
                if isinstance(raw_processed_features, pd.DataFrame) and not raw_processed_features.empty:
                    test_features = raw_processed_features
                else:
                    test_features = None # Or an empty DataFrame if preferred: pd.DataFrame()
            
            # Handle DataFrame boolean ambiguity for labels
            raw_test_labels = context.get("test_labels")
            if isinstance(raw_test_labels, pd.Series) and not raw_test_labels.empty: # Assuming labels are Series
                test_labels = raw_test_labels
            else: # Handles None or empty Series
                raw_labels = context.get("labels")
                if isinstance(raw_labels, pd.Series) and not raw_labels.empty:
                    test_labels = raw_labels
                else:
                    test_labels = None # Or an empty Series if preferred: pd.Series(dtype='object')

            if test_features is None or test_labels is None:
                raise ValueError("No test data found in context")
            
            # Make predictions
            self.logger.info("Generating predictions on test data")
            predictions = self._make_predictions(model, test_features)
            
            # Compute evaluation metrics
            self.logger.info(f"Computing evaluation metrics: {self.metrics}")
            evaluation_results = self._compute_metrics(test_labels, predictions)
            
            # Compare with baseline if provided
            if self.baseline_model:
                baseline_results = self._evaluate_baseline()
                evaluation_results["baseline_comparison"] = self._compare_with_baseline(
                    evaluation_results, baseline_results
                )
            
            # Check performance thresholds
            threshold_results = self._check_thresholds(evaluation_results)
            evaluation_results["threshold_checks"] = threshold_results
            
            # Generate evaluation report
            report = self._generate_report(evaluation_results, context)
            
            # Store results in context
            context.set("evaluation_results", evaluation_results)
            context.set("evaluation_report", report)
            context.set("model_passed_thresholds", all(threshold_results.values()))
            
            # Store evaluation metadata
            metadata = {
                "evaluated_at": datetime.now().isoformat(),
                "model_type": context.get("model_type"),
                "test_samples": len(test_labels) if hasattr(test_labels, '__len__') else 0,
                "metrics_computed": list(evaluation_results.keys()),
                "passed_thresholds": all(threshold_results.values())
            }
            context.set("evaluation_metadata", metadata)
            
            # Save evaluation artifacts if artifact store is available
            if self.artifact_store:
                self._save_evaluation_artifacts(evaluation_results, report, context)
            
            # Log summary results
            self.logger.info("Evaluation completed successfully")
            self.logger.info(f"Key metrics: {self._format_key_metrics(evaluation_results)}")
            if not all(threshold_results.values()):
                self.logger.warning(f"Model failed threshold checks: {threshold_results}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in evaluation stage: {str(e)}")
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after evaluation."""
        self.logger.info(f"Tearing down {self.name} stage")
        
    def _make_predictions(self, model: Any, features: Any) -> Any:
        """
        Make predictions using the trained model.
        
        Args:
            model: Trained model
            features: Test features
            
        Returns:
            Model predictions
        """
        # This is a placeholder implementation
        # In practice, this would use the actual model's predict method
        
        # Simulate predictions based on model type
        model_type = model.get("type", "default") if isinstance(model, dict) else "unknown"
        
        # Generate dummy predictions for demonstration
        num_samples = len(features) if hasattr(features, '__len__') else 100
        
        if model_type == "classification":
            # Binary classification predictions
            predictions = np.random.randint(0, 2, size=num_samples)
        else:
            # Regression or default predictions
            predictions = np.random.randn(num_samples)
            
        return predictions
        
    def _compute_metrics(self, true_labels: Any, predictions: Any) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            true_labels: True labels
            predictions: Model predictions
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        
        # This is a simplified implementation
        # In practice, you'd use sklearn.metrics or similar
        
        # Convert to numpy arrays for easier computation
        y_true = np.array(true_labels) if not isinstance(true_labels, np.ndarray) else true_labels
        y_pred = np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Compute metrics based on requested metrics
        if "accuracy" in self.metrics:
            # For classification
            if y_true.dtype in [np.int32, np.int64, bool]:
                results["accuracy"] = np.mean(y_true == y_pred)
            else:
                # For regression, use R² score
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                results["r2_score"] = 1 - (ss_res / (ss_tot + 1e-10))
                
        if "precision" in self.metrics and y_true.dtype in [np.int32, np.int64, bool]:
            # Binary classification precision
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            predicted_positives = np.sum(y_pred == 1)
            results["precision"] = true_positives / (predicted_positives + 1e-10)
            
        if "recall" in self.metrics and y_true.dtype in [np.int32, np.int64, bool]:
            # Binary classification recall
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            actual_positives = np.sum(y_true == 1)
            results["recall"] = true_positives / (actual_positives + 1e-10)
            
        if "f1" in self.metrics and "precision" in results and "recall" in results:
            # F1 score
            precision = results["precision"]
            recall = results["recall"]
            results["f1"] = 2 * (precision * recall) / (precision + recall + 1e-10)
            
        if "mse" in self.metrics:
            # Mean squared error
            results["mse"] = np.mean((y_true - y_pred) ** 2)
            
        if "mae" in self.metrics:
            # Mean absolute error
            results["mae"] = np.mean(np.abs(y_true - y_pred))
            
        return results
        
    def _evaluate_baseline(self) -> Dict[str, float]:
        """Evaluate baseline model for comparison."""
        # Placeholder for baseline evaluation
        # In practice, this would load and evaluate an actual baseline model
        
        baseline_results = {}
        for metric in self.metrics:
            if metric == "accuracy":
                baseline_results[metric] = 0.5  # Random baseline for binary classification
            elif metric in ["precision", "recall", "f1"]:
                baseline_results[metric] = 0.5
            elif metric == "mse":
                baseline_results[metric] = 1.0
            elif metric == "mae":
                baseline_results[metric] = 0.8
                
        return baseline_results
        
    def _compare_with_baseline(
        self, 
        model_results: Dict[str, float], 
        baseline_results: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare model performance with baseline."""
        comparison = {}
        
        for metric in model_results:
            if metric in baseline_results:
                model_value = model_results[metric]
                baseline_value = baseline_results[metric]
                
                # Calculate improvement
                if metric in ["mse", "mae"]:  # Lower is better
                    improvement = (baseline_value - model_value) / (baseline_value + 1e-10)
                else:  # Higher is better
                    improvement = (model_value - baseline_value) / (baseline_value + 1e-10)
                    
                comparison[metric] = {
                    "model": model_value,
                    "baseline": baseline_value,
                    "improvement_pct": improvement * 100
                }
                
        return comparison
        
    def _check_thresholds(self, evaluation_results: Dict[str, Any]) -> Dict[str, bool]:
        """Check if model meets performance thresholds."""
        threshold_results = {}
        
        for metric, threshold in self.threshold_config.items():
            if metric in evaluation_results:
                value = evaluation_results[metric]
                
                if isinstance(threshold, dict):
                    # Handle min/max thresholds
                    passed = True
                    if "min" in threshold and value < threshold["min"]:
                        passed = False
                    if "max" in threshold and value > threshold["max"]:
                        passed = False
                    threshold_results[metric] = passed
                else:
                    # Simple threshold comparison
                    if metric in ["mse", "mae"]:  # Lower is better
                        threshold_results[metric] = value <= threshold
                    else:  # Higher is better
                        threshold_results[metric] = value >= threshold
                        
        return threshold_results
        
    def _generate_report(self, evaluation_results: Dict[str, Any], context: PipelineContext) -> str:
        """Generate evaluation report."""
        report_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": context.get("model_type"),
            "model_config": context.get("model_config"),
            "evaluation_metrics": evaluation_results,
            "training_metadata": context.get("training_metadata"),
            "threshold_checks": evaluation_results.get("threshold_checks", {}),
            "baseline_comparison": evaluation_results.get("baseline_comparison", {})
        }
        
        if self.report_format == "json":
            return json.dumps(report_data, indent=2)
        elif self.report_format == "html":
            # Simplified HTML report
            html = f"""
            <html>
            <head><title>Model Evaluation Report</title></head>
            <body>
                <h1>Model Evaluation Report</h1>
                <h2>Model: {report_data['model_type']}</h2>
                <h3>Evaluation Metrics</h3>
                <pre>{json.dumps(report_data['evaluation_metrics'], indent=2)}</pre>
                <h3>Generated: {report_data['evaluation_timestamp']}</h3>
            </body>
            </html>
            """
            return html
        else:
            # Default to string representation
            return str(report_data)
            
    def _format_key_metrics(self, evaluation_results: Dict[str, Any]) -> str:
        """Format key metrics for logging."""
        key_metrics = []
        for metric in ["accuracy", "f1", "mse", "r2_score"]:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                if isinstance(value, float):
                    key_metrics.append(f"{metric}={value:.4f}")
                    
        return ", ".join(key_metrics)
        
    def _save_evaluation_artifacts(
        self, 
        evaluation_results: Dict[str, Any], 
        report: str,
        context: PipelineContext
    ) -> None:
        """Save evaluation results and report as artifacts."""
        try:
            run_id = context.get_metadata("run_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save evaluation results as JSON
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(evaluation_results, tmp, indent=2)
                results_path = tmp.name
                
            results_metadata = self.artifact_store.save_artifact(
                artifact_id=f"evaluation_results_{run_id}_{timestamp}",
                artifact_path=results_path,
                artifact_type=ArtifactType.METRICS,
                description="Model evaluation results",
                tags=["evaluation", "metrics", self.name]
            )
            
            # Save evaluation report
            report_suffix = ".html" if self.report_format == "html" else ".txt"
            with tempfile.NamedTemporaryFile(mode='w', suffix=report_suffix, delete=False) as tmp:
                tmp.write(report)
                report_path = tmp.name
                
            report_metadata = self.artifact_store.save_artifact(
                artifact_id=f"evaluation_report_{run_id}_{timestamp}",
                artifact_path=report_path,
                artifact_type=ArtifactType.REPORT,
                description="Model evaluation report",
                tags=["evaluation", "report", self.name]
            )
            
            # Store artifact references in context
            context.set("evaluation_results_artifact", results_metadata.artifact_id)
            context.set("evaluation_report_artifact", report_metadata.artifact_id)
            
            # Clean up temporary files
            Path(results_path).unlink()
            Path(report_path).unlink()
            
            self.logger.info(f"Saved evaluation artifacts: {results_metadata.artifact_id}, {report_metadata.artifact_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save evaluation artifacts: {str(e)}")