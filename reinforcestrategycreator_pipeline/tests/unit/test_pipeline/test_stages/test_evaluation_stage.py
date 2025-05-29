"""Unit tests for EvaluationStage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation import EvaluationStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext


class TestEvaluationStage(unittest.TestCase):
    """Test cases for EvaluationStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "test_data_key": "test_data",
            "baseline_model": None,
            "threshold_config": {
                "accuracy": 0.8,
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.75
            },
            "report_format": "json"
        }
        self.stage = EvaluationStage(config=self.config)
        
        # Reset PipelineContext singleton
        PipelineContext._instance = None
        self.context = PipelineContext.get_instance()
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton
        PipelineContext._instance = None
        
    def test_initialization(self):
        """Test stage initialization."""
        self.assertEqual(self.stage.name, "evaluation")
        self.assertEqual(self.stage.metrics, ["accuracy", "precision", "recall", "f1"])
        self.assertEqual(self.stage.report_format, "json")
        self.assertIsNotNone(self.stage.threshold_config)
        
    def test_setup_without_model(self):
        """Test setup fails without trained model."""
        with self.assertRaises(ValueError) as cm:
            self.stage.setup(self.context)
        self.assertIn("No trained model found", str(cm.exception))
        
    def test_setup_with_model(self):
        """Test successful setup with model."""
        self.context.set("trained_model", {"type": "test_model"})
        self.context.set("artifact_store", Mock())
        
        # Should not raise
        self.stage.setup(self.context)
        self.assertIsNotNone(self.stage.artifact_store)
        
    def test_make_predictions(self):
        """Test making predictions."""
        model = {"type": "classification"}
        features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        predictions = self.stage._make_predictions(model, features)
        
        # Check predictions are generated
        self.assertEqual(len(predictions), 3)
        # For classification, predictions should be 0 or 1
        self.assertTrue(all(p in [0, 1] for p in predictions))
        
    def test_compute_metrics_classification(self):
        """Test computing metrics for classification."""
        # Binary classification case
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        
        metrics = self.stage._compute_metrics(y_true, y_pred)
        
        # Check accuracy
        self.assertIn("accuracy", metrics)
        expected_accuracy = np.mean(y_true == y_pred)
        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy)
        
        # Check precision
        self.assertIn("precision", metrics)
        
        # Check recall
        self.assertIn("recall", metrics)
        
        # Check F1
        self.assertIn("f1", metrics)
        
    def test_compute_metrics_regression(self):
        """Test computing metrics for regression."""
        self.stage.metrics = ["mse", "mae", "accuracy"]
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        metrics = self.stage._compute_metrics(y_true, y_pred)
        
        # Check MSE
        self.assertIn("mse", metrics)
        expected_mse = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(metrics["mse"], expected_mse)
        
        # Check MAE
        self.assertIn("mae", metrics)
        expected_mae = np.mean(np.abs(y_true - y_pred))
        self.assertAlmostEqual(metrics["mae"], expected_mae)
        
        # For regression, accuracy becomes R² score
        self.assertIn("r2_score", metrics)
        
    def test_evaluate_baseline(self):
        """Test baseline evaluation."""
        baseline_results = self.stage._evaluate_baseline()
        
        # Check baseline results structure
        for metric in self.stage.metrics:
            self.assertIn(metric, baseline_results)
            
        # Check baseline values are reasonable
        self.assertEqual(baseline_results["accuracy"], 0.5)
        self.assertEqual(baseline_results["precision"], 0.5)
        
    def test_compare_with_baseline(self):
        """Test comparison with baseline."""
        model_results = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77
        }
        baseline_results = {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5
        }
        
        comparison = self.stage._compare_with_baseline(model_results, baseline_results)
        
        # Check comparison structure
        for metric in model_results:
            self.assertIn(metric, comparison)
            self.assertIn("model", comparison[metric])
            self.assertIn("baseline", comparison[metric])
            self.assertIn("improvement_pct", comparison[metric])
            
        # Check improvement calculation
        expected_improvement = (0.85 - 0.5) / 0.5 * 100
        self.assertAlmostEqual(comparison["accuracy"]["improvement_pct"], expected_improvement)
        
    def test_check_thresholds(self):
        """Test threshold checking."""
        evaluation_results = {
            "accuracy": 0.85,
            "precision": 0.65,  # Below threshold
            "recall": 0.75,
            "f1": 0.70  # Below threshold
        }
        
        threshold_results = self.stage._check_thresholds(evaluation_results)
        
        # Check results
        self.assertTrue(threshold_results["accuracy"])  # Passes
        self.assertFalse(threshold_results["precision"])  # Fails
        self.assertTrue(threshold_results["recall"])  # Passes
        self.assertFalse(threshold_results["f1"])  # Fails
        
    def test_check_thresholds_with_min_max(self):
        """Test threshold checking with min/max values."""
        self.stage.threshold_config = {
            "accuracy": {"min": 0.8, "max": 0.95}
        }
        
        # Test within range
        evaluation_results = {"accuracy": 0.85}
        threshold_results = self.stage._check_thresholds(evaluation_results)
        self.assertTrue(threshold_results["accuracy"])
        
        # Test below min
        evaluation_results = {"accuracy": 0.75}
        threshold_results = self.stage._check_thresholds(evaluation_results)
        self.assertFalse(threshold_results["accuracy"])
        
        # Test above max
        evaluation_results = {"accuracy": 0.98}
        threshold_results = self.stage._check_thresholds(evaluation_results)
        self.assertFalse(threshold_results["accuracy"])
        
    def test_generate_report_json(self):
        """Test JSON report generation."""
        evaluation_results = {
            "accuracy": 0.85,
            "precision": 0.80
        }
        self.context.set("model_type", "test_model")
        self.context.set("model_config", {"param": "value"})
        
        report = self.stage._generate_report(evaluation_results, self.context)
        
        # Parse JSON report
        report_data = json.loads(report)
        
        # Check report structure
        self.assertIn("evaluation_timestamp", report_data)
        self.assertIn("model_type", report_data)
        self.assertIn("evaluation_metrics", report_data)
        self.assertEqual(report_data["model_type"], "test_model")
        
    def test_generate_report_html(self):
        """Test HTML report generation."""
        self.stage.report_format = "html"
        evaluation_results = {"accuracy": 0.85}
        self.context.set("model_type", "test_model")
        
        report = self.stage._generate_report(evaluation_results, self.context)
        
        # Check HTML structure
        self.assertIn("<html>", report)
        self.assertIn("</html>", report)
        self.assertIn("Model Evaluation Report", report)
        self.assertIn("test_model", report)
        
    def test_format_key_metrics(self):
        """Test formatting key metrics for logging."""
        evaluation_results = {
            "accuracy": 0.8567,
            "f1": 0.7234,
            "precision": 0.9123,
            "other_metric": 0.5
        }
        
        formatted = self.stage._format_key_metrics(evaluation_results)
        
        # Check formatting
        self.assertIn("accuracy=0.8567", formatted)
        self.assertIn("f1=0.7234", formatted)
        self.assertNotIn("precision", formatted)  # Not a key metric
        self.assertNotIn("other_metric", formatted)
        
    def test_run_without_test_data(self):
        """Test run fails without test data."""
        self.context.set("trained_model", {"type": "test"})
        self.context.set("artifact_store", Mock())
        self.stage.setup(self.context)
        
        with self.assertRaises(ValueError) as cm:
            self.stage.run(self.context)
        self.assertIn("No test data found", str(cm.exception))
        
    def test_run_success(self):
        """Test successful evaluation run."""
        # Set up context
        self.context.set("trained_model", {"type": "classification"})
        self.context.set("test_features", [[1, 2], [3, 4], [5, 6]])
        self.context.set("test_labels", [0, 1, 0])
        self.context.set("model_type", "test_model")
        self.context.set("artifact_store", Mock())
        
        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)
        
        # Check context updates
        self.assertIsNotNone(result_context.get("evaluation_results"))
        self.assertIsNotNone(result_context.get("evaluation_report"))
        self.assertIsNotNone(result_context.get("model_passed_thresholds"))
        self.assertIsNotNone(result_context.get("evaluation_metadata"))
        
        # Check metadata structure
        metadata = result_context.get("evaluation_metadata")
        self.assertIn("evaluated_at", metadata)
        self.assertIn("model_type", metadata)
        self.assertIn("test_samples", metadata)
        self.assertIn("metrics_computed", metadata)
        
    def test_run_with_baseline(self):
        """Test evaluation with baseline comparison."""
        # Enable baseline
        self.stage.baseline_model = {"type": "baseline"}
        
        # Set up context
        self.context.set("trained_model", {"type": "test"})
        self.context.set("test_features", [[1, 2]])
        self.context.set("test_labels", [0])
        self.context.set("artifact_store", Mock())
        
        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)
        
        # Check baseline comparison was performed
        evaluation_results = result_context.get("evaluation_results")
        self.assertIn("baseline_comparison", evaluation_results)
        
    @patch('json.dump')
    @patch('tempfile.NamedTemporaryFile')
    def test_save_evaluation_artifacts(self, mock_tempfile, mock_json_dump):
        """Test saving evaluation artifacts."""
        # Set up mocks
        mock_results_file = MagicMock()
        mock_report_file = MagicMock()
        mock_tempfile.side_effect = [
            MagicMock(__enter__=lambda self: mock_results_file, __exit__=lambda *args: None),
            MagicMock(__enter__=lambda self: mock_report_file, __exit__=lambda *args: None)
        ]
        mock_results_file.name = "/tmp/results.json"
        mock_report_file.name = "/tmp/report.txt"
        
        mock_artifact_store = Mock()
        mock_artifact_store.save_artifact.side_effect = [
            Mock(artifact_id="results_123"),
            Mock(artifact_id="report_123")
        ]
        self.stage.artifact_store = mock_artifact_store
        
        # Set up data
        evaluation_results = {"accuracy": 0.85}
        report = "Test report"
        self.context.set_metadata("run_id", "test_run")
        
        # Save artifacts
        self.stage._save_evaluation_artifacts(evaluation_results, report, self.context)
        
        # Verify artifact store was called twice
        self.assertEqual(mock_artifact_store.save_artifact.call_count, 2)
        
        # Verify context was updated
        self.assertEqual(self.context.get("evaluation_results_artifact"), "results_123")
        self.assertEqual(self.context.get("evaluation_report_artifact"), "report_123")


if __name__ == "__main__":
    unittest.main()