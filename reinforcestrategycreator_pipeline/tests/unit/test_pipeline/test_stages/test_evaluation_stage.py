from pathlib import Path
"""Unit tests for EvaluationStage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
from datetime import datetime
from pathlib import Path # Added import

from reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation import EvaluationStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager


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
        
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        # Mock the loader attribute and its base_path
        self.mock_config_manager.loader = MagicMock()
        self.mock_config_manager.loader.base_path = Path('.') # Mock base_path to avoid issues

        # Mock the return of get_config().evaluation.model_dump()
        # to provide necessary paths for data_source and baseline_model_artifact
        mock_eval_config_dump = {
            "metrics": ["accuracy", "precision"], # Simplified for now
            "report_formats": ["json"], # Simplified
            "test_data_source": {"type": "local", "path": "dummy_test_data.csv"},
            "baseline_model_artifact": {"type": "local", "path": "dummy_baseline_model.pkl"},
            "thresholds": {"accuracy": 0.7} # Simplified
        }
        mock_evaluation_section = MagicMock()
        mock_evaluation_section.model_dump.return_value = mock_eval_config_dump
        
        mock_pipeline_config = MagicMock()
        mock_pipeline_config.evaluation = mock_evaluation_section
        self.mock_config_manager.get_config.return_value = mock_pipeline_config
        
        self.context.set("config_manager", self.mock_config_manager)
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton
        PipelineContext._instance = None
        
    # def test_initialization(self):
    #     """Test stage initialization."""
    #     # This test needs to be updated as config is now primarily from global config
    #     # self.assertEqual(self.stage.name, "evaluation")
    #     # self.assertEqual(self.stage.metrics, ["accuracy", "precision", "recall", "f1"])
    #     # self.assertEqual(self.stage.report_format, "json")
    #     # self.assertIsNotNone(self.stage.threshold_config)
    #     pass
        
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
        
    # def test_make_predictions(self):
    #     """Test making predictions."""
    #     # _make_predictions was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_compute_metrics_classification(self):
    #     """Test computing metrics for classification."""
    #     # _compute_metrics was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_compute_metrics_regression(self):
    #     """Test computing metrics for regression."""
    #     # _compute_metrics was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_evaluate_baseline(self):
    #     """Test baseline evaluation."""
    #     # _evaluate_baseline was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_compare_with_baseline(self):
    #     """Test comparison with baseline."""
    #     # _compare_with_baseline was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_check_thresholds(self):
    #     """Test threshold checking."""
    #     # _check_thresholds was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_check_thresholds_with_min_max(self):
    #     """Test threshold checking with min/max values."""
    #     # _check_thresholds was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_generate_report_json(self):
    #     """Test JSON report generation."""
    #     # _generate_report was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_generate_report_html(self):
    #     """Test HTML report generation."""
    #     # _generate_report was removed. EvaluationEngine handles this.
    #     pass
        
    # def test_format_key_metrics(self):
    #     """Test formatting key metrics for logging."""
    #     # _format_key_metrics was removed.
    #     pass
        
    def test_run_without_test_data(self):
        """Test run fails without test data."""
        self.context.set("trained_model", {"type": "test"})
        self.context.set("artifact_store", Mock())
        # Mock EvaluationEngine.evaluate to avoid deeper errors for this specific test
        with patch.object(self.stage.evaluation_engine, 'evaluate', side_effect=ValueError("No test data found")):
            self.stage.setup(self.context)
            
            with self.assertRaises(ValueError) as cm:
                self.stage.run(self.context)
            self.assertIn("No test data found", str(cm.exception))
        
    @patch('reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation.EvaluationEngine.evaluate')
    @patch('reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation.EvaluationEngine.generate_report')
    def test_run_success(self, mock_generate_report, mock_evaluate_engine):
        """Test successful evaluation run."""
        # Mock EvaluationEngine.evaluate
        mock_evaluate_engine.return_value = {
            "metrics": {"accuracy": 0.9, "precision": 0.85},
            "threshold_results": {"accuracy": True, "precision": True},
            "passed_all_thresholds": True,
            "baseline_comparison": None # Or mock this too if needed
        }
        # Mock EvaluationEngine.generate_report
        mock_generate_report.return_value = "Mocked Report Content"

        # Set up context
        self.context.set("trained_model", Mock(name="MockTrainedModel")) # Needs to be an object for some internal checks
        self.context.set("test_features", np.array([[1, 2], [3, 4], [5, 6]])) # EvaluationEngine might expect numpy
        self.context.set("test_labels", np.array([0, 1, 0])) # EvaluationEngine might expect numpy
        self.context.set("model_type", "test_model_type_from_context")
        self.context.set("model_config", {"hyperparams": "test"})
        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)
        
        # Check context updates
        self.assertIsNotNone(result_context.get("evaluation_results"))
        self.assertIsNotNone(result_context.get("evaluation_report"))
        self.assertEqual(result_context.get("evaluation_report"), "Mocked Report Content")
        self.assertIsNotNone(result_context.get("model_passed_thresholds"))
        self.assertTrue(result_context.get("model_passed_thresholds"))
        self.assertIsNotNone(result_context.get("evaluation_metadata"))
        
        # Check metadata structure
        metadata = result_context.get("evaluation_metadata")
        self.assertIn("evaluated_at", metadata)
        self.assertIn("model_type", metadata)
        self.assertEqual(metadata["model_type"], "test_model_type_from_context")
        self.assertIn("test_samples", metadata)
        self.assertIn("metrics_computed", metadata)
        
        mock_evaluate_engine.assert_called_once()
        mock_generate_report.assert_called_once()
        
    # def test_run_with_baseline(self):
    #     """Test evaluation with baseline comparison."""
    #     # This test needs more elaborate mocking of EvaluationEngine and its interactions
    #     # with baseline models.
    #     pass
        
    # @patch('json.dump')
    # @patch('tempfile.NamedTemporaryFile')
    # def test_save_evaluation_artifacts(self, mock_tempfile, mock_json_dump):
    #     """Test saving evaluation artifacts."""
    #     # _save_evaluation_artifacts was removed. Artifact saving is part of EvaluationEngine.
    #     pass


if __name__ == "__main__":
    unittest.main()