import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation import EvaluationStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactMetadata, ArtifactType

class TestEvaluationStageIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        self.mock_artifact_store = MagicMock()
        self.context.set("artifact_store", self.mock_artifact_store)
        self.context.set_metadata("run_id", "test_eval_run_789")

        self.mock_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger.return_value = self.mock_logger_instance

        self.stage_config = {
            "metrics": ["accuracy", "precision", "f1"],
            "threshold_config": {"accuracy": 0.75, "f1": 0.7},
            "report_format": "json",
            "baseline_model": None # Can be enabled in specific tests
        }
        self.stage = EvaluationStage(config=self.stage_config)

        # Prepare dummy data for context
        self.trained_model_mock = {"type": "test_classifier", "model_data": "some_weights"}
        self.test_features_df = pd.DataFrame({'x1': [1,0,1,0], 'x2': [0,1,0,1]})
        self.test_labels_series = pd.Series([1,0,1,1])
        
        self.context.set("trained_model", self.trained_model_mock)
        self.context.set("test_features", self.test_features_df)
        self.context.set("test_labels", self.test_labels_series)
        self.context.set("model_type", "test_classifier_from_train_stage")
        self.context.set("model_config", {"param": "value"})


    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.mock_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_evaluation_run_successful_interaction_with_mocked_engine_and_store(self):
        # --- Mocking the behavior of an EvaluationEngine ---
        mock_engine_predictions = np.array([1, 0, 0, 1]) # Example predictions
        mock_engine_metrics = {
            "accuracy": 0.75,
            "precision": 0.66,
            "f1": 0.70,
            # Threshold checks will be done by the stage based on these
        }
        mock_engine_report_str = json.dumps({"report_content": "mock_eval_report", "metrics": mock_engine_metrics})

        # Patch the stage's internal methods that would call the EvaluationEngine
        with patch.object(self.stage, '_make_predictions', return_value=mock_engine_predictions) as mock_make_preds, \
             patch.object(self.stage, '_compute_metrics', return_value=mock_engine_metrics) as mock_compute_metrics, \
             patch.object(self.stage, '_generate_report', return_value=mock_engine_report_str) as mock_gen_report:

            # --- Mocking ArtifactStore ---
            mock_results_artifact_meta = ArtifactMetadata(artifact_id="eval_results_id_1", artifact_type=ArtifactType.METRICS, version="1", description="", created_at=datetime.now(), tags=[])
            mock_report_artifact_meta = ArtifactMetadata(artifact_id="eval_report_id_1", artifact_type=ArtifactType.REPORT, version="1", description="", created_at=datetime.now(), tags=[])
            self.mock_artifact_store.save_artifact.side_effect = [mock_results_artifact_meta, mock_report_artifact_meta]

            # --- Execute Stage ---
            self.stage.setup(self.context)
            result_context = self.stage.run(self.context)

            # --- Assertions ---
            mock_make_preds.assert_called_once_with(self.trained_model_mock, self.test_features_df)
            # In _compute_metrics, labels and predictions are converted to np.array if not already
            # So, we need to assert that the call was made with arguments that would result in these np.arrays
            mock_compute_metrics.assert_called_once()
            call_args_metrics, _ = mock_compute_metrics.call_args
            pd.testing.assert_series_equal(pd.Series(call_args_metrics[0]), self.test_labels_series, check_dtype=False) # true_labels
            np.testing.assert_array_equal(call_args_metrics[1], mock_engine_predictions) # predictions

            mock_gen_report.assert_called_once()
            # Check context updates from EvaluationStage
            self.assertEqual(result_context.get("evaluation_results")["accuracy"], 0.75)
            self.assertEqual(result_context.get("evaluation_report"), mock_engine_report_str)
            self.assertTrue(result_context.get("model_passed_thresholds")) # 0.75 >= 0.75, 0.70 >= 0.7

            eval_metadata = result_context.get("evaluation_metadata")
            self.assertIsNotNone(eval_metadata)
            self.assertEqual(eval_metadata["model_type"], "test_classifier_from_train_stage")
            self.assertTrue(eval_metadata["passed_thresholds"])

            # Verify ArtifactStore interaction
            self.assertEqual(self.mock_artifact_store.save_artifact.call_count, 2)
            
            # Call 1: Results
            args_results, kwargs_results = self.mock_artifact_store.save_artifact.call_args_list[0]
            self.assertTrue(kwargs_results['artifact_id'].startswith("evaluation_results_test_eval_run_789"))
            self.assertEqual(kwargs_results['artifact_type'], ArtifactType.METRICS)
            
            # Call 2: Report
            args_report, kwargs_report = self.mock_artifact_store.save_artifact.call_args_list[1]
            self.assertTrue(kwargs_report['artifact_id'].startswith("evaluation_report_test_eval_run_789"))
            self.assertEqual(kwargs_report['artifact_type'], ArtifactType.REPORT)

            self.assertEqual(result_context.get("evaluation_results_artifact"), "eval_results_id_1")
            self.assertEqual(result_context.get("evaluation_report_artifact"), "eval_report_id_1")

    def test_evaluation_run_metric_computation_fails(self):
        # Patch _make_predictions to return something
        with patch.object(self.stage, '_make_predictions', return_value=np.array([1,0,1,0])) as mock_make_preds:
            # Patch _compute_metrics to simulate EvaluationEngine failure
            with patch.object(self.stage, '_compute_metrics', side_effect=ValueError("Metrics calculation error!")) as mock_compute_metrics:
                
                self.stage.setup(self.context)
                with self.assertRaisesRegex(ValueError, "Metrics calculation error!"):
                    self.stage.run(self.context)

                mock_make_preds.assert_called_once()
                mock_compute_metrics.assert_called_once()
                self.mock_artifact_store.save_artifact.assert_not_called()
                self.mock_logger_instance.error.assert_any_call("Error in evaluation stage: Metrics calculation error!")

    def test_evaluation_fails_threshold_check(self):
        mock_engine_predictions = np.array([1, 0, 0, 1])
        mock_engine_metrics = {"accuracy": 0.60, "f1": 0.55} # Below thresholds
        mock_engine_report_str = json.dumps({"report_content": "failed_threshold_report"})

        with patch.object(self.stage, '_make_predictions', return_value=mock_engine_predictions), \
             patch.object(self.stage, '_compute_metrics', return_value=mock_engine_metrics), \
             patch.object(self.stage, '_generate_report', return_value=mock_engine_report_str):
            
            self.mock_artifact_store.save_artifact.side_effect = [MagicMock(artifact_id="res_fail"), MagicMock(artifact_id="rep_fail")]

            self.stage.setup(self.context)
            result_context = self.stage.run(self.context)

            self.assertFalse(result_context.get("model_passed_thresholds"))
            eval_metadata = result_context.get("evaluation_metadata")
            self.assertFalse(eval_metadata["passed_thresholds"])
            self.mock_logger_instance.warning.assert_any_call(unittest.mock.ANY) # Check warning for threshold failure


if __name__ == '__main__':
    unittest.main()