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
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry
from reinforcestrategycreator_pipeline.src.evaluation.engine import EvaluationEngine # Added import

class TestEvaluationStageIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        self.mock_artifact_store = MagicMock()
        self.context.set("artifact_store", self.mock_artifact_store)

        self.mock_config_manager = MagicMock(spec=ConfigManager)
        # Mock the loader and base_path attributes on the mock_config_manager
        self.mock_config_manager.loader = MagicMock()
        self.mock_config_manager.loader.base_path = self.test_dir
        self.context.set("config_manager", self.mock_config_manager)

        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.context.set("model_registry", self.mock_model_registry)

        self.context.set_metadata("run_id", "test_eval_run_789")

        # Create a dummy CSV file for the mocked CsvDataSource
        self.dummy_csv_path = self.test_dir / "dummy_data.csv"
        dummy_df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'open': [10, 11, 12, 13],
            'high': [12, 13, 14, 15],
            'low': [9, 10, 11, 12],
            'close': [11, 12, 13, 14],
            'volume': [100, 110, 120, 130]
        })
        dummy_df.to_csv(self.dummy_csv_path, index=False)

        # Mock the return values for data source configuration
        self.mock_config_manager.get_config.return_value.data.model_dump.return_value = {
            "source_id": "mock_data_source",
            "source_type": "csv",
            "file_path": str(self.dummy_csv_path), # Provide the path to the dummy CSV
            "cache_enabled": False,
            "auto_adjust": False,
            "interval": "1d"
        }

        self.mock_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger.return_value = self.mock_logger_instance

        # Mock the EvaluationEngine constructor
        self.mock_evaluation_engine = MagicMock(spec=EvaluationEngine)
        self.patcher_evaluation_engine = patch('reinforcestrategycreator_pipeline.src.pipeline.stages.evaluation.EvaluationEngine', new=self.mock_evaluation_engine)
        self.patcher_evaluation_engine.start()
        self.mock_evaluation_engine.return_value = self.mock_evaluation_engine # Make the mock callable and return itself

        self.stage_config = {
            "metrics": ["accuracy", "precision", "f1"],
            "threshold_config": {"accuracy": 0.75, "f1": 0.7},
            "report_format": "json",
            "baseline_model": None # Can be enabled in specific tests
        }
        # Prepare dummy data for context
        self.trained_model_mock = {"type": "test_classifier", "model_data": "some_weights"}
        self.test_features_df = pd.DataFrame({'x1': [1,0,1,0], 'x2': [0,1,0,1]})
        self.test_labels_series = pd.Series([1,0,1,1])
        
        # Create a dummy CSV file for the mocked CsvDataSource
        self.dummy_csv_path = self.test_dir / "dummy_data.csv"
        dummy_df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'open': [10, 11, 12, 13],
            'high': [12, 13, 14, 15],
            'low': [9, 10, 11, 12],
            'close': [11, 12, 13, 14],
            'volume': [100, 110, 120, 130]
        })
        dummy_df.to_csv(self.dummy_csv_path, index=False)

        # Update the mock_data_config to include the file_path
        mock_data_config = MagicMock()
        mock_data_config.get.side_effect = lambda key, default=None: {
            "source_id": "mock_data_source",
            "source_type": "csv",
            "file_path": str(self.dummy_csv_path), # Provide the path to the dummy CSV
            "cache_enabled": False,
            "auto_adjust": False,
            "interval": "1d"
        }.get(key, default)
        self.mock_config_manager.get_config.return_value.data.model_dump.return_value = mock_data_config

        self.stage = EvaluationStage(config=self.stage_config) # This line needs to be after full config mock

        self.context.set("trained_model", self.trained_model_mock)
        self.context.set("test_features", self.test_features_df)
        self.context.set("test_labels", self.test_labels_series)
        self.context.set("model_type", "test_classifier_from_train_stage")
        self.context.set("model_config", {"param": "value"})


    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.mock_logger_patcher.stop()
        self.patcher_evaluation_engine.stop() # Stop the patcher
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_evaluation_run_successful_interaction_with_mocked_engine_and_store(self):
        # --- Mocking the behavior of EvaluationEngine.evaluate() ---
        mock_evaluation_results = {
            "metrics": {
                "accuracy": 0.75,
                "precision": 0.66,
                "f1": 0.70,
            },
            "reports": {
                "json": json.dumps({"report_content": "mock_eval_report", "metrics": {"accuracy": 0.75}}),
                "html": "<html>mock_html_report</html>"
            },
            "timestamp": datetime.now().isoformat(),
            "model": {"version": "v1.0.0"} # Example model info
        }
        self.mock_evaluation_engine.evaluate.return_value = mock_evaluation_results

        # --- Mocking ArtifactStore (EvaluationEngine handles saving now) ---
        # The EvaluationEngine is responsible for calling artifact_store.save_artifact.
        # We need to mock the artifact_store that is passed to the EvaluationEngine.
        # Since EvaluationStage creates EvaluationEngine, we need to ensure our mock
        # is passed through.
        # The EvaluationEngine constructor is mocked in setUp, so we can configure its behavior.
        
        # Simulate artifact_store.save_artifact calls that EvaluationEngine would make
        mock_results_artifact_meta = ArtifactMetadata(artifact_id="eval_results_id_1", artifact_type=ArtifactType.METRICS, version="1", description="", created_at=datetime.now(), tags=[])
        mock_report_artifact_meta = ArtifactMetadata(artifact_id="eval_report_id_1", artifact_type=ArtifactType.REPORT, version="1", description="", created_at=datetime.now(), tags=[])
        self.mock_artifact_store.save_artifact.side_effect = [mock_results_artifact_meta, mock_report_artifact_meta]

        # --- Execute Stage ---
        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)

        # --- Assertions ---
        # Verify EvaluationEngine.evaluate was called with correct arguments
        self.mock_evaluation_engine.evaluate.assert_called_once_with(
            model_id=self.context.get("trained_model_artifact_id"),
            model_version=self.context.get("trained_model_version"),
            data_source_id=self.stage.global_data_config.get("source_id", "dummy_csv_data"), # Use dummy_csv_data as default
            metrics=self.stage.global_evaluation_config.get("metrics"),
            compare_benchmarks=self.stage.global_evaluation_config.get("compare_benchmarks", True),
            save_results=self.stage.global_evaluation_config.get("save_results", True),
            report_formats=self.stage.global_evaluation_config.get("report_formats", ["html", "markdown"]),
            generate_visualizations=self.stage.global_evaluation_config.get("generate_visualizations", True),
            evaluation_name=f"eval_{self.context.get('trained_model_artifact_id')}_{self.context.get('trained_model_version') or 'latest'}"
        )
        
        # Check context updates from EvaluationStage
        self.assertEqual(result_context.get("evaluation_results")["accuracy"], 0.75)
        self.assertEqual(result_context.get("evaluation_reports"), mock_evaluation_results["reports"])
        self.assertTrue(result_context.get("model_passed_thresholds")) # This logic is now within EvaluationStage based on returned metrics
        
        eval_metadata = result_context.get("evaluation_metadata")
        self.assertIsNotNone(eval_metadata)
        self.assertEqual(eval_metadata["model_id_evaluated"], self.context.get("trained_model_artifact_id"))
        self.assertEqual(eval_metadata["model_version_evaluated"], self.context.get("trained_model_version"))
        self.assertTrue(eval_metadata["passed_thresholds"]) # Assuming EvaluationStage sets this based on metrics

        # Verify ArtifactStore interaction (now called by EvaluationEngine)
        # The EvaluationEngine.evaluate method is mocked, so we need to assert on the artifact_store mock directly
        self.assertEqual(self.mock_artifact_store.save_artifact.call_count, 2)
        
        # Call 1: Results (assuming EvaluationEngine saves metrics)
        args_results, kwargs_results = self.mock_artifact_store.save_artifact.call_args_list[0]
        self.assertTrue(kwargs_results['artifact_id'].startswith("eval_results_id_1")) # Check prefix or specific ID
        self.assertEqual(kwargs_results['artifact_type'], ArtifactType.METRICS)
        
        # Call 2: Report (assuming EvaluationEngine saves reports)
        args_report, kwargs_report = self.mock_artifact_store.save_artifact.call_args_list[1]
        self.assertTrue(kwargs_report['artifact_id'].startswith("eval_report_id_1")) # Check prefix or specific ID
        self.assertEqual(kwargs_report['artifact_type'], ArtifactType.REPORT)

        # These context keys are set by EvaluationEngine, not EvaluationStage directly
        # self.assertEqual(result_context.get("evaluation_results_artifact"), "eval_results_id_1")
        # self.assertEqual(result_context.get("evaluation_report_artifact"), "eval_report_id_1")

    def test_evaluation_run_metric_computation_fails(self):
        # Mock EvaluationEngine.evaluate to raise an error
        self.mock_evaluation_engine.evaluate.side_effect = ValueError("Metrics calculation error!")
        
        self.stage.setup(self.context)
        with self.assertRaisesRegex(ValueError, "Metrics calculation error!"):
            self.stage.run(self.context)

        self.mock_evaluation_engine.evaluate.assert_called_once()
        self.mock_artifact_store.save_artifact.assert_not_called() # EvaluationEngine would not save if it fails early
        self.mock_logger_instance.error.assert_any_call(unittest.mock.ANY, exc_info=True) # Check for error log with exc_info
        self.mock_logger_instance.error.assert_any_call("Error in RL EvaluationStage run: Metrics calculation error!", exc_info=True)

    def test_evaluation_fails_threshold_check(self):
        mock_evaluation_results_failed = {
            "metrics": {"accuracy": 0.60, "f1": 0.55}, # Below thresholds
            "reports": {"json": json.dumps({"report_content": "failed_threshold_report"})},
            "timestamp": datetime.now().isoformat(),
            "model": {"version": "v1.0.0"}
        }
        self.mock_evaluation_engine.evaluate.return_value = mock_evaluation_results_failed
        
        # Simulate artifact_store.save_artifact calls that EvaluationEngine would make
        mock_results_artifact_meta = MagicMock(artifact_id="res_fail", artifact_type=ArtifactType.METRICS)
        mock_report_artifact_meta = MagicMock(artifact_id="rep_fail", artifact_type=ArtifactType.REPORT)
        self.mock_artifact_store.save_artifact.side_effect = [mock_results_artifact_meta, mock_report_artifact_meta]

        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)

        # EvaluationStage itself checks thresholds based on results from EvaluationEngine
        self.assertFalse(result_context.get("model_passed_thresholds"))
        eval_metadata = result_context.get("evaluation_metadata")
        self.assertFalse(eval_metadata["passed_thresholds"])
        self.mock_logger_instance.warning.assert_any_call(unittest.mock.ANY) # Check warning for threshold failure
        self.mock_logger_instance.info.assert_any_call(f"Evaluation stage completed. Metrics: {mock_evaluation_results_failed['metrics']}")


if __name__ == '__main__':
    unittest.main()