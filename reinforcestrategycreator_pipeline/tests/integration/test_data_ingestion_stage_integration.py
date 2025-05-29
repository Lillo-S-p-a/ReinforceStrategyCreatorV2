import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from datetime import datetime # Added import

from reinforcestrategycreator_pipeline.src.pipeline.stages.data_ingestion import DataIngestionStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactMetadata, ArtifactType

class TestDataIngestionStageIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp()) # Create a temporary directory for test files
        
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        self.mock_artifact_store = MagicMock()
        self.context.set("artifact_store", self.mock_artifact_store)
        self.context.set_metadata("run_id", "test_integration_run_123")

        # Mock logger for the stage
        self.mock_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger.return_value = self.mock_logger_instance

    def tearDown(self):
        shutil.rmtree(self.test_dir) # Clean up the temporary directory
        self.mock_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def _create_dummy_csv(self, filename="data.csv", rows=3):
        file_path = self.test_dir / filename
        df = pd.DataFrame({
            'col_a': range(rows),
            'col_b': [f"text_{i}" for i in range(rows)],
            'col_c': [float(i) * 1.1 for i in range(rows)]
        })
        df.to_csv(file_path, index=False)
        return file_path

    def _create_dummy_parquet(self, filename="data.parquet", rows=3):
        file_path = self.test_dir / filename
        df = pd.DataFrame({
            'feature1': range(rows),
            'feature2': [f"item_{i}" for i in range(rows)]
        })
        df.to_parquet(file_path, index=False)
        return file_path

    @patch('pandas.DataFrame.to_parquet')
    def test_ingest_csv_successful_with_artifact_saving(self, mock_to_parquet): # Added mock_to_parquet
        csv_path = self._create_dummy_csv(rows=5)
        stage_config = {
            "source_path": str(csv_path),
            "source_type": "csv",
            "validation_rules": {"required_columns": ["col_a", "col_b"]}
        }
        stage = DataIngestionStage(config=stage_config)

        mock_artifact_meta = ArtifactMetadata(
            artifact_id="raw_data_test_integration_run_123_csv",
            artifact_type=ArtifactType.DATASET,
            version="1",
            description="Raw ingested data",
            created_at=datetime.now(), # Added proper datetime
            # properties={"source": str(csv_path)}, # 'metadata' arg to save_artifact would populate this
            tags=["raw", "ingested", stage.name]
            # source_info could also be used for {"source": str(csv_path)}
        )
        self.mock_artifact_store.save_artifact.return_value = mock_artifact_meta

        stage.setup(self.context)
        result_context = stage.run(self.context)

        # Check context for raw_data
        raw_data = result_context.get("raw_data")
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertEqual(len(raw_data), 5)
        self.assertListEqual(list(raw_data.columns), ['col_a', 'col_b', 'col_c'])

        # Check context for metadata
        data_metadata = result_context.get("data_metadata")
        self.assertEqual(data_metadata["row_count"], 5)
        self.assertEqual(data_metadata["source_path"], str(csv_path))
        self.assertTrue(data_metadata["validation_passed"]) # Basic validation

        # Check artifact store interaction
        self.mock_artifact_store.save_artifact.assert_called_once()
        mock_to_parquet.assert_called_once() # Verify to_parquet was called
        call_args = self.mock_artifact_store.save_artifact.call_args
        
        # The artifact_id is now constructed using the run_id from context
        expected_artifact_id = f"raw_data_{self.context.get_metadata('run_id')}"
        # The call_args for kwargs is call_args.kwargs, not call_args[1]
        self.assertEqual(call_args.kwargs['artifact_id'], expected_artifact_id)
        self.assertEqual(call_args.kwargs['artifact_type'], ArtifactType.DATASET)
        self.assertEqual(result_context.get("raw_data_artifact"), mock_artifact_meta.artifact_id) # Use the ID from the returned meta
        
        self.mock_logger_instance.info.assert_any_call(f"Successfully ingested data: 5 rows, 3 columns")
        self.mock_logger_instance.info.assert_any_call(f"Saved raw data artifact: {mock_artifact_meta.artifact_id}")


    def test_ingest_parquet_successful(self):
        try:
            parquet_path = self._create_dummy_parquet(rows=10) # Moved inside try
            stage_config = {
                "source_path": str(parquet_path),
                "source_type": "parquet"
            }
            stage = DataIngestionStage(config=stage_config)
            
            # For this test, don't focus on artifact saving details, assume it works if called
            self.mock_artifact_store.save_artifact.return_value = MagicMock(artifact_id="dummy_parquet_id")

            stage.setup(self.context)
            result_context = stage.run(self.context)

            raw_data = result_context.get("raw_data")
            self.assertIsInstance(raw_data, pd.DataFrame)
            self.assertEqual(len(raw_data), 10)
            self.assertListEqual(list(raw_data.columns), ['feature1', 'feature2'])
            self.mock_artifact_store.save_artifact.assert_called_once() # Ensure it was called
        except ImportError as e:
            self.skipTest(f"Skipping Parquet test due to missing engine: {e}")
            return

    def test_ingestion_file_not_found(self):
        stage_config = {
            "source_path": str(self.test_dir / "non_existent_file.csv"),
            "source_type": "csv"
        }
        stage = DataIngestionStage(config=stage_config)

        with self.assertRaises(FileNotFoundError):
            stage.setup(self.context) # Error should be raised in setup

    def test_ingestion_with_sampling(self):
        csv_path = self._create_dummy_csv(rows=100)
        stage_config = {
            "source_path": str(csv_path),
            "source_type": "csv",
            "sample_size": 10
        }
        stage = DataIngestionStage(config=stage_config)
        self.mock_artifact_store.save_artifact.return_value = MagicMock(artifact_id="dummy_sampled_id")

        stage.setup(self.context)
        result_context = stage.run(self.context)

        raw_data = result_context.get("raw_data")
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertEqual(len(raw_data), 10) # Check if sampling was applied
        self.mock_logger_instance.info.assert_any_call("Sampling 10 rows from 100 total rows")

    def test_ingestion_validation_fails_missing_columns(self):
        csv_path = self._create_dummy_csv(filename="validation_test.csv") # col_a, col_b, col_c
        stage_config = {
            "source_path": str(csv_path),
            "source_type": "csv",
            "validation_rules": {"required_columns": ["col_a", "col_d_missing"]}
        }
        stage = DataIngestionStage(config=stage_config)
        self.mock_artifact_store.save_artifact.return_value = MagicMock(artifact_id="dummy_validation_fail_id")

        stage.setup(self.context)
        result_context = stage.run(self.context)
        
        data_metadata = result_context.get("data_metadata")
        self.assertFalse(data_metadata["validation_passed"])
        
        validation_results = result_context.get("data_validation_results")
        self.assertIn("Missing required columns: {'col_d_missing'}", validation_results["errors"][0])
        self.mock_logger_instance.warning.assert_any_call(unittest.mock.ANY) # Check that a warning was logged

    def test_ingestion_no_artifact_store_in_context(self):
        csv_path = self._create_dummy_csv(filename="no_store.csv")
        stage_config = {
            "source_path": str(csv_path),
            "source_type": "csv"
        }
        # Remove artifact_store from context for this test
        self.context.delete("artifact_store") 
        
        stage = DataIngestionStage(config=stage_config)

        stage.setup(self.context) # Should still work, artifact_store is optional for setup
        result_context = stage.run(self.context)

        # Check that data is still ingested
        raw_data = result_context.get("raw_data")
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertEqual(len(raw_data), 3)
        
        # Check that save_artifact was NOT called
        self.mock_artifact_store.save_artifact.assert_not_called()
        self.assertIsNone(result_context.get("raw_data_artifact"))


if __name__ == '__main__':
    unittest.main()