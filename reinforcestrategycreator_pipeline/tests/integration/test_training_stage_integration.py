import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stages.training import TrainingStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactMetadata, ArtifactType
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry # Added import

class TestTrainingStageIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        self.mock_artifact_store = MagicMock()
        self.context.set("artifact_store", self.mock_artifact_store)

        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.context.set("config_manager", self.mock_config_manager)

        self.mock_model_registry = MagicMock(spec=ModelRegistry) # Added
        self.context.set("model_registry", self.mock_model_registry) # Added
        
        self.context.set_metadata("run_id", "test_train_run_456")

        self.mock_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger.return_value = self.mock_logger_instance

        self.stage_config = {
            "model_type": "test_nn",
            "model_config": {"layers": [64, 32], "activation": "relu"},
            "training_config": {"epochs": 3, "batch_size": 16, "learning_rate": 0.01},
            "validation_split": 0.25,
            "early_stopping": {"enabled": False}
        }
        self.stage = TrainingStage(config=self.stage_config)

        # Prepare some dummy data for the context
        self.features_df = pd.DataFrame({'f1': [1,2,3,4,5,6,7,8], 'f2': [8,7,6,5,4,3,2,1]})
        self.labels_series = pd.Series([0,1,0,1,0,1,0,1])
        self.context.set("processed_features", self.features_df)
        self.context.set("labels", self.labels_series)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.mock_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_training_run_successful_interaction_with_mocked_engine_and_store(self):
        # --- Mocking the behavior of a TrainingEngine ---
        mock_engine_model = {"engine_model_type": "test_nn", "trained_params": "mock_weights"}
        mock_engine_history = {
            "train_loss": [0.5, 0.3, 0.1],
            "val_loss": [0.6, 0.4, 0.2],
            "train_accuracy": [0.7, 0.8, 0.9],
            "val_accuracy": [0.65, 0.75, 0.85]
        }

        # Patch the stage's internal methods that would call the TrainingEngine
        with patch.object(self.stage, '_initialize_model', return_value=None) as mock_init_model_call:
            # Simulate _initialize_model setting self.model as if an engine did
            def set_model_on_stage(*args, **kwargs):
                self.stage.model = mock_engine_model 
            mock_init_model_call.side_effect = set_model_on_stage
            
            with patch.object(self.stage, '_train_model', return_value=mock_engine_history) as mock_train_model_call:
                
                # --- Mocking ArtifactStore ---
                mock_saved_artifact_meta = ArtifactMetadata(
                    artifact_id=f"model_{self.stage.model_type}_test_train_run_456_timestamp",
                    artifact_type=ArtifactType.MODEL,
                    # path="dummy/path/model.pkl", # Removed path
                    version="1",
                    description=f"Trained {self.stage.model_type} model",
                    created_at=datetime.now(), # Corrected to datetime object
                    # metadata={}, # Removed metadata
                    tags=["trained", self.stage.model_type, self.stage.name]
                )
                self.mock_artifact_store.save_artifact.return_value = mock_saved_artifact_meta

                # --- Execute Stage ---
                self.stage.setup(self.context) # Calls _initialize_model
                result_context = self.stage.run(self.context) # Calls _train_model & _save_model_artifact

                # --- Assertions ---
                # Verify _initialize_model was called by setup
                mock_init_model_call.assert_called_once()
                self.assertEqual(self.stage.model, mock_engine_model) # Check if self.model was set by the patched method

                # Verify _train_model was called by run with expected data (simplified check)
                mock_train_model_call.assert_called_once()
                args_train, _ = mock_train_model_call.call_args
                # args_train[0] is train_data, args_train[1] is val_data
                # We can't easily check the exact split data without reimplementing _split_data logic here
                # So, we'll trust _split_data (unit tested elsewhere or assumed simple) and focus on the call
                self.assertIsNotNone(args_train[0]) # train_data
                self.assertIsNotNone(args_train[1]) # val_data
                
                # Check context updates from TrainingStage
                self.assertEqual(result_context.get("trained_model"), mock_engine_model)
                self.assertEqual(result_context.get("training_history"), mock_engine_history)
                self.assertEqual(result_context.get("model_type"), self.stage_config["model_type"])
                self.assertEqual(result_context.get("model_config"), self.stage_config["model_config"])
                
                training_metadata = result_context.get("training_metadata")
                self.assertIsNotNone(training_metadata)
                self.assertEqual(training_metadata["model_type"], self.stage_config["model_type"])
                self.assertEqual(training_metadata["hyperparameters"], self.stage_config["training_config"])
                self.assertEqual(training_metadata["final_metrics"]["final_val_accuracy"], 0.85)

                # Verify ArtifactStore interaction
                self.mock_artifact_store.save_artifact.assert_called_once()
                save_call_args = self.mock_artifact_store.save_artifact.call_args[1]
                self.assertTrue(save_call_args['artifact_id'].startswith(f"model_{self.stage.model_type}_test_train_run_456"))
                self.assertEqual(save_call_args['artifact_type'], ArtifactType.MODEL)
                
                # Check model_artifact in context
                self.assertEqual(result_context.get("model_artifact"), mock_saved_artifact_meta.artifact_id)
                self.mock_logger_instance.info.assert_any_call(f"Saved model artifact: {mock_saved_artifact_meta.artifact_id}")

    def test_training_run_engine_train_fails(self):
        # Patch _initialize_model to set a dummy model
        with patch.object(self.stage, '_initialize_model', return_value=None) as mock_init_model_call:
            self.stage.model = {"type": "dummy_engine_model"} # Simulate model init
            
            # Patch _train_model to simulate TrainingEngine failure
            with patch.object(self.stage, '_train_model', side_effect=RuntimeError("Engine training failed!")) as mock_train_model_call:
                
                self.stage.setup(self.context)
                with self.assertRaisesRegex(RuntimeError, "Engine training failed!"):
                    self.stage.run(self.context)

                mock_train_model_call.assert_called_once()
                self.mock_artifact_store.save_artifact.assert_not_called() # Should not attempt to save
                self.mock_logger_instance.error.assert_any_call("Error in training stage: Engine training failed!")

    def test_training_run_artifact_store_fails(self):
        mock_engine_model = {"engine_model_type": "test_nn", "trained_params": "mock_weights"}
        mock_engine_history = {"train_loss": [0.1], "val_loss": [0.2]}

        with patch.object(self.stage, '_initialize_model', return_value=None) as mock_init_model_call:
            def set_model_on_stage(*args, **kwargs): self.stage.model = mock_engine_model
            mock_init_model_call.side_effect = set_model_on_stage
            
            with patch.object(self.stage, '_train_model', return_value=mock_engine_history) as mock_train_model_call:
                # Simulate ArtifactStore failure
                self.mock_artifact_store.save_artifact.side_effect = IOError("Failed to connect to store")

                self.stage.setup(self.context)
                result_context = self.stage.run(self.context) # Should not raise from run, but log warning

                # Model training and context updates should still happen
                self.assertEqual(result_context.get("trained_model"), mock_engine_model)
                self.assertEqual(result_context.get("training_history"), mock_engine_history)
                
                # Artifact saving was attempted
                self.mock_artifact_store.save_artifact.assert_called_once()
                # Context should not contain the artifact ID
                self.assertIsNone(result_context.get("model_artifact"))
                self.mock_logger_instance.warning.assert_any_call("Failed to save model artifact: Failed to connect to store")


if __name__ == '__main__':
    unittest.main()