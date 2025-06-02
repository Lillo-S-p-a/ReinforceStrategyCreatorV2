"""Unit tests for TrainingStage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from reinforcestrategycreator_pipeline.src.pipeline.stages.training import TrainingStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext


class TestTrainingStage(unittest.TestCase):
    """Test cases for TrainingStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model_type": "random_forest",
            "model_config": {"n_estimators": 100},
            "training_config": {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "validation_split": 0.2,
            "early_stopping": {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.001,
                "monitor": "val_loss"
            }
        }
        self.stage = TrainingStage(config=self.config)
        
        # Reset PipelineContext singleton
        PipelineContext._instance = None
        self.context = PipelineContext.get_instance()
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton
        PipelineContext._instance = None
        
    def test_initialization(self):
        """Test stage initialization."""
        self.assertEqual(self.stage.name, "training")
        self.assertEqual(self.stage.model_type, "random_forest")
        self.assertEqual(self.stage.model_config, {"n_estimators": 100})
        self.assertEqual(self.stage.validation_split, 0.2)
        self.assertTrue(self.stage.early_stopping["enabled"])
        
    def test_setup_without_features(self):
        """Test setup fails without processed features."""
        with self.assertRaises(ValueError) as cm:
            self.stage.setup(self.context)
        self.assertIn("No processed features found", str(cm.exception))
        
    def test_setup_with_features(self):
        """Test successful setup with features."""
        self.context.set("processed_features", [[1, 2, 3], [4, 5, 6]])
        self.context.set("artifact_store", Mock())
        
        # Should not raise
        self.stage.setup(self.context)
        self.assertIsNotNone(self.stage.model)
        
    def test_initialize_model_types(self):
        """Test model initialization for different types."""
        # Test random forest
        self.stage._initialize_model()
        self.assertEqual(self.stage.model["type"], "random_forest")
        self.assertEqual(self.stage.model["params"], self.config["model_config"])
        
        # Test neural network
        self.stage.model_type = "neural_network"
        self.stage._initialize_model()
        self.assertEqual(self.stage.model["type"], "neural_network")
        
        # Test gradient boosting
        self.stage.model_type = "gradient_boosting"
        self.stage._initialize_model()
        self.assertEqual(self.stage.model["type"], "gradient_boosting")
        
        # Test default
        self.stage.model_type = "unknown"
        self.stage._initialize_model()
        self.assertEqual(self.stage.model["type"], "default")
        
    def test_split_data(self):
        """Test data splitting."""
        features = list(range(100))
        labels = list(range(100))
        
        train_data, val_data = self.stage._split_data(features, labels)
        
        # Check structure
        self.assertEqual(len(train_data), 2)  # (features, labels)
        self.assertEqual(len(val_data), 2)    # (features, labels)
        
    def test_train_model(self):
        """Test model training."""
        train_data = ([1, 2, 3], [0, 1, 0])
        val_data = ([4, 5, 6], [1, 0, 1])
        
        history = self.stage._train_model(train_data, val_data)
        
        # Check history structure
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("train_accuracy", history)
        self.assertIn("val_accuracy", history)
        
        # Check history length
        expected_epochs = self.config["training_config"]["epochs"]
        self.assertEqual(len(history["train_loss"]), expected_epochs)
        
    def test_early_stopping(self):
        """Test early stopping logic."""
        # Test with early stopping disabled
        self.stage.early_stopping["enabled"] = False
        history = {"val_loss": [1.0, 0.9, 0.8, 0.7]}
        self.assertFalse(self.stage._should_stop_early(history))
        
        # Test with early stopping enabled but not enough epochs
        self.stage.early_stopping["enabled"] = True
        self.stage.early_stopping["patience"] = 5
        self.assertFalse(self.stage._should_stop_early(history))
        
        # Test with early stopping triggered
        self.stage.early_stopping["patience"] = 3
        history = {"val_loss": [1.0, 0.5, 0.4, 0.4, 0.4]}  # No improvement for 3 epochs
        self.assertTrue(self.stage._should_stop_early(history))
        
        # Test with improvement within patience
        history = {"val_loss": [1.0, 0.5, 0.4, 0.3, 0.2]}  # Continuous improvement
        self.assertFalse(self.stage._should_stop_early(history))
        
    def test_get_final_metrics(self):
        """Test getting final metrics from training history."""
        self.stage.training_history = {
            "train_loss": [1.0, 0.5, 0.3],
            "val_loss": [1.1, 0.6, 0.4],
            "train_accuracy": [0.5, 0.7, 0.9],
            "val_accuracy": [0.4, 0.6, 0.8]
        }
        
        final_metrics = self.stage._get_final_metrics()
        
        self.assertEqual(final_metrics["final_train_loss"], 0.3)
        self.assertEqual(final_metrics["final_val_loss"], 0.4)
        self.assertEqual(final_metrics["final_train_accuracy"], 0.9)
        self.assertEqual(final_metrics["final_val_accuracy"], 0.8)
        
    def test_run_without_labels(self):
        """Test run fails without labels."""
        self.context.set("processed_features", [[1, 2, 3]])
        self.context.set("artifact_store", Mock())
        self.stage.setup(self.context)
        
        with self.assertRaises(ValueError) as cm:
            self.stage.run(self.context)
        self.assertIn("No labels found", str(cm.exception))
        
    def test_run_success(self):
        """Test successful training run."""
        # Set up context
        self.context.set("processed_features", [[1, 2, 3], [4, 5, 6]])
        self.context.set("labels", [0, 1])
        self.context.set("artifact_store", Mock())
        
        self.stage.setup(self.context)
        result_context = self.stage.run(self.context)
        
        # Check context updates
        self.assertIsNotNone(result_context.get("trained_model"))
        self.assertIsNotNone(result_context.get("training_history"))
        self.assertEqual(result_context.get("model_type"), "random_forest")
        self.assertIsNotNone(result_context.get("training_metadata"))
        
        # Check metadata structure
        metadata = result_context.get("training_metadata")
        self.assertIn("model_type", metadata)
        self.assertIn("training_duration_seconds", metadata)
        self.assertIn("final_metrics", metadata)
        self.assertIn("hyperparameters", metadata)
        
    @patch('pickle.dump')
    @patch('tempfile.NamedTemporaryFile')
    def test_save_model_artifact(self, mock_tempfile, mock_pickle):
        """Test saving model artifact."""
        # Set up mocks
        mock_file = MagicMock()
        mock_tempfile.return_value.__enter__.return_value = mock_file
        mock_file.name = "/tmp/test_model.pkl"
        
        mock_artifact_store = Mock()
        mock_artifact_store.save_artifact.return_value = Mock(artifact_id="model_123")
        self.stage.artifact_store = mock_artifact_store
        
        # Set up context
        self.context.set_metadata("run_id", "test_run")
        self.stage.model = {"type": "test"}
        self.stage.training_history = {"loss": [1.0]}
        
        # Save artifact
        self.stage._save_model_artifact(self.context)
        
        # Verify artifact store was called
        mock_artifact_store.save_artifact.assert_called_once()
        call_args = mock_artifact_store.save_artifact.call_args[1]
        self.assertIn("model_random_forest_test_run", call_args["artifact_id"])
        self.assertEqual(call_args["artifact_path"], "/tmp/test_model.pkl")
        
        # Verify context was updated
        self.assertEqual(self.context.get("model_artifact"), "model_123")
        
    def test_teardown(self):
        """Test teardown method."""
        # Create a temporary directory
        self.stage.temp_checkpoint_dir = Path(tempfile.mkdtemp())
        self.assertTrue(self.stage.temp_checkpoint_dir.exists())
        
        # Teardown should remove it
        self.stage.teardown(self.context)
        self.assertFalse(self.stage.temp_checkpoint_dir.exists())


if __name__ == "__main__":
    unittest.main()