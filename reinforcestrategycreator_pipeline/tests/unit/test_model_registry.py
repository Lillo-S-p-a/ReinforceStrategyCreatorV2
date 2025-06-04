"""Unit tests for the Model Registry."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np

from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry
from reinforcestrategycreator_pipeline.src.models.base import ModelBase
from reinforcestrategycreator_pipeline.src.models.implementations import DQN
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactStore, ArtifactType, ArtifactMetadata


class MockModel(ModelBase):
    """Mock model for testing."""
    
    model_type = "MockModel"
    
    def build(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def train(self, train_data, validation_data=None, **kwargs):
        self.is_trained = True
        return {"loss": 0.1}
    
    def predict(self, data, **kwargs):
        return np.array([0.5])
    
    def evaluate(self, test_data, **kwargs):
        return {"accuracy": 0.95}
    
    def get_model_state(self):
        return {"mock_state": True, "trained": self.is_trained}
    
    def set_model_state(self, state):
        self.is_trained = state.get("trained", False)


class TestModelRegistry:
    """Test cases for ModelRegistry."""
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create a mock artifact store."""
        store = Mock(spec=ArtifactStore)
        return store
    
    @pytest.fixture
    def registry(self, mock_artifact_store):
        """Create a model registry with mock artifact store."""
        return ModelRegistry(mock_artifact_store)
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model instance."""
        model = MockModel({"hyperparameters": {"learning_rate": 0.001}})
        model.is_trained = True
        return model
    
    def test_registry_initialization(self, mock_artifact_store):
        """Test registry initialization."""
        registry = ModelRegistry(mock_artifact_store)
        assert registry.artifact_store == mock_artifact_store
        assert hasattr(registry, 'factory')
    
    def test_register_model(self, registry, mock_model, mock_artifact_store):
        """Test registering a model."""
        # Setup mock artifact store
        mock_metadata = ArtifactMetadata(
            artifact_id="model_test_model_12345678",
            artifact_type=ArtifactType.MODEL,
            version="v_20250529_151000",
            created_at=datetime.now()
        )
        mock_artifact_store.save_artifact.return_value = mock_metadata
        
        # Register model
        with patch('reinforcestrategycreator_pipeline.src.models.registry.Path') as mock_path:
            with patch('reinforcestrategycreator_pipeline.src.models.registry.shutil'):
                model_id = registry.register_model(
                    model=mock_model,
                    model_name="test_model",
                    tags=["test", "mock"],
                    description="Test model",
                    metrics={"accuracy": 0.95}
                )
        
        # Verify
        assert model_id.startswith("model_test_model_")
        mock_artifact_store.save_artifact.assert_called_once()
        
        # Check call arguments
        call_args = mock_artifact_store.save_artifact.call_args
        assert call_args.kwargs["artifact_type"] == ArtifactType.MODEL
        assert call_args.kwargs["tags"] == ["test", "mock"]
        assert call_args.kwargs["description"] == "Test model"
        
        # Check metadata
        metadata = call_args.kwargs["metadata"]
        assert metadata["model_name"] == "test_model"
        assert metadata["model_type"] == "MockModel"
        assert metadata["is_trained"] == True
        assert metadata["metrics"]["accuracy"] == 0.95
    
    def test_register_model_with_parent(self, registry, mock_model, mock_artifact_store):
        """Test registering a model with parent lineage."""
        mock_metadata = ArtifactMetadata(
            artifact_id="model_child_12345678",
            artifact_type=ArtifactType.MODEL,
            version="v_20250529_151000",
            created_at=datetime.now()
        )
        mock_artifact_store.save_artifact.return_value = mock_metadata
        
        with patch('reinforcestrategycreator_pipeline.src.models.registry.Path') as mock_path:
            with patch('reinforcestrategycreator_pipeline.src.models.registry.shutil'):
                model_id = registry.register_model(
                    model=mock_model,
                    model_name="child_model",
                    parent_model_id="model_parent_87654321"
                )
        
        # Check parent ID in metadata
        call_args = mock_artifact_store.save_artifact.call_args
        metadata = call_args.kwargs["metadata"]
        assert metadata["parent_model_id"] == "model_parent_87654321"
    
    def test_load_model(self, registry, mock_artifact_store):
        """Test loading a model from registry."""
        # Setup mock artifact store
        temp_dir = tempfile.mkdtemp()
        try:
            # Create mock model files
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            
            # Create config file
            config = {
                "model_type": "MockModel",
                "hyperparameters": {"learning_rate": 0.001}
            }
            with open(model_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            # Create model state file
            with open(model_dir / "model.pkl", "wb") as f:
                import pickle
                pickle.dump({"mock_state": True, "trained": True}, f)
            
            # Create metadata file
            metadata = {
                "created_at": datetime.now().isoformat(),
                "model_type": "MockModel",
                "is_trained": True
            }
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # Setup mocks
            mock_artifact_store.load_artifact.return_value = model_dir
            mock_metadata = ArtifactMetadata(
                artifact_id="model_test_12345678",
                artifact_type=ArtifactType.MODEL,
                version="v_20250529_151000",
                created_at=datetime.now(),
                properties={"model_type": "MockModel"}
            )
            mock_artifact_store.get_artifact_metadata.return_value = mock_metadata
            
            # Register MockModel with factory
            with patch('reinforcestrategycreator_pipeline.src.models.registry.get_factory') as mock_get_factory:
                mock_factory = Mock()
                mock_factory.create_model.return_value = MockModel(config)
                mock_get_factory.return_value = mock_factory
                
                # Load model
                model = registry.load_model("model_test_12345678")
            
            # Verify
            assert isinstance(model, MockModel)
            assert model.is_trained == True
            mock_artifact_store.load_artifact.assert_called_once_with(
                artifact_id="model_test_12345678",
                version=None,
                destination_path=None
            )
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_model_metadata(self, registry, mock_artifact_store):
        """Test getting model metadata."""
        # Setup mock
        mock_metadata = ArtifactMetadata(
            artifact_id="model_test_12345678",
            artifact_type=ArtifactType.MODEL,
            version="v_20250529_151000",
            created_at=datetime.now(),
            tags=["test"],
            description="Test model",
            properties={
                "model_name": "test_model",
                "model_type": "DQN",
                "metrics": {"accuracy": 0.95}
            }
        )
        mock_artifact_store.get_artifact_metadata.return_value = mock_metadata
        
        # Get metadata
        metadata = registry.get_model_metadata("model_test_12345678")
        
        # Verify
        assert metadata["model_id"] == "model_test_12345678"
        assert metadata["version"] == "v_20250529_151000"
        assert metadata["tags"] == ["test"]
        assert metadata["description"] == "Test model"
        assert metadata["model_name"] == "test_model"
        assert metadata["model_type"] == "DQN"
        assert metadata["metrics"]["accuracy"] == 0.95
    
    def test_list_models(self, registry, mock_artifact_store):
        """Test listing models with filters."""
        # Setup mock artifacts
        artifacts = [
            ArtifactMetadata(
                artifact_id="model_dqn_001",
                artifact_type=ArtifactType.MODEL,
                version="v1",
                created_at=datetime(2025, 5, 29, 10, 0, 0),
                tags=["rl", "dqn"],
                properties={
                    "model_name": "trading_dqn",
                    "model_type": "DQN",
                    "parent_model_id": None
                }
            ),
            ArtifactMetadata(
                artifact_id="model_ppo_002",
                artifact_type=ArtifactType.MODEL,
                version="v1",
                created_at=datetime(2025, 5, 29, 11, 0, 0),
                tags=["rl", "ppo"],
                properties={
                    "model_name": "trading_ppo",
                    "model_type": "PPO",
                    "parent_model_id": None
                }
            ),
            ArtifactMetadata(
                artifact_id="model_dqn_003",
                artifact_type=ArtifactType.MODEL,
                version="v2",
                created_at=datetime(2025, 5, 29, 12, 0, 0),
                tags=["rl", "dqn", "improved"],
                properties={
                    "model_name": "trading_dqn",
                    "model_type": "DQN",
                    "parent_model_id": "model_dqn_001"
                }
            )
        ]
        mock_artifact_store.list_artifacts.return_value = artifacts
        
        # Test listing all models
        all_models = registry.list_models()
        assert len(all_models) == 3
        
        # Test filtering by model_name
        dqn_models = registry.list_models(model_name="trading_dqn")
        assert len(dqn_models) == 2
        assert all(m["model_name"] == "trading_dqn" for m in dqn_models)
        
        # Test filtering by model_type
        ppo_models = registry.list_models(model_type="PPO")
        assert len(ppo_models) == 1
        assert ppo_models[0]["model_type"] == "PPO"
        
        # Test filtering by parent_model_id
        child_models = registry.list_models(parent_model_id="model_dqn_001")
        assert len(child_models) == 1
        assert child_models[0]["model_id"] == "model_dqn_003"
        
        # Test sorting (newest first)
        assert all_models[0]["model_id"] == "model_dqn_003"  # Latest
        assert all_models[-1]["model_id"] == "model_dqn_001"  # Oldest
    
    def test_list_model_versions(self, registry, mock_artifact_store):
        """Test listing model versions."""
        # Setup mock
        mock_artifact_store.list_versions.return_value = ["v1", "v2", "v3"]
        
        # Mock get_model_metadata for each version
        with patch.object(registry, 'get_model_metadata') as mock_get_metadata:
            mock_get_metadata.side_effect = [
                {"model_id": "model_test", "version": "v1", "created_at": "2025-05-29T10:00:00"},
                {"model_id": "model_test", "version": "v2", "created_at": "2025-05-29T11:00:00"},
                {"model_id": "model_test", "version": "v3", "created_at": "2025-05-29T12:00:00"}
            ]
            
            versions = registry.list_model_versions("model_test")
        
        assert len(versions) == 3
        assert versions[0]["version"] == "v1"
        assert versions[1]["version"] == "v2"
        assert versions[2]["version"] == "v3"
    
    def test_get_model_lineage(self, registry, mock_artifact_store):
        """Test getting model lineage."""
        # Setup current model metadata
        with patch.object(registry, 'get_model_metadata') as mock_get_metadata:
            mock_get_metadata.side_effect = [
                # Current model
                {
                    "model_id": "model_child",
                    "parent_model_id": "model_parent",
                    "version": "v1"
                },
                # Parent model
                {
                    "model_id": "model_parent",
                    "parent_model_id": "model_grandparent",
                    "version": "v1",
                    "created_at": "2025-05-29T10:00:00"
                },
                # Grandparent model
                {
                    "model_id": "model_grandparent",
                    "parent_model_id": None,
                    "version": "v1",
                    "created_at": "2025-05-29T09:00:00"
                }
            ]
            
            # Mock list_models for children
            with patch.object(registry, 'list_models') as mock_list_models:
                mock_list_models.return_value = [
                    {
                        "model_id": "model_grandchild1",
                        "version": "v1",
                        "created_at": "2025-05-29T12:00:00"
                    },
                    {
                        "model_id": "model_grandchild2",
                        "version": "v1",
                        "created_at": "2025-05-29T13:00:00"
                    }
                ]
                
                lineage = registry.get_model_lineage("model_child")
        
        # Verify lineage structure
        assert lineage["model_id"] == "model_child"
        assert lineage["parent_model_id"] == "model_parent"
        assert len(lineage["children"]) == 2
        assert len(lineage["ancestors"]) == 2
        
        # Check ancestors order (immediate parent first)
        assert lineage["ancestors"][0]["model_id"] == "model_parent"
        assert lineage["ancestors"][1]["model_id"] == "model_grandparent"
    
    def test_delete_model(self, registry, mock_artifact_store):
        """Test deleting a model."""
        mock_artifact_store.delete_artifact.return_value = True
        
        result = registry.delete_model("model_test_12345678", version="v1")
        
        assert result == True
        mock_artifact_store.delete_artifact.assert_called_once_with(
            artifact_id="model_test_12345678",
            version="v1"
        )
    
    def test_model_exists(self, registry, mock_artifact_store):
        """Test checking if model exists."""
        mock_artifact_store.artifact_exists.return_value = True
        
        exists = registry.model_exists("model_test_12345678")
        
        assert exists == True
        mock_artifact_store.artifact_exists.assert_called_once_with(
            artifact_id="model_test_12345678",
            version=None
        )
    
    def test_compare_models(self, registry):
        """Test comparing multiple models."""
        # Mock get_model_metadata
        with patch.object(registry, 'get_model_metadata') as mock_get_metadata:
            mock_get_metadata.side_effect = [
                {
                    "model_id": "model1",
                    "model_type": "DQN",
                    "metrics": {"accuracy": 0.85, "loss": 0.15},
                    "hyperparameters": {"learning_rate": 0.001, "batch_size": 32}
                },
                {
                    "model_id": "model2",
                    "model_type": "PPO",
                    "metrics": {"accuracy": 0.90, "reward": 100},
                    "hyperparameters": {"learning_rate": 0.0003, "clip_range": 0.2}
                },
                {
                    "model_id": "model3",
                    "model_type": "A2C",
                    "metrics": {"accuracy": 0.88, "loss": 0.12, "reward": 95},
                    "hyperparameters": {"learning_rate": 0.0007, "batch_size": 64}
                }
            ]
            
            comparison = registry.compare_models(["model1", "model2", "model3"])
        
        # Verify comparison structure
        assert len(comparison["models"]) == 3
        
        # Check metrics comparison
        assert "accuracy" in comparison["metrics_comparison"]
        assert comparison["metrics_comparison"]["accuracy"] == [0.85, 0.90, 0.88]
        assert "loss" in comparison["metrics_comparison"]
        assert comparison["metrics_comparison"]["loss"] == [0.15, None, 0.12]
        assert "reward" in comparison["metrics_comparison"]
        assert comparison["metrics_comparison"]["reward"] == [None, 100, 95]
        
        # Check hyperparameters comparison
        assert "learning_rate" in comparison["hyperparameters_comparison"]
        assert comparison["hyperparameters_comparison"]["learning_rate"] == [0.001, 0.0003, 0.0007]
        assert "batch_size" in comparison["hyperparameters_comparison"]
        assert comparison["hyperparameters_comparison"]["batch_size"] == [32, None, 64]
    
    def test_register_model_error_handling(self, registry, mock_model, mock_artifact_store):
        """Test error handling during model registration."""
        # Simulate error during save
        mock_artifact_store.save_artifact.side_effect = Exception("Storage error")
        
        with patch('reinforcestrategycreator_pipeline.src.models.registry.Path') as mock_path:
            with patch('reinforcestrategycreator_pipeline.src.models.registry.shutil') as mock_shutil:
                # Create a real temp directory for the test
                temp_dir = tempfile.mkdtemp()
                mock_path.return_value = Path(temp_dir)
                mock_path.return_value.mkdir = Mock()
                mock_path.return_value.exists.return_value = True
                
                try:
                    with pytest.raises(Exception, match="Storage error"):
                        registry.register_model(mock_model, "test_model")
                    
                    # Verify cleanup was attempted
                    mock_shutil.rmtree.assert_called()
                finally:
                    if Path(temp_dir).exists():
                        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])