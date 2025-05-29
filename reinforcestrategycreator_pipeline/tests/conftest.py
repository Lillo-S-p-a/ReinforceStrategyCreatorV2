"""Shared fixtures for pytest."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.artifact_store.base import ArtifactStore, ArtifactMetadata, ArtifactType

@pytest.fixture
def mock_artifact_store() -> MagicMock:
    """Provides a MagicMock for the ArtifactStore."""
    store_mock = MagicMock(spec=ArtifactStore)
    
    # Default return values for common methods, can be overridden in tests
    store_mock.artifact_exists.return_value = True
    store_mock.load_artifact.return_value = Path("/tmp/mock_loaded_artifact_package.tar.gz")
    
    # Mocking save_artifact to return a mock ArtifactMetadata
    mock_metadata = MagicMock(spec=ArtifactMetadata)
    mock_metadata.artifact_id = "mock_eval_id_123"
    mock_metadata.version = "v_mock"
    mock_metadata.artifact_type = ArtifactType.EVALUATION # Default type
    
    store_mock.save_artifact.return_value = mock_metadata
    
    # Mocking list_artifacts to return an empty list by default
    store_mock.list_artifacts.return_value = []
    
    return store_mock