"""Unit tests for the artifact store module."""

import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import pytest

from src.artifact_store import (
    ArtifactStore,
    ArtifactMetadata,
    ArtifactType,
    LocalFileSystemStore
)


class TestArtifactMetadata:
    """Test the ArtifactMetadata data class."""
    
    def test_metadata_creation(self):
        """Test creating artifact metadata."""
        metadata = ArtifactMetadata(
            artifact_id="test-model",
            artifact_type=ArtifactType.MODEL,
            version="v1.0",
            created_at=datetime.now(),
            created_by="test-user",
            description="Test model artifact",
            tags=["test", "model"],
            properties={"accuracy": 0.95},
            source_info={"training_data": "dataset-v1"}
        )
        
        assert metadata.artifact_id == "test-model"
        assert metadata.artifact_type == ArtifactType.MODEL
        assert metadata.version == "v1.0"
        assert metadata.created_by == "test-user"
        assert metadata.description == "Test model artifact"
        assert "test" in metadata.tags
        assert metadata.properties["accuracy"] == 0.95
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        created_at = datetime.now()
        metadata = ArtifactMetadata(
            artifact_id="test-dataset",
            artifact_type=ArtifactType.DATASET,
            version="v2.0",
            created_at=created_at,
            tags=["training", "processed"]
        )
        
        data = metadata.to_dict()
        assert data["artifact_id"] == "test-dataset"
        assert data["artifact_type"] == "dataset"
        assert data["version"] == "v2.0"
        assert data["created_at"] == created_at.isoformat()
        assert data["tags"] == ["training", "processed"]
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        created_at = datetime.now()
        data = {
            "artifact_id": "test-report",
            "artifact_type": "report",
            "version": "v1.1",
            "created_at": created_at.isoformat(),
            "created_by": "analyzer",
            "description": "Performance report",
            "tags": ["evaluation"],
            "properties": {"metric": "accuracy"},
            "source_info": {"model": "model-v1"}
        }
        
        metadata = ArtifactMetadata.from_dict(data)
        assert metadata.artifact_id == "test-report"
        assert metadata.artifact_type == ArtifactType.REPORT
        assert metadata.version == "v1.1"
        assert metadata.created_by == "analyzer"
        assert metadata.description == "Performance report"
        assert metadata.tags == ["evaluation"]


class TestLocalFileSystemStore:
    """Test the LocalFileSystemStore implementation."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary artifact store."""
        temp_dir = tempfile.mkdtemp()
        store = LocalFileSystemStore(temp_dir)
        yield store
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_file(self):
        """Create a sample file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Sample artifact content")
            temp_path = f.name
        yield Path(temp_path)
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_directory(self):
        """Create a sample directory for testing."""
        temp_dir = tempfile.mkdtemp()
        # Create some files in the directory
        (Path(temp_dir) / "file1.txt").write_text("Content 1")
        (Path(temp_dir) / "file2.txt").write_text("Content 2")
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("Content 3")
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_save_file_artifact(self, temp_store, sample_file):
        """Test saving a file artifact."""
        metadata = temp_store.save_artifact(
            artifact_id="test-file",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER,
            description="Test file artifact",
            tags=["test", "file"]
        )
        
        assert metadata.artifact_id == "test-file"
        assert metadata.artifact_type == ArtifactType.OTHER
        assert metadata.description == "Test file artifact"
        assert "test" in metadata.tags
        assert metadata.source_info["is_directory"] is False
        
        # Check that the artifact was saved
        assert temp_store.artifact_exists("test-file")
    
    def test_save_directory_artifact(self, temp_store, sample_directory):
        """Test saving a directory artifact."""
        metadata = temp_store.save_artifact(
            artifact_id="test-dataset",
            artifact_path=sample_directory,
            artifact_type=ArtifactType.DATASET,
            version="v1.0",
            metadata={"num_files": 3}
        )
        
        assert metadata.artifact_id == "test-dataset"
        assert metadata.version == "v1.0"
        assert metadata.properties["num_files"] == 3
        assert metadata.source_info["is_directory"] is True
    
    def test_load_artifact(self, temp_store, sample_file):
        """Test loading an artifact."""
        # Save artifact first
        temp_store.save_artifact(
            artifact_id="test-load",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER
        )
        
        # Load without destination
        loaded_path = temp_store.load_artifact("test-load")
        assert loaded_path.exists()
        assert loaded_path.read_text() == "Sample artifact content"
        
        # Load with destination
        dest_path = Path(tempfile.mkdtemp()) / "loaded_artifact.txt"
        loaded_path2 = temp_store.load_artifact("test-load", destination_path=dest_path)
        assert loaded_path2 == dest_path
        assert dest_path.read_text() == "Sample artifact content"
    
    def test_load_specific_version(self, temp_store, sample_file):
        """Test loading a specific version of an artifact."""
        # Save multiple versions
        v1_metadata = temp_store.save_artifact(
            artifact_id="versioned-artifact",
            artifact_path=sample_file,
            artifact_type=ArtifactType.MODEL,
            version="v1.0"
        )
        
        # Modify the file and save as v2
        sample_file.write_text("Updated content")
        v2_metadata = temp_store.save_artifact(
            artifact_id="versioned-artifact",
            artifact_path=sample_file,
            artifact_type=ArtifactType.MODEL,
            version="v2.0"
        )
        
        # Load specific versions
        v1_path = temp_store.load_artifact("versioned-artifact", version="v1.0")
        assert v1_path.read_text() == "Sample artifact content"
        
        v2_path = temp_store.load_artifact("versioned-artifact", version="v2.0")
        assert v2_path.read_text() == "Updated content"
        
        # Load latest (should be v2.0)
        latest_path = temp_store.load_artifact("versioned-artifact")
        assert latest_path.read_text() == "Updated content"
    
    def test_get_artifact_metadata(self, temp_store, sample_file):
        """Test retrieving artifact metadata."""
        saved_metadata = temp_store.save_artifact(
            artifact_id="test-metadata",
            artifact_path=sample_file,
            artifact_type=ArtifactType.CONFIG,
            description="Configuration file",
            tags=["config", "test"]
        )
        
        # Get metadata
        retrieved_metadata = temp_store.get_artifact_metadata("test-metadata")
        assert retrieved_metadata.artifact_id == saved_metadata.artifact_id
        assert retrieved_metadata.version == saved_metadata.version
        assert retrieved_metadata.description == "Configuration file"
        assert set(retrieved_metadata.tags) == {"config", "test"}
    
    def test_list_artifacts(self, temp_store, sample_file):
        """Test listing artifacts."""
        # Save multiple artifacts
        temp_store.save_artifact(
            artifact_id="model-1",
            artifact_path=sample_file,
            artifact_type=ArtifactType.MODEL,
            tags=["production"]
        )
        
        temp_store.save_artifact(
            artifact_id="dataset-1",
            artifact_path=sample_file,
            artifact_type=ArtifactType.DATASET,
            tags=["training", "production"]
        )
        
        temp_store.save_artifact(
            artifact_id="model-2",
            artifact_path=sample_file,
            artifact_type=ArtifactType.MODEL,
            tags=["experimental"]
        )
        
        # List all artifacts
        all_artifacts = temp_store.list_artifacts()
        assert len(all_artifacts) == 3
        
        # List by type
        models = temp_store.list_artifacts(artifact_type=ArtifactType.MODEL)
        assert len(models) == 2
        assert all(m.artifact_type == ArtifactType.MODEL for m in models)
        
        # List by tags
        production = temp_store.list_artifacts(tags=["production"])
        assert len(production) == 2
        assert all("production" in a.tags for a in production)
        
        # List by type and tags
        prod_models = temp_store.list_artifacts(
            artifact_type=ArtifactType.MODEL,
            tags=["production"]
        )
        assert len(prod_models) == 1
        assert prod_models[0].artifact_id == "model-1"
    
    def test_list_versions(self, temp_store, sample_file):
        """Test listing versions of an artifact."""
        # Save multiple versions
        for i in range(3):
            temp_store.save_artifact(
                artifact_id="multi-version",
                artifact_path=sample_file,
                artifact_type=ArtifactType.MODEL,
                version=f"v1.{i}"
            )
        
        versions = temp_store.list_versions("multi-version")
        assert len(versions) == 3
        assert versions == ["v1.0", "v1.1", "v1.2"]
    
    def test_delete_artifact_version(self, temp_store, sample_file):
        """Test deleting a specific version."""
        # Save multiple versions
        temp_store.save_artifact(
            artifact_id="deletable",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER,
            version="v1.0"
        )
        temp_store.save_artifact(
            artifact_id="deletable",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER,
            version="v2.0"
        )
        
        # Delete v1.0
        assert temp_store.delete_artifact("deletable", version="v1.0")
        assert not temp_store.artifact_exists("deletable", version="v1.0")
        assert temp_store.artifact_exists("deletable", version="v2.0")
        
        # Latest should still be v2.0
        metadata = temp_store.get_artifact_metadata("deletable")
        assert metadata.version == "v2.0"
    
    def test_delete_all_versions(self, temp_store, sample_file):
        """Test deleting all versions of an artifact."""
        # Save artifact
        temp_store.save_artifact(
            artifact_id="delete-all",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER
        )
        
        assert temp_store.artifact_exists("delete-all")
        assert temp_store.delete_artifact("delete-all")
        assert not temp_store.artifact_exists("delete-all")
    
    def test_artifact_exists(self, temp_store, sample_file):
        """Test checking artifact existence."""
        assert not temp_store.artifact_exists("non-existent")
        
        temp_store.save_artifact(
            artifact_id="exists-test",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER,
            version="v1.0"
        )
        
        assert temp_store.artifact_exists("exists-test")
        assert temp_store.artifact_exists("exists-test", version="v1.0")
        assert not temp_store.artifact_exists("exists-test", version="v2.0")
    
    def test_auto_version_generation(self, temp_store, sample_file):
        """Test automatic version generation."""
        metadata1 = temp_store.save_artifact(
            artifact_id="auto-version",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER
        )
        
        # Version should be timestamp-based
        assert metadata1.version is not None
        assert len(metadata1.version) == 15  # YYYYMMDD_HHMMSS format
        
        # Save another version
        import time
        time.sleep(1)  # Ensure different timestamp
        
        metadata2 = temp_store.save_artifact(
            artifact_id="auto-version",
            artifact_path=sample_file,
            artifact_type=ArtifactType.OTHER
        )
        
        assert metadata2.version != metadata1.version
        assert metadata2.version > metadata1.version  # Later timestamp
    
    def test_error_handling(self, temp_store):
        """Test error handling."""
        # Non-existent artifact path
        with pytest.raises(FileNotFoundError):
            temp_store.save_artifact(
                artifact_id="error-test",
                artifact_path="/non/existent/path",
                artifact_type=ArtifactType.OTHER
            )
        
        # Load non-existent artifact
        with pytest.raises(ValueError):
            temp_store.load_artifact("non-existent")
        
        # Get metadata for non-existent artifact
        with pytest.raises(ValueError):
            temp_store.get_artifact_metadata("non-existent")