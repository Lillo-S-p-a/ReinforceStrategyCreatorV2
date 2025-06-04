"""Unit tests for Data Manager."""

import pytest
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from reinforcestrategycreator_pipeline.src.data.manager import DataManager
from reinforcestrategycreator_pipeline.src.data.base import DataSource, DataSourceMetadata
from reinforcestrategycreator_pipeline.src.data.csv_source import CsvDataSource
from reinforcestrategycreator_pipeline.src.data.api_source import ApiDataSource
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactStore, ArtifactType, ArtifactMetadata
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager


class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    def __init__(self, source_id: str, config: dict):
        super().__init__(source_id, config)
        self.load_data_called = False
        self.load_data_kwargs = None
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        self.load_data_called = True
        self.load_data_kwargs = kwargs
        return pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
    
    def validate_config(self) -> bool:
        return True
    
    def get_schema(self) -> dict:
        return {"col1": "int64", "col2": "object"}


class TestDataManager:
    """Test DataManager class."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_config.return_value = {
            "data": {
                "cache_enabled": True,
                "cache_dir": "./test_cache",
                "cache_ttl_hours": 24
            }
        }
        return config_manager
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create mock artifact store."""
        return Mock(spec=ArtifactStore)
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def data_manager(self, mock_config_manager, mock_artifact_store, temp_cache_dir):
        """Create DataManager instance for testing."""
        return DataManager(
            config_manager=mock_config_manager,
            artifact_store=mock_artifact_store,
            cache_dir=temp_cache_dir
        )
    
    def test_initialization(self, mock_config_manager, mock_artifact_store):
        """Test DataManager initialization."""
        manager = DataManager(mock_config_manager, mock_artifact_store)
        
        assert manager.config_manager == mock_config_manager
        assert manager.artifact_store == mock_artifact_store
        assert manager.cache_enabled is True
        assert manager.cache_ttl_hours == 24
        assert len(manager.data_sources) == 0
        assert len(manager.lineage) == 0
    
    def test_initialization_with_custom_cache_dir(self, mock_config_manager, mock_artifact_store, temp_cache_dir):
        """Test initialization with custom cache directory."""
        manager = DataManager(
            mock_config_manager,
            mock_artifact_store,
            cache_dir=temp_cache_dir
        )
        
        assert manager.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
    
    def test_register_source_csv(self, data_manager, tmp_path):
        """Test registering a CSV data source."""
        # Create a test CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,a\n2,b\n")
        
        config = {"file_path": str(csv_file)}
        source = data_manager.register_source("csv_source", "csv", config)
        
        assert isinstance(source, CsvDataSource)
        assert source.source_id == "csv_source"
        assert "csv_source" in data_manager.data_sources
        assert data_manager.data_sources["csv_source"] == source
        
        # Check lineage
        assert "csv_source" in data_manager.lineage
        lineage = data_manager.lineage["csv_source"]
        assert len(lineage) == 1
        assert lineage[0]["operation"] == "register"
        assert lineage[0]["details"]["source_type"] == "csv"
    
    def test_register_source_api(self, data_manager):
        """Test registering an API data source."""
        config = {"endpoint": "https://api.example.com/data"}
        source = data_manager.register_source("api_source", "api", config)
        
        assert isinstance(source, ApiDataSource)
        assert source.source_id == "api_source"
        assert "api_source" in data_manager.data_sources
    
    def test_register_source_invalid_type(self, data_manager):
        """Test registering with invalid source type."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            data_manager.register_source("bad_source", "invalid_type", {})
    
    def test_load_data_not_registered(self, data_manager):
        """Test loading from unregistered source."""
        with pytest.raises(ValueError, match="Data source not registered"):
            data_manager.load_data("unknown_source")
    
    def test_load_data_no_cache(self, data_manager):
        """Test loading data without cache."""
        # Register mock source
        mock_source = MockDataSource("test_source", {})
        data_manager.data_sources["test_source"] = mock_source
        
        # Load data with cache disabled
        df = data_manager.load_data("test_source", use_cache=False)
        
        assert len(df) == 3
        assert mock_source.load_data_called is True
        
        # Check lineage
        lineage = data_manager.lineage.get("test_source", [])
        load_op = next((op for op in lineage if op["operation"] == "load_from_source"), None)
        assert load_op is not None
        assert load_op["details"]["rows"] == 3
    
    def test_load_data_with_cache_miss(self, data_manager):
        """Test loading data with cache miss."""
        # Register mock source
        mock_source = MockDataSource("test_source", {})
        data_manager.data_sources["test_source"] = mock_source
        
        # First load should miss cache
        df1 = data_manager.load_data("test_source")
        
        assert len(df1) == 3
        assert mock_source.load_data_called is True
        
        # Check cache file was created
        cache_files = list(data_manager.cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1
    
    def test_load_data_with_cache_hit(self, data_manager):
        """Test loading data with cache hit."""
        # Register mock source
        mock_source = MockDataSource("test_source", {})
        data_manager.data_sources["test_source"] = mock_source
        
        # First load to populate cache
        df1 = data_manager.load_data("test_source")
        mock_source.load_data_called = False  # Reset flag
        
        # Second load should hit cache
        df2 = data_manager.load_data("test_source")
        
        assert len(df2) == 3
        assert mock_source.load_data_called is False  # Should not call source
        pd.testing.assert_frame_equal(df1, df2)
        
        # Check lineage for cache hit
        lineage = data_manager.lineage.get("test_source", [])
        cache_hit = next((op for op in lineage if op["operation"] == "load_from_cache"), None)
        assert cache_hit is not None
        assert cache_hit["details"]["cache_hit"] is True
    
    def test_load_data_cache_expiry(self, data_manager):
        """Test cache expiry."""
        # Register mock source
        mock_source = MockDataSource("test_source", {})
        data_manager.data_sources["test_source"] = mock_source
        
        # Create expired cache entry
        cache_key = data_manager._get_cache_key("test_source", {})
        cache_file = data_manager.cache_dir / f"{cache_key}.pkl"
        meta_file = data_manager.cache_dir / f"{cache_key}.meta"
        
        # Save data
        test_df = pd.DataFrame({"old": [1, 2, 3]})
        test_df.to_pickle(cache_file)
        
        # Save expired metadata
        old_time = datetime.now() - timedelta(hours=25)  # Older than TTL
        meta = {
            "source_id": "test_source",
            "timestamp": old_time.isoformat(),
            "kwargs": {},
            "shape": [3, 1]
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        
        # Load should miss expired cache
        df = data_manager.load_data("test_source")
        
        assert len(df) == 3
        assert mock_source.load_data_called is True
        assert "col1" in df.columns  # New data, not cached "old" column
    
    def test_load_data_with_kwargs(self, data_manager):
        """Test loading data with additional kwargs."""
        # Register mock source
        mock_source = MockDataSource("test_source", {})
        data_manager.data_sources["test_source"] = mock_source
        
        # Load with kwargs
        df = data_manager.load_data("test_source", param1="value1", param2=42)
        
        assert mock_source.load_data_kwargs == {"param1": "value1", "param2": 42}
    
    def test_save_version(self, data_manager, mock_artifact_store):
        """Test saving a versioned dataset."""
        # Setup mock artifact store
        mock_metadata = ArtifactMetadata(
            artifact_id="data_test_source",
            artifact_type=ArtifactType.DATASET,
            version="20230101_120000",
            created_at=datetime.now()
        )
        mock_artifact_store.save_artifact.return_value = mock_metadata
        
        # Create test data
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        
        # Save version
        version = data_manager.save_version(
            "test_source",
            df,
            description="Test dataset",
            tags=["test", "sample"]
        )
        
        # Check version format
        assert len(version) == 15  # YYYYMMDD_HHMMSS format
        
        # Verify artifact store was called
        mock_artifact_store.save_artifact.assert_called_once()
        call_args = mock_artifact_store.save_artifact.call_args
        
        assert call_args.kwargs["artifact_id"] == "data_test_source"
        assert call_args.kwargs["artifact_type"] == ArtifactType.DATASET
        assert call_args.kwargs["description"] == "Test dataset"
        assert call_args.kwargs["tags"] == ["test", "sample"]
        
        # Check metadata
        metadata = call_args.kwargs["metadata"]
        assert metadata["source_id"] == "test_source"
        assert metadata["shape"] == [3, 2]
        assert metadata["columns"] == ["col1", "col2"]
        assert "dtypes" in metadata
        assert "lineage" in metadata
    
    def test_save_version_with_custom_version(self, data_manager, mock_artifact_store):
        """Test saving with custom version string."""
        mock_artifact_store.save_artifact.return_value = Mock()
        
        df = pd.DataFrame({"col1": [1, 2, 3]})
        version = data_manager.save_version("test_source", df, version="v1.0.0")
        
        assert version == "v1.0.0"
        
        # Verify custom version was used
        call_args = mock_artifact_store.save_artifact.call_args
        assert call_args.kwargs["version"] == "v1.0.0"
    
    def test_load_version(self, data_manager, mock_artifact_store, temp_cache_dir):
        """Test loading a specific version."""
        # Create test data file
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["x", "y", "z"]})
        version_file = temp_cache_dir / "versioned_data.pkl"
        test_df.to_pickle(version_file)
        
        # Mock artifact store to return the file path
        mock_artifact_store.load_artifact.return_value = version_file
        
        # Load version
        df = data_manager._load_version("test_source", "v1.0.0")
        
        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2"]
        pd.testing.assert_frame_equal(df, test_df)
        
        # Verify artifact store was called correctly
        mock_artifact_store.load_artifact.assert_called_once_with(
            artifact_id="data_test_source",
            version="v1.0.0"
        )
        
        # Check lineage
        lineage = data_manager.lineage.get("test_source", [])
        load_version_op = next((op for op in lineage if op["operation"] == "load_version"), None)
        assert load_version_op is not None
        assert load_version_op["details"]["version"] == "v1.0.0"
    
    def test_list_versions(self, data_manager, mock_artifact_store):
        """Test listing versions."""
        mock_artifact_store.list_versions.return_value = ["v1.0.0", "v1.1.0", "v2.0.0"]
        
        versions = data_manager.list_versions("test_source")
        
        assert versions == ["v1.0.0", "v1.1.0", "v2.0.0"]
        mock_artifact_store.list_versions.assert_called_once_with("data_test_source")
    
    def test_get_lineage(self, data_manager):
        """Test getting lineage information."""
        # Add some lineage entries
        data_manager._track_lineage("test_source", "operation1", {"detail": "value1"})
        data_manager._track_lineage("test_source", "operation2", {"detail": "value2"})
        
        # Also add a mock source with its own lineage
        mock_source = MockDataSource("test_source", {})
        mock_source.update_lineage("source_op", {"source_detail": "value"})
        data_manager.data_sources["test_source"] = mock_source
        
        # Get combined lineage
        lineage = data_manager.get_lineage("test_source")
        
        assert len(lineage) >= 3
        ops = [op["operation"] for op in lineage]
        assert "operation1" in ops
        assert "operation2" in ops
        assert "source_op" in ops
    
    def test_get_source_metadata(self, data_manager):
        """Test getting source metadata."""
        # Register a source
        mock_source = MockDataSource("test_source", {"param": "value"})
        data_manager.data_sources["test_source"] = mock_source
        
        metadata = data_manager.get_source_metadata("test_source")
        
        assert metadata is not None
        assert metadata.source_id == "test_source"
        assert metadata.source_type == "MockDataSource"
        assert metadata.properties == {"param": "value"}
        
        # Test non-existent source
        assert data_manager.get_source_metadata("unknown") is None
    
    def test_clear_cache_all(self, data_manager):
        """Test clearing all cache."""
        # Create some cache files
        for i in range(3):
            cache_file = data_manager.cache_dir / f"test_{i}.pkl"
            meta_file = data_manager.cache_dir / f"test_{i}.meta"
            cache_file.write_text("data")
            meta_file.write_text("{}")
        
        # Clear all cache
        count = data_manager.clear_cache()
        
        assert count == 3
        assert len(list(data_manager.cache_dir.glob("*.pkl"))) == 0
        assert len(list(data_manager.cache_dir.glob("*.meta"))) == 0
    
    def test_clear_cache_specific_source(self, data_manager):
        """Test clearing cache for specific source."""
        # Create cache files for different sources
        cache_files = [
            ("source1_abc123.pkl", "source1_abc123.meta"),
            ("source1_def456.pkl", "source1_def456.meta"),
            ("source2_ghi789.pkl", "source2_ghi789.meta"),
        ]
        
        for pkl, meta in cache_files:
            (data_manager.cache_dir / pkl).write_text("data")
            (data_manager.cache_dir / meta).write_text("{}")
        
        # Clear cache for source1 only
        count = data_manager.clear_cache("source1")
        
        assert count == 2
        assert not (data_manager.cache_dir / "source1_abc123.pkl").exists()
        assert not (data_manager.cache_dir / "source1_def456.pkl").exists()
        assert (data_manager.cache_dir / "source2_ghi789.pkl").exists()
    
    def test_cache_key_generation(self, data_manager):
        """Test cache key generation is deterministic."""
        kwargs1 = {"param1": "value1", "param2": 42, "param3": [1, 2, 3]}
        kwargs2 = {"param2": 42, "param3": [1, 2, 3], "param1": "value1"}  # Different order
        kwargs3 = {"param1": "value2", "param2": 42, "param3": [1, 2, 3]}  # Different value
        
        key1 = data_manager._get_cache_key("test_source", kwargs1)
        key2 = data_manager._get_cache_key("test_source", kwargs2)
        key3 = data_manager._get_cache_key("test_source", kwargs3)
        
        # Same kwargs in different order should produce same key
        assert key1 == key2
        # Different kwargs should produce different key
        assert key1 != key3
        
        # Key should include source_id
        assert key1.startswith("test_source_")
    
    def test_integration_csv_to_versioned(self, data_manager, mock_artifact_store, tmp_path):
        """Integration test: Register CSV source, load data, save version."""
        # Create test CSV
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        
        # Register source
        source = data_manager.register_source(
            "csv_test",
            "csv",
            {"file_path": str(csv_file)}
        )
        
        # Load data
        df = data_manager.load_data("csv_test")
        assert len(df) == 3
        
        # Save version
        mock_artifact_store.save_artifact.return_value = Mock()
        version = data_manager.save_version(
            "csv_test",
            df,
            description="Test CSV data"
        )
        
        # Verify full flow
        assert version is not None
        assert "csv_test" in data_manager.data_sources
        assert len(data_manager.lineage.get("csv_test", [])) > 0