"""Unit tests for data source base classes."""

import pytest
from datetime import datetime
from typing import Dict

import pandas as pd

from reinforcestrategycreator_pipeline.src.data.base import DataSource, DataSourceMetadata


class MockDataSource(DataSource):
    """Mock data source for testing."""
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load mock data."""
        return pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
    
    def validate_config(self) -> bool:
        """Validate mock config."""
        return True
    
    def get_schema(self) -> Dict[str, str]:
        """Get mock schema."""
        return {
            "col1": "int64",
            "col2": "object"
        }


class TestDataSourceMetadata:
    """Test DataSourceMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating metadata instance."""
        metadata = DataSourceMetadata(
            source_id="test_source",
            source_type="csv",
            version="1.0.0",
            created_at=datetime.now(),
            description="Test data source",
            schema={"col1": "int", "col2": "str"},
            properties={"path": "/data/test.csv"},
            lineage={"operations": []}
        )
        
        assert metadata.source_id == "test_source"
        assert metadata.source_type == "csv"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test data source"
        assert metadata.schema == {"col1": "int", "col2": "str"}
        assert metadata.properties == {"path": "/data/test.csv"}
        assert metadata.lineage == {"operations": []}
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        created_at = datetime.now()
        metadata = DataSourceMetadata(
            source_id="test_source",
            source_type="csv",
            version="1.0.0",
            created_at=created_at
        )
        
        data_dict = metadata.to_dict()
        
        assert data_dict["source_id"] == "test_source"
        assert data_dict["source_type"] == "csv"
        assert data_dict["version"] == "1.0.0"
        assert data_dict["created_at"] == created_at.isoformat()
        assert data_dict["description"] is None
        assert data_dict["schema"] is None
        assert data_dict["properties"] == {}
        assert data_dict["lineage"] == {}
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        created_at = datetime.now()
        data_dict = {
            "source_id": "test_source",
            "source_type": "api",
            "version": "2.0.0",
            "created_at": created_at.isoformat(),
            "description": "API source",
            "schema": {"data": "array"},
            "properties": {"endpoint": "https://api.example.com"},
            "lineage": {"operations": [{"op": "fetch"}]}
        }
        
        metadata = DataSourceMetadata.from_dict(data_dict)
        
        assert metadata.source_id == "test_source"
        assert metadata.source_type == "api"
        assert metadata.version == "2.0.0"
        assert metadata.created_at.isoformat() == created_at.isoformat()
        assert metadata.description == "API source"
        assert metadata.schema == {"data": "array"}
        assert metadata.properties == {"endpoint": "https://api.example.com"}
        assert metadata.lineage == {"operations": [{"op": "fetch"}]}


class TestDataSource:
    """Test DataSource abstract base class."""
    
    def test_data_source_initialization(self):
        """Test initializing a data source."""
        config = {"param1": "value1", "param2": 42}
        source = MockDataSource("mock_source", config)
        
        assert source.source_id == "mock_source"
        assert source.config == config
        assert source._metadata is None
    
    def test_get_metadata(self):
        """Test getting metadata from data source."""
        source = MockDataSource("mock_source", {"test": True})
        metadata = source.get_metadata()
        
        assert metadata.source_id == "mock_source"
        assert metadata.source_type == "MockDataSource"
        assert metadata.version == "1.0.0"
        assert metadata.schema == {"col1": "int64", "col2": "object"}
        assert metadata.properties == {"test": True}
        
        # Should return same instance on subsequent calls
        metadata2 = source.get_metadata()
        assert metadata is metadata2
    
    def test_update_lineage(self):
        """Test updating lineage information."""
        source = MockDataSource("mock_source", {})
        
        # Update lineage
        source.update_lineage("load", {"rows": 100})
        source.update_lineage("filter", {"condition": "col1 > 0"})
        
        metadata = source.get_metadata()
        operations = metadata.lineage.get("operations", [])
        
        assert len(operations) == 2
        assert operations[0]["operation"] == "load"
        assert operations[0]["details"] == {"rows": 100}
        assert operations[1]["operation"] == "filter"
        assert operations[1]["details"] == {"condition": "col1 > 0"}
        
        # Check timestamps exist
        for op in operations:
            assert "timestamp" in op
            # Verify it's a valid ISO format timestamp
            datetime.fromisoformat(op["timestamp"])
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            DataSource("test", {})