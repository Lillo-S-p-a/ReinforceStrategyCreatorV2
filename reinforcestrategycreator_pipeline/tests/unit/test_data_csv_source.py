"""Unit tests for CSV data source."""

import pytest
from pathlib import Path
import tempfile
import pandas as pd

from reinforcestrategycreator_pipeline.src.data.csv_source import CsvDataSource


class TestCsvDataSource:
    """Test CsvDataSource class."""
    
    @pytest.fixture
    def temp_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,value\n")
            f.write("1,Alice,100.5\n")
            f.write("2,Bob,200.75\n")
            f.write("3,Charlie,300.0\n")
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def temp_tsv_file(self):
        """Create a temporary TSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id\tname\tvalue\n")
            f.write("1\tAlice\t100.5\n")
            f.write("2\tBob\t200.75\n")
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    def test_csv_source_initialization(self, temp_csv_file):
        """Test initializing CSV data source."""
        config = {
            "file_path": str(temp_csv_file),
            "delimiter": ",",
            "encoding": "utf-8"
        }
        
        source = CsvDataSource("test_csv", config)
        
        assert source.source_id == "test_csv"
        assert source.file_path == temp_csv_file
        assert source.delimiter == ","
        assert source.encoding == "utf-8"
    
    def test_validate_config_success(self, temp_csv_file):
        """Test successful config validation."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_csv", config)
        
        # Should not raise
        assert source.validate_config() is True
    
    def test_validate_config_missing_path(self):
        """Test validation with missing file path."""
        config = {}
        
        with pytest.raises(ValueError, match="requires a valid 'source_path' or 'file_path' in config"):
            CsvDataSource("test_csv", config)
    
    def test_validate_config_file_not_found(self):
        """Test validation with non-existent file."""
        config = {"file_path": "/non/existent/file.csv"}
        
        with pytest.raises(FileNotFoundError, match="CSV file not found"): # Changed ValueError to FileNotFoundError
            CsvDataSource("test_csv", config)
    
    def test_validate_config_not_a_file(self, tmp_path):
        """Test validation with directory instead of file."""
        config = {"file_path": str(tmp_path)}
        
        with pytest.raises(ValueError, match="Path is not a file"):
            CsvDataSource("test_csv", config)
    
    def test_validate_config_wrong_extension(self, tmp_path):
        """Test validation with wrong file extension."""
        wrong_file = tmp_path / "data.json"
        wrong_file.write_text("{}")
        
        config = {"file_path": str(wrong_file)}
        
        with pytest.raises(ValueError, match="does not appear to be a CSV"):
            CsvDataSource("test_csv", config)
    
    def test_load_data_basic(self, temp_csv_file):
        """Test basic data loading."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_csv", config)
        
        df = source.load_data()
        
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "value"]
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert df["value"].tolist() == [100.5, 200.75, 300.0]
    
    def test_load_data_with_delimiter(self, temp_tsv_file):
        """Test loading TSV file with tab delimiter."""
        config = {
            "file_path": str(temp_tsv_file),
            "delimiter": "\t"
        }
        source = CsvDataSource("test_tsv", config)
        
        df = source.load_data()
        
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
    
    def test_load_data_with_parse_dates(self, tmp_path):
        """Test loading with date parsing."""
        csv_file = tmp_path / "dates.csv"
        csv_file.write_text("date,value\n2023-01-01,100\n2023-01-02,200\n")
        
        config = {
            "file_path": str(csv_file),
            "parse_dates": ["date"]
        }
        source = CsvDataSource("test_dates", config)
        
        df = source.load_data()
        
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].iloc[0] == pd.Timestamp("2023-01-01")
    
    def test_load_data_with_index_col(self, temp_csv_file):
        """Test loading with index column."""
        config = {
            "file_path": str(temp_csv_file),
            "index_col": "id"
        }
        source = CsvDataSource("test_index", config)
        
        df = source.load_data()
        
        assert df.index.name == "id"
        assert list(df.index) == [1, 2, 3]
    
    def test_load_data_with_dtype(self, temp_csv_file):
        """Test loading with specified dtypes."""
        config = {
            "file_path": str(temp_csv_file),
            "dtype": {"id": str, "value": float}
        }
        source = CsvDataSource("test_dtype", config)
        
        df = source.load_data()
        
        assert df["id"].dtype == object  # str type
        assert df["value"].dtype == float
    
    def test_load_data_with_kwargs(self, temp_csv_file):
        """Test loading with additional kwargs."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_kwargs", config)
        
        # Load only first 2 rows
        df = source.load_data(nrows=2)
        
        assert len(df) == 2
        assert df["name"].tolist() == ["Alice", "Bob"]
    
    def test_get_schema(self, temp_csv_file):
        """Test getting schema from CSV."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_schema", config)
        
        schema = source.get_schema()
        
        assert "id" in schema
        assert "name" in schema
        assert "value" in schema
        assert schema["id"] == "int64"
        assert schema["name"] == "object"
        assert schema["value"] == "float64"
    
    def test_get_file_info(self, temp_csv_file):
        """Test getting file information."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_info", config)
        
        info = source.get_file_info()
        
        assert info["file_path"] == str(temp_csv_file)
        assert info["file_size"] > 0
        assert "modified_time" in info
        assert "created_time" in info
    
    def test_lineage_tracking(self, temp_csv_file):
        """Test that lineage is tracked during operations."""
        config = {"file_path": str(temp_csv_file)}
        source = CsvDataSource("test_lineage", config)
        
        # Load data
        df = source.load_data()
        
        # Check lineage
        metadata = source.get_metadata()
        operations = metadata.lineage.get("operations", [])
        
        assert len(operations) >= 2  # load_data and load_complete
        
        # Check load_data operation
        load_op = next(op for op in operations if op["operation"] == "load_data")
        assert load_op["details"]["file_path"] == str(temp_csv_file)
        assert "file_size" in load_op["details"]
        
        # Check load_complete operation
        complete_op = next(op for op in operations if op["operation"] == "load_complete")
        assert complete_op["details"]["rows"] == 3
        assert complete_op["details"]["columns"] == ["id", "name", "value"]
        assert "memory_usage" in complete_op["details"]
    
    def test_load_data_error_handling(self, tmp_path):
        """Test error handling during load."""
        # Create a malformed CSV that will cause pandas to fail
        bad_csv = tmp_path / "bad.csv"
        # Create a CSV with invalid content that pandas can't parse
        bad_csv.write_text("col1,col2\n\"unclosed quote,value2\n")
        
        config = {"file_path": str(bad_csv)}
        source = CsvDataSource("test_error", config)
        
        # Should raise but also track in lineage
        with pytest.raises(Exception):
            source.load_data()
        
        # Check error was tracked
        metadata = source.get_metadata()
        operations = metadata.lineage.get("operations", [])
        error_op = next((op for op in operations if op["operation"] == "load_error"), None)
        
        assert error_op is not None
        assert "error_type" in error_op["details"]
        assert "error_message" in error_op["details"]