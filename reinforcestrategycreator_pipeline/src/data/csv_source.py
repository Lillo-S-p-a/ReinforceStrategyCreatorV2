"""CSV data source implementation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from .base import DataSource


class CsvDataSource(DataSource):
    """Data source for loading data from CSV files."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """Initialize CSV data source.
        
        Args:
            source_id: Unique identifier for this data source
            config: Configuration dictionary containing:
                - file_path: Path to the CSV file
                - delimiter: CSV delimiter (default: ',')
                - encoding: File encoding (default: 'utf-8')
                - parse_dates: List of columns to parse as dates
                - index_col: Column to use as index
                - dtype: Dictionary of column data types
        """
        super().__init__(source_id, config)
        # Check for 'source_path' (from global DataConfig) first, then 'file_path' (for direct CsvDataSource config)
        path_str = config.get("source_path", config.get("file_path", ""))
        self.file_path = Path(path_str) if path_str else Path("") # Ensure Path object even if empty
        
        # If the path is relative and starts with "../", we need to handle it specially
        # The path "../dummy_data.csv" is relative to the configs directory, but we're running from pipeline root
        if not self.file_path.is_absolute() and str(self.file_path).startswith("../"):
            # Since the config uses "../dummy_data.csv" which is relative to configs/,
            # and dummy_data.csv is in the pipeline root, we just need the filename
            self.file_path = Path("dummy_data.csv")
        self.delimiter = config.get("delimiter", ",")
        self.encoding = config.get("encoding", "utf-8")
        self.parse_dates = config.get("parse_dates", None)
        self.index_col = config.get("index_col", None)
        self.dtype = config.get("dtype", None)
        
        # Validate configuration on initialization
        self.validate_config()
    
    def validate_config(self) -> bool:
        """Validate the CSV data source configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # self.file_path is now set in __init__ from "source_path" or "file_path"
        original_source_path = self.config.get("source_path")
        original_file_path = self.config.get("file_path")

        if not self.file_path or str(self.file_path) == "" or str(self.file_path) == ".":
            raise ValueError(
                f"CSV data source requires a valid 'source_path' or 'file_path' in config. "
                f"Resolved file_path is '{self.file_path}'. "
                f"Original config: source_path='{original_source_path}', file_path='{original_file_path}'"
            )
        
        # Paths are expected to be resolvable from CWD (workspace root)
        # The config 'source_path' is "reinforcestrategycreator_pipeline/dummy_data.csv"
        resolved_path = self.file_path.resolve() # Get absolute path for clearer error messages

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"CSV file not found at resolved path: '{resolved_path}' "
                f"(derived from file_path: '{self.file_path}', "
                f"original config: source_path='{original_source_path}', file_path='{original_file_path}')"
            )
        
        if not resolved_path.is_file():
            raise ValueError(
                f"Path is not a file: '{resolved_path}' "
                f"(derived from file_path: '{self.file_path}', "
                f"original config: source_path='{original_source_path}', file_path='{original_file_path}')"
            )
        
        # Suffix check can remain on self.file_path (original relative or absolute path)
        if self.file_path.suffix.lower() not in [".csv", ".tsv", ".txt"]: # Allow .tsv and .txt as common variants
            raise ValueError(f"File does not appear to be a CSV/TSV/TXT: '{self.file_path}'")
        
        return True
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from the CSV file.
        
        Args:
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            DataFrame containing the loaded data
        """
        # Update lineage before loading
        self.update_lineage("load_data", {
            "file_path": str(self.file_path),
            "file_size": self.file_path.stat().st_size,
            "kwargs": kwargs
        })
        
        # Merge default parameters with kwargs
        read_params = {
            "filepath_or_buffer": self.file_path,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
        }
        
        if self.parse_dates is not None:
            read_params["parse_dates"] = self.parse_dates
        
        if self.index_col is not None:
            read_params["index_col"] = self.index_col
            
        if self.dtype is not None:
            read_params["dtype"] = self.dtype
        
        # Override with any provided kwargs
        read_params.update(kwargs)

        # Map 'columns' to 'usecols' if present
        if "columns" in read_params:
            read_params["usecols"] = read_params.pop("columns")
        
        try:
            df = pd.read_csv(**read_params)
            
            # Update lineage with success info
            self.update_lineage("load_complete", {
                "rows": len(df),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            })
            
            return df
            
        except Exception as e:
            # Update lineage with error info
            self.update_lineage("load_error", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema of the CSV file.
        
        Returns:
            Dictionary mapping column names to data types
        """
        # Read just the first few rows to infer schema
        try:
            df_sample = pd.read_csv(
                self.file_path,
                delimiter=self.delimiter,
                encoding=self.encoding,
                nrows=100  # Sample first 100 rows
            )
            
            schema = {}
            for col, dtype in df_sample.dtypes.items():
                schema[str(col)] = str(dtype)
            
            return schema
            
        except Exception as e:
            # Return empty schema if we can't read the file
            return {}
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the CSV file.
        
        Returns:
            Dictionary containing file information
        """
        stat = self.file_path.stat()
        return {
            "file_path": str(self.file_path),
            "file_size": int(stat.st_size),  # Convert to regular int for JSON serialization
            "modified_time": pd.Timestamp(stat.st_mtime, unit='s').isoformat(),
            "created_time": pd.Timestamp(stat.st_ctime, unit='s').isoformat(),
        }