"""Data ingestion stage implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import json

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactType


class DataIngestionStage(PipelineStage):
    """
    Stage responsible for ingesting data from various sources.
    
    This stage handles:
    - Loading data from files (CSV, JSON, Parquet)
    - Validating data schema
    - Basic data quality checks
    - Storing raw data as artifacts
    """
    
    def __init__(self, name: str = "data_ingestion", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data ingestion stage.
        
        Args:
            name: Stage name
            config: Stage configuration containing:
                - source_path: Path to data source
                - source_type: Type of data source (csv, json, parquet)
                - validation_rules: Optional data validation rules
                - sample_size: Optional sample size for large datasets
        """
        super().__init__(name, config or {})
        # These will be properly initialized in setup() using context
        self.source_path: Optional[Path] = None
        self.source_type: str = "csv" # Default, will be overridden in setup
        self.validation_rules = self.config.get("validation_rules", {}) # Stage-specific overrides
        self.sample_size = self.config.get("sample_size") # Stage-specific overrides
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the data ingestion stage."""
        self.logger.info(f"Setting up {self.name} stage")

        # Get ConfigManager from context to access global configurations
        config_manager = context.get("config_manager")
        if not config_manager:
            self.logger.warning("ConfigManager not found in context. Using stage-specific config only for data source.")
            # Fallback to existing logic if no ConfigManager in context (though it should be)
            self.source_path_str = self.config.get("source_path", "")
            self.source_type = self.config.get("source_type", "csv").lower()
        else:
            global_pipeline_config = config_manager.get_config() # Gets the PipelineConfig object
            global_data_config = global_pipeline_config.data   # Access the DataConfig model

            # Prioritize stage-specific config, then global, then default for source_type
            self.source_type = self.config.get("source_type", global_data_config.source_type.value).lower()
            
            # Prioritize stage-specific config, then global for source_path
            # Note: global_data_config.source_path might be None if source_type is API
            self.source_path_str = self.config.get("source_path", global_data_config.source_path)

        if self.source_path_str:
            self.source_path = Path(self.source_path_str)
        else:
            self.source_path = None

        # Validate source path exists if type is CSV
        if self.source_type == "csv":
            if not self.source_path:
                raise ValueError(f"Source type is CSV but no source_path is configured.")
            
            # Resolve relative paths based on ConfigManager's loader base_path
            if not self.source_path.is_absolute() and config_manager and hasattr(config_manager, 'loader'):
                # If path is relative, resolve it relative to the config file's location
                self.logger.info(f"ConfigManager loader base_path: {config_manager.loader.base_path}")
                self.logger.info(f"Source path (relative): {self.source_path}")
                resolved_source_path = (config_manager.loader.base_path / self.source_path).resolve()
                self.logger.info(f"Resolved source path: {resolved_source_path}")
            else:
                # If path is absolute or no ConfigManager, use as-is
                resolved_source_path = self.source_path.resolve()
            if not resolved_source_path.exists():
                raise FileNotFoundError(f"Data source CSV file not found: {resolved_source_path} (original: {self.source_path})")
            # Store the resolved path for use in loading
            self.resolved_source_path = resolved_source_path
        elif self.source_type == "api":
            # Check if we have config_manager and can access global config
            if config_manager:
                global_pipeline_config = config_manager.get_config()
                global_data_config = global_pipeline_config.data
                if not global_data_config.api_endpoint:
                    raise ValueError("Source type is API but no api_endpoint is configured in global data config.")
            
        # Get artifact store from context if available
        self.artifact_store = context.get("artifact_store")
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the data ingestion stage.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated pipeline context with ingested data
        """
        self.logger.info(f"Running {self.name} stage")
        
        try:
            # Load data based on source type
            data = self._load_data()
            
            # Perform basic validation
            validation_results = self._validate_data(data)
            
            # Log validation results
            if validation_results["errors"]:
                self.logger.warning(f"Data validation warnings: {validation_results['errors']}")
            
            # Store raw data in context
            context.set("raw_data", data)
            context.set("data_validation_results", validation_results)
            
            # Store metadata
            metadata = {
                "source_path": str(self.source_path),
                "source_type": self.source_type,
                "row_count": len(data),
                "column_count": len(data.columns) if hasattr(data, 'columns') else 0,
                "columns": list(data.columns) if hasattr(data, 'columns') else [],
                "validation_passed": len(validation_results["errors"]) == 0
            }
            context.set("data_metadata", metadata)
            
            # Save as artifact if artifact store is available
            if self.artifact_store:
                self._save_artifact(data, context)
            
            self.logger.info(f"Successfully ingested data: {metadata['row_count']} rows, "
                           f"{metadata['column_count']} columns")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in data ingestion: {str(e)}")
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after data ingestion."""
        self.logger.info(f"Tearing down {self.name} stage")
        
    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified source."""
        # Use resolved path for CSV files
        load_path = getattr(self, 'resolved_source_path', self.source_path)
        self.logger.info(f"Loading data from {load_path} (type: {self.source_type})")
        
        if self.source_type == "csv":
            data = pd.read_csv(load_path)
        elif self.source_type == "json":
            data = pd.read_json(self.source_path)
        elif self.source_type == "parquet":
            data = pd.read_parquet(self.source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
            
        # Apply sampling if specified
        if self.sample_size and len(data) > self.sample_size:
            self.logger.info(f"Sampling {self.sample_size} rows from {len(data)} total rows")
            data = data.sample(n=self.sample_size, random_state=42)
            
        return data
        
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic data validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check for empty dataset
        if len(data) == 0:
            results["errors"].append("Dataset is empty")
            
        # Check for missing columns
        required_columns = self.validation_rules.get("required_columns", [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            results["errors"].append(f"Missing required columns: {missing_columns}")
            
        # Check for null values
        null_counts = data.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if len(columns_with_nulls) > 0:
            results["warnings"].append(f"Columns with null values: {columns_with_nulls.to_dict()}")
            
        # Check data types
        expected_dtypes = self.validation_rules.get("expected_dtypes", {})
        for col, expected_dtype in expected_dtypes.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                if not actual_dtype.startswith(expected_dtype):
                    results["warnings"].append(
                        f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                    )
                    
        # Check for duplicates
        if self.validation_rules.get("check_duplicates", True):
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                results["warnings"].append(f"Found {duplicate_count} duplicate rows")
                
        return results
        
    def _save_artifact(self, data: pd.DataFrame, context: PipelineContext) -> None:
        """Save the ingested data as an artifact."""
        try:
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as tmp:
                temp_path = tmp.name
                
            # Save data to temporary file
            data.to_parquet(temp_path, index=False)
            
            # Save to artifact store
            run_id = context.get_metadata("run_id", "unknown")
            artifact_metadata = self.artifact_store.save_artifact(
                artifact_id=f"raw_data_{run_id}",
                artifact_path=temp_path,
                artifact_type=ArtifactType.DATASET,
                description="Raw ingested data",
                tags=["raw", "ingested", self.name]
            )
            
            # Store artifact reference in context
            context.set("raw_data_artifact", artifact_metadata.artifact_id)
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            self.logger.info(f"Saved raw data artifact: {artifact_metadata.artifact_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save artifact: {str(e)}")