"""Base interface and data models for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd


@dataclass
class DataSourceMetadata:
    """Metadata for a data source."""
    
    source_id: str
    source_type: str  # csv, api, database
    version: str
    created_at: datetime
    description: Optional[str] = None
    schema: Optional[Dict[str, str]] = None  # column_name -> data_type
    properties: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)  # Track data lineage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "schema": self.schema,
            "properties": self.properties,
            "lineage": self.lineage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSourceMetadata":
        """Create metadata from dictionary."""
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description"),
            schema=data.get("schema"),
            properties=data.get("properties", {}),
            lineage=data.get("lineage", {})
        )


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """Initialize data source.
        
        Args:
            source_id: Unique identifier for this data source
            config: Configuration dictionary for the data source
        """
        self.source_id = source_id
        self.config = config
        self._metadata: Optional[DataSourceMetadata] = None
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from the source.
        
        Args:
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame containing the loaded data
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the data source configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Get the schema of the data source.
        
        Returns:
            Dictionary mapping column names to data types
        """
        pass
    
    def get_metadata(self) -> DataSourceMetadata:
        """Get metadata for this data source.
        
        Returns:
            DataSourceMetadata object
        """
        if self._metadata is None:
            self._metadata = DataSourceMetadata(
                source_id=self.source_id,
                source_type=self.__class__.__name__,
                version="1.0.0",
                created_at=datetime.now(),
                schema=self.get_schema(),
                properties=self.config.copy()
            )
        return self._metadata
    
    def update_lineage(self, operation: str, details: Dict[str, Any]) -> None:
        """Update data lineage information.
        
        Args:
            operation: Name of the operation performed
            details: Details about the operation
        """
        metadata = self.get_metadata()
        if "operations" not in metadata.lineage:
            metadata.lineage["operations"] = []
        
        metadata.lineage["operations"].append({
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })