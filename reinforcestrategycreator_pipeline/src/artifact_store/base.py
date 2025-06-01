"""Base interface and data models for artifact storage."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


class ArtifactType(Enum):
    """Types of artifacts that can be stored."""
    MODEL = "model"
    DATASET = "dataset"
    REPORT = "report"
    CONFIG = "config"
    METRICS = "metrics"
    VISUALIZATION = "visualization"
    EVALUATION = "evaluation"
    OTHER = "other"


@dataclass
class ArtifactMetadata:
    """Metadata associated with an artifact."""
    
    artifact_id: str
    artifact_type: ArtifactType
    version: str
    created_at: datetime
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "properties": self.properties,
            "source_info": self.source_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        """Create metadata from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            description=data.get("description"),
            tags=data.get("tags", []),
            properties=data.get("properties", {}),
            source_info=data.get("source_info", {})
        )


class ArtifactStore(ABC):
    """Abstract base class for artifact storage implementations."""
    
    @abstractmethod
    def save_artifact(
        self,
        artifact_id: str,
        artifact_path: Union[str, Path],
        artifact_type: ArtifactType,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> ArtifactMetadata:
        """Save an artifact to the store.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_path: Path to the artifact file or directory
            artifact_type: Type of the artifact
            version: Version string (auto-generated if not provided)
            metadata: Additional metadata to store with the artifact
            tags: List of tags for categorization
            description: Human-readable description
            
        Returns:
            ArtifactMetadata object with complete artifact information
        """
        pass
    
    @abstractmethod
    def load_artifact(
        self,
        artifact_id: str,
        artifact_type: ArtifactType, # Added artifact_type
        version: Optional[str] = None,
        destination_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Load an artifact from the store.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of the artifact to load
            version: Specific version to load (latest if not provided)
            destination_path: Where to save the artifact (temp location if not provided)
            
        Returns:
            Path to the loaded artifact
        """
        pass
    
    @abstractmethod
    def get_artifact_metadata(
        self,
        artifact_id: str,
        artifact_type: ArtifactType, # Added artifact_type
        version: Optional[str] = None
    ) -> ArtifactMetadata:
        """Get metadata for an artifact.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of the artifact
            version: Specific version (latest if not provided)
            
        Returns:
            ArtifactMetadata object
        """
        pass
    
    @abstractmethod
    def list_artifacts(
        self,
        artifact_type_filter: Optional[ArtifactType] = None, # Renamed for clarity
        tags: Optional[List[str]] = None
    ) -> List[ArtifactMetadata]:
        """List artifacts in the store.
        
        Args:
            artifact_type_filter: Filter by artifact type
            tags: Filter by tags (artifacts must have all specified tags)
            
        Returns:
            List of ArtifactMetadata objects
        """
        pass
    
    @abstractmethod
    def list_versions(self, artifact_id: str, artifact_type: ArtifactType) -> List[str]: # Added artifact_type
        """List all versions of an artifact.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of the artifact
            
        Returns:
            List of version strings, sorted by creation time
        """
        pass
    
    @abstractmethod
    def delete_artifact(
        self,
        artifact_id: str,
        artifact_type: ArtifactType, # Added artifact_type
        version: Optional[str] = None
    ) -> bool:
        """Delete an artifact from the store.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of the artifact
            version: Specific version to delete (all versions if not provided)
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def artifact_exists(
        self,
        artifact_id: str,
        artifact_type: ArtifactType, # Added artifact_type
        version: Optional[str] = None
    ) -> bool:
        """Check if an artifact exists in the store.
        
        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of the artifact
            version: Specific version to check
            
        Returns:
            True if the artifact exists
        """
        pass