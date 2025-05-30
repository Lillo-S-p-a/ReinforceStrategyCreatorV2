"""Local file system implementation of the ArtifactStore interface."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import os
import numpy as np

from .base import ArtifactStore, ArtifactMetadata, ArtifactType


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class LocalFileSystemStore(ArtifactStore):
    """Local file system implementation of artifact storage.
    
    Directory structure:
    artifacts_root/
    ├── {artifact_id}/
    │   ├── versions/
    │   │   ├── {version}/
    │   │   │   ├── data/          # The actual artifact files
    │   │   │   └── metadata.json  # Version metadata
    │   └── latest.json            # Points to latest version
    """
    
    def __init__(self, root_path: Union[str, Path]):
        """Initialize the local file system store.
        
        Args:
            root_path: Root directory for storing artifacts
        """
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
    
    def _get_artifact_path(self, artifact_id: str) -> Path:
        """Get the base path for an artifact."""
        return self.root_path / artifact_id
    
    def _get_version_path(self, artifact_id: str, version: str) -> Path:
        """Get the path for a specific version of an artifact."""
        return self._get_artifact_path(artifact_id) / "versions" / version
    
    def _get_data_path(self, artifact_id: str, version: str) -> Path:
        """Get the data path for a specific version."""
        return self._get_version_path(artifact_id, version) / "data"
    
    def _get_metadata_path(self, artifact_id: str, version: str) -> Path:
        """Get the metadata path for a specific version."""
        return self._get_version_path(artifact_id, version) / "metadata.json"
    
    def _get_latest_path(self, artifact_id: str) -> Path:
        """Get the path to the latest version file."""
        return self._get_artifact_path(artifact_id) / "latest.json"
    
    def _generate_version(self) -> str:
        """Generate a version string based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _update_latest_version(self, artifact_id: str, version: str) -> None:
        """Update the latest version pointer."""
        latest_path = self._get_latest_path(artifact_id)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_path, 'w') as f:
            json.dump({"version": version}, f)
    
    def _get_latest_version(self, artifact_id: str) -> Optional[str]:
        """Get the latest version of an artifact."""
        latest_path = self._get_latest_path(artifact_id)
        if latest_path.exists():
            with open(latest_path, 'r') as f:
                return json.load(f)["version"]
        
        # Fallback: find the most recent version
        versions_dir = self._get_artifact_path(artifact_id) / "versions"
        if versions_dir.exists():
            versions = sorted(versions_dir.iterdir(), key=lambda p: p.name)
            if versions:
                return versions[-1].name
        return None
    
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
        """Save an artifact to the store."""
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version()
        
        # Create artifact metadata
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            version=version,
            created_at=datetime.now(),
            created_by=os.environ.get("USER", "unknown"),
            description=description,
            tags=tags or [],
            properties=metadata or {},
            source_info={
                "original_path": str(artifact_path),
                "is_directory": artifact_path.is_dir()
            }
        )
        
        # Create version directory
        version_path = self._get_version_path(artifact_id, version)
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Copy artifact data
        data_path = self._get_data_path(artifact_id, version)
        if artifact_path.is_dir():
            shutil.copytree(artifact_path, data_path, dirs_exist_ok=True)
        else:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, data_path)
        
        # Save metadata
        metadata_path = self._get_metadata_path(artifact_id, version)
        with open(metadata_path, 'w') as f:
            json.dump(artifact_metadata.to_dict(), f, indent=2, cls=NumpyEncoder)
        
        # Update latest version
        self._update_latest_version(artifact_id, version)
        
        return artifact_metadata
    
    def load_artifact(
        self,
        artifact_id: str,
        version: Optional[str] = None,
        destination_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Load an artifact from the store."""
        # Get version
        if version is None:
            version = self._get_latest_version(artifact_id)
            if version is None:
                raise ValueError(f"No versions found for artifact: {artifact_id}")
        
        # Check if version exists
        data_path = self._get_data_path(artifact_id, version)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {artifact_id} version {version}"
            )
        
        # Determine destination
        if destination_path is None:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix=f"{artifact_id}_{version}_")
            destination_path = Path(temp_dir) / data_path.name
        else:
            destination_path = Path(destination_path)
        
        # Copy artifact
        if data_path.is_dir():
            shutil.copytree(data_path, destination_path, dirs_exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_path, destination_path)
        
        return destination_path
    
    def get_artifact_metadata(
        self,
        artifact_id: str,
        version: Optional[str] = None
    ) -> ArtifactMetadata:
        """Get metadata for an artifact."""
        # Get version
        if version is None:
            version = self._get_latest_version(artifact_id)
            if version is None:
                raise ValueError(f"No versions found for artifact: {artifact_id}")
        
        # Load metadata
        metadata_path = self._get_metadata_path(artifact_id, version)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found for artifact: {artifact_id} version {version}"
            )
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return ArtifactMetadata.from_dict(metadata_dict)
    
    def list_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None,
        tags: Optional[List[str]] = None
    ) -> List[ArtifactMetadata]:
        """List artifacts in the store."""
        artifacts = []
        
        # Iterate through all artifact directories
        for artifact_dir in self.root_path.iterdir():
            if not artifact_dir.is_dir():
                continue
            
            # Get latest version metadata
            try:
                latest_version = self._get_latest_version(artifact_dir.name)
                if latest_version:
                    metadata = self.get_artifact_metadata(
                        artifact_dir.name, latest_version
                    )
                    
                    # Apply filters
                    if artifact_type and metadata.artifact_type != artifact_type:
                        continue
                    
                    if tags and not all(tag in metadata.tags for tag in tags):
                        continue
                    
                    artifacts.append(metadata)
            except Exception:
                # Skip artifacts with issues
                continue
        
        # Sort by creation time
        artifacts.sort(key=lambda m: m.created_at, reverse=True)
        return artifacts
    
    def list_versions(self, artifact_id: str) -> List[str]:
        """List all versions of an artifact."""
        versions_dir = self._get_artifact_path(artifact_id) / "versions"
        if not versions_dir.exists():
            return []
        
        versions = []
        for version_dir in versions_dir.iterdir():
            if version_dir.is_dir():
                versions.append(version_dir.name)
        
        # Sort by version name (timestamp format ensures chronological order)
        return sorted(versions)
    
    def delete_artifact(
        self,
        artifact_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Delete an artifact from the store."""
        artifact_path = self._get_artifact_path(artifact_id)
        
        if version is None:
            # Delete entire artifact
            if artifact_path.exists():
                shutil.rmtree(artifact_path)
                return True
            return False
        else:
            # Delete specific version
            version_path = self._get_version_path(artifact_id, version)
            if version_path.exists():
                shutil.rmtree(version_path)
                
                # Update latest version if needed
                latest_version = self._get_latest_version(artifact_id)
                if latest_version == version:
                    # Find new latest version
                    remaining_versions = self.list_versions(artifact_id)
                    if remaining_versions:
                        self._update_latest_version(
                            artifact_id, remaining_versions[-1]
                        )
                    else:
                        # No versions left, remove latest.json
                        latest_path = self._get_latest_path(artifact_id)
                        if latest_path.exists():
                            latest_path.unlink()
                
                return True
            return False
    
    def artifact_exists(
        self,
        artifact_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Check if an artifact exists in the store."""
        if version is None:
            # Check if any version exists
            artifact_path = self._get_artifact_path(artifact_id)
            return artifact_path.exists() and any(
                (artifact_path / "versions").iterdir()
            ) if (artifact_path / "versions").exists() else False
        else:
            # Check specific version
            version_path = self._get_version_path(artifact_id, version)
            return version_path.exists()