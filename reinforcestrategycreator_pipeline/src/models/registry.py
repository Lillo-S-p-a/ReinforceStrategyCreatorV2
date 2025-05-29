"""Model registry for tracking model versions and metadata."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import uuid

from ..artifact_store.base import ArtifactStore, ArtifactType, ArtifactMetadata
from .base import ModelBase
from .factory import ModelFactory, get_factory


class ModelRegistry:
    """Registry for tracking model versions and metadata.
    
    The registry uses an ArtifactStore for persistence and provides
    functionality for versioning, metadata tracking, and model lineage.
    """
    
    def __init__(self, artifact_store: ArtifactStore):
        """Initialize the model registry.
        
        Args:
            artifact_store: ArtifactStore instance for persistence
        """
        self.artifact_store = artifact_store
        self.factory = get_factory()
    
    def register_model(
        self,
        model: ModelBase,
        model_name: str,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        parent_model_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a model in the registry.
        
        Args:
            model: Model instance to register
            model_name: Name for the model
            version: Version string (auto-generated if not provided)
            tags: List of tags for categorization
            description: Human-readable description
            parent_model_id: ID of parent model (for lineage tracking)
            metrics: Performance metrics from evaluation
            dataset_info: Information about training dataset
            additional_metadata: Any additional metadata to store
            
        Returns:
            Unique model ID for the registered model
        """
        # Generate model ID
        model_id = f"model_{model_name}_{uuid.uuid4().hex[:8]}"
        
        # Auto-generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"
        
        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "model_type": model.model_type,
            "hyperparameters": model.get_hyperparameters(),
            "is_trained": model.is_trained,
            "parent_model_id": parent_model_id,
            "metrics": metrics or {},
            "dataset_info": dataset_info or {},
            "registration_time": datetime.now().isoformat()
        }
        
        # Add model-specific metadata
        metadata.update(model.get_metadata())
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create temporary directory for model files
        temp_dir = Path(f"/tmp/{model_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model to temporary directory
            model.save(temp_dir)
            
            # Save model artifacts to artifact store
            artifact_metadata = self.artifact_store.save_artifact(
                artifact_id=model_id,
                artifact_path=temp_dir,
                artifact_type=ArtifactType.MODEL,
                version=version,
                metadata=metadata,
                tags=tags or [],
                description=description
            )
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
            return model_id
            
        except Exception as e:
            # Clean up on error
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            raise e
    
    def load_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        destination_path: Optional[Union[str, Path]] = None
    ) -> ModelBase:
        """Load a model from the registry.
        
        Args:
            model_id: Unique model ID
            version: Specific version to load (latest if not provided)
            destination_path: Where to extract model files
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model not found or cannot be loaded
        """
        # Load model artifacts from store
        model_path = self.artifact_store.load_artifact(
            artifact_id=model_id,
            version=version,
            destination_path=destination_path
        )
        
        # Get metadata to determine model type
        metadata = self.artifact_store.get_artifact_metadata(
            artifact_id=model_id,
            version=version
        )
        
        model_type = metadata.properties.get("model_type")
        if not model_type:
            raise ValueError(f"Model type not found in metadata for {model_id}")
        
        # Load configuration
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model instance
        model = self.factory.create_model(model_type, config)
        
        # Load model state
        model.load(model_path)
        
        return model
    
    def get_model_metadata(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metadata for a model.
        
        Args:
            model_id: Unique model ID
            version: Specific version (latest if not provided)
            
        Returns:
            Dictionary containing model metadata
        """
        artifact_metadata = self.artifact_store.get_artifact_metadata(
            artifact_id=model_id,
            version=version
        )
        
        return {
            "model_id": artifact_metadata.artifact_id,
            "version": artifact_metadata.version,
            "created_at": artifact_metadata.created_at.isoformat(),
            "tags": artifact_metadata.tags,
            "description": artifact_metadata.description,
            **artifact_metadata.properties
        }
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List models in the registry.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            tags: Filter by tags
            parent_model_id: Filter by parent model
            
        Returns:
            List of model metadata dictionaries
        """
        # Get all model artifacts
        all_models = self.artifact_store.list_artifacts(
            artifact_type=ArtifactType.MODEL,
            tags=tags
        )
        
        # Apply filters
        filtered_models = []
        for artifact_metadata in all_models:
            props = artifact_metadata.properties
            
            # Apply model_name filter
            if model_name and props.get("model_name") != model_name:
                continue
            
            # Apply model_type filter
            if model_type and props.get("model_type") != model_type:
                continue
            
            # Apply parent_model_id filter
            if parent_model_id and props.get("parent_model_id") != parent_model_id:
                continue
            
            # Convert to dict and add to results
            filtered_models.append({
                "model_id": artifact_metadata.artifact_id,
                "version": artifact_metadata.version,
                "created_at": artifact_metadata.created_at.isoformat(),
                "tags": artifact_metadata.tags,
                "description": artifact_metadata.description,
                **props
            })
        
        # Sort by creation time (newest first)
        filtered_models.sort(
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        return filtered_models
    
    def list_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List all versions of a model.
        
        Args:
            model_id: Unique model ID
            
        Returns:
            List of version metadata dictionaries
        """
        versions = self.artifact_store.list_versions(model_id)
        
        version_metadata = []
        for version in versions:
            try:
                metadata = self.get_model_metadata(model_id, version)
                version_metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Could not load metadata for {model_id} v{version}: {e}")
        
        return version_metadata
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get the lineage of a model (parent/child relationships).
        
        Args:
            model_id: Unique model ID
            
        Returns:
            Dictionary containing lineage information
        """
        # Get model metadata
        metadata = self.get_model_metadata(model_id)
        
        lineage = {
            "model_id": model_id,
            "parent_model_id": metadata.get("parent_model_id"),
            "children": []
        }
        
        # Find children (models that have this model as parent)
        all_models = self.list_models(parent_model_id=model_id)
        lineage["children"] = [
            {
                "model_id": child["model_id"],
                "version": child["version"],
                "created_at": child["created_at"]
            }
            for child in all_models
        ]
        
        # Trace back to root parent
        ancestors = []
        current_parent = metadata.get("parent_model_id")
        while current_parent:
            try:
                parent_metadata = self.get_model_metadata(current_parent)
                ancestors.append({
                    "model_id": current_parent,
                    "version": parent_metadata["version"],
                    "created_at": parent_metadata["created_at"]
                })
                current_parent = parent_metadata.get("parent_model_id")
            except Exception:
                break
        
        lineage["ancestors"] = ancestors
        
        return lineage
    
    def delete_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Delete a model from the registry.
        
        Args:
            model_id: Unique model ID
            version: Specific version to delete (all versions if not provided)
            
        Returns:
            True if deletion was successful
        """
        return self.artifact_store.delete_artifact(
            artifact_id=model_id,
            version=version
        )
    
    def model_exists(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Check if a model exists in the registry.
        
        Args:
            model_id: Unique model ID
            version: Specific version to check
            
        Returns:
            True if the model exists
        """
        return self.artifact_store.artifact_exists(
            artifact_id=model_id,
            version=version
        )
    
    def compare_models(
        self,
        model_ids: List[str],
        versions: Optional[List[Optional[str]]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models side by side.
        
        Args:
            model_ids: List of model IDs to compare
            versions: List of versions (parallel to model_ids)
            
        Returns:
            Dictionary containing comparison data
        """
        if versions is None:
            versions = [None] * len(model_ids)
        
        comparison = {
            "models": [],
            "metrics_comparison": {},
            "hyperparameters_comparison": {}
        }
        
        all_metrics = set()
        all_hyperparams = set()
        
        # Gather data for each model
        for model_id, version in zip(model_ids, versions):
            try:
                metadata = self.get_model_metadata(model_id, version)
                comparison["models"].append(metadata)
                
                # Collect all metric names
                metrics = metadata.get("metrics", {})
                all_metrics.update(metrics.keys())
                
                # Collect all hyperparameter names
                hyperparams = metadata.get("hyperparameters", {})
                all_hyperparams.update(hyperparams.keys())
                
            except Exception as e:
                print(f"Warning: Could not load {model_id}: {e}")
        
        # Build comparison tables
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = []
            for model_data in comparison["models"]:
                value = model_data.get("metrics", {}).get(metric, None)
                comparison["metrics_comparison"][metric].append(value)
        
        for param in all_hyperparams:
            comparison["hyperparameters_comparison"][param] = []
            for model_data in comparison["models"]:
                value = model_data.get("hyperparameters", {}).get(param, None)
                comparison["hyperparameters_comparison"][param].append(value)
        
        return comparison
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ModelRegistry(artifact_store={self.artifact_store})"