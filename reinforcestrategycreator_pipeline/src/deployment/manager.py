"""Deployment manager for handling model deployments."""

import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .packager import ModelPackager
from ..models.registry import ModelRegistry
from ..artifact_store.base import ArtifactStore, ArtifactType


class DeploymentStatus(Enum):
    """Status of a deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    DIRECT = "direct"


class DeploymentManager:
    """Manages model deployments across different environments.
    
    The deployment manager handles:
    - Deploying packaged models to target environments
    - Managing deployment configurations
    - Tracking deployment status
    - Implementing rollback capabilities
    - Managing multiple deployment versions
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        artifact_store: ArtifactStore,
        deployment_root: Optional[Union[str, Path]] = None
    ):
        """Initialize the deployment manager.
        
        Args:
            model_registry: Registry for accessing models
            artifact_store: Store for deployment artifacts
            deployment_root: Root directory for deployments (default: ./deployments)
        """
        self.model_registry = model_registry
        self.artifact_store = artifact_store
        self.packager = ModelPackager(model_registry, artifact_store)
        
        # Set deployment root directory
        if deployment_root is None:
            deployment_root = Path.cwd() / "deployments"
        self.deployment_root = Path(deployment_root)
        self.deployment_root.mkdir(parents=True, exist_ok=True)
        
        # Deployment state file
        self.state_file = self.deployment_root / "deployment_state.json"
        self.state = self._load_state()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def deploy(
        self,
        model_id: str,
        target_environment: str,
        model_version: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        strategy: Union[str, DeploymentStrategy] = DeploymentStrategy.ROLLING,
        package_id: Optional[str] = None,
        force: bool = False
    ) -> str:
        """Deploy a model to a target environment.
        
        Args:
            model_id: ID of the model to deploy
            target_environment: Target environment (e.g., "production", "staging")
            model_version: Specific model version to deploy
            deployment_config: Deployment-specific configuration
            strategy: Deployment strategy to use
            package_id: Use existing package instead of creating new one
            force: Force deployment even if same version already deployed
            
        Returns:
            Deployment ID
            
        Raises:
            ValueError: If deployment fails or validation errors occur
        """
        # Convert strategy to enum if string
        if isinstance(strategy, str):
            strategy = DeploymentStrategy(strategy)
        
        # Generate deployment ID
        deployment_id = f"deploy_{model_id}_{target_environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if model is already deployed in this environment
        current_deployment = self._get_current_deployment(model_id, target_environment)
        if current_deployment and not force:
            current_version = current_deployment.get("model_version")
            if current_version == model_version:
                raise ValueError(
                    f"Model {model_id} version {model_version} is already deployed "
                    f"to {target_environment}. Use force=True to redeploy."
                )
        
        # Initialize deployment record
        deployment_record = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "model_version": model_version,
            "target_environment": target_environment,
            "strategy": strategy.value,
            "status": DeploymentStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "deployment_config": deployment_config or {},
            "package_id": package_id,
            "previous_deployment": current_deployment
        }
        
        try:
            # Update status
            deployment_record["status"] = DeploymentStatus.DEPLOYING.value
            self._update_deployment_record(deployment_record)
            
            # Create or use existing package
            if package_id is None:
                self.logger.info(f"Creating deployment package for {model_id}")
                package_id = self.packager.package_model(
                    model_id=model_id,
                    model_version=model_version,
                    deployment_config=deployment_config,
                    tags=[target_environment, strategy.value],
                    description=f"Deployment to {target_environment}"
                )
            else:
                # Verify package exists
                if not self.artifact_store.artifact_exists(package_id):
                    raise ValueError(f"Package {package_id} not found")
            
            deployment_record["package_id"] = package_id
            
            # Execute deployment based on strategy
            deployment_path = self._execute_deployment(
                deployment_record=deployment_record,
                package_id=package_id,
                strategy=strategy
            )
            
            # Update deployment record
            deployment_record["status"] = DeploymentStatus.DEPLOYED.value
            deployment_record["deployed_at"] = datetime.now().isoformat()
            deployment_record["deployment_path"] = str(deployment_path)
            
            # Update current deployment for environment
            self._set_current_deployment(model_id, target_environment, deployment_record)
            
            self.logger.info(
                f"Successfully deployed {model_id} to {target_environment} "
                f"(deployment_id: {deployment_id})"
            )
            
            return deployment_id
            
        except Exception as e:
            # Update status to failed
            deployment_record["status"] = DeploymentStatus.FAILED.value
            deployment_record["error"] = str(e)
            self._update_deployment_record(deployment_record)
            
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def rollback(
        self,
        model_id: str,
        target_environment: str,
        to_deployment_id: Optional[str] = None
    ) -> str:
        """Rollback a deployment to a previous version.
        
        Args:
            model_id: Model ID
            target_environment: Target environment
            to_deployment_id: Specific deployment to rollback to (previous if None)
            
        Returns:
            Deployment ID of the rollback
            
        Raises:
            ValueError: If rollback fails or no previous deployment exists
        """
        # Get current deployment
        current_deployment = self._get_current_deployment(model_id, target_environment)
        if not current_deployment:
            raise ValueError(f"No current deployment found for {model_id} in {target_environment}")
        
        # Determine target deployment for rollback
        if to_deployment_id:
            # Verify deployment exists
            target_deployment = self._get_deployment_record(to_deployment_id)
            if not target_deployment:
                raise ValueError(f"Deployment {to_deployment_id} not found")
        else:
            # Use previous deployment
            previous_deployment = current_deployment.get("previous_deployment")
            if not previous_deployment:
                raise ValueError(f"No previous deployment found for rollback")
            target_deployment = previous_deployment
        
        # Create rollback deployment record
        rollback_id = f"rollback_{model_id}_{target_environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        rollback_record = {
            "deployment_id": rollback_id,
            "model_id": model_id,
            "model_version": target_deployment["model_version"],
            "target_environment": target_environment,
            "strategy": "direct",  # Rollbacks use direct strategy
            "status": DeploymentStatus.DEPLOYING.value,
            "created_at": datetime.now().isoformat(),
            "deployment_config": target_deployment.get("deployment_config", {}),
            "package_id": target_deployment["package_id"],
            "is_rollback": True,
            "rollback_from": current_deployment["deployment_id"],
            "rollback_to": target_deployment["deployment_id"]
        }
        
        try:
            # Execute rollback deployment
            deployment_path = self._execute_deployment(
                deployment_record=rollback_record,
                package_id=target_deployment["package_id"],
                strategy=DeploymentStrategy.DIRECT
            )
            
            # Update rollback record
            rollback_record["status"] = DeploymentStatus.DEPLOYED.value
            rollback_record["deployed_at"] = datetime.now().isoformat()
            rollback_record["deployment_path"] = str(deployment_path)
            
            # Update current deployment
            self._set_current_deployment(model_id, target_environment, rollback_record)
            
            # Mark original deployment as rolled back
            current_deployment["status"] = DeploymentStatus.ROLLED_BACK.value
            self._update_deployment_record(current_deployment)
            
            self.logger.info(
                f"Successfully rolled back {model_id} in {target_environment} "
                f"from {current_deployment['deployment_id']} to {target_deployment['deployment_id']}"
            )
            
            return rollback_id
            
        except Exception as e:
            rollback_record["status"] = DeploymentStatus.FAILED.value
            rollback_record["error"] = str(e)
            self._update_deployment_record(rollback_record)
            raise
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get the status of a deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment status information
        """
        deployment = self._get_deployment_record(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        return {
            "deployment_id": deployment_id,
            "status": deployment["status"],
            "model_id": deployment["model_id"],
            "model_version": deployment["model_version"],
            "target_environment": deployment["target_environment"],
            "created_at": deployment["created_at"],
            "deployed_at": deployment.get("deployed_at"),
            "is_rollback": deployment.get("is_rollback", False),
            "error": deployment.get("error")
        }
    
    def list_deployments(
        self,
        model_id: Optional[str] = None,
        target_environment: Optional[str] = None,
        status: Optional[Union[str, DeploymentStatus]] = None,
        include_rollbacks: bool = True
    ) -> List[Dict[str, Any]]:
        """List deployments with optional filters.
        
        Args:
            model_id: Filter by model ID
            target_environment: Filter by environment
            status: Filter by status
            include_rollbacks: Whether to include rollback deployments
            
        Returns:
            List of deployment records
        """
        deployments = []
        
        # Convert status to string if enum
        if isinstance(status, DeploymentStatus):
            status = status.value
        
        # Filter deployments
        for deployment_id, deployment in self.state.get("deployments", {}).items():
            # Apply filters
            if model_id and deployment["model_id"] != model_id:
                continue
            if target_environment and deployment["target_environment"] != target_environment:
                continue
            if status and deployment["status"] != status:
                continue
            if not include_rollbacks and deployment.get("is_rollback", False):
                continue
            
            deployments.append(deployment)
        
        # Sort by creation time (newest first)
        deployments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return deployments
    
    def get_current_deployment(
        self,
        model_id: str,
        target_environment: str
    ) -> Optional[Dict[str, Any]]:
        """Get the current active deployment for a model in an environment.
        
        Args:
            model_id: Model ID
            target_environment: Target environment
            
        Returns:
            Current deployment record or None
        """
        return self._get_current_deployment(model_id, target_environment)
    
    def _execute_deployment(
        self,
        deployment_record: Dict[str, Any],
        package_id: str,
        strategy: DeploymentStrategy
    ) -> Path:
        """Execute the actual deployment.
        
        Args:
            deployment_record: Deployment record
            package_id: Package to deploy
            strategy: Deployment strategy
            
        Returns:
            Path to deployed model
        """
        # Create deployment directory
        env_dir = self.deployment_root / deployment_record["target_environment"]
        model_dir = env_dir / deployment_record["model_id"]
        deployment_dir = model_dir / deployment_record["deployment_id"]
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract package
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load package from artifact store
            package_path = self.artifact_store.load_artifact(
                artifact_id=package_id,
                artifact_type=ArtifactType.OTHER,  # Added missing artifact_type
                destination_path=temp_dir
            )
            self.logger.info(f"Package path from load_artifact: {package_path}") # DEBUG LOG
            self.logger.info(f"Is package_path a directory? {Path(package_path).is_dir()}") # DEBUG LOG
            self.logger.info(f"Is package_path a file? {Path(package_path).is_file()}") # DEBUG LOG
            
            # Extract tarball
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(deployment_dir)
            
            # Find the extracted package directory
            package_dirs = list(deployment_dir.iterdir())
            if len(package_dirs) != 1:
                raise ValueError(f"Expected single directory in package, found {len(package_dirs)}")
            
            extracted_dir = package_dirs[0]
            
            # Apply deployment strategy
            if strategy == DeploymentStrategy.DIRECT:
                # Direct deployment - just use extracted directory
                final_deployment_path = extracted_dir
            elif strategy == DeploymentStrategy.ROLLING:
                # Rolling deployment - create symlink for zero-downtime switch
                current_link = model_dir / "current"
                new_link = model_dir / "new"
                
                # Create new symlink
                if new_link.exists():
                    new_link.unlink()
                new_link.symlink_to(extracted_dir)
                
                # Atomic switch
                if current_link.exists():
                    current_link.unlink()
                current_link.symlink_to(extracted_dir)
                
                # Clean up new link
                new_link.unlink()
                
                final_deployment_path = current_link
            else:
                # Other strategies not implemented yet
                raise NotImplementedError(f"Strategy {strategy} not implemented")
            
            # Create deployment info file
            info_file = deployment_dir / "deployment_info.json"
            with open(info_file, "w") as f:
                json.dump({
                    "deployment_id": deployment_record["deployment_id"],
                    "deployed_at": datetime.now().isoformat(),
                    "package_id": package_id,
                    "strategy": strategy.value
                }, f, indent=2)
            
            return final_deployment_path
    
    def _get_current_deployment(self, model_id: str, target_environment: str) -> Optional[Dict[str, Any]]:
        """Get current deployment for a model in an environment."""
        key = f"{model_id}:{target_environment}"
        deployment_id = self.state.get("current_deployments", {}).get(key)
        if deployment_id:
            return self._get_deployment_record(deployment_id)
        return None
    
    def _set_current_deployment(self, model_id: str, target_environment: str, deployment: Dict[str, Any]) -> None:
        """Set current deployment for a model in an environment."""
        key = f"{model_id}:{target_environment}"
        if "current_deployments" not in self.state:
            self.state["current_deployments"] = {}
        self.state["current_deployments"][key] = deployment["deployment_id"]
        self._update_deployment_record(deployment)
    
    def _get_deployment_record(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get a deployment record by ID."""
        return self.state.get("deployments", {}).get(deployment_id)
    
    def _update_deployment_record(self, deployment: Dict[str, Any]) -> None:
        """Update a deployment record and save state."""
        if "deployments" not in self.state:
            self.state["deployments"] = {}
        self.state["deployments"][deployment["deployment_id"]] = deployment
        self._save_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load deployment state from file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"deployments": {}, "current_deployments": {}}
    
    def _save_state(self) -> None:
        """Save deployment state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)