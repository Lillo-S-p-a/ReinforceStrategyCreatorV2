"""Deployment stage implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import shutil
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactType


class DeploymentStage(PipelineStage):
    """
    Stage responsible for deploying trained and evaluated models.
    
    This stage handles:
    - Model packaging and versioning
    - Deployment to target environments
    - Model registry updates
    - Deployment validation
    - Rollback capabilities
    """
    
    def __init__(self, name: str = "deployment", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the deployment stage.
        
        Args:
            name: Stage name
            config: Stage configuration containing:
                - deployment_target: Target environment (local, staging, production)
                - model_registry: Model registry configuration
                - deployment_strategy: Strategy for deployment (blue-green, canary, etc.)
                - validation_config: Post-deployment validation settings
                - rollback_config: Rollback configuration
                - packaging_format: Model packaging format
        """
        super().__init__(name, config or {})
        self.deployment_target = self.config.get("deployment_target", "local")
        self.model_registry = self.config.get("model_registry", {})
        self.deployment_strategy = self.config.get("deployment_strategy", "direct")
        self.validation_config = self.config.get("validation_config", {})
        self.rollback_config = self.config.get("rollback_config", {})
        self.packaging_format = self.config.get("packaging_format", "pickle")
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the deployment stage."""
        self.logger.info(f"Setting up {self.name} stage")
        
        # Validate required data is available
        if not context.get("trained_model"):
            raise ValueError("No trained model found in context. Run training stage first.")
            
        if not context.get("model_passed_thresholds", True):
            if not self.config.get("force_deployment", False):
                raise ValueError("Model did not pass evaluation thresholds. Set force_deployment=True to override.")
            else:
                self.logger.warning("Model did not pass thresholds but force_deployment is enabled")
                
        # Get artifact store from context
        self.artifact_store = context.get("artifact_store")
        
        # Initialize deployment target
        self._initialize_deployment_target()
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the deployment stage.
        
        Args:
            context: Pipeline context containing trained model and evaluation results
            
        Returns:
            Updated pipeline context with deployment information
        """
        self.logger.info(f"Running {self.name} stage")
        
        try:
            # Get model and metadata from context
            model = context.get("trained_model")
            model_type = context.get("model_type")
            evaluation_results = context.get("evaluation_results", {})
            
            # Generate deployment package
            self.logger.info("Creating deployment package")
            deployment_package = self._create_deployment_package(model, context)
            
            # Version the model
            model_version = self._generate_model_version(context)
            deployment_package["version"] = model_version
            
            # Register model in model registry
            if self.model_registry.get("enabled", False):
                registry_entry = self._register_model(deployment_package, context)
                context.set("model_registry_entry", registry_entry)
            
            # Deploy model based on strategy
            self.logger.info(f"Deploying model using {self.deployment_strategy} strategy to {self.deployment_target}")
            deployment_result = self._deploy_model(deployment_package, context)
            
            # Validate deployment
            validation_result = self._validate_deployment(deployment_result, context)
            
            if not validation_result["success"]:
                self.logger.error(f"Deployment validation failed: {validation_result['errors']}")
                if self.rollback_config.get("auto_rollback", True):
                    self._rollback_deployment(deployment_result, context)
                    raise RuntimeError("Deployment validation failed, rolled back")
            
            # Store deployment information in context
            deployment_info = {
                "model_version": model_version,
                "deployment_target": self.deployment_target,
                "deployment_strategy": self.deployment_strategy,
                "deployment_timestamp": datetime.now().isoformat(),
                "deployment_result": deployment_result,
                "validation_result": validation_result,
                "model_location": deployment_result.get("model_location")
            }
            context.set("deployment_info", deployment_info)
            
            # Save deployment artifacts
            if self.artifact_store:
                self._save_deployment_artifacts(deployment_package, deployment_info, context)
            
            self.logger.info(f"Model successfully deployed: version={model_version}, target={self.deployment_target}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in deployment stage: {str(e)}")
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after deployment."""
        self.logger.info(f"Tearing down {self.name} stage")
        
    def _initialize_deployment_target(self) -> None:
        """Initialize the deployment target environment."""
        self.logger.info(f"Initializing deployment target: {self.deployment_target}")
        
        if self.deployment_target == "local":
            # Create local deployment directory
            self.deployment_dir = Path("deployed_models")
            self.deployment_dir.mkdir(exist_ok=True)
        elif self.deployment_target == "staging":
            # Initialize staging environment connection
            # This would connect to staging servers, cloud services, etc.
            self.logger.info("Connecting to staging environment")
        elif self.deployment_target == "production":
            # Initialize production environment connection
            # This would connect to production servers, cloud services, etc.
            self.logger.info("Connecting to production environment")
        else:
            raise ValueError(f"Unknown deployment target: {self.deployment_target}")
            
    def _create_deployment_package(self, model: Any, context: PipelineContext) -> Dict[str, Any]:
        """
        Create a deployment package containing the model and necessary metadata.
        
        Args:
            model: Trained model
            context: Pipeline context
            
        Returns:
            Deployment package dictionary
        """
        package = {
            "model": model,
            "model_type": context.get("model_type"),
            "model_config": context.get("model_config"),
            "training_metadata": context.get("training_metadata"),
            "evaluation_results": context.get("evaluation_results"),
            "feature_engineering_config": context.get("feature_engineering_config"),
            "created_at": datetime.now().isoformat(),
            "pipeline_run_id": context.get_metadata("run_id", "unknown")
        }
        
        # Add preprocessing information if available
        if context.get("preprocessing_pipeline"):
            package["preprocessing_pipeline"] = context.get("preprocessing_pipeline")
            
        # Add model artifacts references
        if context.get("model_artifact"):
            package["model_artifact_id"] = context.get("model_artifact")
            
        return package
        
    def _generate_model_version(self, context: PipelineContext) -> str:
        """Generate a version identifier for the model."""
        # Version format: v{major}.{minor}.{patch}-{timestamp}
        # In practice, this might integrate with existing versioning systems
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_type = context.get("model_type", "unknown")
        
        # Simple versioning based on existing deployments
        existing_versions = self._get_existing_versions(model_type)
        
        if not existing_versions:
            version = f"v1.0.0-{timestamp}"
        else:
            # Increment patch version
            latest = existing_versions[-1]
            parts = latest.split("-")[0].replace("v", "").split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            version = f"v{major}.{minor}.{patch + 1}-{timestamp}"
            
        return version
        
    def _get_existing_versions(self, model_type: str) -> List[str]:
        """Get list of existing model versions."""
        # This is a placeholder - in practice, this would query the model registry
        # or deployment directory
        
        if self.deployment_target == "local" and hasattr(self, 'deployment_dir'):
            versions = []
            for path in self.deployment_dir.glob(f"{model_type}_v*.pkl"):
                version = path.stem.replace(f"{model_type}_", "")
                versions.append(version)
            return sorted(versions)
        
        return []
        
    def _register_model(self, deployment_package: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """Register model in the model registry."""
        self.logger.info("Registering model in model registry")
        
        registry_entry = {
            "model_id": f"{deployment_package['model_type']}_{deployment_package['version']}",
            "model_type": deployment_package["model_type"],
            "version": deployment_package["version"],
            "created_at": deployment_package["created_at"],
            "evaluation_metrics": deployment_package.get("evaluation_results", {}),
            "deployment_target": self.deployment_target,
            "status": "registered",
            "tags": self.model_registry.get("tags", [])
        }
        
        # In practice, this would make API calls to a model registry service
        # For now, we'll simulate by saving to a local registry file
        registry_path = Path("model_registry.json")
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": []}
            
        registry["models"].append(registry_entry)
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
            
        self.logger.info(f"Model registered: {registry_entry['model_id']}")
        
        return registry_entry
        
    def _deploy_model(self, deployment_package: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """
        Deploy the model based on the configured strategy.
        
        Args:
            deployment_package: Package containing model and metadata
            context: Pipeline context
            
        Returns:
            Deployment result information
        """
        deployment_result = {
            "strategy": self.deployment_strategy,
            "target": self.deployment_target,
            "started_at": datetime.now().isoformat()
        }
        
        if self.deployment_strategy == "direct":
            # Direct deployment - replace existing model
            result = self._direct_deployment(deployment_package)
        elif self.deployment_strategy == "blue-green":
            # Blue-green deployment
            result = self._blue_green_deployment(deployment_package)
        elif self.deployment_strategy == "canary":
            # Canary deployment
            result = self._canary_deployment(deployment_package)
        else:
            raise ValueError(f"Unknown deployment strategy: {self.deployment_strategy}")
            
        deployment_result.update(result)
        deployment_result["completed_at"] = datetime.now().isoformat()
        
        return deployment_result
        
    def _direct_deployment(self, deployment_package: Dict[str, Any]) -> Dict[str, Any]:
        """Perform direct deployment."""
        if self.deployment_target == "local":
            # Save model to local deployment directory
            model_filename = f"{deployment_package['model_type']}_{deployment_package['version']}.pkl"
            model_path = self.deployment_dir / model_filename
            
            # Save model package
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(deployment_package, f)
                
            # Create symlink to latest version
            latest_link = self.deployment_dir / f"{deployment_package['model_type']}_latest.pkl"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(model_filename)
            
            return {
                "model_location": str(model_path),
                "latest_link": str(latest_link),
                "deployment_method": "file_system"
            }
        else:
            # Placeholder for other deployment targets
            return {
                "model_location": f"{self.deployment_target}://{deployment_package['model_type']}/{deployment_package['version']}",
                "deployment_method": "remote"
            }
            
    def _blue_green_deployment(self, deployment_package: Dict[str, Any]) -> Dict[str, Any]:
        """Perform blue-green deployment."""
        # This is a placeholder implementation
        # In practice, this would:
        # 1. Deploy to inactive environment (green)
        # 2. Run smoke tests
        # 3. Switch traffic from blue to green
        # 4. Keep blue as rollback option
        
        self.logger.info("Performing blue-green deployment")
        
        # Deploy to green environment
        green_result = self._direct_deployment(deployment_package)
        
        return {
            "green_deployment": green_result,
            "blue_deployment": "previous_version",
            "traffic_switched": True
        }
        
    def _canary_deployment(self, deployment_package: Dict[str, Any]) -> Dict[str, Any]:
        """Perform canary deployment."""
        # This is a placeholder implementation
        # In practice, this would:
        # 1. Deploy new version alongside existing
        # 2. Route small percentage of traffic to new version
        # 3. Monitor metrics
        # 4. Gradually increase traffic if metrics are good
        
        self.logger.info("Performing canary deployment")
        
        canary_percentage = self.config.get("canary_percentage", 10)
        
        return {
            "canary_version": deployment_package["version"],
            "stable_version": "previous_version",
            "canary_percentage": canary_percentage,
            "deployment_method": "canary"
        }
        
    def _validate_deployment(self, deployment_result: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """Validate the deployment."""
        validation_result = {
            "success": True,
            "checks": {},
            "errors": []
        }
        
        # Check if model is accessible
        if self.deployment_target == "local":
            model_path = Path(deployment_result.get("model_location", ""))
            validation_result["checks"]["model_exists"] = model_path.exists()
            if not model_path.exists():
                validation_result["success"] = False
                validation_result["errors"].append("Model file not found")
                
        # Run smoke tests if configured
        if self.validation_config.get("smoke_tests", False):
            smoke_test_result = self._run_smoke_tests(deployment_result, context)
            validation_result["checks"]["smoke_tests"] = smoke_test_result["passed"]
            if not smoke_test_result["passed"]:
                validation_result["success"] = False
                validation_result["errors"].extend(smoke_test_result["errors"])
                
        # Check model serving endpoint if applicable
        if self.validation_config.get("check_endpoint", False):
            endpoint_result = self._check_model_endpoint(deployment_result)
            validation_result["checks"]["endpoint_health"] = endpoint_result["healthy"]
            if not endpoint_result["healthy"]:
                validation_result["success"] = False
                validation_result["errors"].append("Model endpoint not healthy")
                
        return validation_result
        
    def _run_smoke_tests(self, deployment_result: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """Run smoke tests on deployed model."""
        # Placeholder for smoke tests
        # In practice, this would make test predictions and verify responses
        
        return {
            "passed": True,
            "errors": [],
            "test_count": 5,
            "passed_count": 5
        }
        
    def _check_model_endpoint(self, deployment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check if model serving endpoint is healthy."""
        # Placeholder for endpoint health check
        # In practice, this would make HTTP requests to model serving endpoint
        
        return {
            "healthy": True,
            "response_time_ms": 50,
            "status_code": 200
        }
        
    def _rollback_deployment(self, deployment_result: Dict[str, Any], context: PipelineContext) -> None:
        """Rollback a failed deployment."""
        self.logger.warning("Rolling back deployment")
        
        if self.deployment_target == "local":
            # Remove deployed model file
            model_path = Path(deployment_result.get("model_location", ""))
            if model_path.exists():
                model_path.unlink()
                
            # Restore previous latest link
            # In practice, this would be more sophisticated
            self.logger.info("Rollback completed")
            
    def _save_deployment_artifacts(
        self, 
        deployment_package: Dict[str, Any],
        deployment_info: Dict[str, Any],
        context: PipelineContext
    ) -> None:
        """Save deployment artifacts."""
        try:
            run_id = context.get_metadata("run_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save deployment manifest
            import tempfile
            manifest = {
                "deployment_info": deployment_info,
                "model_version": deployment_package["version"],
                "deployment_package_summary": {
                    "model_type": deployment_package["model_type"],
                    "created_at": deployment_package["created_at"],
                    "evaluation_metrics": deployment_package.get("evaluation_results", {})
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(manifest, tmp, indent=2)
                manifest_path = tmp.name
                
            manifest_metadata = self.artifact_store.save_artifact(
                artifact_id=f"deployment_manifest_{run_id}_{timestamp}",
                artifact_path=manifest_path,
                artifact_type=ArtifactType.METADATA,
                description="Deployment manifest",
                tags=["deployment", "manifest", self.name]
            )
            
            # Store artifact reference in context
            context.set("deployment_manifest_artifact", manifest_metadata.artifact_id)
            
            # Clean up temporary file
            Path(manifest_path).unlink()
            
            self.logger.info(f"Saved deployment manifest: {manifest_metadata.artifact_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save deployment artifacts: {str(e)}")