"""Example script demonstrating model deployment functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment import DeploymentManager, DeploymentStatus, DeploymentStrategy
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalArtifactStore
from src.models.factory import get_factory
from src.models.base import ModelBase


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyModel(ModelBase):
    """Dummy model for demonstration purposes."""
    
    def build(self, input_shape, output_shape):
        """Build model architecture."""
        self.input_shape = input_shape
        self.output_shape = output_shape
        logger.info(f"Built model with input shape {input_shape} and output shape {output_shape}")
    
    def train(self, train_data, validation_data=None, **kwargs):
        """Train the model."""
        self.is_trained = True
        return {
            "loss": 0.1,
            "accuracy": 0.95,
            "epochs": 10
        }
    
    def predict(self, data, **kwargs):
        """Make predictions."""
        return [0.5] * len(data)
    
    def evaluate(self, test_data, **kwargs):
        """Evaluate the model."""
        return {
            "test_loss": 0.15,
            "test_accuracy": 0.93
        }
    
    def get_model_state(self):
        """Get model state."""
        return {
            "weights": "dummy_weights",
            "is_trained": self.is_trained
        }
    
    def set_model_state(self, state):
        """Set model state."""
        self.is_trained = state.get("is_trained", False)


def create_and_register_model(model_registry):
    """Create a dummy model and register it."""
    logger.info("Creating and registering a dummy model...")
    
    # Register dummy model type with factory
    factory = get_factory()
    factory.register_model("DummyModel", DummyModel)
    
    # Create model instance
    model = factory.create_model("DummyModel", {
        "model_type": "DummyModel",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    })
    
    # Build and train model
    model.build(input_shape=(10,), output_shape=(1,))
    training_metrics = model.train([1, 2, 3, 4, 5])
    
    # Register model
    model_id = model_registry.register_model(
        model=model,
        model_name="example_model",
        version="v1.0",
        tags=["example", "demo"],
        description="Example model for deployment demonstration",
        metrics=training_metrics,
        dataset_info={
            "name": "dummy_dataset",
            "size": 1000,
            "features": 10
        }
    )
    
    logger.info(f"Model registered with ID: {model_id}")
    return model_id


def demonstrate_deployment(deployment_manager, model_id):
    """Demonstrate deployment functionality."""
    
    # 1. Deploy to staging environment
    logger.info("\n=== Deploying to Staging ===")
    staging_deployment_id = deployment_manager.deploy(
        model_id=model_id,
        target_environment="staging",
        model_version="v1.0",
        deployment_config={
            "replicas": 1,
            "resources": {
                "cpu": "1 core",
                "memory": "2GB"
            },
            "health_check": {
                "endpoint": "/health",
                "interval": 30
            }
        },
        strategy=DeploymentStrategy.DIRECT
    )
    logger.info(f"Deployed to staging: {staging_deployment_id}")
    
    # Check deployment status
    status = deployment_manager.get_deployment_status(staging_deployment_id)
    logger.info(f"Deployment status: {json.dumps(status, indent=2)}")
    
    # 2. Deploy to production with rolling strategy
    logger.info("\n=== Deploying to Production ===")
    prod_deployment_id = deployment_manager.deploy(
        model_id=model_id,
        target_environment="production",
        model_version="v1.0",
        deployment_config={
            "replicas": 3,
            "resources": {
                "cpu": "2 cores",
                "memory": "4GB",
                "gpu": "optional"
            },
            "strategy": "rolling",
            "rolling_update": {
                "max_surge": 1,
                "max_unavailable": 1
            }
        },
        strategy=DeploymentStrategy.ROLLING
    )
    logger.info(f"Deployed to production: {prod_deployment_id}")
    
    # 3. List all deployments
    logger.info("\n=== Listing All Deployments ===")
    all_deployments = deployment_manager.list_deployments()
    for deployment in all_deployments:
        logger.info(f"- {deployment['deployment_id']}: {deployment['target_environment']} ({deployment['status']})")
    
    # 4. Create and deploy a new version
    logger.info("\n=== Creating and Deploying New Version ===")
    
    # Register a new version of the model
    factory = get_factory()
    model_v2 = factory.create_model("DummyModel", {
        "model_type": "DummyModel",
        "hyperparameters": {
            "learning_rate": 0.0005,  # Changed hyperparameter
            "batch_size": 64,
            "epochs": 20
        }
    })
    model_v2.build(input_shape=(10,), output_shape=(1,))
    model_v2.train([1, 2, 3, 4, 5])
    
    model_v2_id = deployment_manager.model_registry.register_model(
        model=model_v2,
        model_name="example_model",
        version="v2.0",
        tags=["example", "demo", "v2"],
        description="Version 2 with improved hyperparameters",
        parent_model_id=model_id,  # Link to parent model
        metrics={"loss": 0.08, "accuracy": 0.97}
    )
    
    # Deploy v2 to production
    prod_v2_deployment_id = deployment_manager.deploy(
        model_id=model_v2_id,
        target_environment="production",
        model_version="v2.0",
        deployment_config={
            "replicas": 3,
            "resources": {
                "cpu": "2 cores",
                "memory": "4GB"
            }
        },
        strategy=DeploymentStrategy.ROLLING
    )
    logger.info(f"Deployed v2 to production: {prod_v2_deployment_id}")
    
    # 5. Demonstrate rollback
    logger.info("\n=== Demonstrating Rollback ===")
    logger.info("Simulating issue with v2, rolling back to v1...")
    
    rollback_id = deployment_manager.rollback(
        model_id=model_v2_id,
        target_environment="production"
    )
    logger.info(f"Rollback completed: {rollback_id}")
    
    # Check current deployment
    current_prod = deployment_manager.get_current_deployment(model_id, "production")
    if current_prod:
        logger.info(f"Current production deployment: {current_prod['deployment_id']} (version: {current_prod['model_version']})")
    
    # 6. List deployment packages
    logger.info("\n=== Listing Deployment Packages ===")
    packages = deployment_manager.packager.list_packages()
    for package in packages:
        logger.info(f"- Package: {package['package_id']} for model {package['model_id']} v{package['model_version']}")
    
    return staging_deployment_id, prod_deployment_id


def demonstrate_package_inspection(deployment_manager, model_id):
    """Demonstrate package inspection functionality."""
    logger.info("\n=== Package Inspection ===")
    
    # Create a custom package
    package_id = deployment_manager.packager.package_model(
        model_id=model_id,
        model_version="v1.0",
        deployment_config={
            "target_environment": "edge",
            "hardware_requirements": {
                "cpu": "ARM",
                "memory": "512MB",
                "storage": "1GB"
            }
        },
        package_name="edge_deployment_package",
        tags=["edge", "iot", "lightweight"],
        description="Optimized package for edge deployment"
    )
    
    logger.info(f"Created custom package: {package_id}")
    
    # Get package information
    package_info = deployment_manager.packager.get_package_info(package_id)
    logger.info(f"Package manifest: {json.dumps(package_info['manifest'], indent=2)}")
    
    return package_id


def main():
    """Main demonstration function."""
    logger.info("Starting Deployment Manager Demonstration")
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    artifact_store_path = base_dir / "artifacts"
    deployment_root = base_dir / "deployments"
    
    # Initialize components
    logger.info("Initializing components...")
    artifact_store = LocalArtifactStore(base_path=artifact_store_path)
    model_registry = ModelRegistry(artifact_store=artifact_store)
    deployment_manager = DeploymentManager(
        model_registry=model_registry,
        artifact_store=artifact_store,
        deployment_root=deployment_root
    )
    
    try:
        # Create and register a model
        model_id = create_and_register_model(model_registry)
        
        # Demonstrate deployment functionality
        staging_id, prod_id = demonstrate_deployment(deployment_manager, model_id)
        
        # Demonstrate package inspection
        package_id = demonstrate_package_inspection(deployment_manager, model_id)
        
        # Summary
        logger.info("\n=== Demonstration Summary ===")
        logger.info(f"✓ Created and registered model: {model_id}")
        logger.info(f"✓ Deployed to staging: {staging_id}")
        logger.info(f"✓ Deployed to production: {prod_id}")
        logger.info(f"✓ Demonstrated rollback functionality")
        logger.info(f"✓ Created custom package: {package_id}")
        logger.info(f"\nDeployment files can be found in: {deployment_root}")
        logger.info(f"Artifact store location: {artifact_store_path}")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())