"""Model packager for creating deployment artifacts."""

import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

from ..models.registry import ModelRegistry
from ..artifact_store.base import ArtifactStore, ArtifactType


class ModelPackager:
    """Packages models into self-contained deployment artifacts.
    
    The packager creates deployment-ready artifacts that include:
    - Model files (weights, configuration)
    - Dependencies specification
    - Deployment metadata
    - Version information
    - Execution environment details
    """
    
    def __init__(self, model_registry: ModelRegistry, artifact_store: ArtifactStore):
        """Initialize the model packager.
        
        Args:
            model_registry: Registry for accessing models
            artifact_store: Store for saving deployment packages
        """
        self.model_registry = model_registry
        self.artifact_store = artifact_store
    
    def package_model(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        include_dependencies: bool = True,
        package_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> str:
        """Package a model for deployment.
        
        Args:
            model_id: ID of the model to package
            model_version: Specific version to package (latest if None)
            deployment_config: Deployment-specific configuration
            include_dependencies: Whether to include dependency specifications
            package_name: Custom name for the package (auto-generated if None)
            tags: Tags for the deployment package
            description: Description of the deployment package
            
        Returns:
            Package ID of the created deployment artifact
            
        Raises:
            ValueError: If model not found or packaging fails
        """
        # Verify model exists
        if not self.model_registry.model_exists(model_id, model_version):
            raise ValueError(f"Model {model_id} (version: {model_version}) not found")
        
        # Get model metadata
        model_metadata = self.model_registry.get_model_metadata(model_id, model_version)
        
        # Generate package name if not provided
        if package_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"deployment_{model_id}_{timestamp}"
        
        # Create temporary directory for packaging
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / package_name
            package_path.mkdir(parents=True)
            
            # Load model to temporary location
            model_path = self.model_registry.artifact_store.load_artifact(
                artifact_id=model_id,
                version=model_version,
                destination_path=package_path / "model"
            )
            
            # Create deployment manifest
            manifest = self._create_deployment_manifest(
                model_id=model_id,
                model_version=model_metadata["version"],
                model_metadata=model_metadata,
                deployment_config=deployment_config or {},
                package_name=package_name
            )
            
            # Save manifest
            manifest_path = package_path / "deployment_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            
            # Include dependencies if requested
            if include_dependencies:
                self._create_dependencies_file(package_path, model_metadata)
            
            # Create deployment scripts
            self._create_deployment_scripts(package_path, manifest)
            
            # Create README
            self._create_readme(package_path, manifest)
            
            # Calculate package checksum
            checksum = self._calculate_checksum(package_path)
            manifest["checksum"] = checksum
            
            # Update manifest with checksum
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            
            # Create tarball
            tarball_path = Path(temp_dir) / f"{package_name}.tar.gz"
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(package_path, arcname=package_name)
            
            # Save to artifact store
            package_metadata = self.artifact_store.save_artifact(
                artifact_id=package_name,
                artifact_path=tarball_path,
                artifact_type=ArtifactType.OTHER,
                metadata={
                    "package_type": "model_deployment",
                    "model_id": model_id,
                    "model_version": model_metadata["version"],
                    "manifest": manifest,
                    "checksum": checksum
                },
                tags=tags or [],
                description=description or f"Deployment package for {model_id}"
            )
            
            return package_metadata.artifact_id
    
    def _create_deployment_manifest(
        self,
        model_id: str,
        model_version: str,
        model_metadata: Dict[str, Any],
        deployment_config: Dict[str, Any],
        package_name: str
    ) -> Dict[str, Any]:
        """Create deployment manifest with all necessary metadata.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            model_metadata: Model metadata from registry
            deployment_config: Deployment configuration
            package_name: Name of the deployment package
            
        Returns:
            Deployment manifest dictionary
        """
        return {
            "package_name": package_name,
            "package_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "model": {
                "id": model_id,
                "version": model_version,
                "type": model_metadata.get("model_type"),
                "hyperparameters": model_metadata.get("hyperparameters", {}),
                "metrics": model_metadata.get("metrics", {}),
                "created_at": model_metadata.get("created_at")
            },
            "deployment": {
                "config": deployment_config,
                "target_environment": deployment_config.get("target_environment", "production"),
                "deployment_strategy": deployment_config.get("strategy", "rolling"),
                "health_check": deployment_config.get("health_check", {
                    "endpoint": "/health",
                    "interval": 30,
                    "timeout": 5
                })
            },
            "runtime": {
                "python_version": "3.8+",
                "framework": model_metadata.get("framework", "unknown"),
                "hardware_requirements": deployment_config.get("hardware_requirements", {
                    "cpu": "2 cores",
                    "memory": "4GB",
                    "gpu": "optional"
                })
            }
        }
    
    def _create_dependencies_file(self, package_path: Path, model_metadata: Dict[str, Any]) -> None:
        """Create requirements.txt with model dependencies.
        
        Args:
            package_path: Path to package directory
            model_metadata: Model metadata containing dependency info
        """
        requirements = [
            "numpy>=1.19.0",
            "pandas>=1.2.0",
            "scikit-learn>=0.24.0"
        ]
        
        # Add framework-specific dependencies
        framework = model_metadata.get("framework", "").lower()
        if "tensorflow" in framework:
            requirements.append("tensorflow>=2.4.0")
        elif "pytorch" in framework:
            requirements.append("torch>=1.7.0")
        elif "stable-baselines3" in framework or "sb3" in framework:
            requirements.append("stable-baselines3>=1.0")
        
        # Add any custom dependencies from metadata
        custom_deps = model_metadata.get("dependencies", [])
        requirements.extend(custom_deps)
        
        # Write requirements file
        requirements_path = package_path / "requirements.txt"
        with open(requirements_path, "w") as f:
            for req in sorted(set(requirements)):
                f.write(f"{req}\n")
    
    def _create_deployment_scripts(self, package_path: Path, manifest: Dict[str, Any]) -> None:
        """Create deployment helper scripts.
        
        Args:
            package_path: Path to package directory
            manifest: Deployment manifest
        """
        # Create run script
        run_script = """#!/usr/bin/env python3
\"\"\"Run the deployed model.\"\"\"

import json
import sys
from pathlib import Path

def load_model():
    \"\"\"Load the packaged model.\"\"\"
    # This is a placeholder - actual implementation depends on model type
    model_path = Path(__file__).parent / "model"
    config_path = model_path / "config.json"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Model loaded: {config.get('model_type', 'unknown')}")
    return config

def main():
    \"\"\"Main entry point.\"\"\"
    print("Loading deployment manifest...")
    manifest_path = Path(__file__).parent / "deployment_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print(f"Package: {manifest['package_name']}")
    print(f"Model: {manifest['model']['id']} v{manifest['model']['version']}")
    print(f"Target Environment: {manifest['deployment']['target_environment']}")
    
    # Load model
    model_config = load_model()
    
    # Placeholder for actual model serving
    print("\\nModel is ready for serving!")
    print("Implement your serving logic here.")

if __name__ == "__main__":
    main()
"""
        
        run_script_path = package_path / "run.py"
        with open(run_script_path, "w") as f:
            f.write(run_script)
        
        # Make script executable
        run_script_path.chmod(0o755)
        
        # Create health check script
        health_script = """#!/usr/bin/env python3
\"\"\"Health check for deployed model.\"\"\"

import json
import sys
from pathlib import Path

def check_model_files():
    \"\"\"Check if all model files are present.\"\"\"
    required_files = [
        "model/config.json",
        "model/metadata.json",
        "model/model.pkl",
        "deployment_manifest.json"
    ]
    
    base_path = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        if not (base_path / file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def main():
    \"\"\"Run health check.\"\"\"
    print("Running deployment health check...")
    
    # Check files
    files_ok, missing = check_model_files()
    if not files_ok:
        print(f"ERROR: Missing files: {missing}")
        sys.exit(1)
    
    print("✓ All required files present")
    
    # Load and validate manifest
    manifest_path = Path(__file__).parent / "deployment_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print(f"✓ Deployment manifest valid")
    print(f"  Package: {manifest['package_name']}")
    print(f"  Model: {manifest['model']['id']}")
    
    print("\\nHealth check passed!")
    sys.exit(0)

if __name__ == "__main__":
    main()
"""
        
        health_script_path = package_path / "health_check.py"
        with open(health_script_path, "w") as f:
            f.write(health_script)
        
        health_script_path.chmod(0o755)
    
    def _create_readme(self, package_path: Path, manifest: Dict[str, Any]) -> None:
        """Create README file for the deployment package.
        
        Args:
            package_path: Path to package directory
            manifest: Deployment manifest
        """
        readme_content = f"""# Deployment Package: {manifest['package_name']}

## Overview
This is a deployment package for model `{manifest['model']['id']}` version `{manifest['model']['version']}`.

Created: {manifest['created_at']}

## Model Information
- **Type**: {manifest['model']['type']}
- **Version**: {manifest['model']['version']}
- **Created**: {manifest['model']['created_at']}

## Deployment Configuration
- **Target Environment**: {manifest['deployment']['target_environment']}
- **Strategy**: {manifest['deployment']['deployment_strategy']}

## Hardware Requirements
- **CPU**: {manifest['runtime']['hardware_requirements']['cpu']}
- **Memory**: {manifest['runtime']['hardware_requirements']['memory']}
- **GPU**: {manifest['runtime']['hardware_requirements']['gpu']}

## Usage

### 1. Extract Package
```bash
tar -xzf {manifest['package_name']}.tar.gz
cd {manifest['package_name']}
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Health Check
```bash
python health_check.py
```

### 4. Run Model
```bash
python run.py
```

## Directory Structure
```
{manifest['package_name']}/
├── deployment_manifest.json  # Deployment metadata
├── requirements.txt         # Python dependencies
├── run.py                  # Main execution script
├── health_check.py         # Health check script
├── README.md              # This file
└── model/                 # Model files
    ├── config.json       # Model configuration
    ├── metadata.json     # Model metadata
    └── model.pkl        # Model weights
```

## Notes
- Ensure Python {manifest['runtime']['python_version']} is installed
- The model was trained with framework: {manifest['runtime']['framework']}
"""
        
        readme_path = package_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
    
    def _calculate_checksum(self, directory: Path) -> str:
        """Calculate SHA256 checksum of directory contents.
        
        Args:
            directory: Directory to checksum
            
        Returns:
            Hex string of SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        
        # Sort files for consistent ordering
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                # Include relative path in hash
                relative_path = file_path.relative_to(directory)
                sha256_hash.update(str(relative_path).encode())
                
                # Include file contents
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def list_packages(
        self,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List deployment packages.
        
        Args:
            model_id: Filter by source model ID
            tags: Filter by tags
            
        Returns:
            List of package metadata dictionaries
        """
        # Get all deployment packages from artifact store
        all_artifacts = self.artifact_store.list_artifacts(
            artifact_type=ArtifactType.OTHER,
            tags=tags
        )
        
        packages = []
        for artifact in all_artifacts:
            # Check if it's a deployment package
            if artifact.properties.get("package_type") == "model_deployment":
                # Apply model_id filter if specified
                if model_id and artifact.properties.get("model_id") != model_id:
                    continue
                
                packages.append({
                    "package_id": artifact.artifact_id,
                    "model_id": artifact.properties.get("model_id"),
                    "model_version": artifact.properties.get("model_version"),
                    "created_at": artifact.created_at.isoformat(),
                    "tags": artifact.tags,
                    "description": artifact.description
                })
        
        return packages
    
    def get_package_info(self, package_id: str) -> Dict[str, Any]:
        """Get detailed information about a deployment package.
        
        Args:
            package_id: ID of the deployment package
            
        Returns:
            Package information including manifest
        """
        metadata = self.artifact_store.get_artifact_metadata(package_id)
        
        if metadata.properties.get("package_type") != "model_deployment":
            raise ValueError(f"{package_id} is not a deployment package")
        
        return {
            "package_id": metadata.artifact_id,
            "created_at": metadata.created_at.isoformat(),
            "manifest": metadata.properties.get("manifest", {}),
            "checksum": metadata.properties.get("checksum"),
            "tags": metadata.tags,
            "description": metadata.description
        }