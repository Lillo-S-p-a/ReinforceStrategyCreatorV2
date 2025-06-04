"""Unit tests for the model packager."""

import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from reinforcestrategycreator_pipeline.src.deployment.packager import ModelPackager
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactMetadata, ArtifactType


class TestModelPackager:
    """Test cases for ModelPackager."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = Mock()
        registry.model_exists.return_value = True
        registry.get_model_metadata.return_value = {
            "model_id": "test_model",
            "version": "v1.0",
            "model_type": "PPO",
            "created_at": "2025-01-01T00:00:00",
            "hyperparameters": {"learning_rate": 0.001},
            "metrics": {"total_reward": 100.0},
            "framework": "stable-baselines3"
        }
        
        # Mock artifact store within registry
        registry.artifact_store = Mock()
        return registry
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create a mock artifact store."""
        store = Mock()
        
        # Mock save_artifact to return metadata
        def save_artifact_side_effect(**kwargs):
            return ArtifactMetadata(
                artifact_id=kwargs["artifact_id"],
                artifact_type=kwargs["artifact_type"],
                version="v1",
                created_at=datetime.now(),
                tags=kwargs.get("tags", []),
                description=kwargs.get("description", "")
            )
        
        store.save_artifact.side_effect = save_artifact_side_effect
        return store
    
    @pytest.fixture
    def model_packager(self, mock_registry, mock_artifact_store):
        """Create a model packager instance."""
        return ModelPackager(
            model_registry=mock_registry,
            artifact_store=mock_artifact_store
        )
    
    @pytest.fixture
    def mock_model_files(self, tmp_path):
        """Create mock model files."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        # Create model files
        (model_dir / "model.pkl").write_text("mock model data")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "PPO"}))
        (model_dir / "metadata.json").write_text(json.dumps({"is_trained": True}))
        
        return model_dir
    
    def test_package_model_success(self, model_packager, mock_registry, mock_artifact_store, mock_model_files):
        """Test successful model packaging."""
        # Mock registry to load model files
        mock_registry.artifact_store.load_artifact.return_value = mock_model_files
        
        # Package model
        package_id = model_packager.package_model(
            model_id="test_model",
            model_version="v1.0",
            deployment_config={"target_environment": "production"},
            tags=["production", "v1.0"],
            description="Test deployment package"
        )
        
        # Verify package was created
        assert package_id.startswith("deployment_test_model_")
        
        # Verify artifact store was called
        mock_artifact_store.save_artifact.assert_called_once()
        call_args = mock_artifact_store.save_artifact.call_args[1]
        
        # Check artifact metadata
        assert call_args["artifact_type"] == ArtifactType.OTHER
        assert "production" in call_args["tags"]
        assert call_args["description"] == "Test deployment package"
        
        # Check package metadata
        package_metadata = call_args["metadata"]
        assert package_metadata["package_type"] == "model_deployment"
        assert package_metadata["model_id"] == "test_model"
        assert package_metadata["model_version"] == "v1.0"
    
    def test_package_model_not_found(self, model_packager, mock_registry):
        """Test error when model not found."""
        mock_registry.model_exists.return_value = False
        
        with pytest.raises(ValueError, match="Model test_model.*not found"):
            model_packager.package_model(
                model_id="test_model",
                model_version="v1.0"
            )
    
    def test_create_deployment_manifest(self, model_packager):
        """Test deployment manifest creation."""
        model_metadata = {
            "model_type": "PPO",
            "hyperparameters": {"learning_rate": 0.001},
            "metrics": {"reward": 100},
            "created_at": "2025-01-01T00:00:00",
            "framework": "stable-baselines3"
        }
        
        deployment_config = {
            "target_environment": "production",
            "strategy": "blue_green",
            "hardware_requirements": {
                "cpu": "4 cores",
                "memory": "8GB",
                "gpu": "required"
            }
        }
        
        manifest = model_packager._create_deployment_manifest(
            model_id="test_model",
            model_version="v1.0",
            model_metadata=model_metadata,
            deployment_config=deployment_config,
            package_name="test_package"
        )
        
        # Verify manifest structure
        assert manifest["package_name"] == "test_package"
        assert manifest["package_version"] == "1.0.0"
        assert "created_at" in manifest
        
        # Verify model info
        assert manifest["model"]["id"] == "test_model"
        assert manifest["model"]["version"] == "v1.0"
        assert manifest["model"]["type"] == "PPO"
        assert manifest["model"]["hyperparameters"]["learning_rate"] == 0.001
        
        # Verify deployment config
        assert manifest["deployment"]["target_environment"] == "production"
        assert manifest["deployment"]["deployment_strategy"] == "blue_green"
        
        # Verify runtime info
        assert manifest["runtime"]["framework"] == "stable-baselines3"
        assert manifest["runtime"]["hardware_requirements"]["gpu"] == "required"
    
    def test_create_dependencies_file(self, model_packager, tmp_path):
        """Test dependencies file creation."""
        package_path = tmp_path / "package"
        package_path.mkdir()
        
        # Test with stable-baselines3 framework
        model_metadata = {"framework": "stable-baselines3"}
        model_packager._create_dependencies_file(package_path, model_metadata)
        
        requirements_file = package_path / "requirements.txt"
        assert requirements_file.exists()
        
        requirements = requirements_file.read_text().strip().split("\n")
        assert "numpy>=1.19.0" in requirements
        assert "pandas>=1.2.0" in requirements
        assert "stable-baselines3>=1.0" in requirements
        
        # Test with tensorflow framework
        package_path2 = tmp_path / "package2"
        package_path2.mkdir()
        
        model_metadata = {"framework": "tensorflow", "dependencies": ["custom-lib==1.0"]}
        model_packager._create_dependencies_file(package_path2, model_metadata)
        
        requirements_file2 = package_path2 / "requirements.txt"
        requirements2 = requirements_file2.read_text().strip().split("\n")
        assert "tensorflow>=2.4.0" in requirements2
        assert "custom-lib==1.0" in requirements2
    
    def test_create_deployment_scripts(self, model_packager, tmp_path):
        """Test deployment script creation."""
        package_path = tmp_path / "package"
        package_path.mkdir()
        
        manifest = {
            "package_name": "test_package",
            "model": {"id": "test_model", "version": "v1.0"},
            "deployment": {"target_environment": "production"}
        }
        
        model_packager._create_deployment_scripts(package_path, manifest)
        
        # Check run script
        run_script = package_path / "run.py"
        assert run_script.exists()
        assert run_script.stat().st_mode & 0o111  # Check executable
        
        # Check health check script
        health_script = package_path / "health_check.py"
        assert health_script.exists()
        assert health_script.stat().st_mode & 0o111  # Check executable
        
        # Verify script content
        run_content = run_script.read_text()
        assert "load_model" in run_content
        assert "deployment_manifest.json" in run_content
        
        health_content = health_script.read_text()
        assert "check_model_files" in health_content
        assert "Health check passed" in health_content
    
    def test_create_readme(self, model_packager, tmp_path):
        """Test README creation."""
        package_path = tmp_path / "package"
        package_path.mkdir()
        
        manifest = {
            "package_name": "test_package",
            "created_at": "2025-01-01T00:00:00",
            "model": {
                "id": "test_model",
                "version": "v1.0",
                "type": "PPO",
                "created_at": "2025-01-01T00:00:00"
            },
            "deployment": {
                "target_environment": "production",
                "deployment_strategy": "rolling"
            },
            "runtime": {
                "python_version": "3.8+",
                "framework": "stable-baselines3",
                "hardware_requirements": {
                    "cpu": "2 cores",
                    "memory": "4GB",
                    "gpu": "optional"
                }
            }
        }
        
        model_packager._create_readme(package_path, manifest)
        
        readme_file = package_path / "README.md"
        assert readme_file.exists()
        
        readme_content = readme_file.read_text()
        assert "# Deployment Package: test_package" in readme_content
        assert "Model Information" in readme_content
        assert "Type**: PPO" in readme_content
        assert "Target Environment**: production" in readme_content
        assert "pip install -r requirements.txt" in readme_content
        assert "python health_check.py" in readme_content
    
    def test_calculate_checksum(self, model_packager, tmp_path):
        """Test checksum calculation."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        
        # Create test files
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")
        
        # Calculate checksum
        checksum1 = model_packager._calculate_checksum(test_dir)
        assert len(checksum1) == 64  # SHA256 hex length
        
        # Verify checksum changes with content
        (test_dir / "file1.txt").write_text("modified content")
        checksum2 = model_packager._calculate_checksum(test_dir)
        assert checksum1 != checksum2
        
        # Verify checksum is consistent
        (test_dir / "file1.txt").write_text("content1")
        checksum3 = model_packager._calculate_checksum(test_dir)
        assert checksum1 == checksum3
    
    def test_list_packages(self, model_packager, mock_artifact_store):
        """Test listing deployment packages."""
        # Mock artifact store response
        artifacts = [
            ArtifactMetadata(
                artifact_id="package1",
                artifact_type=ArtifactType.OTHER,
                version="v1",
                created_at=datetime.now(),
                properties={
                    "package_type": "model_deployment",
                    "model_id": "model_a",
                    "model_version": "v1.0"
                },
                tags=["production"],
                description="Production package"
            ),
            ArtifactMetadata(
                artifact_id="package2",
                artifact_type=ArtifactType.OTHER,
                version="v1",
                created_at=datetime.now(),
                properties={
                    "package_type": "model_deployment",
                    "model_id": "model_b",
                    "model_version": "v2.0"
                },
                tags=["staging"],
                description="Staging package"
            ),
            ArtifactMetadata(
                artifact_id="other_artifact",
                artifact_type=ArtifactType.OTHER,
                version="v1",
                created_at=datetime.now(),
                properties={"package_type": "other"},
                tags=[],
                description="Not a deployment package"
            )
        ]
        
        mock_artifact_store.list_artifacts.return_value = artifacts
        
        # List all packages
        packages = model_packager.list_packages()
        assert len(packages) == 2
        assert all(p["package_id"] in ["package1", "package2"] for p in packages)
        
        # List packages for specific model
        packages = model_packager.list_packages(model_id="model_a")
        assert len(packages) == 1
        assert packages[0]["model_id"] == "model_a"
        
        # List packages by tags
        packages = model_packager.list_packages(tags=["staging"])
        mock_artifact_store.list_artifacts.assert_called_with(
            artifact_type=ArtifactType.OTHER,
            tags=["staging"]
        )
    
    def test_get_package_info(self, model_packager, mock_artifact_store):
        """Test getting package information."""
        # Mock artifact metadata
        metadata = ArtifactMetadata(
            artifact_id="test_package",
            artifact_type=ArtifactType.OTHER,
            version="v1",
            created_at=datetime.now(),
            properties={
                "package_type": "model_deployment",
                "model_id": "test_model",
                "model_version": "v1.0",
                "manifest": {"package_name": "test_package"},
                "checksum": "abc123"
            },
            tags=["production"],
            description="Test package"
        )
        
        mock_artifact_store.get_artifact_metadata.return_value = metadata
        
        # Get package info
        info = model_packager.get_package_info("test_package")
        
        assert info["package_id"] == "test_package"
        assert info["manifest"]["package_name"] == "test_package"
        assert info["checksum"] == "abc123"
        assert "production" in info["tags"]
        
        # Test error for non-deployment package
        metadata.properties["package_type"] = "other"
        with pytest.raises(ValueError, match="not a deployment package"):
            model_packager.get_package_info("test_package")
    
    def test_package_model_integration(self, model_packager, mock_registry, mock_artifact_store, tmp_path):
        """Test full packaging workflow."""
        # Create mock model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.pkl").write_text("model data")
        (model_dir / "config.json").write_text(json.dumps({"model_type": "PPO"}))
        (model_dir / "metadata.json").write_text(json.dumps({"is_trained": True}))
        
        # Mock registry to return model directory
        mock_registry.artifact_store.load_artifact.return_value = model_dir
        
        # Mock artifact store to save tarball
        saved_tarball_path = None
        def save_artifact_side_effect(**kwargs):
            nonlocal saved_tarball_path
            saved_tarball_path = kwargs["artifact_path"]
            return ArtifactMetadata(
                artifact_id=kwargs["artifact_id"],
                artifact_type=kwargs["artifact_type"],
                version="v1",
                created_at=datetime.now()
            )
        
        mock_artifact_store.save_artifact.side_effect = save_artifact_side_effect
        
        # Package model
        package_id = model_packager.package_model(
            model_id="test_model",
            model_version="v1.0",
            package_name="custom_package"
        )
        
        assert package_id == "custom_package"
        
        # Verify tarball was created and contains expected files
        assert saved_tarball_path is not None
        with tarfile.open(saved_tarball_path, "r:gz") as tar:
            members = tar.getnames()
            assert any("deployment_manifest.json" in m for m in members)
            assert any("requirements.txt" in m for m in members)
            assert any("run.py" in m for m in members)
            assert any("health_check.py" in m for m in members)
            assert any("README.md" in m for m in members)
            assert any("model/model.pkl" in m for m in members)