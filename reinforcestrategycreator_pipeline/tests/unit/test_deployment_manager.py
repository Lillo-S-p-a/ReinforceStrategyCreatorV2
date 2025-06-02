"""Unit tests for the deployment manager."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.deployment import DeploymentManager, DeploymentStatus, DeploymentStrategy
from src.artifact_store.base import ArtifactMetadata, ArtifactType


class TestDeploymentManager:
    """Test cases for DeploymentManager."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry."""
        registry = Mock()
        registry.model_exists.return_value = True
        registry.get_model_metadata.return_value = {
            "model_id": "test_model",
            "version": "v1.0",
            "model_type": "PPO",
            "created_at": "2025-01-01T00:00:00"
        }
        return registry
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create a mock artifact store."""
        store = Mock()
        store.artifact_exists.return_value = True
        store.load_artifact.return_value = Path("/tmp/test_package.tar.gz")
        return store
    
    @pytest.fixture
    def deployment_manager(self, mock_registry, mock_artifact_store, tmp_path):
        """Create a deployment manager instance."""
        return DeploymentManager(
            model_registry=mock_registry,
            artifact_store=mock_artifact_store,
            deployment_root=tmp_path / "deployments"
        )
    
    def test_initialization(self, deployment_manager, tmp_path):
        """Test deployment manager initialization."""
        assert deployment_manager.model_registry is not None
        assert deployment_manager.artifact_store is not None
        assert deployment_manager.packager is not None
        assert deployment_manager.deployment_root.exists()
        assert deployment_manager.state_file.exists()
    
    def test_deploy_success(self, deployment_manager, mock_registry, mock_artifact_store):
        """Test successful model deployment."""
        # Mock packager
        with patch.object(deployment_manager.packager, 'package_model') as mock_package:
            mock_package.return_value = "test_package_id"
            
            # Mock execute deployment
            with patch.object(deployment_manager, '_execute_deployment') as mock_execute:
                mock_execute.return_value = Path("/deployments/test")
                
                # Deploy model
                deployment_id = deployment_manager.deploy(
                    model_id="test_model",
                    target_environment="staging",
                    model_version="v1.0",
                    deployment_config={"replicas": 2}
                )
                
                # Verify deployment ID format
                assert deployment_id.startswith("deploy_test_model_staging_")
                
                # Verify packager was called
                mock_package.assert_called_once()
                
                # Verify deployment was executed
                mock_execute.assert_called_once()
                
                # Check deployment state
                deployment = deployment_manager._get_deployment_record(deployment_id)
                assert deployment is not None
                assert deployment["status"] == DeploymentStatus.DEPLOYED.value
                assert deployment["model_id"] == "test_model"
                assert deployment["target_environment"] == "staging"
    
    def test_deploy_with_existing_package(self, deployment_manager):
        """Test deployment using existing package."""
        with patch.object(deployment_manager, '_execute_deployment') as mock_execute:
            mock_execute.return_value = Path("/deployments/test")
            
            # Deploy with existing package
            deployment_id = deployment_manager.deploy(
                model_id="test_model",
                target_environment="production",
                package_id="existing_package_id"
            )
            
            # Verify packager was not called
            with patch.object(deployment_manager.packager, 'package_model') as mock_package:
                mock_package.assert_not_called()
            
            # Check deployment used existing package
            deployment = deployment_manager._get_deployment_record(deployment_id)
            assert deployment["package_id"] == "existing_package_id"
    
    def test_deploy_already_deployed_error(self, deployment_manager):
        """Test error when deploying already deployed version."""
        # Set up existing deployment
        deployment_manager._set_current_deployment(
            "test_model",
            "production",
            {
                "deployment_id": "existing_deploy",
                "model_version": "v1.0",
                "status": DeploymentStatus.DEPLOYED.value
            }
        )
        
        # Try to deploy same version
        with pytest.raises(ValueError, match="already deployed"):
            deployment_manager.deploy(
                model_id="test_model",
                target_environment="production",
                model_version="v1.0"
            )
    
    def test_deploy_force_redeploy(self, deployment_manager):
        """Test force redeployment of same version."""
        # Set up existing deployment
        deployment_manager._set_current_deployment(
            "test_model",
            "production",
            {
                "deployment_id": "existing_deploy",
                "model_version": "v1.0",
                "status": DeploymentStatus.DEPLOYED.value
            }
        )
        
        with patch.object(deployment_manager.packager, 'package_model') as mock_package:
            mock_package.return_value = "new_package_id"
            
            with patch.object(deployment_manager, '_execute_deployment') as mock_execute:
                mock_execute.return_value = Path("/deployments/test")
                
                # Deploy with force=True
                deployment_id = deployment_manager.deploy(
                    model_id="test_model",
                    target_environment="production",
                    model_version="v1.0",
                    force=True
                )
                
                assert deployment_id is not None
    
    def test_rollback_success(self, deployment_manager):
        """Test successful rollback."""
        # Set up deployments
        previous_deployment = {
            "deployment_id": "deploy_1",
            "model_id": "test_model",
            "model_version": "v1.0",
            "package_id": "package_1",
            "status": DeploymentStatus.DEPLOYED.value,
            "deployment_config": {}
        }
        
        current_deployment = {
            "deployment_id": "deploy_2",
            "model_id": "test_model",
            "model_version": "v2.0",
            "package_id": "package_2",
            "status": DeploymentStatus.DEPLOYED.value,
            "previous_deployment": previous_deployment,
            "deployment_config": {}
        }
        
        deployment_manager._update_deployment_record(previous_deployment)
        deployment_manager._update_deployment_record(current_deployment)
        deployment_manager._set_current_deployment("test_model", "production", current_deployment)
        
        with patch.object(deployment_manager, '_execute_deployment') as mock_execute:
            mock_execute.return_value = Path("/deployments/rollback")
            
            # Perform rollback
            rollback_id = deployment_manager.rollback(
                model_id="test_model",
                target_environment="production"
            )
            
            # Verify rollback ID format
            assert rollback_id.startswith("rollback_test_model_production_")
            
            # Check rollback deployment
            rollback = deployment_manager._get_deployment_record(rollback_id)
            assert rollback["is_rollback"] is True
            assert rollback["model_version"] == "v1.0"
            assert rollback["rollback_from"] == "deploy_2"
            assert rollback["rollback_to"] == "deploy_1"
    
    def test_rollback_no_previous_deployment(self, deployment_manager):
        """Test rollback error when no previous deployment exists."""
        current_deployment = {
            "deployment_id": "deploy_1",
            "model_id": "test_model",
            "model_version": "v1.0",
            "status": DeploymentStatus.DEPLOYED.value
        }
        
        deployment_manager._set_current_deployment("test_model", "production", current_deployment)
        
        with pytest.raises(ValueError, match="No previous deployment"):
            deployment_manager.rollback(
                model_id="test_model",
                target_environment="production"
            )
    
    def test_get_deployment_status(self, deployment_manager):
        """Test getting deployment status."""
        deployment = {
            "deployment_id": "test_deploy",
            "model_id": "test_model",
            "model_version": "v1.0",
            "target_environment": "staging",
            "status": DeploymentStatus.DEPLOYED.value,
            "created_at": "2025-01-01T00:00:00",
            "deployed_at": "2025-01-01T00:01:00"
        }
        
        deployment_manager._update_deployment_record(deployment)
        
        status = deployment_manager.get_deployment_status("test_deploy")
        
        assert status["deployment_id"] == "test_deploy"
        assert status["status"] == DeploymentStatus.DEPLOYED.value
        assert status["model_id"] == "test_model"
        assert status["deployed_at"] == "2025-01-01T00:01:00"
    
    def test_list_deployments(self, deployment_manager):
        """Test listing deployments with filters."""
        # Create test deployments
        deployments = [
            {
                "deployment_id": "deploy_1",
                "model_id": "model_a",
                "target_environment": "staging",
                "status": DeploymentStatus.DEPLOYED.value,
                "created_at": "2025-01-01T00:00:00"
            },
            {
                "deployment_id": "deploy_2",
                "model_id": "model_a",
                "target_environment": "production",
                "status": DeploymentStatus.DEPLOYED.value,
                "created_at": "2025-01-02T00:00:00"
            },
            {
                "deployment_id": "deploy_3",
                "model_id": "model_b",
                "target_environment": "staging",
                "status": DeploymentStatus.FAILED.value,
                "created_at": "2025-01-03T00:00:00"
            }
        ]
        
        for deployment in deployments:
            deployment_manager._update_deployment_record(deployment)
        
        # Test filter by model_id
        results = deployment_manager.list_deployments(model_id="model_a")
        assert len(results) == 2
        assert all(d["model_id"] == "model_a" for d in results)
        
        # Test filter by environment
        results = deployment_manager.list_deployments(target_environment="staging")
        assert len(results) == 2
        assert all(d["target_environment"] == "staging" for d in results)
        
        # Test filter by status
        results = deployment_manager.list_deployments(status=DeploymentStatus.FAILED)
        assert len(results) == 1
        assert results[0]["status"] == DeploymentStatus.FAILED.value
    
    def test_execute_deployment_direct_strategy(self, deployment_manager, tmp_path):
        """Test direct deployment strategy execution."""
        # Create a mock package
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        
        package_content_dir = package_dir / "test_package"
        package_content_dir.mkdir()
        (package_content_dir / "model.pkl").touch()
        
        # Create tarball
        import tarfile
        tarball_path = tmp_path / "test_package.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(package_content_dir, arcname="test_package")
        
        # Mock artifact store to return our tarball
        deployment_manager.artifact_store.load_artifact.return_value = tarball_path
        
        deployment_record = {
            "deployment_id": "test_deploy",
            "model_id": "test_model",
            "target_environment": "staging"
        }
        
        # Execute deployment
        deployment_path = deployment_manager._execute_deployment(
            deployment_record=deployment_record,
            package_id="test_package_id",
            strategy=DeploymentStrategy.DIRECT
        )
        
        # Verify deployment structure
        assert deployment_path.exists()
        assert (deployment_path / "model.pkl").exists()
        
        # Verify deployment info file
        deployment_dir = deployment_manager.deployment_root / "staging" / "test_model" / "test_deploy"
        info_file = deployment_dir / "deployment_info.json"
        assert info_file.exists()
        
        with open(info_file) as f:
            info = json.load(f)
            assert info["deployment_id"] == "test_deploy"
            assert info["package_id"] == "test_package_id"
            assert info["strategy"] == "direct"
    
    def test_state_persistence(self, tmp_path):
        """Test deployment state persistence."""
        # Create first manager instance
        manager1 = DeploymentManager(
            model_registry=Mock(),
            artifact_store=Mock(),
            deployment_root=tmp_path / "deployments"
        )
        
        # Add deployment
        deployment = {
            "deployment_id": "test_deploy",
            "model_id": "test_model",
            "status": DeploymentStatus.DEPLOYED.value
        }
        manager1._update_deployment_record(deployment)
        
        # Create second manager instance (should load state)
        manager2 = DeploymentManager(
            model_registry=Mock(),
            artifact_store=Mock(),
            deployment_root=tmp_path / "deployments"
        )
        
        # Verify state was loaded
        loaded_deployment = manager2._get_deployment_record("test_deploy")
        assert loaded_deployment is not None
        assert loaded_deployment["model_id"] == "test_model"