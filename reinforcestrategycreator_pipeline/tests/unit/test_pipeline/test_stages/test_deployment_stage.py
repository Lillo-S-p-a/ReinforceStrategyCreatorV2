"""Unit tests for DeploymentStage."""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
import tempfile
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stages.deployment import DeploymentStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext


class TestDeploymentStage(unittest.TestCase):
    """Test cases for DeploymentStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "deployment_target": "local",
            "model_registry": {
                "enabled": True,
                "tags": ["test", "unit-test"]
            },
            "deployment_strategy": "direct",
            "validation_config": {
                "smoke_tests": True,
                "check_endpoint": False
            },
            "rollback_config": {
                "auto_rollback": True
            },
            "packaging_format": "pickle"
        }
        self.stage = DeploymentStage(config=self.config)
        
        # Reset PipelineContext singleton
        PipelineContext._instance = None
        self.context = PipelineContext.get_instance()
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset singleton
        PipelineContext._instance = None
        
        # Clean up any created directories
        if hasattr(self.stage, 'deployment_dir') and self.stage.deployment_dir.exists():
            import shutil
            shutil.rmtree(self.stage.deployment_dir)
            
    def test_initialization(self):
        """Test stage initialization."""
        self.assertEqual(self.stage.name, "deployment")
        self.assertEqual(self.stage.deployment_target, "local")
        self.assertEqual(self.stage.deployment_strategy, "direct")
        self.assertTrue(self.stage.model_registry["enabled"])
        self.assertTrue(self.stage.rollback_config["auto_rollback"])
        
    def test_setup_without_model(self):
        """Test setup fails without trained model."""
        with self.assertRaises(ValueError) as cm:
            self.stage.setup(self.context)
        self.assertIn("No trained model found", str(cm.exception))
        
    def test_setup_failed_thresholds_no_force(self):
        """Test setup fails when model didn't pass thresholds."""
        self.context.set("trained_model", {"type": "test"})
        self.context.set("model_passed_thresholds", False)
        
        with self.assertRaises(ValueError) as cm:
            self.stage.setup(self.context)
        self.assertIn("Model did not pass evaluation thresholds", str(cm.exception))
        
    def test_setup_failed_thresholds_with_force(self):
        """Test setup succeeds with force_deployment."""
        self.stage.config["force_deployment"] = True
        self.context.set("trained_model", {"type": "test"})
        self.context.set("model_passed_thresholds", False)
        self.context.set("artifact_store", Mock())
        
        # Should not raise
        self.stage.setup(self.context)
        
    def test_setup_success(self):
        """Test successful setup."""
        self.context.set("trained_model", {"type": "test"})
        self.context.set("model_passed_thresholds", True)
        self.context.set("artifact_store", Mock())
        
        self.stage.setup(self.context)
        
        # Check deployment directory was created for local target
        self.assertTrue(self.stage.deployment_dir.exists())
        
    def test_initialize_deployment_targets(self):
        """Test initialization of different deployment targets."""
        # Test local
        self.stage._initialize_deployment_target()
        self.assertTrue(self.stage.deployment_dir.exists())
        self.assertEqual(self.stage.deployment_dir.name, "deployed_models")
        
        # Test staging
        self.stage.deployment_target = "staging"
        self.stage._initialize_deployment_target()  # Should not raise
        
        # Test production
        self.stage.deployment_target = "production"
        self.stage._initialize_deployment_target()  # Should not raise
        
        # Test unknown
        self.stage.deployment_target = "unknown"
        with self.assertRaises(ValueError):
            self.stage._initialize_deployment_target()
            
    def test_create_deployment_package(self):
        """Test deployment package creation."""
        model = {"type": "test_model", "params": {}}
        self.context.set("model_type", "random_forest")
        self.context.set("model_config", {"n_estimators": 100})
        self.context.set("training_metadata", {"epochs": 10})
        self.context.set("evaluation_results", {"accuracy": 0.85})
        self.context.set_metadata("run_id", "test_run_123")
        
        package = self.stage._create_deployment_package(model, self.context)
        
        # Check package structure
        self.assertEqual(package["model"], model)
        self.assertEqual(package["model_type"], "random_forest")
        self.assertEqual(package["model_config"]["n_estimators"], 100)
        self.assertEqual(package["training_metadata"]["epochs"], 10)
        self.assertEqual(package["pipeline_run_id"], "test_run_123")
        self.assertIn("created_at", package)
        
    def test_generate_model_version(self):
        """Test model version generation."""
        self.context.set("model_type", "test_model")
        
        # Test first version
        with patch.object(self.stage, '_get_existing_versions', return_value=[]):
            version = self.stage._generate_model_version(self.context)
            self.assertTrue(version.startswith("v1.0.0-"))
            
        # Test incremental version
        with patch.object(self.stage, '_get_existing_versions', return_value=["v1.0.0-20250101", "v1.0.1-20250102"]):
            version = self.stage._generate_model_version(self.context)
            self.assertTrue(version.startswith("v1.0.2-"))
            
    def test_get_existing_versions(self):
        """Test getting existing model versions."""
        # Create test deployment directory
        self.stage.deployment_dir = Path(tempfile.mkdtemp())
        
        # Create some dummy model files
        (self.stage.deployment_dir / "test_model_v1.0.0-20250101.pkl").touch()
        (self.stage.deployment_dir / "test_model_v1.0.1-20250102.pkl").touch()
        (self.stage.deployment_dir / "other_model_v2.0.0-20250103.pkl").touch()
        
        versions = self.stage._get_existing_versions("test_model")
        
        # Check versions
        self.assertEqual(len(versions), 2)
        self.assertIn("v1.0.0-20250101", versions)
        self.assertIn("v1.0.1-20250102", versions)
        self.assertNotIn("v2.0.0-20250103", versions)  # Different model
        
        # Clean up
        import shutil
        shutil.rmtree(self.stage.deployment_dir)
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"models": []}')
    @patch('json.dump')
    def test_register_model(self, mock_json_dump, mock_file):
        """Test model registration."""
        package = {
            "model_type": "test_model",
            "version": "v1.0.0-20250101",
            "created_at": "2025-01-01T00:00:00",
            "evaluation_results": {"accuracy": 0.85}
        }
        
        registry_entry = self.stage._register_model(package, self.context)
        
        # Check registry entry
        self.assertEqual(registry_entry["model_id"], "test_model_v1.0.0-20250101")
        self.assertEqual(registry_entry["model_type"], "test_model")
        self.assertEqual(registry_entry["version"], "v1.0.0-20250101")
        self.assertEqual(registry_entry["status"], "registered")
        self.assertEqual(registry_entry["tags"], ["test", "unit-test"])
        
        # Verify file operations
        mock_file.assert_called()
        mock_json_dump.assert_called_once()
        
    def test_deploy_model_strategies(self):
        """Test different deployment strategies."""
        package = {
            "model_type": "test_model",
            "version": "v1.0.0"
        }
        
        # Test direct deployment
        with patch.object(self.stage, '_direct_deployment', return_value={"method": "direct"}):
            result = self.stage._deploy_model(package, self.context)
            self.assertEqual(result["strategy"], "direct")
            self.assertIn("started_at", result)
            self.assertIn("completed_at", result)
            
        # Test blue-green deployment
        self.stage.deployment_strategy = "blue-green"
        with patch.object(self.stage, '_blue_green_deployment', return_value={"method": "blue-green"}):
            result = self.stage._deploy_model(package, self.context)
            self.assertEqual(result["strategy"], "blue-green")
            
        # Test canary deployment
        self.stage.deployment_strategy = "canary"
        with patch.object(self.stage, '_canary_deployment', return_value={"method": "canary"}):
            result = self.stage._deploy_model(package, self.context)
            self.assertEqual(result["strategy"], "canary")
            
        # Test unknown strategy
        self.stage.deployment_strategy = "unknown"
        with self.assertRaises(ValueError):
            self.stage._deploy_model(package, self.context)
            
    @patch('pickle.dump')
    def test_direct_deployment_local(self, mock_pickle):
        """Test direct deployment to local filesystem."""
        self.stage.deployment_dir = Path(tempfile.mkdtemp())
        
        package = {
            "model_type": "test_model",
            "version": "v1.0.0-20250101"
        }
        
        result = self.stage._direct_deployment(package)
        
        # Check result
        self.assertIn("model_location", result)
        self.assertIn("latest_link", result)
        self.assertEqual(result["deployment_method"], "file_system")
        
        # Check files were created
        model_path = Path(result["model_location"])
        self.assertTrue(model_path.name, "test_model_v1.0.0-20250101.pkl")
        
        # Clean up
        import shutil
        shutil.rmtree(self.stage.deployment_dir)
        
    def test_validate_deployment_success(self):
        """Test successful deployment validation."""
        # Create temporary deployment
        self.stage.deployment_dir = Path(tempfile.mkdtemp())
        model_file = self.stage.deployment_dir / "test_model.pkl"
        model_file.touch()
        
        deployment_result = {
            "model_location": str(model_file)
        }
        
        # Mock smoke tests
        with patch.object(self.stage, '_run_smoke_tests', return_value={"passed": True, "errors": []}):
            validation_result = self.stage._validate_deployment(deployment_result, self.context)
            
        # Check validation passed
        self.assertTrue(validation_result["success"])
        self.assertTrue(validation_result["checks"]["model_exists"])
        self.assertTrue(validation_result["checks"]["smoke_tests"])
        self.assertEqual(len(validation_result["errors"]), 0)
        
        # Clean up
        import shutil
        shutil.rmtree(self.stage.deployment_dir)
        
    def test_validate_deployment_failure(self):
        """Test failed deployment validation."""
        deployment_result = {
            "model_location": "/non/existent/path.pkl"
        }
        
        validation_result = self.stage._validate_deployment(deployment_result, self.context)
        
        # Check validation failed
        self.assertFalse(validation_result["success"])
        self.assertFalse(validation_result["checks"]["model_exists"])
        self.assertIn("Model file not found", validation_result["errors"])
        
    def test_rollback_deployment(self):
        """Test deployment rollback."""
        # Create temporary deployment
        self.stage.deployment_dir = Path(tempfile.mkdtemp())
        model_file = self.stage.deployment_dir / "test_model.pkl"
        model_file.touch()
        
        deployment_result = {
            "model_location": str(model_file)
        }
        
        # Perform rollback
        self.stage._rollback_deployment(deployment_result, self.context)
        
        # Check model was removed
        self.assertFalse(model_file.exists())
        
        # Clean up
        import shutil
        shutil.rmtree(self.stage.deployment_dir)
        
    def test_run_success(self):
        """Test successful deployment run."""
        # Set up context
        self.context.set("trained_model", {"type": "test"})
        self.context.set("model_type", "test_model")
        self.context.set("model_passed_thresholds", True)
        self.context.set("evaluation_results", {"accuracy": 0.85})
        self.context.set("artifact_store", Mock())
        
        # Mock deployment operations
        with patch.object(self.stage, '_create_deployment_package') as mock_package:
            with patch.object(self.stage, '_generate_model_version', return_value="v1.0.0"):
                with patch.object(self.stage, '_register_model'):
                    with patch.object(self.stage, '_deploy_model', return_value={"model_location": "/path/to/model"}):
                        with patch.object(self.stage, '_validate_deployment', return_value={"success": True, "errors": []}):
                            mock_package.return_value = {"model": "test"}
                            
                            self.stage.setup(self.context)
                            result_context = self.stage.run(self.context)
        
        # Check context updates
        deployment_info = result_context.get("deployment_info")
        self.assertIsNotNone(deployment_info)
        self.assertEqual(deployment_info["model_version"], "v1.0.0")
        self.assertEqual(deployment_info["deployment_target"], "local")
        self.assertEqual(deployment_info["deployment_strategy"], "direct")
        self.assertIn("deployment_timestamp", deployment_info)
        
    def test_run_validation_failure_with_rollback(self):
        """Test deployment with validation failure and rollback."""
        # Set up context
        self.context.set("trained_model", {"type": "test"})
        self.context.set("model_type", "test_model")
        self.context.set("model_passed_thresholds", True)
        self.context.set("artifact_store", Mock())
        
        # Mock deployment operations
        with patch.object(self.stage, '_create_deployment_package', return_value={"model": "test"}):
            with patch.object(self.stage, '_generate_model_version', return_value="v1.0.0"):
                with patch.object(self.stage, '_deploy_model', return_value={"model_location": "/path"}):
                    with patch.object(self.stage, '_validate_deployment', return_value={"success": False, "errors": ["Test error"]}):
                        with patch.object(self.stage, '_rollback_deployment'):
                            self.stage.setup(self.context)
                            
                            with self.assertRaises(RuntimeError) as cm:
                                self.stage.run(self.context)
                            self.assertIn("Deployment validation failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()