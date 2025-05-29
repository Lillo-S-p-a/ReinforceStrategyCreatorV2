import unittest
from unittest.mock import MagicMock, patch, call, mock_open
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stages.deployment import DeploymentStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactMetadata, ArtifactType

class TestDeploymentStageIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None
        self.context = PipelineContext.get_instance()

        self.mock_artifact_store = MagicMock()
        self.context.set("artifact_store", self.mock_artifact_store)
        self.context.set_metadata("run_id", "test_deploy_run_abc")

        self.mock_logger_patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger.return_value = self.mock_logger_instance

        self.stage_config = {
            "deployment_target": "local", # Focus on local for easier testing of file ops
            "model_registry": {"enabled": True, "tags": ["int-test"]},
            "deployment_strategy": "direct",
            "validation_config": {"smoke_tests": True},
            "rollback_config": {"auto_rollback": True},
            "packaging_format": "pickle"
        }
        self.stage = DeploymentStage(config=self.stage_config)

        # Prepare dummy data for context
        self.context.set("trained_model", {"model_content": "dummy_trained_model_data"})
        self.context.set("model_type", "super_classifier")
        self.context.set("model_config", {"layers": 3})
        self.context.set("training_metadata", {"epochs_trained": 100})
        self.context.set("evaluation_results", {"accuracy": 0.95})
        self.context.set("model_passed_thresholds", True)
        
        # Ensure deployment_dir is set for local target if _initialize_deployment_target is called in setup
        # For these tests, we'll often mock _initialize_deployment_target or ensure it runs safely.
        # For local, it creates 'deployed_models'. We'll use a temp dir for self.stage.deployment_dir
        self.stage.deployment_dir = self.test_dir / "deployed_models_test" 
        # self.stage.deployment_dir.mkdir(exist_ok=True) # _initialize_deployment_target would do this

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.mock_logger_patcher.stop()
        if PipelineContext._instance:
            PipelineContext._instance.reset()
            PipelineContext._instance = None

    def test_deployment_run_successful_direct_local(self):
        # --- Mock internal helper methods to simulate service interactions ---
        mock_package = {
            "model": self.context.get("trained_model"), 
            "model_type": self.context.get("model_type"),
            "version": "v1.0.0-testts", # This will be set by _generate_model_version patch
            "created_at": "test_creation_time"
        }
        mock_model_version = "v1.0.0-testts123"
        mock_registry_entry = {"model_id": f"{self.context.get('model_type')}_{mock_model_version}", "status": "registered"}
        
        # For _direct_deployment, we want it to actually interact with the temp file system
        # So, we won't mock _direct_deployment itself, but ensure deployment_dir is set.
        # We will mock _get_existing_versions to control version generation.
        mock_deployment_location = str(self.stage.deployment_dir / f"{self.context.get('model_type')}_{mock_model_version}.pkl")
        mock_deployment_result = {
            "model_location": mock_deployment_location,
            "latest_link": str(self.stage.deployment_dir / f"{self.context.get('model_type')}_latest.pkl"),
            "deployment_method": "file_system"
        }
        mock_validation_result = {"success": True, "checks": {"model_exists": True, "smoke_tests": True}, "errors": []}

        # Mock the local model_registry.json file interaction
        mock_registry_file_path = Path("model_registry.json") # Relative to where test runs
        if mock_registry_file_path.exists(): mock_registry_file_path.unlink() # Clean before test

        with patch.object(self.stage, '_create_deployment_package', return_value=mock_package) as mock_create_pkg, \
             patch.object(self.stage, '_generate_model_version', return_value=mock_model_version) as mock_gen_ver, \
             patch.object(self.stage, '_register_model', return_value=mock_registry_entry) as mock_reg_model, \
             patch.object(self.stage, '_validate_deployment', return_value=mock_validation_result) as mock_validate, \
             patch.object(self.stage, '_save_deployment_artifacts') as mock_save_artifacts, \
             patch.object(self.stage, '_initialize_deployment_target') as mock_init_target: # Mock init to control deployment_dir
            
            # Ensure deployment_dir is set correctly by the stage logic if _initialize_deployment_target was not mocked
            # For this test, we set it manually and mock _initialize_deployment_target to prevent side effects.
            # If _initialize_deployment_target is called, it would use self.stage.deployment_dir
            self.stage.deployment_dir.mkdir(parents=True, exist_ok=True) # Ensure the test-specific deployment dir exists
            
            # --- Execute Stage ---
            self.stage.setup(self.context) # Calls _initialize_deployment_target
            result_context = self.stage.run(self.context)

            # --- Assertions ---
            mock_init_target.assert_called_once() # Called during setup
            mock_create_pkg.assert_called_once_with(self.context.get("trained_model"), self.context)
            mock_gen_ver.assert_called_once_with(self.context)
            
            # Check that the package passed to _register_model has the version from _generate_model_version
            actual_package_for_register = mock_reg_model.call_args[0][0]
            self.assertEqual(actual_package_for_register["version"], mock_model_version)
            mock_reg_model.assert_called_once_with(unittest.mock.ANY, self.context) # ANY for package with updated version

            # _deploy_model calls _direct_deployment. Check if the file was created.
            deployed_file = Path(mock_deployment_location)
            self.assertTrue(deployed_file.exists())
            latest_symlink = Path(mock_deployment_result["latest_link"])
            self.assertTrue(latest_symlink.is_symlink())
            self.assertEqual(Path(latest_symlink.resolve()), deployed_file.resolve())

            mock_validate.assert_called_once()
            mock_save_artifacts.assert_called_once()

            # Check context updates
            deployment_info = result_context.get("deployment_info")
            self.assertIsNotNone(deployment_info)
            self.assertEqual(deployment_info["model_version"], mock_model_version)
            self.assertEqual(deployment_info["deployment_target"], "local")
            self.assertEqual(deployment_info["model_location"], mock_deployment_location)
            self.assertEqual(result_context.get("model_registry_entry"), mock_registry_entry)

        if mock_registry_file_path.exists(): mock_registry_file_path.unlink() # Clean after test


    def test_deployment_run_validation_fails_and_rolls_back_local(self):
        mock_package = {"model_type": "rollback_model", "version": "v0.0.1-fail"} # version will be updated
        mock_model_version = "v0.0.1-faildeploy"
        mock_registry_entry = {"model_id": f"rollback_model_{mock_model_version}"}
        
        # Simulate _direct_deployment creating a file
        deployed_file_path = self.stage.deployment_dir / f"rollback_model_{mock_model_version}.pkl"
        
        mock_deployment_result_before_validation = {
            "model_location": str(deployed_file_path),
            "latest_link": str(self.stage.deployment_dir / "rollback_model_latest.pkl"),
            "deployment_method": "file_system"
        }
        mock_validation_result_failed = {"success": False, "checks": {"smoke_tests": False}, "errors": ["Smoke test kaboom!"]}

        with patch.object(self.stage, '_create_deployment_package', return_value=mock_package), \
             patch.object(self.stage, '_generate_model_version', return_value=mock_model_version), \
             patch.object(self.stage, '_register_model', return_value=mock_registry_entry), \
             patch.object(self.stage, '_deploy_model', return_value=mock_deployment_result_before_validation) as mock_deploy_call, \
             patch.object(self.stage, '_validate_deployment', return_value=mock_validation_result_failed) as mock_validate, \
             patch.object(self.stage, '_initialize_deployment_target'):

            # Simulate file creation by _deploy_model (which calls _direct_deployment)
            def simulate_deploy_and_create_file(*args, **kwargs):
                self.stage.deployment_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                deployed_file_path.touch() # Create the dummy file
                return mock_deployment_result_before_validation
            mock_deploy_call.side_effect = simulate_deploy_and_create_file
            
            self.stage.setup(self.context)
            with self.assertRaisesRegex(RuntimeError, "Deployment validation failed, rolled back"):
                self.stage.run(self.context)

            mock_validate.assert_called_once()
            # We can't assert mock_rollback was called if we don't mock it.
            # Instead, we rely on the RuntimeError and the file non-existence.
            
            # Check if the file that was "deployed" is now removed by the rollback logic
            # This assumes _rollback_deployment for local target actually unlinks the file.
            # The actual _rollback_deployment in source does this.
            self.assertFalse(deployed_file_path.exists())


if __name__ == '__main__':
    unittest.main()