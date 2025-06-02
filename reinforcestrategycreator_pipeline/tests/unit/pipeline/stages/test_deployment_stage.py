import unittest
from unittest.mock import MagicMock, patch
import pandas as pd # Added for mock data

from reinforcestrategycreator_pipeline.src.pipeline.stages.deployment import DeploymentStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext, PipelineContextError
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.config.models import (
    PipelineConfig, DeploymentConfig, DataConfig, ModelConfig, TrainingConfig,
    EvaluationConfig, MonitoringConfig, ArtifactStoreConfig, ModelType, DataSourceType,
    EnvironmentType, ArtifactStoreType, MetadataBackend, CleanupPolicyConfig
)
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactStore
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry
from reinforcestrategycreator_pipeline.src.data.manager import DataManager
from reinforcestrategycreator_pipeline.src.monitoring.service import MonitoringService


class TestDeploymentStage(unittest.TestCase):

    def setUp(self):
        self.stage_name = "test_deployment_stage"
        self.stage_config = {}  # Basic config, can be overridden per test
        
        # Patch get_logger at the source to affect all stage instances
        self.patcher = patch('reinforcestrategycreator_pipeline.src.pipeline.stage.get_logger')
        self.mock_get_logger = self.patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger_instance

        self.deployment_stage = DeploymentStage(name=self.stage_name, config=self.stage_config)
        # self.deployment_stage.logger is now self.mock_logger_instance due to the patch

        # Reset PipelineContext singleton for each test
        PipelineContext._instance = None
        try:
            self.mock_pipeline_context = PipelineContext()
        except PipelineContextError: # Should not happen if _instance is reset
            PipelineContext.clear_instance() # Alternative way if direct _instance access is problematic
            self.mock_pipeline_context = PipelineContext()

        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.mock_artifact_store = MagicMock(spec=ArtifactStore)
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_data_manager = MagicMock(spec=DataManager)
        self.mock_monitoring_service = MagicMock(spec=MonitoringService)

        self.mock_pipeline_context.set('config_manager', self.mock_config_manager)
        self.mock_pipeline_context.set('artifact_store', self.mock_artifact_store)
        self.mock_pipeline_context.set('model_registry', self.mock_model_registry)
        self.mock_pipeline_context.set('data_manager', self.mock_data_manager)
        self.mock_pipeline_context.set('monitoring_service', self.mock_monitoring_service)

        # Basic pipeline config
        self.mock_data_config = DataConfig(source_id="test_source", source_type=DataSourceType.CSV, source_path="dummy.csv")
        self.mock_model_config = ModelConfig(model_type=ModelType.DQN)
        self.mock_training_config = TrainingConfig()
        self.mock_evaluation_config = EvaluationConfig()
        self.mock_deployment_config = DeploymentConfig(mode="paper_trading", initial_cash=100000.0)
        self.mock_monitoring_config = MonitoringConfig(enabled=True)
        self.mock_artifact_store_config = ArtifactStoreConfig(
            type=ArtifactStoreType.LOCAL,
            root_path="./artifacts",
            metadata_backend=MetadataBackend.JSON,
            cleanup_policy=CleanupPolicyConfig(enabled=False)
        )

        self.mock_pipeline_config = PipelineConfig(
            name="test_pipeline",
            version="1.0",
            environment=EnvironmentType.TESTING,
            data=self.mock_data_config,
            model=self.mock_model_config,
            training=self.mock_training_config,
            evaluation=self.mock_evaluation_config,
            deployment=self.mock_deployment_config,
            monitoring=self.mock_monitoring_config,
            artifact_store=self.mock_artifact_store_config,
            random_seed=42
        )
        self.mock_config_manager.get_config.return_value = self.mock_pipeline_config
        
        self.mock_pipeline_context.set('trained_model_artifact_id', 'test_model_id_123')
        self.mock_pipeline_context.set('trained_model_version', '1.0.0')

    def test_initialization(self):
        self.assertEqual(self.deployment_stage.name, self.stage_name)
        self.assertEqual(self.deployment_stage.config, self.stage_config)
        self.assertIsNotNone(self.deployment_stage.logger)

    def test_setup_successful(self):
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.assertEqual(self.deployment_stage.deployment_config, self.mock_deployment_config)
        self.assertEqual(self.deployment_stage.model_artifact_id, 'test_model_id_123')
        self.assertEqual(self.deployment_stage.model_version, '1.0.0')
        self.assertEqual(self.deployment_stage.monitoring_service, self.mock_monitoring_service)
        self.mock_config_manager.get_config.assert_called_once()
        # Check for log messages (optional, but good practice)
        # self.deployment_stage.logger.info.assert_any_call(...)

    def test_setup_missing_config_manager(self):
        self.mock_pipeline_context.set('config_manager', None)
        with self.assertRaisesRegex(ValueError, "ConfigManager not found in PipelineContext during DeploymentStage setup."):
            self.deployment_stage.setup(self.mock_pipeline_context)

    def test_setup_missing_deployment_config_in_pipeline_config(self):
        # Simulate pipeline_config.deployment being None
        mock_pipeline_config_no_deployment = MagicMock(spec=PipelineConfig)
        mock_pipeline_config_no_deployment.deployment = None
        self.mock_config_manager.get_config.return_value = mock_pipeline_config_no_deployment
        
        # No need for with patch.object here as self.mock_logger is already a MagicMock
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.assertIsNone(self.deployment_stage.deployment_config) # Should be None as per the mock_pipeline_config_no_deployment
        self.mock_logger_instance.warning.assert_any_call("No 'deployment' configuration found in pipeline_config.deployment.")

    def test_setup_missing_trained_model_artifact_id(self):
        self.mock_pipeline_context.set('trained_model_artifact_id', None)
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.assertIsNone(self.deployment_stage.model_artifact_id)
        self.mock_logger_instance.warning.assert_any_call("Trained model artifact ID not found in PipelineContext. Deployment may not be possible.")

    def test_setup_with_monitoring_service(self):
        # Default setup already includes monitoring service
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.assertEqual(self.deployment_stage.monitoring_service, self.mock_monitoring_service)
        self.mock_logger_instance.info.assert_any_call("MonitoringService retrieved from context.")

    def test_setup_missing_monitoring_service(self):
        self.mock_pipeline_context.set('monitoring_service', None)
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.assertIsNone(self.deployment_stage.monitoring_service)
        self.mock_logger_instance.warning.assert_any_call("MonitoringService not found in context. Monitoring will be disabled for this stage.")

    def test_run_successful_paper_trading(self):
        # Modify the existing mock_deployment_config for this specific test
        self.mock_deployment_config.paper_trading_data_source_id = "paper_trade_data"
        self.mock_deployment_config.paper_trading_data_params = {}
        
        # Re-initialize PipelineConfig with the modified deployment config for this test
        current_pipeline_config = PipelineConfig(
            name="test_pipeline_paper_trading",
            version="1.0",
            environment=EnvironmentType.TESTING,
            data=self.mock_data_config,
            model=self.mock_model_config,
            training=self.mock_training_config,
            evaluation=self.mock_evaluation_config,
            deployment=self.mock_deployment_config, # Use the modified one
            monitoring=self.mock_monitoring_config,
            artifact_store=self.mock_artifact_store_config,
            random_seed=42
        )
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        
        # Setup stage first
        self.deployment_stage.setup(self.mock_pipeline_context)

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [1, 0, 1] # Example predictions (buy, hold, buy)
        self.mock_model_registry.load_model.return_value = mock_model_instance

        # Mock data for paper trading
        mock_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        
        self.mock_data_manager.load_data.return_value = mock_data
        
        # Ensure 'evaluation_data_output' is None in context to force loading via paper_trading_data_source_id
        self.mock_pipeline_context.set("evaluation_data_output", None)


        returned_context = self.deployment_stage.run(self.mock_pipeline_context)

        self.mock_model_registry.load_model.assert_called_once_with(
            model_id='test_model_id_123',
            version='1.0.0'
        )
        self.mock_data_manager.load_data.assert_called_once_with("paper_trade_data") # Check if called with correct source ID
        mock_model_instance.predict.assert_called_once_with(mock_data)
        
        # Check monitoring calls
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.started", description=f"Stage {self.stage_name} started.")
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.paper_trade.executed", description="Simulated BUY trade executed for DUMMY_ASSET.", level="info", context={"asset": "DUMMY_ASSET", "action": "BUY", "shares": 10, "price": 100.0})
        self.mock_monitoring_service.log_metric.assert_any_call(f"{self.stage_name}.paper_trade.cash", 99000.0) # 100000 - 10*100
        self.mock_monitoring_service.log_metric.assert_any_call(f"{self.stage_name}.paper_trade.holdings.DUMMY_ASSET.shares", 10)
        self.mock_monitoring_service.log_metric.assert_any_call(f"{self.stage_name}.paper_trade.pnl", 0.0)
        self.mock_monitoring_service.log_metric.assert_any_call(f"{self.stage_name}.paper_trade.portfolio_value", 100000.0) # 99000 cash + 10*100 asset value
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.completed", description=f"Stage {self.stage_name} completed successfully.", level="info")

        # Check portfolio in context
        final_portfolio = returned_context.get(f"{self.stage_name}_paper_trading_portfolio")
        self.assertIsNotNone(final_portfolio)
        self.assertEqual(final_portfolio['cash'], 99000.0)
        self.assertIn('DUMMY_ASSET', final_portfolio['holdings'])
        self.assertEqual(final_portfolio['holdings']['DUMMY_ASSET']['shares'], 10)
        self.assertEqual(len(final_portfolio['trades']), 1)
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "completed")

    def test_run_no_model_artifact_id(self):
        # No need to override model_artifact_id on the instance, setup handles it from context
        self.mock_pipeline_context.set('trained_model_artifact_id', None)
        self.deployment_stage.setup(self.mock_pipeline_context) # Re-run setup

        returned_context = self.deployment_stage.run(self.mock_pipeline_context)
        
        self.mock_logger_instance.error.assert_any_call("No model_artifact_id found in setup. Cannot proceed with deployment.")
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.failed", description="No model_artifact_id found.", level="error")
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_no_model_id")

    def test_run_no_model_registry(self):
        self.mock_pipeline_context.set('model_registry', None)
        self.deployment_stage.setup(self.mock_pipeline_context)

        returned_context = self.deployment_stage.run(self.mock_pipeline_context)

        self.mock_logger_instance.error.assert_any_call("ModelRegistry not found in PipelineContext.")
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.failed", description="ModelRegistry not found.", level="error")
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_no_model_registry")

    def test_run_model_load_failure(self):
        self.deployment_stage.setup(self.mock_pipeline_context)
        self.mock_model_registry.load_model.side_effect = Exception("Model load error")

        returned_context = self.deployment_stage.run(self.mock_pipeline_context)

        self.mock_logger_instance.error.assert_any_call("Failed to load model 'test_model_id_123': Model load error")
        self.mock_monitoring_service.log_event.assert_any_call(event_type=f"{self.stage_name}.model_load.failed", description="Failed to load model: Model load error", level="error", context={"model_id": "test_model_id_123", "error_details": "Model load error"})
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_model_load_failed: Model load error")

    def test_run_paper_trading_no_data_manager(self):
        self.mock_pipeline_context.set('data_manager', None)
        self.deployment_stage.setup(self.mock_pipeline_context)
        
        # Mock model loading to proceed to data fetching part
        mock_model_instance = MagicMock()
        self.mock_model_registry.load_model.return_value = mock_model_instance

        returned_context = self.deployment_stage.run(self.mock_pipeline_context)

        self.mock_logger_instance.error.assert_any_call("DataManager not found in PipelineContext. Cannot fetch data for paper trading.")
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_no_data_manager")

    def test_run_paper_trading_data_load_failure(self):
        # Modify deployment_config for this test
        failing_deployment_config = self.mock_deployment_config.model_copy(deep=True)
        failing_deployment_config.paper_trading_data_source_id = "failing_data_source"
        current_pipeline_config = self.mock_pipeline_config.model_copy(update={"deployment": failing_deployment_config})
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)
        
        mock_model_instance = MagicMock()
        self.mock_model_registry.load_model.return_value = mock_model_instance
        
        self.mock_data_manager.load_data.side_effect = Exception("Data load error")
        self.mock_pipeline_context.set("evaluation_data_output", None)


        returned_context = self.deployment_stage.run(self.mock_pipeline_context)

        self.mock_logger_instance.error.assert_any_call("Failed to load data for paper trading from source 'failing_data_source': Data load error")
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_data_load_failed: Data load error")

    def test_run_paper_trading_no_data_available(self):
        no_data_source_deployment_config = self.mock_deployment_config.model_copy(deep=True)
        no_data_source_deployment_config.paper_trading_data_source_id = None
        current_pipeline_config = self.mock_pipeline_config.model_copy(update={"deployment": no_data_source_deployment_config})
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)

        mock_model_instance = MagicMock()
        self.mock_model_registry.load_model.return_value = mock_model_instance
        
        self.mock_pipeline_context.set("evaluation_data_output", None)

        returned_context = self.deployment_stage.run(self.mock_pipeline_context)
        self.mock_logger_instance.error.assert_any_call("No data available for paper trading (neither in context via 'evaluation_data_output' nor via 'paper_trading_data_source_id').")
        self.assertEqual(returned_context.get_metadata(f"{self.stage_name}_status"), "error_no_paper_trading_data")


    def test_run_paper_trading_model_no_predict_method(self):
        # Specific deployment config for this test
        current_deployment_config = self.mock_deployment_config.model_copy(deep=True)
        current_deployment_config.paper_trading_data_source_id = "some_data"
        current_pipeline_config = self.mock_pipeline_config.model_copy(update={"deployment": current_deployment_config})
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)

        mock_model_no_predict = MagicMock()
        del mock_model_no_predict.predict
        self.mock_model_registry.load_model.return_value = mock_model_no_predict

        mock_data = pd.DataFrame({'feature1': [1, 2, 3]})
        self.mock_data_manager.load_data.return_value = mock_data
        self.mock_pipeline_context.set("evaluation_data_output", None)

        self.deployment_stage.run(self.mock_pipeline_context)
        self.mock_logger_instance.warning.assert_any_call("Model does not have a predict method or no data for paper trading. Skipping simulation loop.")

    def test_run_live_trading_mode_not_implemented(self):
        live_deployment_config = self.mock_deployment_config.model_copy(update={"mode": "live_trading"})
        current_pipeline_config = self.mock_pipeline_config.model_copy(update={"deployment": live_deployment_config})
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)
        
        mock_model_instance = MagicMock()
        self.mock_model_registry.load_model.return_value = mock_model_instance

        self.deployment_stage.run(self.mock_pipeline_context)
        self.mock_logger_instance.warning.assert_any_call("Live trading mode is configured but not yet implemented in this stage. Skipping.")

    def test_run_unknown_deployment_mode(self):
        unknown_mode_deployment_config = self.mock_deployment_config.model_copy(update={"mode": "unknown_mode"})
        current_pipeline_config = self.mock_pipeline_config.model_copy(update={"deployment": unknown_mode_deployment_config})
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)

        mock_model_instance = MagicMock()
        self.mock_model_registry.load_model.return_value = mock_model_instance

        self.deployment_stage.run(self.mock_pipeline_context)
        self.mock_logger_instance.warning.assert_any_call(
            f"Deployment mode 'unknown_mode' is not "
            f"recognized or not yet implemented for this stage. Skipping."
        )
            
    def test_run_general_exception_handling(self):
        # Setup to a point where paper trading would normally complete or be skipped
        self.mock_deployment_config.mode = "paper_trading" # Ensure a known path
        # Make paper trading data source None so it skips the inner simulation loop quickly
        self.mock_deployment_config.paper_trading_data_source_id = None
        self.mock_pipeline_context.set("evaluation_data_output", None)

        current_pipeline_config = self.mock_pipeline_config.model_copy(
            update={"deployment": self.mock_deployment_config}
        )
        self.mock_config_manager.get_config.return_value = current_pipeline_config
        self.deployment_stage.setup(self.mock_pipeline_context)
        
        # Mock model loading to succeed
        self.mock_model_registry.load_model.return_value = MagicMock()

        # Make the monitoring service fail on the *final* log_event call for stage completion
        # This ensures the exception happens outside the inner try-except e_sim block
        # but within the main try block of the run method.
        expected_exception_message = "Monitoring service final log failed"
        self.mock_monitoring_service.log_event.side_effect = [
            None,  # For stage.started
            # Potentially other log_event calls from paper trading if it ran (it will be skipped here)
            Exception(expected_exception_message) # This will be for stage.completed
        ]
        
        # If paper trading runs and makes calls, we need to account for them in side_effect:
        # For this specific setup (no data source, no eval data), paper trading simulation is skipped.
        # So, the side_effect list should be [None (for started), Exception (for completed)]

        with self.assertRaisesRegex(Exception, expected_exception_message):
            self.deployment_stage.run(self.mock_pipeline_context)
        
        # Check that the outer exception handler in run() logged the event
        self.mock_monitoring_service.log_event.assert_any_call(
            event_type=f"{self.stage_name}.failed",
            description=f"Stage {self.stage_name} failed: {expected_exception_message}",
            level="error",
            context={"error_details": expected_exception_message}
        )
        self.assertEqual(self.mock_pipeline_context.get_metadata(f"{self.stage_name}_status"), f"error_stage_run: {expected_exception_message}")

    def tearDown(self):
        self.patcher.stop()

    def test_teardown(self):
        # Setup and run are not strictly necessary for teardown unless it depends on their state
        # self.deployment_stage.setup(self.mock_pipeline_context)
        # self.deployment_stage.run(self.mock_pipeline_context)
        
        self.deployment_stage.teardown(self.mock_pipeline_context)
        self.mock_logger_instance.info.assert_any_call(f"Tearing down stage: {self.stage_name}")
        # Add more assertions if teardown logic is implemented

if __name__ == '__main__':
    unittest.main()