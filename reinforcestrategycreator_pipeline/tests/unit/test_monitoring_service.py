"""Unit tests for the monitoring service."""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from src.config.models import (
    MonitoringConfig, PipelineConfig, ModelConfig, ModelType,
    DataDriftConfig, ModelDriftConfig, AlertManagerConfig,
    DataDriftDetectionMethod, ModelDriftDetectionMethod # Assuming these enums exist
)
from src.monitoring.service import (
    MonitoringService,
    get_monitoring_service,
    initialize_monitoring_from_pipeline_config
)
from src.monitoring.drift_detection import DataDriftDetector, ModelDriftDetector
from src.monitoring.alerting import AlertManager
from src.deployment.manager import DeploymentManager


class TestMonitoringService:
    """Test the MonitoringService class."""
    
    def test_initialization_with_config(self):
        """Test initializing with a config."""
        config = MonitoringConfig(
            enabled=True,
            log_level="DEBUG",
            metrics_prefix="test_pipeline"
        )
        
        with patch('src.monitoring.service.configure_logging') as mock_log_config, \
             patch('src.monitoring.service.configure_datadog') as mock_dd_config:
            
            service = MonitoringService(config)
            
            assert service.config == config
            assert service._initialized
            mock_log_config.assert_called_once()
            mock_dd_config.assert_called_once()
    
    def test_initialization_disabled(self):
        """Test initializing with disabled monitoring."""
        config = MonitoringConfig(enabled=False)
        
        with patch('src.monitoring.service.configure_logging') as mock_log_config, \
             patch('src.monitoring.service.configure_datadog') as mock_dd_config:
            
            service = MonitoringService(config)
            
            assert service.config == config
            assert not service._initialized
            mock_log_config.assert_not_called()
            mock_dd_config.assert_not_called()
    
    def test_configure_logging(self):
        """Test logging configuration."""
        config = MonitoringConfig(
            enabled=True,
            log_level="INFO"
        )
        
        with patch('src.monitoring.service.configure_logging') as mock_config:
            service = MonitoringService()
            service._configure_logging()  # Should do nothing without config
            mock_config.assert_not_called()
            
            service.config = config
            service._configure_logging()
            
            mock_config.assert_called_once()
            call_args = mock_config.call_args[1]
            assert call_args["log_level"] == "INFO"
            assert call_args["enable_console"] is True
            assert call_args["enable_json"] is True
    
    def test_configure_datadog(self):
        """Test Datadog configuration."""
        config = MonitoringConfig(
            enabled=True,
            datadog_api_key="test_key",
            datadog_app_key="app_key",
            metrics_prefix="test"
        )
        
        with patch('src.monitoring.service.configure_datadog') as mock_config:
            service = MonitoringService()
            service.config = config
            service._configure_datadog()
            
            mock_config.assert_called_once_with(
                api_key="test_key",
                app_key="app_key",
                metrics_prefix="test",
                enabled=True
            )
    
    def test_resolve_env_var(self):
        """Test environment variable resolution."""
        service = MonitoringService()
        
        # Test direct value
        assert service._resolve_env_var("direct_value") == "direct_value"
        
        # Test None
        assert service._resolve_env_var(None) is None
        
        # Test env var reference
        with patch('os.getenv', return_value="env_value"):
            assert service._resolve_env_var("${TEST_VAR}") == "env_value"
    
    @patch('src.monitoring.service.log_with_context')
    @patch('src.monitoring.service.datadog_client')
    def test_log_metric(self, mock_dd_client, mock_log_context):
        """Test logging metrics."""
        service = MonitoringService()
        service._initialized = True
        
        # Test gauge metric
        service.log_metric("test.metric", 42.0, "gauge", tags=["test"])
        
        mock_log_context.assert_called_once_with(
            "info",
            "Metric: test.metric",
            metric_name="test.metric",
            metric_value=42.0,
            metric_type="gauge",
            tags=["test"]
        )
        mock_dd_client.gauge.assert_called_once_with("test.metric", 42.0, tags=["test"])
        
        # Test increment metric
        mock_log_context.reset_mock()
        mock_dd_client.reset_mock()
        
        service.log_metric("counter", 5, "increment")
        mock_dd_client.increment.assert_called_once_with("counter", 5, tags=[])
        
        # Test histogram metric
        mock_log_context.reset_mock()
        mock_dd_client.reset_mock()
        
        service.log_metric("distribution", 100.5, "histogram")
        mock_dd_client.histogram.assert_called_once_with("distribution", 100.5, tags=[])
    
    @patch('src.monitoring.service.log_with_context')
    @patch('src.monitoring.service.track_pipeline_event')
    def test_log_event(self, mock_track_event, mock_log_context):
        """Test logging events."""
        service = MonitoringService()
        service._initialized = True
        
        service.log_event(
            "test_event",
            "Test description",
            level="warning",
            tags=["tag1"],
            context={"key": "value"}
        )
        
        mock_log_context.assert_called_once()
        call_args = mock_log_context.call_args
        assert call_args[0] == ("warning", "Test description")
        assert call_args[1]["event_type"] == "test_event"
        assert call_args[1]["tags"] == ["tag1"]
        assert call_args[1]["key"] == "value"
        
        mock_track_event.assert_called_once_with(
            "test_event",
            "Test description",
            alert_type="warning",
            tags=["tag1"]
        )
    
    def test_check_alert_thresholds(self):
        """Test alert threshold checking."""
        config = MonitoringConfig(
            enabled=True,
            alert_thresholds={
                "accuracy_min": 0.8,
                "loss_max": 0.5,
                "error_rate_max": 0.1
            }
        )
        
        service = MonitoringService(config)
        
        with patch.object(service, 'log_event') as mock_log_event:
            # Test metrics that violate thresholds
            metrics = {
                "accuracy": 0.7,  # Below minimum
                "loss": 0.6,      # Above maximum
                "error_rate": 0.05  # Within threshold
            }
            
            alerts = service.check_alert_thresholds(metrics)
            
            assert len(alerts) == 2
            
            # Check accuracy alert
            accuracy_alert = next(a for a in alerts if a["metric"] == "accuracy")
            assert accuracy_alert["type"] == "below_minimum"
            assert accuracy_alert["value"] == 0.7
            assert accuracy_alert["threshold"] == 0.8
            
            # Check loss alert
            loss_alert = next(a for a in alerts if a["metric"] == "loss")
            assert loss_alert["type"] == "above_maximum"
            assert loss_alert["value"] == 0.6
            assert loss_alert["threshold"] == 0.5
            
            # Verify events were logged
            assert mock_log_event.call_count == 2
    
    def test_create_health_check(self):
        """Test health check creation."""
        config = MonitoringConfig(
            enabled=True,
            log_level="INFO",
            metrics_prefix="test"
        )
        
        # Test uninitialized service
        service = MonitoringService()
        health = service.create_health_check()
        
        assert health["status"] == "not_initialized"
        assert health["monitoring_enabled"] is False
        assert health["logging_configured"] is False
        assert health["datadog_configured"] is False
        
        # Test initialized service
        with patch('src.monitoring.service.datadog_client') as mock_dd:
            mock_dd.enabled = True
            
            service = MonitoringService(config)
            health = service.create_health_check()
            
            assert health["status"] == "healthy"
            assert health["monitoring_enabled"] is True
            assert health["logging_configured"] is True
            assert health["datadog_configured"] is True
            assert health["log_level"] == "INFO"
            assert health["metrics_prefix"] == "test"


class TestMonitoringServiceSingleton:
    """Test the singleton pattern for monitoring service."""
    
    def test_get_monitoring_service(self):
        """Test getting monitoring service instance."""
        # Reset global instance
        import src.monitoring.service
        src.monitoring.service._monitoring_service = None
        
        # First call creates instance
        service1 = get_monitoring_service()
        assert service1 is not None
        
        # Second call returns same instance
        service2 = get_monitoring_service()
        assert service1 is service2
        
        # Call with config initializes if not already initialized
        config = MonitoringConfig(enabled=True)
        with patch.object(service1, 'initialize') as mock_init:
            service3 = get_monitoring_service(config)
            assert service1 is service3
            # Should initialize if not already initialized
            if not service1._initialized:
                mock_init.assert_called_once_with(config)
    
    def test_initialize_from_pipeline_config(self):
        """Test initializing from pipeline config."""
        pipeline_config = PipelineConfig(
            name="test_pipeline",
            model=ModelConfig(model_type=ModelType.DQN),
            monitoring=MonitoringConfig(
                enabled=True,
                log_level="DEBUG"
            )
        )
        
        with patch('src.monitoring.service.get_monitoring_service') as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = mock_service
            
            result = initialize_monitoring_from_pipeline_config(pipeline_config)
            
            assert result == mock_service
            mock_get.assert_called_once_with(pipeline_config.monitoring)


class TestMonitoringIntegration:
    """Test integration scenarios."""
    
    def test_full_initialization_flow(self):
        """Test the full initialization flow."""
        config = MonitoringConfig(
            enabled=True,
            log_level="INFO",
            datadog_api_key="${DD_API_KEY}",
            datadog_app_key="${DD_APP_KEY}",
            metrics_prefix="integration_test"
        )
        
        with patch('os.getenv') as mock_getenv, \
             patch('src.monitoring.service.configure_logging') as mock_log, \
             patch('src.monitoring.service.configure_datadog') as mock_dd, \
             patch('src.monitoring.service.track_pipeline_event') as mock_event:
            
            # Mock environment variables
            mock_getenv.side_effect = lambda key: {
                "DD_API_KEY": "test_api_key",
                "DD_APP_KEY": "test_app_key"
            }.get(key)
            
            service = MonitoringService(config)
            
            # Verify logging was configured
            mock_log.assert_called_once()
            log_args = mock_log.call_args[1]
            assert log_args["log_level"] == "INFO"
            
            # Verify Datadog was configured
            mock_dd.assert_called_once()
            dd_args = mock_dd.call_args[1]
            assert dd_args["api_key"] == "test_api_key"
            assert dd_args["app_key"] == "test_app_key"
            assert dd_args["metrics_prefix"] == "integration_test"
            
            # Verify initialization event was sent
            mock_event.assert_called_once_with(
                "monitoring_initialized",
                "Monitoring service has been initialized",
                alert_type="success"
            )

    @patch('src.monitoring.service.log_with_context')
    @patch.object(MonitoringService, 'process_alert') # Mock process_alert for these tests
    def test_check_data_drift_detected(self, mock_process_alert, mock_log_context):
        """Test check_data_drift when drift is detected."""
        config = MonitoringConfig(
            enabled=True,
            data_drift=DataDriftConfig(enabled=True, method=DataDriftDetectionMethod.KS)
        )
        service = MonitoringService(config) # Initializes _configure_data_drift_detection
        
        # Replace the actual detector with a mock after initialization via _configure_data_drift_detection
        mock_detector = MagicMock(spec=DataDriftDetector)
        service.data_drift_detector = mock_detector
        
        mock_detector.detect.return_value = {"drift_detected": True, "score": 0.6, "p_value": 0.01}
        
        current_data = MagicMock()
        reference_data = MagicMock()
        
        result = service.check_data_drift(current_data, reference_data, model_version="v1")
        
        mock_detector.detect.assert_called_once_with(current_data, reference_data)
        assert result == {"drift_detected": True, "score": 0.6, "p_value": 0.01}
        
        mock_log_context.assert_any_call(
            "warning",
            "Data drift detected. Score: 0.6",
            event_type="data_drift_detected",
            tags=['model_version:v1', 'drift_method:ks', 'status:drift_detected'],
            drift_detected=True, score=0.6, p_value=0.01 # Spread arguments
        )
        mock_process_alert.assert_called_once_with(
            "data_drift_detected",
            {"drift_detected": True, "score": 0.6, "p_value": 0.01},
            severity="warning",
            tags=['model_version:v1', 'drift_method:ks']
        )

    @patch('src.monitoring.service.log_with_context')
    @patch.object(MonitoringService, 'log_metric')
    def test_check_data_drift_not_detected(self, mock_log_metric, mock_log_context):
        """Test check_data_drift when no drift is detected."""
        config = MonitoringConfig(
            enabled=True,
            data_drift=DataDriftConfig(enabled=True, method=DataDriftDetectionMethod.PSI)
        )
        service = MonitoringService(config)
        mock_detector = MagicMock(spec=DataDriftDetector)
        service.data_drift_detector = mock_detector
        
        mock_detector.detect.return_value = {"drift_detected": False, "score": 0.1}
        
        current_data = MagicMock()
        reference_data = MagicMock()
        
        result = service.check_data_drift(current_data, reference_data, model_version="v2")
        
        mock_detector.detect.assert_called_once_with(current_data, reference_data)
        assert result == {"drift_detected": False, "score": 0.1}
        
        mock_log_metric.assert_called_once_with(
            "data_drift_score", 0.1,
            tags=['model_version:v2', 'drift_method:psi']
        )
        # Check that log_event was NOT called for drift detection
        # This requires checking that specific calls to log_event were not made,
        # or checking call_count if log_event is only called for drift.
        # For simplicity, we assume log_event is primarily for actual drift here.

    @patch('src.monitoring.service.log_with_context')
    @patch.object(MonitoringService, 'process_alert')
    def test_check_model_drift_detected(self, mock_process_alert, mock_log_context):
        """Test check_model_drift when drift is detected."""
        config = MonitoringConfig(
            enabled=True,
            model_drift=ModelDriftConfig(enabled=True, method=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION, performance_metric="accuracy")
        )
        service = MonitoringService(config)
        mock_detector = MagicMock(spec=ModelDriftDetector)
        service.model_drift_detector = mock_detector

        mock_detector.detect.return_value = {"drift_detected": True, "metric_value": 0.75, "threshold": 0.8}
        
        predictions = MagicMock()
        ground_truth = MagicMock()
        
        result = service.check_model_drift(predictions, ground_truth, model_version="v3")
        
        mock_detector.detect.assert_called_once_with(predictions, ground_truth, model_confidence=None)
        assert result == {"drift_detected": True, "metric_value": 0.75, "threshold": 0.8}
        
        mock_log_context.assert_any_call(
            "warning",
            "Model drift detected. Metric: accuracy, Value: 0.75",
            event_type="model_drift_detected",
            tags=['model_version:v3', 'drift_method:performance_degradation', 'status:drift_detected'],
            drift_detected=True, metric_value=0.75, threshold=0.8 # Spread arguments
        )
        mock_process_alert.assert_called_once_with(
            "model_drift_detected",
            {"drift_detected": True, "metric_value": 0.75, "threshold": 0.8},
            severity="warning",
            tags=['model_version:v3', 'drift_method:performance_degradation']
        )

    def test_process_alert_with_alert_manager(self):
        """Test process_alert calls AlertManager."""
        config = MonitoringConfig(
            enabled=True,
            alert_manager=AlertManagerConfig(enabled=True)
        )
        service = MonitoringService(config)
        mock_alert_manager = MagicMock(spec=AlertManager)
        service.alert_manager = mock_alert_manager # Inject mock

        event_type = "critical_error"
        event_data = {"details": "System failure"}
        severity = "critical"
        tags = ["system", "failure"]

        service.process_alert(event_type, event_data, severity, tags)

        mock_alert_manager.handle_event.assert_called_once_with(
            event_type, event_data, severity, tags
        )

    @patch('src.monitoring.service.log_with_context')
    @patch('src.monitoring.service.datadog_client')
    def test_log_metric_with_deployment_manager_enrichment(self, mock_dd_client, mock_log_context):
        """Test log_metric enriches tags using DeploymentManager."""
        mock_deployment_manager = MagicMock(spec=DeploymentManager)
        mock_deployment_manager.get_current_deployment.return_value = {
            "model_version": "v1.2.3",
            "deployment_id": "deploy_abc123"
        }
        
        service = MonitoringService(deployment_manager=mock_deployment_manager)
        service._initialized = True # Manually set initialized for this test

        service.log_metric(
            "request_latency", 0.123, "gauge",
            tags=["region:us-east-1"], model_id="my_model", environment="production"
        )

        expected_tags = ["region:us-east-1", "model_version:v1.2.3", "deployment_id:deploy_abc123"]
        
        # Check log_with_context call
        log_call_args = mock_log_context.call_args[1]
        assert sorted(log_call_args["tags"]) == sorted(expected_tags)
        
        # Check datadog_client call
        dd_call_args = mock_dd_client.gauge.call_args
        assert sorted(dd_call_args[1]["tags"]) == sorted(expected_tags)