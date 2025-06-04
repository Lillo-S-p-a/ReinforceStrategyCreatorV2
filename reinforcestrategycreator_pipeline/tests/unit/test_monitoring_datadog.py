"""Unit tests for the Datadog client."""

import pytest
import os
from unittest.mock import patch, MagicMock, call

from reinforcestrategycreator_pipeline.src.monitoring.datadog_client import (
    DatadogClient,
    configure_datadog,
    track_metric,
    track_model_metrics,
    track_pipeline_event,
    DATADOG_AVAILABLE
)


class TestDatadogClient:
    """Test the DatadogClient class."""
    
    def test_singleton_pattern(self):
        """Test that DatadogClient follows singleton pattern."""
        client1 = DatadogClient()
        client2 = DatadogClient()
        assert client1 is client2
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.initialize')
    def test_configure_with_api_key(self, mock_initialize):
        """Test configuring with API key."""
        client = DatadogClient()
        client.configure(
            api_key="test_api_key",
            app_key="test_app_key",
            metrics_prefix="test_prefix",
            enabled=True
        )
        
        if DATADOG_AVAILABLE:
            assert client.enabled
            assert client.metrics_prefix == "test_prefix"
            mock_initialize.assert_called_once()
        else:
            assert not client.enabled
    
    def test_configure_without_api_key(self):
        """Test configuring without API key."""
        client = DatadogClient()
        client.configure(enabled=True)
        
        assert not client.enabled
    
    def test_configure_disabled(self):
        """Test configuring with disabled flag."""
        client = DatadogClient()
        client.configure(
            api_key="test_api_key",
            enabled=False
        )
        
        assert not client.enabled
    
    @patch.dict(os.environ, {"DATADOG_API_KEY": "env_api_key"})
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.initialize')
    def test_configure_with_env_var(self, mock_initialize):
        """Test configuring with environment variable."""
        client = DatadogClient()
        client.configure(enabled=True)
        
        if DATADOG_AVAILABLE:
            assert client.enabled
            call_args = mock_initialize.call_args[1]
            assert call_args["api_key"] == "env_api_key"
    
    def test_format_metric_name(self):
        """Test metric name formatting."""
        client = DatadogClient()
        client.metrics_prefix = "test_prefix"
        
        formatted = client._format_metric_name("my.metric")
        assert formatted == "test_prefix.my.metric"
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_increment(self, mock_statsd):
        """Test increment metric."""
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        client.increment("counter", value=5, tags=["tag1"])
        
        mock_statsd.increment.assert_called_once_with(
            "test.counter",
            value=5,
            tags=["tag1"]
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_gauge(self, mock_statsd):
        """Test gauge metric."""
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        client.gauge("metric", 42.5, tags=["tag1", "tag2"])
        
        mock_statsd.gauge.assert_called_once_with(
            "test.metric",
            value=42.5,
            tags=["tag1", "tag2"]
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_histogram(self, mock_statsd):
        """Test histogram metric."""
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        client.histogram("distribution", 100.0)
        
        mock_statsd.histogram.assert_called_once_with(
            "test.distribution",
            value=100.0,
            tags=None
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_timing(self, mock_statsd):
        """Test timing metric."""
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        client.timing("duration", 1500.0, tags=["operation:test"])
        
        mock_statsd.timing.assert_called_once_with(
            "test.duration",
            value=1500.0,
            tags=["operation:test"]
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.time')
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_timed_context_manager(self, mock_statsd, mock_time):
        """Test timed context manager."""
        # Mock time to return predictable values
        mock_time.time.side_effect = [1000.0, 1002.5]  # 2.5 seconds
        
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        with client.timed("operation.duration", tags=["test"]):
            pass
        
        # Should record 2500ms
        mock_statsd.timing.assert_called_once_with(
            "test.operation.duration",
            value=2500.0,
            tags=["test"]
        )
    
    def test_metrics_disabled(self):
        """Test that metrics are not sent when disabled."""
        client = DatadogClient()
        client.enabled = False
        
        with patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd') as mock_statsd:
            client.increment("test")
            client.gauge("test", 1)
            client.histogram("test", 1)
            client.timing("test", 1)
            
            # No calls should be made
            mock_statsd.increment.assert_not_called()
            mock_statsd.gauge.assert_not_called()
            mock_statsd.histogram.assert_not_called()
            mock_statsd.timing.assert_not_called()
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.api')
    def test_event(self, mock_api):
        """Test sending events."""
        client = DatadogClient()
        client.enabled = True
        
        client.event(
            title="Test Event",
            text="Event description",
            alert_type="warning",
            tags=["test"],
            aggregation_key="test_key"
        )
        
        if DATADOG_AVAILABLE:
            mock_api.Event.create.assert_called_once_with(
                title="Test Event",
                text="Event description",
                alert_type="warning",
                tags=["test"],
                aggregation_key="test_key"
            )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.statsd')
    def test_send_metrics(self, mock_statsd):
        """Test sending multiple metrics."""
        client = DatadogClient()
        client.enabled = True
        client.metrics_prefix = "test"
        
        metrics = {
            "metric1": 10.0,
            "metric2": 20.0,
            "metric3": 30.0
        }
        
        client.send_metrics(metrics, tags=["batch"])
        
        # Should call gauge for each metric
        expected_calls = [
            call("test.metric1", value=10.0, tags=["batch"]),
            call("test.metric2", value=20.0, tags=["batch"]),
            call("test.metric3", value=30.0, tags=["batch"])
        ]
        
        assert mock_statsd.gauge.call_count == 3
        for expected_call in expected_calls:
            assert expected_call in mock_statsd.gauge.call_args_list


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.datadog_client')
    def test_configure_datadog_function(self, mock_client):
        """Test configure_datadog function."""
        configure_datadog(
            api_key="key",
            app_key="app",
            metrics_prefix="prefix",
            enabled=True
        )
        
        mock_client.configure.assert_called_once_with(
            api_key="key",
            app_key="app",
            metrics_prefix="prefix",
            enabled=True
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.datadog_client')
    def test_track_model_metrics(self, mock_client):
        """Test track_model_metrics function."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95
        }
        
        track_model_metrics("my_model", metrics, epoch=10, tags=["experiment:1"])
        
        # Should call gauge for each metric with proper tags
        expected_calls = [
            call("model.my_model.loss", 0.5, tags=["experiment:1", "model:my_model", "epoch:10"]),
            call("model.my_model.accuracy", 0.95, tags=["experiment:1", "model:my_model", "epoch:10"])
        ]
        
        assert mock_client.gauge.call_count == 2
        mock_client.gauge.assert_has_calls(expected_calls, any_order=True)
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.datadog_client')
    def test_track_pipeline_event(self, mock_client):
        """Test track_pipeline_event function."""
        track_pipeline_event(
            "test_event",
            "Test description",
            alert_type="info",
            tags=["test"]
        )
        
        mock_client.event.assert_called_once_with(
            title="Pipeline Event: test_event",
            text="Test description",
            alert_type="info",
            tags=["test"],
            aggregation_key="pipeline_test_event"
        )


class TestTrackMetricDecorator:
    """Test the track_metric decorator."""
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.datadog_client')
    def test_track_metric_success(self, mock_client):
        """Test track_metric decorator with successful execution."""
        @track_metric("gauge")
        def test_function():
            return 42.0
        
        result = test_function()
        
        assert result == 42.0
        
        # Check increment calls
        assert mock_client.increment.call_count == 2
        mock_client.increment.assert_any_call("tests.unit.test_monitoring_datadog.test_function.calls")
        mock_client.increment.assert_any_call("tests.unit.test_monitoring_datadog.test_function.success")
        
        # Check gauge call for result
        mock_client.gauge.assert_called_once_with(
            "tests.unit.test_monitoring_datadog.test_function.value",
            42.0
        )
    
    @patch('reinforcestrategycreator_pipeline.src.monitoring.datadog_client.datadog_client')
    def test_track_metric_failure(self, mock_client):
        """Test track_metric decorator with failed execution."""
        @track_metric("histogram")
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        # Check increment calls
        assert mock_client.increment.call_count == 2
        mock_client.increment.assert_any_call("tests.unit.test_monitoring_datadog.test_function.calls")
        mock_client.increment.assert_any_call("tests.unit.test_monitoring_datadog.test_function.errors")
        
        # Check event call
        mock_client.event.assert_called_once()
        event_call = mock_client.event.call_args
        assert event_call[1]["title"] == "Error in test_function"
        assert "Test error" in event_call[1]["text"]
        assert event_call[1]["alert_type"] == "error"