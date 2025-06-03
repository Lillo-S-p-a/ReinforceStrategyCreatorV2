"""Datadog integration for monitoring metrics and events."""

import os
import time
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
from functools import wraps
import logging

try:
    from datadog import initialize, statsd, api
    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    # Create mock objects for when datadog is not installed
    class MockStatsd:
        def increment(self, *args, **kwargs): pass
        def decrement(self, *args, **kwargs): pass
        def gauge(self, *args, **kwargs): pass
        def histogram(self, *args, **kwargs): pass
        def timing(self, *args, **kwargs): pass
        def timed(self, *args, **kwargs):
            return lambda f: f
        def set(self, *args, **kwargs): pass
    
    statsd = MockStatsd()

from .logger import get_logger


class DatadogClient:
    """Client for sending metrics and events to Datadog."""
    
    _instance: Optional['DatadogClient'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'DatadogClient':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Datadog client."""
        if not self._initialized:
            self.logger = get_logger("datadog")
            self.enabled = False
            self.metrics_prefix = "model_pipeline"
            self._initialized = True
    
    def configure(
        self,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        metrics_prefix: str = "model_pipeline",
        enabled: bool = True
    ) -> None:
        """
        Configure the Datadog client.
        
        Args:
            api_key: Datadog API key (can also be set via DATADOG_API_KEY env var)
            app_key: Datadog app key (can also be set via DATADOG_APP_KEY env var)
            metrics_prefix: Prefix for all metrics
            enabled: Whether to enable Datadog integration
        """
        self.enabled = enabled and DATADOG_AVAILABLE
        self.metrics_prefix = metrics_prefix
        
        if not self.enabled:
            if not DATADOG_AVAILABLE:
                self.logger.warning(
                    "Datadog library not installed. Install with: pip install datadog"
                )
            else:
                self.logger.info("Datadog integration disabled")
            return
        
        # Get API keys from parameters or environment
        api_key = api_key or os.getenv("DATADOG_API_KEY")
        app_key = app_key or os.getenv("DATADOG_APP_KEY")
        
        if not api_key:
            self.logger.error("Datadog API key not provided")
            self.enabled = False
            return
        
        try:
            # Initialize Datadog
            options = {
                "api_key": api_key,
                "app_key": app_key,
                "statsd_host": os.getenv("DATADOG_STATSD_HOST", "localhost"),
                "statsd_port": int(os.getenv("DATADOG_STATSD_PORT", "8125"))
            }
            initialize(**options)
            
            self.logger.info(
                f"Datadog client configured with prefix: {self.metrics_prefix}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Datadog: {e}")
            self.enabled = False
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name with prefix."""
        return f"{self.metrics_prefix}.{metric}"
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric: Metric name
            value: Value to increment by
            tags: Optional tags
        """
        if not self.enabled:
            return
        
        try:
            statsd.increment(
                self._format_metric_name(metric),
                value=value,
                tags=tags
            )
        except Exception as e:
            self.logger.error(f"Failed to send increment metric: {e}")
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Send a gauge metric.
        
        Args:
            metric: Metric name
            value: Gauge value
            tags: Optional tags
        """
        if not self.enabled:
            return
        
        try:
            statsd.gauge(
                self._format_metric_name(metric),
                value=value,
                tags=tags
            )
        except Exception as e:
            self.logger.error(f"Failed to send gauge metric: {e}")
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Send a histogram metric.
        
        Args:
            metric: Metric name
            value: Value to record
            tags: Optional tags
        """
        if not self.enabled:
            return
        
        try:
            statsd.histogram(
                self._format_metric_name(metric),
                value=value,
                tags=tags
            )
        except Exception as e:
            self.logger.error(f"Failed to send histogram metric: {e}")
    
    def timing(
        self,
        metric: str,
        value: float,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Send a timing metric.
        
        Args:
            metric: Metric name
            value: Time in milliseconds
            tags: Optional tags
        """
        if not self.enabled:
            return
        
        try:
            statsd.timing(
                self._format_metric_name(metric),
                value=value,
                tags=tags
            )
        except Exception as e:
            self.logger.error(f"Failed to send timing metric: {e}")
    
    @contextmanager
    def timed(self, metric: str, tags: Optional[List[str]] = None):
        """
        Context manager for timing code execution.
        
        Args:
            metric: Metric name
            tags: Optional tags
            
        Example:
            with datadog_client.timed("model.training.duration"):
                train_model()
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.timing(metric, duration_ms, tags=tags)
    
    def event(
        self,
        title: str,
        text: str,
        alert_type: str = "info",
        tags: Optional[List[str]] = None,
        aggregation_key: Optional[str] = None
    ) -> None:
        """
        Send an event to Datadog.
        
        Args:
            title: Event title
            text: Event text/description
            alert_type: Type of alert (info, warning, error, success)
            tags: Optional tags
            aggregation_key: Key to group related events
        """
        if not self.enabled or not DATADOG_AVAILABLE:
            return
        
        try:
            api.Event.create(
                title=title,
                text=text,
                alert_type=alert_type,
                tags=tags,
                aggregation_key=aggregation_key
            )
        except Exception as e:
            self.logger.error(f"Failed to send event: {e}")
    
    def send_metrics(self, metrics: Dict[str, float], tags: Optional[List[str]] = None) -> None:
        """
        Send multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            tags: Optional tags to apply to all metrics
        """
        for metric_name, value in metrics.items():
            self.gauge(metric_name, value, tags=tags)
    
    def send_service_check(
        self,
        check_name: str,
        status: int,
        tags: Optional[List[str]] = None,
        message: Optional[str] = None,
        hostname: Optional[str] = None
    ) -> None:
        """
        Send a service check to Datadog.

        Args:
            check_name: Name of the service check.
            status: Status of the check (0: OK, 1: WARNING, 2: CRITICAL, 3: UNKNOWN).
            tags: Optional tags.
            message: Optional message for the service check.
            hostname: Optional hostname for the service check.
        """
        if not self.enabled or not DATADOG_AVAILABLE:
            return

        try:
            api.ServiceCheck.check(
                check=self._format_metric_name(check_name), # Use existing prefixing
                host_name=hostname,
                status=status,
                message=message,
                tags=tags
            )
            self.logger.debug(f"Service check '{check_name}' sent with status {status}")
        except Exception as e:
            self.logger.error(f"Failed to send service check '{check_name}': {e}")


# Singleton instance
datadog_client = DatadogClient()


# Convenience functions
def configure_datadog(
    api_key: Optional[str] = None,
    app_key: Optional[str] = None,
    metrics_prefix: str = "model_pipeline",
    enabled: bool = True
) -> None:
    """Configure the Datadog client."""
    datadog_client.configure(
        api_key=api_key,
        app_key=app_key,
        metrics_prefix=metrics_prefix,
        enabled=enabled
    )


def track_metric(metric_type: str = "gauge"):
    """
    Decorator to track function metrics.
    
    Args:
        metric_type: Type of metric (gauge, increment, histogram)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metric_name = f"{func.__module__}.{func.__name__}"
            
            # Track execution count
            datadog_client.increment(f"{metric_name}.calls")
            
            # Track execution time
            with datadog_client.timed(f"{metric_name}.duration"):
                try:
                    result = func(*args, **kwargs)
                    
                    # Track success
                    datadog_client.increment(f"{metric_name}.success")
                    
                    # If result is numeric and metric_type is appropriate, track it
                    if isinstance(result, (int, float)) and metric_type in ["gauge", "histogram"]:
                        if metric_type == "gauge":
                            datadog_client.gauge(f"{metric_name}.value", result)
                        else:
                            datadog_client.histogram(f"{metric_name}.value", result)
                    
                    return result
                    
                except Exception as e:
                    # Track failure
                    datadog_client.increment(f"{metric_name}.errors")
                    datadog_client.event(
                        title=f"Error in {func.__name__}",
                        text=f"Function {func.__module__}.{func.__name__} failed: {str(e)}",
                        alert_type="error",
                        tags=[f"function:{func.__name__}", f"module:{func.__module__}"]
                    )
                    raise
        
        return wrapper
    return decorator


def track_model_metrics(
    model_name: str,
    metrics: Dict[str, float],
    epoch: Optional[int] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Track model training/evaluation metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metric names and values
        epoch: Optional epoch number
        tags: Optional additional tags
    """
    # Prepare tags
    metric_tags = tags or []
    metric_tags.append(f"model:{model_name}")
    if epoch is not None:
        metric_tags.append(f"epoch:{epoch}")
    
    # Send metrics
    for metric_name, value in metrics.items():
        datadog_client.gauge(
            f"model.{model_name}.{metric_name}",
            value,
            tags=metric_tags
        )


def track_pipeline_event(
    event_type: str,
    description: str,
    alert_type: str = "info",
    tags: Optional[List[str]] = None
) -> None:
    """
    Track a pipeline event.
    
    Args:
        event_type: Type of event (e.g., "training_started", "deployment_completed")
        description: Event description
        alert_type: Alert type (info, warning, error, success)
        tags: Optional tags
    """
    datadog_client.event(
        title=f"Pipeline Event: {event_type}",
        text=description,
        alert_type=alert_type,
        tags=tags,
        aggregation_key=f"pipeline_{event_type}"
    )