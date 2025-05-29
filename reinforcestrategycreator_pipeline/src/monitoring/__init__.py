"""Monitoring package for logging and metrics collection."""

from .logger import (
    get_logger,
    configure_logging,
    log_with_context,
    log_execution_time,
    log_step
)

from .datadog_client import (
    configure_datadog,
    datadog_client,
    track_metric,
    track_model_metrics,
    track_pipeline_event
)

from .service import (
    MonitoringService,
    get_monitoring_service,
    initialize_monitoring_from_pipeline_config
)

__all__ = [
    # Logger exports
    "get_logger",
    "configure_logging",
    "log_with_context",
    "log_execution_time",
    "log_step",
    
    # Datadog exports
    "configure_datadog",
    "datadog_client",
    "track_metric",
    "track_model_metrics",
    "track_pipeline_event",
    
    # Service exports
    "MonitoringService",
    "get_monitoring_service",
    "initialize_monitoring_from_pipeline_config"
]