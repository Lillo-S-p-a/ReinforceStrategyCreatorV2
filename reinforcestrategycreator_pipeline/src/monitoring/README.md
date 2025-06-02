# Monitoring Module

This module provides centralized logging and monitoring capabilities for the reinforcement learning trading pipeline.

## Features

- **Structured Logging**: JSON-formatted logs with contextual information
- **Flexible Configuration**: Configure log levels, output destinations, and formats
- **Datadog Integration**: Send metrics and events to Datadog for monitoring
- **Decorators**: Easy-to-use decorators for tracking execution time and pipeline steps
- **Alert Thresholds**: Automatic alerting when metrics violate configured thresholds
- **Health Checks**: Built-in health check functionality

## Components

### 1. Logger (`logger.py`)

Provides centralized logging with structured output:

```python
from src.monitoring import get_logger, configure_logging

# Configure logging
configure_logging(
    log_level="INFO",
    log_file="./logs/pipeline.log",
    enable_console=True,
    enable_json=True
)

# Get a logger instance
logger = get_logger("my_module")
logger.info("Processing started")

# Log with context
from src.monitoring import log_with_context
log_with_context("info", "User action", user_id=123, action="login")
```

#### Decorators

```python
from src.monitoring import log_execution_time, log_step

@log_execution_time
def process_data():
    # Function execution time will be logged
    return data

@log_step("Data Validation")
def validate_data(data):
    # Step start and completion will be logged
    return validated_data
```

### 2. Datadog Client (`datadog_client.py`)

Integrates with Datadog for metrics and events:

```python
from src.monitoring import configure_datadog, datadog_client

# Configure Datadog
configure_datadog(
    api_key="your_api_key",  # Or use ${DATADOG_API_KEY}
    app_key="your_app_key",  # Or use ${DATADOG_APP_KEY}
    metrics_prefix="model_pipeline"
)

# Send metrics
datadog_client.gauge("model.accuracy", 0.95, tags=["model:dqn"])
datadog_client.increment("pipeline.runs")
datadog_client.histogram("training.duration", 1234.5)

# Track model metrics
from src.monitoring import track_model_metrics
track_model_metrics(
    "my_model",
    {"loss": 0.1, "accuracy": 0.95},
    epoch=10,
    tags=["experiment:1"]
)

# Send events
from src.monitoring import track_pipeline_event
track_pipeline_event(
    "training_completed",
    "Model training finished successfully",
    alert_type="success",
    tags=["model:dqn"]
)
```

### 3. Monitoring Service (`service.py`)

High-level service that integrates logging and metrics:

```python
from src.monitoring import initialize_monitoring_from_pipeline_config
from src.config.manager import ConfigManager

# Initialize from pipeline config
config_manager = ConfigManager()
pipeline_config = config_manager.load_config("development")
monitoring_service = initialize_monitoring_from_pipeline_config(pipeline_config)

# Log metrics
monitoring_service.log_metric("sharpe_ratio", 1.5, "gauge")

# Log events
monitoring_service.log_event(
    "model_deployed",
    "Model deployed to production",
    level="info",
    tags=["version:1.0"]
)

# Check alert thresholds
metrics = {"sharpe_ratio": 0.3, "max_drawdown": 0.25}
alerts = monitoring_service.check_alert_thresholds(metrics)
if alerts:
    print(f"Alerts triggered: {alerts}")

# Health check
health_status = monitoring_service.create_health_check()
```

## Configuration

The monitoring system is configured through the pipeline configuration YAML files:

```yaml
monitoring:
  enabled: true
  datadog_api_key: "${DATADOG_API_KEY}"
  datadog_app_key: "${DATADOG_APP_KEY}"
  metrics_prefix: "model_pipeline"
  log_level: "INFO"
  alert_thresholds:
    sharpe_ratio_min: 0.5
    max_drawdown_max: 0.2
    error_rate_max: 0.05
```

## Environment Variables

- `DATADOG_API_KEY`: Datadog API key
- `DATADOG_APP_KEY`: Datadog application key
- `DATADOG_STATSD_HOST`: StatsD host (default: localhost)
- `DATADOG_STATSD_PORT`: StatsD port (default: 8125)

## Installation

The Datadog integration is optional. To enable it:

```bash
pip install datadog
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage Example

See `examples/monitoring_example.py` for a complete example of using the monitoring utilities.

## Testing

Run the unit tests:

```bash
# Test logging utilities
pytest tests/unit/test_monitoring_logger.py -v

# Test Datadog client
pytest tests/unit/test_monitoring_datadog.py -v

# Test monitoring service
pytest tests/unit/test_monitoring_service.py -v

# Run all monitoring tests
pytest tests/unit/test_monitoring_*.py -v
```

## Best Practices

1. **Use Structured Logging**: Always use the provided logging utilities instead of print statements
2. **Add Context**: Include relevant context in logs (user IDs, request IDs, etc.)
3. **Track Key Metrics**: Monitor important business and technical metrics
4. **Set Alert Thresholds**: Configure thresholds for critical metrics
5. **Use Decorators**: Leverage decorators for consistent logging of functions and steps
6. **Handle Errors**: Ensure errors are properly logged with stack traces
7. **Environment Variables**: Use environment variables for sensitive configuration

## Troubleshooting

### Datadog Not Sending Metrics

1. Check that Datadog is installed: `pip install datadog`
2. Verify API keys are set correctly
3. Check network connectivity to Datadog
4. Enable debug logging to see detailed errors

### Logs Not Appearing

1. Check log level configuration
2. Verify file permissions for log files
3. Ensure logging is configured before use
4. Check that handlers are properly attached

### Performance Impact

- Logging and metrics collection is designed to be lightweight
- Use sampling for high-frequency metrics if needed
- Consider async logging for high-throughput scenarios