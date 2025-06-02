Monitoring Deployed Models
==========================

Effective monitoring is crucial for ensuring that deployed trading models perform as expected and for detecting issues like performance degradation, data drift, or system health problems. The ReinforceStrategyCreator Pipeline includes a ``MonitoringService`` to facilitate this.

Key Monitoring Components
-------------------------

*   **``MonitoringService`` (``src.monitoring.service.MonitoringService``)**:
    *   The central service for managing all monitoring activities.
    *   It initializes and coordinates various monitoring sub-components based on the ``monitoring`` section of your pipeline configuration.
*   **Structured Logging (``src.monitoring.logger``)**:
    *   The pipeline uses a centralized logging system that can output structured JSON logs. This makes logs easier to parse, search, and integrate with log management systems.
    *   Logs include timestamps, levels, logger names, messages, and can also include custom context, metrics, and exception information.
*   **Datadog Integration (``src.monitoring.datadog_client``)**:
    *   If configured with API keys, the ``MonitoringService`` can send metrics and events to Datadog.
    *   This allows for creating dashboards in Datadog to visualize model performance, system health, and pipeline events.
    *   Example Datadog dashboard JSON definitions are provided in ``reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/`` and can be imported into your Datadog environment.
*   **Drift Detection (``src.monitoring.drift_detection``)**:
    *   **Data Drift Detector**: Monitors input data for significant changes compared to a reference dataset (e.g., training data). Configurable via ``monitoring.data_drift`` in ``pipeline.yaml``.
    *   **Model Drift Detector**: Monitors the model's predictive performance or output distribution for significant changes over time. Configurable via ``monitoring.model_drift`` in ``pipeline.yaml``.
*   **Alerting (``src.monitoring.alerting.AlertManager``)**:
    *   Manages alert rules and dispatches notifications through configured channels (e.g., email, Slack) when certain conditions are met (e.g., metric thresholds breached, drift detected). Configurable via ``monitoring.alert_manager`` in ``pipeline.yaml``.

Configuration
-------------

Monitoring is configured via the ``monitoring`` section in your main ``pipeline.yaml`` or environment-specific configuration files. Key settings include:

*   ``enabled``: (Boolean) Master switch to enable or disable all monitoring.
*   ``datadog_api_key`` / ``datadog_app_key``: API keys for Datadog integration (use environment variables like ``${DATADOG_API_KEY}``).
*   ``metrics_prefix``: A prefix for all metrics sent to Datadog (e.g., "model_pipeline").
*   ``log_level``: Default logging level (e.g., "INFO", "DEBUG").
*   ``log_file``: Path to a central log file (e.g., ``./logs/pipeline.log``).
*   ``alert_thresholds``: Define thresholds for key metrics that trigger alerts (e.g., ``sharpe_ratio_min: 0.5``).
*   ``data_drift``: Configuration for data drift detection (method, parameters, reference data).
*   ``model_drift``: Configuration for model drift detection (method, performance metric, parameters).
*   ``alert_manager``: Configuration for alert rules and notification channels.

Refer to the :doc:`../../user_guide/configuration` guide for more details on these settings.

What to Monitor
---------------

1.  **Model Performance Metrics**:
    *   Track key trading metrics over time (e.g., Sharpe ratio, total return, win rate, max drawdown) for deployed models.
    *   Compare these against benchmarks and expected performance.
    *   The ``MonitoringService.log_metric()`` method is used by pipeline components to record these.

2.  **Data Drift**:
    *   Monitor the statistical properties of incoming market data fed to the model.
    *   Significant drift from the data the model was trained on can degrade performance.
    *   The ``MonitoringService.check_data_drift()`` method is used. Alerts are triggered if drift exceeds configured thresholds.

3.  **Model Drift (Concept & Performance Degradation)**:
    *   **Concept Drift**: The underlying relationship between input features and the target variable changes over time.
    *   **Performance Degradation**: Directly track the model's predictive accuracy or relevant performance metric on new data.
    *   The ``MonitoringService.check_model_drift()`` method is used. Alerts are triggered if drift or degradation is significant.

4.  **System Health & Pipeline Events**:
    *   Monitor the health of the deployment environment (CPU, memory, disk space).
    *   Track pipeline execution events (starts, completions, failures of stages).
    *   Log errors and exceptions.
    *   The ``MonitoringService.log_event()`` method is used.

Interpreting Dashboards and Alerts
----------------------------------

*   **Dashboards (e.g., in Datadog)**:
    *   Visualize trends in model performance metrics over time.
    *   Track data drift scores and model drift indicators.
    *   Monitor system resource usage.
    *   Look for anomalies, sudden changes, or consistent degradation.
    *   The project may include predefined Datadog dashboard JSON files in ``reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/`` which can be imported into Datadog.

*   **Alerts**:
    *   Alerts are triggered by the ``AlertManager`` when predefined conditions are met (e.g., a performance metric drops below a threshold, significant data/model drift is detected, a pipeline stage fails).
    *   Investigate alerts promptly. They often indicate an issue that requires attention, such as::

        *   Model retraining needed due to drift.
        *   Problems with the data feed.
        *   Bugs in the model or deployment code.
        *   Issues with the execution environment.

Accessing Logs
--------------
Pipeline logs, including monitoring events, are typically written to:
*   The console during execution.
*   A specified log file (e.g., ``./logs/pipeline.log``), often in a structured JSON format for easier parsing.
*   Log management systems if integrated (e.g., Datadog Logs).

Review these logs for detailed error messages, context around events, and a historical record of pipeline and model behavior.

By actively monitoring deployed models and the pipeline itself, you can maintain performance, identify issues early, and ensure the reliability of your trading strategies.