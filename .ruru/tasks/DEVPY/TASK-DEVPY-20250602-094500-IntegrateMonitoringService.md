+++
id = "TASK-DEVPY-20250602-094500"
title = "Integrate MonitoringService into ModelPipeline and Stages for Datadog"
status = "🟢 Done"
type = "🌟 Feature"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026" # Current session
created_date = "2025-06-02T09:45:00Z"
updated_date = "2025-06-02T08:18:23Z"
tags = ["pipeline", "monitoring", "datadog", "integration", "feature"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/monitoring/service.py",
    "reinforcestrategycreator_pipeline/src/pipeline/orchestrator.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stage.py",
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/data_ingestion.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/feature_engineering.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/training.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/evaluation.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/deployment.py"
]
+++

## Description

The pipeline has a `MonitoringService` designed to interact with Datadog, and the main configuration (`pipeline.yaml`) includes settings for Datadog integration (API keys, etc.). However, the `ModelPipeline` orchestrator currently does not initialize or use this `MonitoringService`. Consequently, no metrics or events are being sent to Datadog.

This task involves integrating the `MonitoringService` into the `ModelPipeline`'s lifecycle and updating relevant pipeline stages to utilize this service for logging key metrics and events.

## Acceptance Criteria

1.  `MonitoringService` is initialized within `ModelPipeline.__init__` using the global monitoring configuration from `pipeline.yaml`.
2.  The initialized `MonitoringService` instance is added to the `PipelineContext` to be accessible by all stages.
3.  Key pipeline stages (DataIngestion, FeatureEngineering, Training, Evaluation, Deployment) are updated to:
    *   Retrieve the `monitoring_service` from the `PipelineContext` during their `setup` method.
    *   Utilize `monitoring_service.log_metric()` to send relevant quantitative data (e.g., number of rows ingested, model loss, evaluation scores, paper trading P&L).
    *   Utilize `monitoring_service.log_event()` to report significant stage events (e.g., stage start, stage completion, errors encountered).
4.  The pipeline (`full_cycle_pipeline`) runs successfully after these integrations.
5.  If `DATADOG_API_KEY` and `DATADOG_APP_KEY` environment variables are set with valid keys, evidence of metrics/events appearing in Datadog should be verifiable (outside the scope of this task to *set up* Datadog, but the code should correctly attempt submission). If keys are not set, logs should indicate that Datadog was configured but API keys were missing, or that metric/event calls were made to a (potentially non-functional) client.

## Checklist

-   `[✅]` **ModelPipeline Orchestrator (`orchestrator.py`) Changes:**
    -   `[✅]` Import `MonitoringService` and `initialize_monitoring_from_pipeline_config` from `reinforcestrategycreator_pipeline.src.monitoring.service`.
    -   `[✅]` In `ModelPipeline.__init__`, after `ConfigManager` is set up, call `initialize_monitoring_from_pipeline_config(self.config_manager.get_config())` to get a `MonitoringService` instance.
    -   `[✅]` Store the `MonitoringService` instance (e.g., `self.monitoring_service_instance`).
    -   `[✅]` Add the `monitoring_service_instance` to the `PipelineContext` (e.g., `self.context.set("monitoring_service", self.monitoring_service_instance)`).
-   `[✅]` **Pipeline Stage Updates (General Pattern for each relevant stage):**
    -   `[✅]` In the stage's `__init__` or `setup` method, add an attribute to hold the monitoring service (e.g., `self.monitoring_service: Optional[MonitoringService] = None`).
    -   `[✅]` In the stage's `setup(self, context: PipelineContext)` method:
        -   `[✅]` Retrieve the monitoring service: `self.monitoring_service = context.get("monitoring_service")`.
        -   `[✅]` Add a log message if the service is not found (though it should be if the orchestrator is updated).
    -   `[✅]` In the stage's `run(self, context: PipelineContext)` method:
        -   `[✅]` At the beginning, if `self.monitoring_service` is available, call `self.monitoring_service.log_event(event_type=f"{self.name}.started", description=f"Stage {self.name} started.")`.
        -   `[✅]` Identify key metrics generated or processed by the stage. For each, call `self.monitoring_service.log_metric(metric_name=f"{self.name}.my_metric", value=..., tags=[...])`.
        -   `[✅]` Upon successful completion, call `self.monitoring_service.log_event(event_type=f"{self.name}.completed", description=f"Stage {self.name} completed successfully.", level="info")`.
        -   `[✅]` In error handling blocks (`except` clauses), call `self.monitoring_service.log_event(event_type=f"{self.name}.failed", description=f"Stage {self.name} failed: {str(e)}", level="error", context={"error_details": str(e)})`.
-   `[✅]` **Specific Stage Metric/Event Examples (apply general pattern above):**
    -   `[✅]` **DataIngestionStage:**
        -   Metric: `row_count`, `column_count` after loading data.
        -   Event: Validation warnings/errors.
    -   `[✅]` **FeatureEngineeringStage:**
        -   Metric: Number of features created/dropped.
    -   `[✅]` **TrainingStage:**
        -   Metrics: Training loss, validation loss (if applicable), episode rewards, epsilon value over time.
        -   Event: Checkpoint saved.
    -   `[✅]` **EvaluationStage:**
        -   Metrics: All computed evaluation metrics (Sharpe, P&L, drawdown, etc.).
    -   `[✅]` **DeploymentStage (Paper Trading):**
        -   Metrics: Simulated P&L, number of trades, current virtual portfolio value.
        -   Events: Simulated buy/sell orders.
-   `[ ]` Perform an E2E test by running the `full_cycle_pipeline`.
    -   `[ ]` Verify pipeline completes without new errors related to monitoring integration.
    -   `[ ]` Inspect logs for messages from `MonitoringService` indicating metric/event logging attempts (e.g., "Metric: DataIngestion.row_count", "Event: Training.started").
    -   `[ ]` (If Datadog keys are configured locally) Check Datadog to see if metrics/events appear.

## Notes
- Ensure that API keys are handled securely (using environment variable placeholders like `${DATADOG_API_KEY}` is good).
- The focus is on integrating the calls; the actual appearance in Datadog depends on valid keys and Datadog setup, which is outside this task's direct implementation scope but should be testable if keys are available.