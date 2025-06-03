+++
id = "TASK-DEVPY-20250602-174100"
title = "Investigate Empty Datadog Dashboard After Pipeline Run"
status = "🟠 In Progress"
type = "🐞 Bug"
priority = "High"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026"
created_date = "2025-06-02T15:41:00Z" # Approximate UTC
updated_date = "2025-06-02T15:41:00Z" # Approximate UTC
related_tasks = ["TASK-DEVPY-20250602-170100-RunPipelineVerifyDatadog.md"]
related_docs = [
    "reinforcestrategycreator_pipeline/run_main_pipeline.py",
    "reinforcestrategycreator_pipeline/src/monitoring/datadog_client.py",
    "reinforcestrategycreator_pipeline/src/monitoring/logger.py",
    "reinforcestrategycreator_pipeline/src/monitoring/service.py",
    "reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json"
]
tags = ["datadog", "dashboard", "data-ingestion", "metrics", "pipeline", "debugging"]
+++

## 📝 Description

Following a pipeline run (Task [`TASK-DEVPY-20250602-170100-RunPipelineVerifyDatadog.md`](.ruru/tasks/DEVPY/TASK-DEVPY-20250602-170100-RunPipelineVerifyDatadog.md)), the Datadog dashboard ([`drift_detection_dashboard.json`](reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json)) is showing no data. The pipeline script logs indicated successful execution and Datadog integration activity.

The issue might be related to:
- Incorrect metric names being sent.
- Incorrect tags or lack of tags, causing data not to match dashboard queries.
- Problems with the Datadog agent configuration (if applicable).
- Issues within the `datadog_client.py` or other monitoring scripts responsible for sending data.
- The service check `model_pipeline.drift_check` not reporting.

## ✅ Acceptance Criteria

1.  The root cause for the empty Datadog dashboard is identified.
2.  Necessary fixes are implemented in the pipeline scripts (e.g., `datadog_client.py`, `run_main_pipeline.py`, or other relevant monitoring components).
3.  After re-running the pipeline (or a relevant part of it), data appears correctly on the `drift_detection_dashboard.json` in Datadog.
4.  The service check `model_pipeline.drift_check` reports values.

## 📋 Checklist

- [✅] Review the queries used in `drift_detection_dashboard.json` widgets.
- [✅] Review the metric names, tags, and service check names being submitted by the Python pipeline (check `datadog_client.py`, `service.py`, and how they are used in `run_main_pipeline.py`).
- [ ] Verify that the Datadog API key and APP key have the necessary permissions for submitting metrics and service checks. (Assumed to be correct for now as other metrics might be flowing, but good to double-check if issues persist)
- [ ] Check Datadog agent logs on the host running the pipeline, if an agent is used for metric submission. (Assuming direct API usage based on `datadog_client.py`)
- [ ] Add debug logging to the pipeline's Datadog submission parts to see exactly what is being sent. (Implemented implicitly by adding service check calls with logging)
- [✅] Implement fixes based on findings.
- [ ] Re-run the pipeline.
- [ ] Verify data appears on the Datadog dashboard.
- [ ] Verify the `model_pipeline.drift_check` service check is reporting.

## 🪵 Log / Notes

- User provided a screenshot showing the empty dashboard.
- The "Features with Active Drift" widget specifically states "No value reported for service check model_pipeline.drift_check".
- **2025-06-02:** Investigated the issue.
    - Reviewed `drift_detection_dashboard.json`: Confirmed it uses metrics like `model_pipeline.data_drift_psi` and a service check `model_pipeline.drift_check` tagged by feature.
    - Reviewed `datadog_client.py`: Found it was missing a method to send service checks. Added `send_service_check` method utilizing `datadog.api.ServiceCheck.check`.
    - Reviewed `service.py`: Found that `check_data_drift` method was not submitting service checks. Modified it to iterate over feature drift results and use the new `send_service_check` method in `datadog_client.py`.
    - Added missing `DataDriftDetectionMethod` import in `service.py`.
- The primary root cause identified was the lack of functionality to send service checks and the `MonitoringService` not calling such functionality.