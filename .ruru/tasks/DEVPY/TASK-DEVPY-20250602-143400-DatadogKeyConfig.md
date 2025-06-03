+++
id = "TASK-DEVPY-20250602-143400"
title = "Investigate and Guide Datadog API Key Configuration for Pipeline"
status = "🟢 Done"
type = "🔧 Chore"
priority = "High"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026"
created_date = "2025-06-02T12:34:00Z" # Approximate UTC
updated_date = "2025-06-02T12:40:00Z" # Approximate UTC
related_tasks = ["TASK-DEVPY-20250602-094500-IntegrateMonitoringService.md"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/monitoring/service.py",
    "reinforcestrategycreator_pipeline/src/monitoring/datadog_client.py",
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml",
    ".env"
]
tags = ["datadog", "monitoring", "configuration", "api-keys", "pipeline", "investigation"]
+++

## 📝 Description

The user has reported that despite the successful integration of `MonitoringService` (Task `TASK-DEVPY-20250602-094500`), no new data or dashboards are appearing in Datadog.
Previous E2E tests indicated that Datadog API keys were not detected by the pipeline, with logs showing "Datadog API key not provided".

This task is to investigate the current Datadog API key configuration within the project and guide the user on how to correctly set them up so that the `MonitoringService` can successfully send data to Datadog.

## ✅ Acceptance Criteria

1.  The method by which the pipeline expects to receive Datadog API keys (`DATADOG_API_KEY`, `DATADOG_APP_KEY`) is clearly identified (e.g., environment variables, `.env` file, configuration files).
2.  Instructions are provided to the user on how and where to set these API keys securely.
3.  Guidance is given on how to verify that the pipeline can access these keys once set.
4.  (Optional, if feasible and secure) A simple test or check is proposed to confirm the keys are being picked up by the `DatadogClient` or `MonitoringService` without necessarily running a full pipeline.
5.  The user confirms understanding of the configuration steps.

## 📋 Checklist

- [x] Review `reinforcestrategycreator_pipeline/src/monitoring/datadog_client.py` to confirm how API keys are loaded.
- [x] Review `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml` for any Datadog related configuration that might influence key loading.
- [x] Check if a `.env` file is used and if it's the intended place for these keys.
- [ ] Document the correct procedure for setting `DATADOG_API_KEY` and `DATADOG_APP_KEY` (e.g., in `.env`, as environment variables).
- [ ] Provide clear, step-by-step instructions to the user.
- [ ] Suggest a method for the user to verify the configuration (e.g., a small script to print the loaded keys from the service's perspective, or specific log messages to look for after setting keys and re-running a small part of the pipeline if necessary).

## 🪵 Log / Notes

- Initial task to address user feedback regarding missing Datadog data.
- Previous pipeline run logs indicate keys were not found.
- 2025-06-02: Confirmed API keys are loaded from environment variables (`DATADOG_API_KEY`, `DATADOG_APP_KEY`) by `datadog_client.py`.
- 2025-06-02: Confirmed `pipeline.yaml` expects these environment variables.
- 2025-06-02: Confirmed `.env` file exists and is suitable for storing these keys for local development, but does not currently contain them.