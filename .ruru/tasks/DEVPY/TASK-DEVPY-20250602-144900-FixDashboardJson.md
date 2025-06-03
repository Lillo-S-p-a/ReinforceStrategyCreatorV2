+++
id = "TASK-DEVPY-20250602-144900"
title = "Fix Invalid 'title_size' in drift_detection_dashboard.json"
status = "🟡 To Do"
type = "🐞 Bug"
priority = "High"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026"
created_date = "2025-06-02T12:49:00Z" # Approximate UTC
updated_date = "2025-06-02T12:49:00Z" # Approximate UTC
related_tasks = ["TASK-DEVPY-20250602-144600-DotenvImportDashboard.md"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json",
    "import_datadog_dashboard.py"
]
tags = ["datadog", "dashboard", "json", "bugfix", "api-error"]
+++

## 📝 Description

When attempting to import the `drift_detection_dashboard.json` using the `import_datadog_dashboard.py` script, the Datadog API returned a 400 Bad Request error.
The error message is: `"Invalid widget definition at position 0 of type group. Error: Additional properties are not allowed ('title_size' was unexpected)."`

This indicates that the `title_size` property within a group widget in the `drift_detection_dashboard.json` file is not a valid property according to the Datadog API schema for dashboard widgets.

## ✅ Acceptance Criteria

1.  The `drift_detection_dashboard.json` file is analyzed to locate the group widget at position 0.
2.  The unexpected `title_size` property is removed from the widget definition, or corrected if a valid alternative exists and is intended.
3.  After modification, running `python import_datadog_dashboard.py reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json` no longer produces the `title_size` error. (Other errors might still occur if there are further issues, but this specific one should be resolved).

## 📋 Checklist

- [ ] Read the content of `reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json`.
- [ ] Identify the widget at position 0 (the first widget in the main "widgets" array).
- [ ] If it's a "group" type widget, inspect its definition for a `title_size` property.
- [ ] Remove the `title_size` property from that widget's definition.
- [ ] Save the modified `drift_detection_dashboard.json` file.
- [ ] Test the import script again with the modified file to confirm the `title_size` error is resolved.

## 🪵 Log / Notes

- This task is to fix a specific JSON schema validation error reported by the Datadog API.