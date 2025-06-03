+++
id = "TASK-DEVPY-20250602-154200"
title = "Fix Invalid 'table' Widget in Group Widget in drift_detection_dashboard.json"
status = "🟢 Done"
type = "🐞 Bug"
priority = "High"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026"
created_date = "2025-06-02T13:42:00Z" # Approximate UTC
updated_date = "2025-06-02T15:51:45Z" # Approximate UTC
related_tasks = ["TASK-DEVPY-20250602-144900-FixDashboardJson.md"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json",
    "import_datadog_dashboard.py"
]
tags = ["datadog", "dashboard", "json", "bugfix", "api-error", "widget"]
+++

## 📝 Description

After resolving the `title_size` issue in `drift_detection_dashboard.json` (Task `TASK-DEVPY-20250602-144900`), a new error occurred when attempting to import the dashboard using `import_datadog_dashboard.py`.
The Datadog API returned an error: `"Invalid widget definition at position 4 of type group. Error: Widget type 'table' is not allowed in group widgets."`

This indicates that the `drift_detection_dashboard.json` file contains a 'table' widget nested directly within a 'group' widget at position 4, which is not permitted by the Datadog API.

## ✅ Acceptance Criteria

1.  The `drift_detection_dashboard.json` file is analyzed to locate the group widget at position 4.
2.  The 'table' widget nested within this group widget is identified.
3.  The JSON structure is modified to resolve this issue. This might involve:
    *   Moving the 'table' widget out of the group widget to be a top-level widget.
    *   Changing the 'table' widget to a type that is allowed within a group (if appropriate and achieves a similar visual/functional goal).
    *   Removing the 'table' widget if it's deemed non-essential or incorrectly placed.
4.  After modification, running `python import_datadog_dashboard.py reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json` no longer produces the error about 'table' widgets in groups.

## 📋 Checklist

- [x] Read the content of `reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json`.
- [x] Identify the group widget at position 4 (the fifth widget in the main "widgets" array, as it's 0-indexed).
- [x] Inspect its `definition.widgets` (or similar structure for group widgets) for any widget of type 'table'.
- [x] Decide on the best approach to fix the invalid nesting (move, change type, or remove the table widget).
- [x] Modify the `drift_detection_dashboard.json` file with the chosen fix.
- [x] Save the modified `drift_detection_dashboard.json` file.
- [x] Test the import script again with the modified file to confirm the 'table' widget error is resolved.

## 🪵 Log / Notes

- This task addresses a new JSON schema validation error reported by the Datadog API after a previous fix.