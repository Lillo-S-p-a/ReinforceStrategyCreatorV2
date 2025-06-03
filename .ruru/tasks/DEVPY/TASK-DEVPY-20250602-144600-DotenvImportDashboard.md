+++
id = "TASK-DEVPY-20250602-144600"
title = "Modify import_datadog_dashboard.py to use python-dotenv"
status = "🟢 Done"
type = "🔧 Chore"
priority = "High"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026"
created_date = "2025-06-02T12:46:00Z" # Approximate UTC
updated_date = "2025-06-02T12:47:24Z" # Approximate UTC
related_tasks = ["TASK-DEVPY-20250602-143400-DatadogKeyConfig.md"]
related_docs = [
    "import_datadog_dashboard.py",
    ".env",
    "pyproject.toml"
]
tags = ["datadog", "python-dotenv", "environment-variables", "script-modification"]
+++

## 📝 Description

The `import_datadog_dashboard.py` script currently fails because it cannot find the `DATADOG_API_KEY` and `DATADOG_APP_KEY` environment variables. These variables have been added to the `.env` file, but the script does not explicitly load this file.

The `python-dotenv` package is already a project dependency. This task is to modify `import_datadog_dashboard.py` to use `python-dotenv` to load variables from the `.env` file at the beginning of the script.

## ✅ Acceptance Criteria

1.  `import_datadog_dashboard.py` is modified to include `from dotenv import load_dotenv` and `load_dotenv()` at the beginning of its execution flow.
2.  After modification, the script successfully loads environment variables from the `.env` file.
3.  When the script is run (assuming correct API keys in `.env`), it no longer errors out due to missing API keys. (It may still fail if keys are invalid or other Datadog issues occur, but the key loading part should work).

## 📋 Checklist

- [✅] Add `from dotenv import load_dotenv` to the imports in `import_datadog_dashboard.py`.
- [✅] Add `load_dotenv()` call near the beginning of the script's execution, before `os.environ.get()` is used for Datadog keys.
- [✅] Test running the script (e.g., `python import_datadog_dashboard.py reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/drift_detection_dashboard.json`) to ensure it no longer reports missing API keys (assuming `.env` is populated).

## 🪵 Log / Notes

- This task is to fix the environment variable loading mechanism for the dashboard import script.