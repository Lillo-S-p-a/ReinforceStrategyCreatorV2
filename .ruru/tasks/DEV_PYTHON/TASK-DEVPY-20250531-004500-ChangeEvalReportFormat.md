+++
id = "TASK-DEVPY-20250531-004500"
title = "Change Evaluation Report Format to Supported Type"
status = "🟡 To Do"
type = "🛠️ Refactor"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-31T00:45:00Z"
updated_date = "2025-05-31T00:45:00Z"
tags = ["python", "evaluation", "report-generator", "pipeline", "config"]
related_docs = [
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml",
    "reinforcestrategycreator_pipeline/src/visualization/report_generator.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/evaluation.py"
]
+++

## 📝 Description

The pipeline fails in the `EvaluationStage` with `ValueError: Unsupported report format: json`. This is because the `ReportGenerator` in `reinforcestrategycreator_pipeline/src/visualization/report_generator.py` does not currently support generating reports in JSON format.

The `evaluation.report_format` is configured in `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml`.

The simplest solution is to change the configured `report_format` to a supported type, such as "html" or "markdown".

## ✅ Acceptance Criteria

*   The `evaluation.report_format` in `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml` is changed from `"json"` to `"html"` (or `"markdown"` if "html" also presents issues).
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) no longer raises the `ValueError: Unsupported report format: json` in the `EvaluationStage`.
*   The pipeline either completes successfully or fails at a later point for a different reason.
*   An evaluation report file (e.g., `.html` or `.md`) is generated in the expected output directory if the stage completes.

## 📋 Checklist

*   [ ] **Modify `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml`**:
    *   Locate the `evaluation:` section.
    *   Change `report_format:` from `"json"` to `"html"`.
*   [ ] **Test**:
    *   Run `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory.
    *   Verify that the `ValueError` for unsupported report format is resolved.
    *   Check if an evaluation report (e.g., `evaluation_report.html`) is created in the default output location (e.g., `reinforcestrategycreator_pipeline/evaluation_reports/`).