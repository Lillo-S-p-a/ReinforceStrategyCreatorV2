+++
id = "TASK-DEVPY-20250530-223000"
title = "Fix FileNotFoundError for dummy_data.csv in DataIngestionStage"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-30T22:30:00Z"
updated_date = "2025-05-30T22:42:00Z"
tags = ["python", "path-resolution", "data-ingestion", "pipeline", "file-not-found"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/pipeline/stages/data_ingestion.py",
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml"
]
+++

## 📝 Description

The main pipeline (`run_main_pipeline.py`) fails during the `DataIngestionStage` with a `FileNotFoundError`. The path to `dummy_data.csv` is being resolved incorrectly, leading to a duplicated path segment: `/home/alessio/Personal/ReinforceStrategyCreatorV2/reinforcestrategycreator_pipeline/reinforcestrategycreator_pipeline/dummy_data.csv`.

This needs to be fixed by:
1.  Making the `data.source_path` in `pipeline.yaml` relative to the config file's own location.
2.  Updating `DataIngestionStage.setup()` to resolve this relative path correctly using `ConfigManager.base_path`.

## ✅ Acceptance Criteria

*   The `data.source_path` in `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml` is updated to `"../../dummy_data.csv"`.
*   The `setup()` method in `reinforcestrategycreator_pipeline/src/pipeline/stages/data_ingestion.py` correctly resolves the `source_path` for CSV files by prepending `self.context.get("config_manager").base_path` if the `source_path` is relative.
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) no longer raises a `FileNotFoundError` for `dummy_data.csv` in the `DataIngestionStage`.
*   The pipeline proceeds past the `DataIngestionStage` successfully (or fails at a later stage for a different reason).

## 📋 Checklist

*   [✅] **Modify `reinforcestrategycreator_pipeline/configs/base/pipeline.yaml`**:
    *   Locate the `data:` section.
    *   Change `source_path:` from `"reinforcestrategycreator_pipeline/dummy_data.csv"` to `"../dummy_data.csv"` (relative to configs directory).
*   [✅] **Modify `reinforcestrategycreator_pipeline/src/pipeline/stages/data_ingestion.py`**:
    *   In the `DataIngestionStage.setup()` method, specifically where `resolved_source_path` is determined for CSV files:
        *   If `Path(self.source_path)` is not absolute, it should be resolved as:
          `resolved_source_path = (config_manager.loader.base_path / self.source_path).resolve()`
*   [✅] **Additional fix for `reinforcestrategycreator_pipeline/src/data/csv_source.py`**:
    *   Added logic to handle relative paths starting with "../" to ensure the CSV file is found when loaded by the Training stage.
*   [✅] Test the changes by running `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory. Ensure the `FileNotFoundError` in `DataIngestionStage` is resolved.