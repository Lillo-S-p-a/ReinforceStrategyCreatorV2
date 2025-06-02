+++
id = "TASK-DEVPY-20250531-001800"
title = "Fix ValueError in EvaluationStage: Data source not registered"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-31T00:18:00Z"
updated_date = "2025-05-31T00:30:00Z"
tags = ["python", "data-manager", "evaluation-stage", "pipeline", "value-error"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/pipeline/stages/evaluation.py",
    "reinforcestrategycreator_pipeline/src/evaluation/engine.py",
    "reinforcestrategycreator_pipeline/src/data/manager.py",
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml"
]
+++

## 📝 Description

The pipeline fails during the `EvaluationStage` with `ValueError: Data source not registered: dummy_csv_data`. This occurs when `EvaluationEngine.evaluate()` calls `self.data_manager.load_data(data_source_id)`.

The `DataManager` instance within the `EvaluationStage` does not have the "dummy_csv_data" source registered, even though it was likely registered and used in previous stages (`DataIngestionStage`, `TrainingStage`).

The `EvaluationStage` needs to ensure that the data source specified by `data_config.source_id` (from `pipeline.yaml`) is registered with its instance of `DataManager` before attempting to load data.

## ✅ Acceptance Criteria

*   The `setup()` or `run()` method of `EvaluationStage` in `reinforcestrategycreator_pipeline/src/pipeline/stages/evaluation.py` is modified to ensure the required data source (identified by `data_config.source_id`) is registered with its `self.data_manager` instance.
    *   This might involve retrieving `data_config` from the `PipelineContext`, then getting `source_id`, `source_type`, and `source_path`.
    *   Then, calling `self.data_manager.register_data_source()` with these details if the source is not already registered. The path resolution for `source_path` should be consistent with how it's done in `DataIngestionStage` (using `config_manager.loader.base_path`).
*   The `ValueError: Data source not registered: dummy_csv_data` in `EvaluationStage` is resolved.
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) proceeds past the point where this `ValueError` previously occurred in `EvaluationStage`.
*   The pipeline either completes successfully or fails at a later point for a different reason.

## 📋 Checklist

*   [✅] **Modify `reinforcestrategycreator_pipeline/src/pipeline/stages/evaluation.py`**:
    *   In the `EvaluationStage.setup()` method (or at the beginning of `run()`):
        *   Retrieve `config_manager = self.context.get("config_manager")`.
        *   Retrieve `data_config = config_manager.get_config().data`.
        *   Get `source_id = data_config.source_id`, `source_type = data_config.source_type`, `source_path = data_config.source_path`.
        *   Resolve `source_path` correctly: if not absolute, `resolved_source_path = (config_manager.loader.base_path / source_path).resolve()`.
        *   Check if `source_id` is already in `self.data_manager.sources`.
        *   If not, create the appropriate `DataSource` (e.g., `CsvDataSource`) using `resolved_source_path` and other necessary parameters from `data_config`.
        *   Register the data source: `self.data_manager.register_data_source(source_id, data_source_instance)`.
*   [✅] **Test**:
    *   Run `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory.
    *   Verify that the `ValueError` in `EvaluationStage` is resolved.
    *   Confirm that the pipeline progresses further.