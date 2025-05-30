+++
id = "TASK-DEVPY-20250530-234200"
title = "Fix TypeError in TrainingStage: model.save() returns dict instead of path string"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-30T23:42:00Z"
updated_date = "2025-05-30T23:58:00Z"
tags = ["python", "type-error", "model-saving", "pipeline", "training-stage", "rl"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/pipeline/stages/training.py",
    "reinforcestrategycreator_pipeline/src/models/base.py",
    "reinforcestrategycreator_pipeline/src/models/implementations/dqn.py"
]
+++

## 📝 Description

When running the full pipeline (`run_main_pipeline.py`), after the training epochs complete, the `TrainingStage` fails with a `TypeError` when trying to save the model artifact.
The error is: `TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'dict'`.
This occurs at line 255 of `reinforcestrategycreator_pipeline/src/pipeline/stages/training.py`:
```python
artifact_path_to_save = Path(rllib_checkpoint_path)
```
The variable `rllib_checkpoint_path` is the result of `self.trained_model.save(str(temp_model_save_dir))`. This indicates that the `save()` method of the trained model (e.g., `DQN.save()`, which likely calls `BaseModel.save()`) is returning a dictionary instead of the expected string path to the saved checkpoint.

The `BaseModel.save()` method in `reinforcestrategycreator_pipeline/src/models/base.py` should return the `checkpoint_path`.

## ✅ Acceptance Criteria

*   The `save()` method of the `BaseModel` (and by extension, its subclasses like `DQN`) is modified to consistently return a string path to the saved model checkpoint.
*   The `TrainingStage` in `reinforcestrategycreator_pipeline/src/pipeline/stages/training.py` can successfully construct a `Path` object from the return value of `self.trained_model.save()`.
*   The `TypeError` is resolved, and the model artifact is successfully saved by the `TrainingStage`.
*   The `trained_model_artifact_id` is correctly set in the `PipelineContext`.
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) proceeds past the model saving step in `TrainingStage` without this `TypeError`.
*   The `EvaluationStage` no longer fails due to a missing `trained_model_artifact_id`.

## 📋 Checklist

*   [✅] **Investigate `reinforcestrategycreator_pipeline/src/models/base.py`**:
    *   Examine the `BaseModel.save()` method. Ensure it returns `checkpoint_path` (which should be a string).
*   [✅] **Investigate `reinforcestrategycreator_pipeline/src/models/implementations/dqn.py` (and other model implementations if necessary)**:
    *   If model-specific `save()` methods override `BaseModel.save()`, ensure they also return a string path.
    *   The current `DQN.save()` method in `dqn.py` calls `super().save(checkpoint_dir_path)`. Verify what `super().save()` returns.
*   [✅] **Modify `save()` method(s)**:
    *   Ensure the final return value from the `save()` call chain is the string path to the checkpoint.
*   [✅] **Test**:
    *   Run `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory.
    *   Verify that the `TypeError` is resolved.
    *   Confirm that the pipeline progresses to the `EvaluationStage` and that the `trained_model_artifact_id` is available.