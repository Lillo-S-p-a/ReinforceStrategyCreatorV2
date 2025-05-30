+++
id = "TASK-DEVPY-20250530-230300"
title = "Fix AttributeError in DQN get_model_state for model saving"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-30T23:03:00Z"
updated_date = "2025-05-30T23:04:00Z"
tags = ["python", "dqn", "attribute-error", "model-saving", "pipeline", "rl"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/models/implementations/dqn.py",
    "reinforcestrategycreator_pipeline/src/pipeline/stages/training.py",
    "reinforcestrategycreator_pipeline/src/models/base.py"
]
+++

## 📝 Description

When running the full pipeline (`run_main_pipeline.py`), an `AttributeError: 'list' object has no attribute 'tolist'` occurs in the `get_model_state` method of `reinforcestrategycreator_pipeline/src/models/implementations/dqn.py`. This happens when the `TrainingStage` attempts to save the trained DQN model.

The traceback indicates the error is here:
```python
# reinforcestrategycreator_pipeline/src/models/implementations/dqn.py
# In get_model_state method
k: v.tolist() for k, v in state["q_network"]["weights"].items()
   ^^^^^^^^
AttributeError: 'list' object has no attribute 'tolist'
```
This suggests that some values (`v`) within `state["q_network"]["weights"]` are already Python lists, or are not NumPy arrays/tensors that have a `.tolist()` method. The `get_model_state` method should return a state dictionary that is serializable by `pickle`.

This error prevents the model from being saved, which means `trained_model_artifact_id` is not set in the `PipelineContext`. Consequently, the `EvaluationStage` fails because it requires this ID.

## ✅ Acceptance Criteria

*   The `get_model_state` method in `reinforcestrategycreator_pipeline/src/models/implementations/dqn.py` is modified to correctly handle the structure of `state["q_network"]["weights"]` and ensure all parts of the returned state are serializable (e.g., by checking types before calling `.tolist()` or by ensuring weights are consistently stored as NumPy arrays/tensors that support `.tolist()`).
*   The `TrainingStage` can successfully save the DQN model without encountering the `AttributeError`.
*   The `trained_model_artifact_id` is correctly set in the `PipelineContext` after the `TrainingStage`.
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) proceeds past the `TrainingStage`'s model saving step without this `AttributeError`.
*   The `EvaluationStage` no longer fails due to a missing `trained_model_artifact_id` (it may fail for other reasons, but this specific blocker should be removed).

## 📋 Checklist

*   [✅] **Investigate `reinforcestrategycreator_pipeline/src/models/implementations/dqn.py`**:
    *   Examine the `get_model_state` method.
    *   Determine the actual type and structure of `state["q_network"]["weights"]` and its items at the point of failure.
*   [✅] **Modify `get_model_state`**:
    *   Implement a robust way to convert the Q-network weights to a serializable format (e.g., lists of numbers). This might involve checking `isinstance(v, np.ndarray)` or `isinstance(v, torch.Tensor)` before calling `v.tolist()`, and handling plain lists appropriately.
*   [✅] **Test**:
    *   Run `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory.
    *   Verify that the `AttributeError` is resolved.
    *   Confirm that the pipeline progresses to the `EvaluationStage` and that the `trained_model_artifact_id` is available in the context for that stage.