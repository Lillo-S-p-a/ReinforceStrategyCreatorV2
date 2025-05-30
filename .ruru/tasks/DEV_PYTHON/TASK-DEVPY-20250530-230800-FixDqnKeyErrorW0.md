+++
id = "TASK-DEVPY-20250530-230800"
title = "Fix KeyError: 'W0' in DQN model during training"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "dev-python"
coordinator = "roo-commander"
created_date = "2025-05-30T23:08:00Z"
updated_date = "2025-05-30T23:24:00Z"
tags = ["python", "dqn", "key-error", "training", "pipeline", "rl", "neural-network"]
related_docs = [
    "reinforcestrategycreator_pipeline/src/models/implementations/dqn.py",
    "reinforcestrategycreator_pipeline/src/training/engine.py"
]
+++

## 📝 Description

When running the full pipeline (`run_main_pipeline.py`), the `TrainingStage` fails with a `KeyError: 'W0'`. The error occurs within the `_forward` method of the `DQN` model ([`reinforcestrategycreator_pipeline/src/models/implementations/dqn.py`](reinforcestrategycreator_pipeline/src/models/implementations/dqn.py)) when it attempts to access `weights[f"W{i}"]`.

The traceback is:
```
KeyError: 'W0'
  File "/home/alessio/Personal/ReinforceStrategyCreatorV2/reinforcestrategycreator_pipeline/src/models/implementations/dqn.py", line 189, in _forward
    x = np.dot(x, weights[f"W{i}"]) + weights[f"b{i}"]
                  ~~~~~~~^^^^^^^^^
```

This indicates that the `weights` dictionary for the Q-network (or target network) does not contain the expected key 'W0', which likely corresponds to the weights of the first layer of the neural network. This could be due to an issue in the network initialization (`_initialize_network` method) or how the weights are structured and accessed.

## ✅ Acceptance Criteria

*   The `KeyError: 'W0'` in the `DQN._forward` method is resolved.
*   The `DQN` model initializes its network weights correctly, ensuring that keys like 'W0', 'b0', 'W1', 'b1', etc., are present as expected by the `_forward` method for all layers.
*   The `TrainingStage` in the pipeline completes without this `KeyError`.
*   Running `python run_main_pipeline.py` (from the `reinforcestrategycreator_pipeline` directory) proceeds past the point where this `KeyError` previously occurred.

## 📋 Checklist

*   [✅] **Investigate `reinforcestrategycreator_pipeline/src/models/implementations/dqn.py`**:
    *   Examine the `_initialize_network` method to understand how `self.q_network_weights` and `self.target_network_weights` are populated.
    *   Verify that the keys being generated (e.g., `f"W{i}"`, `f"b{i}"`) match how they are accessed in the `_forward` method, especially for the first layer (i=0).
    *   Check the layer indexing and naming consistency.
    *   **Finding**: The issue was not in the initialization but in the `get_model_state` method which was corrupting the weights dictionary during serialization.
*   [✅] **Modify `_initialize_network` or `_forward`**:
    *   Adjust the initialization logic to ensure the `weights` dictionary contains the correct keys (e.g., 'W0', 'b0').
    *   Alternatively, if the key naming convention in `_initialize_network` is different (e.g., starts from 'W1'), adjust the `_forward` method to match. Consistency is key.
    *   **Solution**: Fixed the `get_model_state` method to use deep copy to avoid modifying the original weights. Also added validation in `_forward` and recovery logic in `train` method.
*   [✅] **Test**:
    *   Run `python run_main_pipeline.py` from the `reinforcestrategycreator_pipeline` directory.
    *   Verify that the `KeyError: 'W0'` is resolved and the training proceeds.
    *   **Result**: The KeyError has been resolved. The model now trains successfully through all epochs.