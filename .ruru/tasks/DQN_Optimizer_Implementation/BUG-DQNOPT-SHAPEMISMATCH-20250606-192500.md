+++
id = "BUG-DQNOPT-SHAPEMISMATCH-20250606-192500"
title = "Fix Shape Mismatch in DQN Adam Backprop for Multi-Layer"
status = "🟢 Done"
type = "🐞 Bug"
assigned_to = "util-senior-dev"
coordinator = "TASK-CMD-DQNOPT-20250606-191500" # Related to the previous feature
created_date = "2025-06-06T19:25:00Z"
updated_date = "2025-06-06T19:59:00Z"
related_docs = [
    "reinforcestrategycreator_pipeline/src/models/implementations/dqn.py",
    ".ruru/tasks/DQN_Optimizer_Implementation/TASK-SRDEV-DQNOPT-20250606-191500.md"
]
tags = ["dqn", "optimizer", "backpropagation", "adam", "shape-mismatch", "bugfix", "numpy", "multi-layer-mlp"]
+++

## Description

The recently implemented Adam optimizer and backpropagation in [`dqn.py`](reinforcestrategycreator_pipeline/src/models/implementations/dqn.py) (Task: `TASK-SRDEV-DQNOPT-20250606-191500`) is causing a `ValueError` during training when the network is configured with multiple hidden layers (e.g., `hidden_layers: [256, 128, 64]`).

The error is:
`ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 64 is different from 256)`
This occurs at the line: `Z1 = A0 @ W_out + b_out` in `_train_step`.

The issue appears to be that the backpropagation logic and the caching of activations (`X_batch`, `Z0`, `A0`, `Z1`) in `_train_step` were implemented assuming a simple 2-layer MLP (one hidden layer), while the model's `_initialize_networks` and `_forward` methods correctly handle a dynamic number of hidden layers based on `self.hidden_layers`.

The backpropagation and Adam update steps need to be generalized to work correctly with the actual number of layers defined in `self.hidden_layers` and the corresponding weight matrices (`W0, b0, W1, b1, W2, b2, ... W_out, b_out`).

## Acceptance Criteria

1.  The `ValueError` (shape mismatch) during `_train_step` is resolved.
2.  The backpropagation logic in `_train_step` correctly calculates gradients for all weight matrices in a multi-layer perceptron, as defined by `self.hidden_layers`. This includes:
    *   Correctly caching all intermediate activations and pre-activations (Z values) during the forward pass specific to `_train_step`.
    *   Iteratively backpropagating gradients through each layer, from output to input.
3.  The Adam optimizer update rule is correctly applied to all weight matrices using their respective gradients.
4.  The pipeline (`reinforcestrategycreator_pipeline/run_main_pipeline.py`) runs successfully with the default configuration (`hidden_layers: [256, 128, 64]`).
5.  The training loss shows a trend indicative of learning (e.g., generally decreasing, not NaN or Inf).

## Checklist

-   [x] Modify `_train_step` to perform a full forward pass that caches all necessary activations (`A_layers`) and pre-activations (`Z_layers`) for all layers, consistent with `self.hidden_layers`.
-   [x] Modify the backpropagation gradient calculation in `_train_step` to iterate backward through the layers, correctly calculating `dW` and `db` for each layer (`W_out, b_out, ... W2, b2, W1, b1, W0, b0`).
    -   [x] Ensure correct handling of ReLU derivatives for hidden layers and linear derivative for the output layer.
-   [x] Ensure the Adam optimizer updates are applied correctly to each corresponding weight matrix and its bias.
-   [x] Verify that Adam optimizer state variables (`m`, `v`, `t`) are correctly managed for all weight matrices.
-   [x] Test thoroughly by running the main pipeline with the default multi-layer configuration.
-   [x] Confirm that the loss is a scalar and shows a reasonable trend.

## Notes
- The network structure is: Input -> (Dense -> ReLU) * N_hidden_layers -> Dense (linear output) -> Q-values.
- Weight keys in `self.q_network['weights']` are `W0, b0, W1, b1, ... W{L-1}, b{L-1}` where `L` is the total number of weight layers (number of hidden layers + 1 output layer).
- `self.hidden_layers` is a list of integers, e.g., `[256, 128, 64]`.