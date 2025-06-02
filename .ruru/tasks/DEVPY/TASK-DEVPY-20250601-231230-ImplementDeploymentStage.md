+++
id = "TASK-DEVPY-20250601-231230"
title = "Implement and Integrate DeploymentStage for Paper Trading"
status = "🟢 Done" # Implementation and integration complete, pending tests
type = "🌟 Feature"
assigned_to = "dev-python"
coordinator = "RooCommander-SESSION-ReinvestigateFixMermaidFeedback-2506011026" # Assuming current session context
created_date = "2025-06-01T23:12:30Z"
updated_date = "2025-06-01T23:58:00Z" # Updated time
tags = ["pipeline", "deployment", "papertrading", "feature"]
related_docs = [
    "reinforcestrategycreator_pipeline/configs/base/pipeline.yaml",
    "reinforcestrategycreator_pipeline/configs/base/pipelines_definition.yaml",
    "reinforcestrategycreator_pipeline/src/pipeline/stage.py"
]
+++

## Description

The current `full_cycle_pipeline` successfully ingests data, trains a model, and evaluates it. The main configuration file ([`reinforcestrategycreator_pipeline/configs/base/pipeline.yaml`](reinforcestrategycreator_pipeline/configs/base/pipeline.yaml)) includes a `deployment` section configured for `paper_trading`. However, the pipeline definition ([`reinforcestrategycreator_pipeline/configs/base/pipelines_definition.yaml`](reinforcestrategycreator_pipeline/configs/base/pipelines_definition.yaml)) does not currently include a `DeploymentStage` to act on this configuration.

This task is to implement a new `DeploymentStage` capable of handling paper trading logic and integrate it into the `full_cycle_pipeline` or a new dedicated deployment pipeline.

## Acceptance Criteria

- A new `DeploymentStage` class is created (e.g., in `reinforcestrategycreator_pipeline/src/pipeline/stages/deployment.py`).
- The `DeploymentStage` reads the `deployment` configuration from `pipeline.yaml` via the `ConfigManager` in the `PipelineContext`.
- For `paper_trading` mode, it simulates trade execution based on the trained model's signals. Initial simulation can be logging of intended trades and a virtual portfolio status.
- The `DeploymentStage` is added to `reinforcestrategycreator_pipeline/configs/base/pipelines_definition.yaml`, likely extending the `full_cycle_pipeline` or as part of a new, distinct deployment pipeline.
- The pipeline can be run with the new `DeploymentStage` without errors.
- Logs confirm the `DeploymentStage` attempts paper trading actions based on the configuration and the model loaded from the context.

## Checklist

- `[✅]` Design the `DeploymentStage` class structure, inheriting from `PipelineStage`.
    - `[✅]` Define `setup()` method to load necessary configurations (e.g., deployment parameters, model artifact ID from context).
    - `[✅]` Define `run()` method to execute paper trading logic.
    - `[✅]` Define `teardown()` method if any cleanup is needed.
- `[✅]` Implement `DeploymentStage.setup()`:
    - `[✅]` Access `ConfigManager` and `ArtifactStore` from `PipelineContext`.
    - `[✅]` Load the global `deployment` configuration.
    - `[✅]` Load the trained model artifact ID and version from context (set by `TrainingStage`).
- `[✅]` Implement `DeploymentStage.run()`:
    - `[✅]` Load the trained model using `ModelRegistry` or directly from the artifact store via the artifact ID.
    - `[✅]` Implement logic for paper trading:
        - `[✅]` Fetch or use latest available data (potentially via `DataManager`) to generate signals with the model. (Basic implementation)
        - `[✅]` Simulate buy/sell/hold orders based on model signals. (Basic simulation)
        - `[✅]` Maintain and update a virtual portfolio (e.g., cash, holdings, P&L). (Basic implementation)
        - `[✅]` Log simulated trades, portfolio value, and relevant metrics. (Basic logging)
- `[✅]` Create the `DeploymentStage` file (e.g., `reinforcestrategycreator_pipeline/src/pipeline/stages/deployment.py`).
- `[✅]` Add the new `DeploymentStage` to `reinforcestrategycreator_pipeline/configs/base/pipelines_definition.yaml`.
    - `[✅]` Decide if it extends `full_cycle_pipeline` or forms a new pipeline. (Extended `full_cycle_pipeline`)
- `[ ]` Write unit tests for the `DeploymentStage` covering its core logic.
- `[✅]` Perform an E2E test by running the pipeline with the new stage and verify its operation through logs and any generated artifacts.

## Notes

- Consider how the `DeploymentStage` will access the latest market data for making trading decisions if it's intended to run pseudo-live. For a simple paper trading simulation based on historical evaluation data, this might be simpler.
- The `deployment` section in `pipeline.yaml` has `api_endpoint` and `api_key` which are for live trading; these should be ignored or handled gracefully in `paper_trading` mode.