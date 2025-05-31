# 4. Pipeline Orchestration (`ModelPipeline`)

The `ModelPipeline` class, likely found within `reinforcestrategycreator_pipeline/src/pipeline/orchestrator.py` (based on the Quick Start example), serves as the central orchestrator for the entire process. It is responsible for managing the lifecycle of a trading strategy's development, from initial setup to the execution of various stages.

### 4.1. Role of the Orchestrator
The primary role of the `ModelPipeline` orchestrator includes:
*   **Configuration Loading:** Initializing and managing the pipeline's configuration by loading settings from `pipeline.yaml` and `pipelines_definition.yaml`, and merging any environment-specific overrides.
*   **Stage Management:** Dynamically instantiating and managing the sequence of pipeline stages (e.g., Data Ingestion, Feature Engineering, Training, Evaluation) as defined in `pipelines_definition.yaml`.
*   **Execution Control:** Driving the execution flow by calling each configured stage in the specified order.
*   **Data Flow Coordination:** Ensuring that the output of one stage is correctly passed as input to the next, managing the overall data flow through the pipeline.
*   **Resource Management:** Potentially handling shared resources or context that needs to be available across different stages.
*   **Error Handling and Logging:** Implementing top-level error handling and consistent logging throughout the pipeline's execution.

### 4.2. Initialization
When an instance of `ModelPipeline` is created, it typically performs the following initialization steps:
1.  **Load Configuration:** It reads the main `pipeline.yaml` (specified by `config_path` during instantiation) and the `pipelines_definition.yaml`. If an environment is specified (e.g., via an argument or an environment variable), it merges the corresponding environment-specific configuration file from `configs/environments/`.
2.  **Instantiate Stages:** Based on the `pipelines_definition.yaml`, the orchestrator dynamically loads and instantiates the Python classes for each stage defined in the selected pipeline (e.g., `full_cycle_pipeline`). Each stage instance is typically provided with the relevant portion of the consolidated configuration.
3.  **Setup Services:** It may initialize shared services like logging, monitoring (if enabled), and the artifact store.

### 4.3. Execution Flow
The `run()` method of the `ModelPipeline` orchestrator executes the defined pipeline stages sequentially.
1.  The orchestrator iterates through the list of instantiated stages.
2.  For each stage, it calls a standardized execution method (e.g., `stage.execute()` or `stage.run()`).
3.  The output(s) of a completed stage are typically passed as input(s) to the subsequent stage. The orchestrator manages this data handoff.
4.  Progress, logs, and any errors are recorded throughout the execution.

```mermaid
graph TD
    A[ModelPipeline.run()] --> B{Loop through Stages};
    B --> C(Stage 1: Execute);
    C --> D{Output from Stage 1};
    D --> E(Stage 2: Execute with Input from Stage 1);
    E --> F{Output from Stage 2};
    F --> G(...);
    G --> H(Stage N: Execute with Input from Stage N-1);
    H --> I{Final Output};
```

The modular design allows for flexibility in defining different pipelines by simply altering the `pipelines_definition.yaml` file to include, exclude, or reorder stages without changing the core orchestrator logic.