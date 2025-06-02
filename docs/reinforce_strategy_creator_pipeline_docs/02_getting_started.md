# 2. Getting Started

### 2.1. Installation
Follow these steps to set up the ReinforceStrategyCreator Pipeline environment:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd reinforcestrategycreator_pipeline
    ```
    *(Replace `<repository-url>` with the actual URL of the repository.)*

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install the package and its dependencies:**
    ```bash
    pip install -e .
    ```
    For development purposes, including testing and linting tools, install with the `dev` extras:
    ```bash
    pip install -e ".[dev]"
    ```

### 2.2. Quick Start Example
To run the pipeline with a default configuration, you can use the following Python script. This typically involves initializing the `ModelPipeline` orchestrator with the path to a base configuration file and then calling its `run()` method.

```python
from reinforcestrategycreator_pipeline.src.pipeline.orchestrator import ModelPipeline # Assuming this is the correct path

# Example: Run the pipeline with a base configuration
try:
    pipeline = ModelPipeline(config_path="configs/base/pipeline.yaml")
    pipeline.run()
    print("Pipeline execution completed successfully.")
except Exception as e:
    print(f"An error occurred during pipeline execution: {e}")

```
*(Note: The import path for `ModelPipeline` might need adjustment based on the final project structure. The example in `README.md` was `from pipeline.orchestrator import ModelPipeline` which might be relative if run from the root of `reinforcestrategycreator_pipeline`)*