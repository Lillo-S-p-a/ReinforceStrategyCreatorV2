# 10. How to Extend the Pipeline

The ReinforceStrategyCreator Pipeline is designed with modularity and extensibility in mind, allowing users to customize and enhance its capabilities to suit specific needs. Here are common ways to extend the pipeline:

### 10.1. Adding New Data Sources
To incorporate a new data source (e.g., a different financial data provider API, a new database):
1.  **Implement a Data Fetcher:** Create a new Python class or set of functions within the `src/data/` directory (or a relevant subdirectory) responsible for connecting to the new source, fetching data for specified symbols and date ranges, and transforming it into a standard format (e.g., a Pandas DataFrame with OHLCV columns).
2.  **Update Configuration:**
    *   Modify `pipeline.yaml` to include a new `source_type` (e.g., `"my_new_api"`) and any necessary parameters for the new source (e.g., API endpoint, credentials).
    *   The `DataIngestionStage` would need to be updated or designed to recognize this new `source_type` and delegate to your new data fetcher.
3.  **Register (if applicable):** If the pipeline uses a factory pattern for data sources, register your new data fetcher with it.

### 10.2. Adding New Feature Engineering Steps
To add custom feature engineering logic:
1.  **Implement Transformation Logic:** Write Python functions or classes that take the market data (typically a Pandas DataFrame) as input and return the DataFrame augmented with new features. These should ideally be placed in a relevant module within `src/pipeline/stages/feature_engineering.py` or a dedicated `src/features/` directory.
2.  **Integrate into `FeatureEngineeringStage`:**
    *   Modify the `FeatureEngineeringStage` class to call your new transformation functions.
    *   Alternatively, if the stage is designed for pluggable transformations, you might configure it to use your new steps via `pipeline.yaml` (e.g., by adding a new entry under `data.transformation` that specifies your custom function/class and its parameters).
3.  **Ensure Compatibility:** Make sure your new features are in a format suitable for the RL model's observation space.

### 10.3. Adding New RL Models
To introduce a new reinforcement learning algorithm:
1.  **Implement the Model:** Create a new Python class for your RL model within `src/models/implementations/` (or a similar path). This class should typically:
    *   Define the neural network architecture.
    *   Implement the learning algorithm (how it updates its policy/value functions).
    *   Provide methods for action selection, training, saving, and loading.
    *   Adhere to a common interface expected by the `TrainingStage` and `ModelFactory`.
2.  **Register with Model Factory:** If a `ModelFactory` (e.g., in `src/models/factory.py`) is used, register your new model class with it, associating a string identifier (e.g., `"MyCustomRLModel"`) with your class.
3.  **Update Configuration:** In `pipeline.yaml`, set `model.model_type` to the string identifier of your new model and provide any necessary `model.hyperparameters`.

### 10.4. Adding New Evaluation Metrics
To evaluate strategies using custom performance metrics:
1.  **Implement Metric Calculation:** Write a Python function that takes the trade history, equity curve, or other relevant evaluation data as input and returns the calculated metric value.
2.  **Integrate into `EvaluationStage`:**
    *   Modify the `EvaluationStage` to call your new metric calculation function.
    *   If the stage supports custom metrics via configuration, you might add the name of your new metric to the `evaluation.metrics` list in `pipeline.yaml` and ensure the stage can discover and execute your function.
3.  **Update Reporting:** Ensure your new metric is included in the generated evaluation reports and visualizations.

### 10.5. Customizing Pipeline Stages
For more significant changes, you might need to customize existing pipeline stages or create entirely new ones:
1.  **Subclass Existing Stages:** If you need to modify the behavior of an existing stage slightly, consider creating a subclass and overriding specific methods.
2.  **Create New Stages:** For entirely new processing steps, define a new stage class adhering to the pipeline's stage interface (e.g., having an `execute()` method and handling configuration).
3.  **Update `pipelines_definition.yaml`:** Add your new or customized stage to the desired pipeline definition in `pipelines_definition.yaml`, specifying its `name`, `module`, and `class`.
4.  **Manage Data Flow:** Ensure the inputs and outputs of your new/customized stage are compatible with adjacent stages in the pipeline.

When extending the pipeline, always consider writing corresponding unit and integration tests to ensure your changes are correct and do not break existing functionality.