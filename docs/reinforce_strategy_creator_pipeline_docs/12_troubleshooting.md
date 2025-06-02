# 12. Troubleshooting

This section provides guidance on common issues that might arise when working with the ReinforceStrategyCreator Pipeline and suggests potential solutions.

*   **Issue: Configuration Errors (e.g., `FileNotFoundError` for `pipeline.yaml` or `pipelines_definition.yaml`)**
    *   **Cause:** The path provided to the `ModelPipeline` orchestrator for configuration files might be incorrect, or the files might be missing. Paths in `pipeline.yaml` (like `data.source_path` or `model.checkpoint_dir`) might be incorrect relative to the execution context.
    *   **Solution:**
        *   Verify that the `config_path` passed during `ModelPipeline` instantiation points to the correct `pipeline.yaml`.
        *   Ensure `pipelines_definition.yaml` exists in the expected location (usually `configs/base/`).
        *   Check that all relative paths within `pipeline.yaml` (e.g., for data sources, checkpoint directories, log directories) are correct with respect to where the pipeline is being run or how the `ConfigManager` resolves them.
        *   Ensure YAML syntax is correct in all configuration files. Use a YAML linter if necessary.

*   **Issue: Dependency Conflicts or Missing Dependencies**
    *   **Cause:** The Python environment may not have all the required packages installed, or there might be version conflicts between packages.
    *   **Solution:**
        *   Ensure you are working within the correct activated virtual environment.
        *   Re-install dependencies using `pip install -e .` (and `pip install -e ".[dev]"` for development tools).
        *   Check `pyproject.toml` (or `requirements.txt` / `setup.py`) for specific version constraints. If conflicts arise, you might need to adjust versions or create a fresh environment.

*   **Issue: Data Ingestion Failures (e.g., cannot fetch data, CSV parsing errors)**
    *   **Cause:**
        *   For API sources: Incorrect API endpoint, invalid API key, network connectivity issues, rate limiting by the provider.
        *   For CSV sources: Incorrect file path, malformed CSV file, incorrect delimiter or encoding.
    *   **Solution:**
        *   Verify API credentials and endpoint URLs. Check network connectivity.
        *   Ensure the CSV file exists at the specified `data.source_path` and is readable.
        *   Inspect the CSV file for formatting issues.
        *   Check logs from the `DataIngestionStage` for more specific error messages.

*   **Issue: Model Training Errors (e.g., `CUDA out of memory`, shape mismatches, slow training)**
    *   **Cause:**
        *   `CUDA out of memory`: GPU memory is insufficient for the model size and batch size.
        *   Shape mismatches: The dimensions of data, model layers, or environment observation/action spaces are incompatible.
        *   Slow training: Inefficient data loading, large model, unoptimized hyperparameters, CPU-bound operations.
    *   **Solution:**
        *   Reduce `training.batch_size` or simplify model architecture (`model.hyperparameters.hidden_layers`) if encountering memory issues.
        *   Carefully check the shapes of inputs and outputs at each layer of your model and ensure they align with the environment's expected shapes. Debugging tools or print statements can help here.
        *   Profile the training loop to identify bottlenecks. Consider optimizing data preprocessing, using more efficient operations, or exploring distributed training if applicable.
        *   Ensure that PyTorch/TensorFlow is correctly configured to use the GPU if available.

*   **Issue: Errors during Hyperparameter Optimization (HPO)**
    *   **Cause:** Incorrect HPO configuration (search space, trial scheduler, pruner), issues with the underlying HPO framework (Ray Tune, Optuna), or errors within individual training trials.
    *   **Solution:**
        *   Double-check the HPO configuration in `pipeline.yaml` or the dedicated HPO config file.
        *   Examine logs from the HPO framework and individual trial logs for specific error messages.
        *   Start with a simpler HPO setup (smaller search space, fewer trials) to isolate issues.

*   **Issue: Evaluation Metrics Seem Incorrect or Unexpected**
    *   **Cause:** Bugs in metric calculation logic, issues with test data preparation, incorrect benchmark implementation, or the model is genuinely performing poorly.
    *   **Solution:**
        *   Review the implementation of each evaluation metric.
        *   Verify that the test data is being loaded and preprocessed correctly and is truly unseen by the model during training.
        *   Ensure benchmark calculations are accurate.
        *   Analyze detailed trade logs or episode data from the evaluation to understand the model's behavior.

*   **General Troubleshooting Steps:**
    *   **Check Logs:** The primary source of information for diagnosing issues. Increase `monitoring.log_level` to "DEBUG" in `pipeline.yaml` for more detailed output. Check both console output and file logs (e.g., in `logs/` or `training.log_dir`).
    *   **Simplify Configuration:** Temporarily revert to a minimal, known-good configuration to see if the issue persists. This can help isolate problematic settings.
    *   **Test Stages Individually:** If possible, try to run or test individual pipeline stages in isolation to pinpoint where an error is occurring.
    *   **Consult `README.md` and Existing Documentation:** The project's main `README.md` or other existing documentation might contain solutions to common problems.
    *   **Review Code:** If logs are unhelpful, stepping through the relevant code sections with a debugger can be invaluable.

This list is not exhaustive. Always refer to specific error messages and logs for the most direct clues.