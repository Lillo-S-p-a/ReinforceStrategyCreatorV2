Troubleshooting Guide
=====================

This guide provides solutions and tips for common issues you might encounter while working with the ReinforceStrategyCreator Pipeline.

Common Issues
-------------

1.  **Configuration Errors**
    *   **Symptom**: Pipeline fails to start, errors related to YAML parsing, missing keys, or incorrect data types.
    *   **Troubleshooting**:
            *   **Validate YAML Syntax**: Ensure your ``.yaml`` files (``pipeline.yaml``, environment configs, etc.) have correct YAML syntax. Use a YAML linter or online validator.
            *   **Check Key Names**: Verify that all configuration keys match those expected by the pipeline components (refer to the :doc:`user_guide/configuration` guide).
            *   **Data Types**: Ensure values have the correct data types (e.g., boolean ``true``/``false``, numbers, strings quoted if necessary).
            *   **Environment Variables**: If using placeholders like ``${MY_API_KEY}``, ensure the corresponding environment variables are set in your execution environment.
            *   **Merge Order**: Understand how configurations are merged (base -> environment -> experiment). An override in a later file might be causing unexpected behavior. Use the ``ConfigManager`` debug logs (if available) or print parts of the loaded config to verify.

2.  **Data Access Problems**
    *   **Symptom**: Errors during data ingestion, "File not found," API connection errors, authentication failures.
    *   **Troubleshooting**:
            *   **File Paths**: If loading from local files (e.g., CSV), ensure paths in the configuration are correct relative to the pipeline's execution directory or are absolute paths.
            *   **API Credentials**: For API data sources, double-check that API keys, secrets, and endpoint URLs are correct and that the necessary environment variables (e.g., ``${DATA_API_KEY}``) are set and accessible by the pipeline.
            *   **Network Issues**: Ensure the machine running the pipeline has network access to the API endpoints. Check firewalls or proxy settings.
            *   **Permissions**: If reading local files or writing to cache, ensure the pipeline has the necessary read/write permissions for those directories.

3.  **Python Environment & Dependencies**
    *   **Symptom**: ``ModuleNotFoundError``, ``ImportError``, version conflicts.
    *   **Troubleshooting**:
            *   **Virtual Environment**: Always use a dedicated virtual environment for the project to avoid conflicts with system-wide packages. Activate it before running any pipeline scripts.
            *   **Install Requirements**: Ensure all dependencies are installed using ``pip install -r requirements.txt`` from the ``reinforcestrategycreator_pipeline`` directory.
            *   **PYTHONPATH**: If running scripts from outside the main project directory or if modules are not found, you might need to adjust the ``PYTHONPATH`` environment variable to include the project's ``src`` directory or the project root. For example: ``export PYTHONPATH=/path/to/your/reinforcestrategycreator_pipeline:$PYTHONPATH``.

4.  **Sphinx Documentation Build Errors**
    *   **Symptom**: ``make html`` fails, warnings about unknown modules, or formatting errors.
    *   **Troubleshooting**:
            *   **PYTHONPATH for Sphinx**: When building documentation, Sphinx needs to be able to import your source code. Run the build from the ``docs/`` directory and ensure ``PYTHONPATH`` is set correctly to point to the project root (e.g., ``PYTHONPATH=.. make html``).
            *   **Source Code Imports**: ``ModuleNotFoundError`` during Sphinx build often indicates that Python files within your ``src`` directory have import statements that Sphinx cannot resolve with the current ``PYTHONPATH``. These usually need to be fixed in the source code by ``dev-python`` (e.g., ensuring consistent use of relative or absolute imports that work from the project root).
            *   **RST Formatting**: Pay close attention to Sphinx warnings about reStructuredText syntax (e.g., "Title underline too short," "Unexpected indentation"). These often require minor adjustments to spacing, blank lines, or directive syntax in your ``.rst`` files.

5.  **Model Training Issues**
    *   **Symptom**: Training runs very slowly, loss becomes NaN, metrics don't improve.
    *   **Troubleshooting**:
            *   **Hyperparameters**: Experiment with different learning rates, batch sizes, network architectures, and other hyperparameters defined in your model configuration.
            *   **Data Quality**: Ensure your training data is clean, correctly preprocessed, and representative of the problem. Data issues are a common cause of training problems.
            *   **Reward Function**: (For Reinforcement Learning) Critically evaluate your reward function. A poorly designed reward function can lead to unexpected agent behavior or failure to learn.
            *   **Exploration vs. Exploitation**: (For RL) Adjust exploration parameters (e.g., epsilon in DQN) to ensure the agent explores enough before exploiting.
            *   **Check Logs**: Examine training logs and TensorBoard for patterns in loss curves and metrics.
            *   **Resource Limits**: Ensure your machine has enough CPU, GPU (if applicable), and RAM for the training task and batch size.

6.  **Deployment Issues**
    *   **Symptom**: Errors during model packaging, deployment to paper/live trading fails, model does not execute trades as expected in deployed environment.
    *   **Troubleshooting**:
            *   **Package Contents**: If using ``ModelPackager``, inspect the contents of the generated ``.tar.gz`` file to ensure all necessary model files, configurations, and scripts are included.
            *   **Environment Consistency**: Ensure the deployment environment has all the same Python dependencies (versions) as the training environment.
            *   **API Credentials for Trading**: For paper or live trading, double-check that the correct API keys and endpoints are configured for the ``DeploymentManager`` or ``PaperTradingDeployer`` and accessible in the deployment environment.
            *   **Broker API Limits**: Be aware of any rate limits or restrictions imposed by the brokerage API.
            *   **Permissions**: Ensure the deployment process has permissions to write to deployment directories or interact with necessary services.
            *   **Logs**: Check logs from the ``DeploymentManager`` and the deployed model/application itself for specific error messages.

Contacting Support
------------------

If you encounter issues not covered here, or if the troubleshooting steps do not resolve your problem, please consult with the ``dev-python`` team or the project maintainers, providing as much detail as possible:
*   The exact command or script you were running.
*   The full error message and traceback.
*   Your configuration files (omitting any secrets).
*   Steps you've already tried.