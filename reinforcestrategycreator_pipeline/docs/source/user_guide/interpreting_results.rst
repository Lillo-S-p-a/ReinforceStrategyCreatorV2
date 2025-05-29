Interpreting Results and Reports
================================

After running a pipeline, especially training or evaluation, various outputs are generated. This section guides you on where to find these results and how to interpret them.

Location of Results
-------------------

*   **Results Directory**: If configured (e.g., ``evaluation.save_results: true`` and ``evaluation.results_dir: "./results"`` in ``pipeline.yaml``), detailed evaluation outputs, metrics files, and plots created by components like the ``EvaluationEngine`` (using ``PerformanceVisualizer`` and ``ReportGenerator`` from ``src.visualization``) will be saved to the specified directory.
*   **Artifact Store**: Key pipeline artifacts like trained models, processed datasets, and comprehensive evaluation reports are versioned and stored in the configured Artifact Store.
    *   If using the ``LocalArtifactStore`` (default, ``artifact_store.type: "local"``), these are typically found under the ``artifact_store.root_path`` (e.g., ``./artifacts``). Artifacts are organized by ``artifact_id`` (e.g., model name, dataset name), and then by ``version``. Each version will have the artifact file itself and a ``.meta.json`` file containing its metadata.
    *   The ``ModelRegistry`` (from ``src.models.registry``) is the primary interface for managing models within the Artifact Store. You would typically use its methods (e.g., ``register_model``, ``get_model_version``, ``list_model_versions``) programmatically to interact with stored models. For example, after training, the ``TrainingEngine`` uses the ``ModelRegistry`` to save the new model version. For evaluation or deployment, you'd use the ``ModelRegistry`` to retrieve a specific model version by its ID and version string.
*   **Log Directory**: Training logs, including TensorBoard logs (if ``training.use_tensorboard: true``), are typically saved in the directory specified by ``training.log_dir`` (e.g., ``./logs``). Model checkpoints from training are saved in the directory specified by ``model.checkpoint_dir`` (e.g., ``./checkpoints/my_model_training``).
*   **Console Output**: The pipeline will also print summary information and progress to the console during execution.

Common Outputs
--------------

1.  **Metrics Files**:
    *   Usually saved as CSV or JSON files in the results directory or logged by the ``ArtifactStore``.
    *   Contain values for the metrics specified in the ``evaluation.metrics`` configuration (e.g., Sharpe ratio, total return).
    *   Training history (loss, metrics per epoch/step) might also be saved.

2.  **Plots**:
    *   If ``evaluation.generate_plots: true``, various performance plots are generated.
    *   Common plots include:

        *   Portfolio value over time.
        *   Returns distribution.
        *   Drawdown chart.
        *   Comparison against benchmarks.
        *   For HPO: Parallel coordinate plots, hyperparameter importance.

3.  **Logs**:
    *   **Pipeline Logs**: Detailed logs of the pipeline execution, including information from each stage, warnings, and errors. These are typically found in the main log file or console output.
    *   **TensorBoard Logs**: If enabled (``training.use_tensorboard: true``), you can launch TensorBoard to visualize training progress in real-time or after completion. Point TensorBoard to the ``training.log_dir``:

        .. code-block:: bash

           tensorboard --logdir ./logs/my_training_run

       Replace ``./logs/my_training_run`` with the actual path to your run's log directory. TensorBoard provides interactive visualizations of metrics, model graphs, and hyperparameter distributions.

4.  **Trained Models**:
    *   Saved either as checkpoints in ``model.checkpoint_dir`` or as versioned artifacts in the ``ArtifactStore`` via the ``ModelRegistry``:

        *   These can be loaded for further evaluation, deployment, or resuming training.
    
    5.  **Hyperparameter Optimization (HPO) Results**:
        *   If HPO was run using the ``HPOptimizer``, the results will typically include:
            *   The set of best hyperparameters found.
            *   The performance metric achieved with these best hyperparameters.
            *   Detailed logs or data from all trials conducted (e.g., in a CSV file or through Ray Tune's output directories).

        *   These results are crucial for understanding which hyperparameter configurations work best and for configuring a final model for production training. The ``HPOptimizer`` might save these results to a specified output directory or log them.


Interpreting Key Metrics
------------------------

The pipeline can calculate various performance metrics, as defined in the ``evaluation.metrics`` section of your configuration. Here are some common ones:

*   **Sharpe Ratio**: Measures risk-adjusted return. A higher Sharpe ratio is generally better, indicating a better return for the amount of risk taken.
*   **Total Return**: The overall percentage gain or loss over the evaluation period.
*   **Max Drawdown**: The largest peak-to-trough decline during a specific period, indicating the downside risk. A smaller (less negative) drawdown is preferred.
*   **Win Rate**: The percentage of trades or periods that were profitable.
*   **Profit Factor**: Gross profits divided by gross losses. A value greater than 1 indicates profitability.
*   **Sortino Ratio**: Similar to Sharpe ratio, but it only penalizes downside volatility, not upside.
*   **Annualized Return**: The geometric average amount of money earned by an investment each year over a given time period.
*   **Volatility**: The degree of variation of a trading price series over time, usually measured by the standard deviation of returns.

For detailed financial interpretation and implications of these metrics in the context of your specific trading strategy, consult with a financial expert or refer to quantitative finance literature. The ``dev-python`` team can provide details on how these metrics are calculated within the pipeline.

6.  **Monitoring Outputs (e.g., Datadog)**:
    *   If the ``MonitoringService`` is enabled and configured (e.g., for Datadog via ``monitoring.datadog_api_key``), real-time metrics, logs, and events from pipeline runs will be sent to the configured monitoring platform.
    *   Users should consult their Datadog dashboards (or equivalent) to observe:
        *   System health metrics (CPU, memory usage of pipeline components).
        *   Model performance metrics during training and evaluation (e.g., loss, rewards, specific evaluation metrics).
        *   Data drift or model drift alerts if configured.
        *   Custom business metrics logged by the pipeline.

    *   The project includes example Datadog dashboard JSON definitions in ``reinforcestrategycreator_pipeline/src/monitoring/datadog_dashboards/`` which can be imported into Datadog to provide pre-built visualizations for:
        *   Model Performance (``model_performance_dashboard.json``)
        *   Drift Detection (``drift_detection_dashboard.json``)
        *   Production Monitoring (``production_monitoring_dashboard.json``)
        *   System Health (``system_health_dashboard.json``)

Reports
-------
The pipeline may also generate comprehensive HTML or PDF reports summarizing the performance, including metrics, plots, and configuration details. These are typically found in the ``evaluation.results_dir`` or the ``ArtifactStore``.

(Further details on specific report formats will be added as the reporting capabilities are finalized.)