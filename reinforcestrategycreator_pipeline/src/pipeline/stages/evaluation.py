"""Evaluation stage implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd # Added import
import json
import numpy as np
from datetime import datetime

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactType
from reinforcestrategycreator_pipeline.src.evaluation.engine import EvaluationEngine
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry
from reinforcestrategycreator_pipeline.src.data.manager import DataManager
from reinforcestrategycreator_pipeline.src.data.csv_source import CsvDataSource
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager # For type hinting


class EvaluationStage(PipelineStage):
    """
    Stage responsible for evaluating trained models.
    
    This stage handles:
    - Loading trained models and test data
    - Computing evaluation metrics
    - Generating performance reports
    - Comparing against baseline models
    - Saving evaluation results as artifacts
    """
    
    def __init__(self, name: str = "evaluation", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation stage.
        
        Args:
            name: Stage name
            config: Stage configuration containing:
                - metrics: List of metrics to compute
                - test_data_key: Key to retrieve test data from context
                - baseline_model: Optional baseline model for comparison
                - threshold_config: Performance thresholds for pass/fail
                - report_format: Format for evaluation report (json, html, etc.)
        """
        super().__init__(name, config or {})
        # Configs will be fetched from global config via context in setup
        self.global_evaluation_config: Optional[Dict[str, Any]] = None
        self.global_data_config: Optional[Dict[str, Any]] = None # For evaluation data source

        self.model_registry: Optional[ModelRegistry] = None
        self.data_manager: Optional[DataManager] = None
        self.artifact_store = None # Will be fetched from context
        self.evaluation_engine: Optional[EvaluationEngine] = None
        
    def setup(self, context: PipelineContext) -> None:
        """Set up the evaluation stage."""
        self.logger.info(f"Setting up {self.name} stage")
        
        # Get ConfigManager and ArtifactStore from context
        config_manager: Optional[ConfigManager] = context.get("config_manager")
        self.artifact_store = context.get("artifact_store")

        if not config_manager:
            raise ValueError("ConfigManager not found in pipeline context for EvaluationStage.")
        if not self.artifact_store:
            # Depending on EvaluationEngine, this might be critical or optional if not saving artifacts
            self.logger.warning("ArtifactStore not found in pipeline context for EvaluationStage.")

        # Fetch global configurations
        pipeline_cfg = config_manager.get_config()
        self.global_evaluation_config = pipeline_cfg.evaluation.model_dump() if pipeline_cfg.evaluation else {}
        # Data config for evaluation might be different, or reuse training data config for now
        self.global_data_config = pipeline_cfg.data.model_dump() if pipeline_cfg.data else {}
        
        self.logger.info(f"Global Evaluation Config: {self.global_evaluation_config}")
        self.logger.info(f"Global Data Config (for eval): {self.global_data_config}")

        # Initialize ModelRegistry, DataManager, EvaluationEngine
        self.model_registry = ModelRegistry(artifact_store=self.artifact_store)
        self.data_manager = DataManager(config_manager=config_manager, artifact_store=self.artifact_store)
        
        # EvaluationEngine needs metrics_config, benchmark_config etc. from global_evaluation_config
        metrics_cfg = self.global_evaluation_config.get("metrics_calculator_config",
                                                       {"metrics": self.global_evaluation_config.get("metrics", [])})
        benchmark_cfg = self.global_evaluation_config.get("benchmark_evaluator_config",
                                                         {"benchmark_symbols": self.global_evaluation_config.get("benchmark_symbols", [])})
        report_cfg = self.global_evaluation_config.get("report_generator_config", {})
        # visualization_config is not explicitly in pipeline.yaml, pass empty for now
        visualization_cfg = self.global_evaluation_config.get("visualization_config", {})


        self.evaluation_engine = EvaluationEngine(
            model_registry=self.model_registry,
            data_manager=self.data_manager,
            artifact_store=self.artifact_store,
            metrics_config=metrics_cfg,
            benchmark_config=benchmark_cfg,
            visualization_config=visualization_cfg,
            report_config=report_cfg
        )
        self.logger.info("ModelRegistry, DataManager, and EvaluationEngine initialized.")

        # Register the data source with DataManager
        if self.global_data_config:
            source_id = self.global_data_config.get("source_id", "dummy_csv_data")
            source_type = self.global_data_config.get("source_type", "csv")
            source_path = self.global_data_config.get("source_path")
            
            # Check if the data source is already registered
            if source_id not in self.data_manager.data_sources:
                self.logger.info(f"Registering data source '{source_id}' with DataManager")
                
                # Resolve the source path if it's relative
                if source_path and not Path(source_path).is_absolute():
                    # The path is relative to the config file location
                    resolved_source_path = (config_manager.loader.base_path / source_path).resolve()
                    self.logger.info(f"Resolved source path: {resolved_source_path}")
                else:
                    resolved_source_path = Path(source_path).resolve() if source_path else None
                
                # Prepare the configuration for the data source
                data_source_config = {
                    "source_path": str(resolved_source_path) if resolved_source_path else None,
                    "symbols": self.global_data_config.get("symbols", []),
                    "start_date": self.global_data_config.get("start_date"),
                    "end_date": self.global_data_config.get("end_date"),
                }
                
                # Register the data source
                try:
                    self.data_manager.register_source(
                        source_id=source_id,
                        source_type=source_type,
                        config=data_source_config
                    )
                    self.logger.info(f"Successfully registered data source '{source_id}'")
                except Exception as e:
                    self.logger.error(f"Failed to register data source '{source_id}': {str(e)}")
                    raise
            else:
                self.logger.info(f"Data source '{source_id}' is already registered")

        # Validate required data from previous stage is available
        if not context.get("trained_model"): # This is the actual model object
            raise ValueError("No trained_model object found in context. Run training stage first.")
        if not context.get("model_type") or not context.get("model_config"):
             self.logger.warning("model_type or model_config not found in context from TrainingStage.")
        # The EvaluationEngine.evaluate expects model_id and version.
        # TrainingStage needs to put these in context, or EvaluationEngine needs to be adapted.
        # For now, we'll assume TrainingStage puts 'trained_model_artifact_id' and 'trained_model_version'.
        if not context.get("trained_model_artifact_id"):
            self.logger.warning("trained_model_artifact_id not found in context. EvaluationEngine might rely on this.")
        
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the evaluation stage.
        
        Args:
            context: Pipeline context containing trained model and test data
            
        Returns:
            Updated pipeline context with evaluation results
        """
        self.logger.info(f"Running {self.name} stage using RL EvaluationEngine.")

        if not self.evaluation_engine:
            raise RuntimeError("EvaluationEngine not initialized in EvaluationStage. Call setup() first.")

        try:
            # Get necessary info from context, set by TrainingStage
            # EvaluationEngine.evaluate expects model_id and model_version
            model_artifact_id = context.get("trained_model_artifact_id")
            model_version = context.get("trained_model_version")
            
            # The actual trained model object is also in context, but EvaluationEngine
            # is designed to load from registry/artifact store.
            # If model_artifact_id is not set (e.g., if TrainingStage didn't save to artifact_store),
            # this stage might not work as EvaluationEngine expects to load a registered model.
            if not model_artifact_id:
                # Fallback or error: For now, let's try to use the direct model object if ID is missing.
                # This requires adapting EvaluationEngine or how this stage calls it.
                # For this iteration, we'll assume model_artifact_id IS set by TrainingStage.
                # If not, EvaluationEngine.evaluate will fail when trying to load.
                self.logger.error("trained_model_artifact_id not found in context. EvaluationEngine cannot load the model by ID.")
                raise ValueError("trained_model_artifact_id is required by EvaluationEngine.")

            # Data source for evaluation: use global_data_config's source_id
            # This assumes evaluation uses the same data source definition as training,
            # or that data_config is general enough.
            eval_data_source_id = self.global_data_config.get("source_id", "default_eval_data")
            if eval_data_source_id == "default_eval_data" and "source_id" not in self.global_data_config:
                 self.logger.warning("No specific source_id in global_data_config for evaluation, using default.")
            
            # Parameters for EvaluationEngine.evaluate:
            # These can come from self.global_evaluation_config or stage-specific self.config
            metrics_to_calc = self.global_evaluation_config.get("metrics") # List of metric names
            compare_benchmarks_flag = self.global_evaluation_config.get("compare_benchmarks", True)
            save_results_flag = self.global_evaluation_config.get("save_results", True)
            report_formats_list = self.global_evaluation_config.get("report_formats", ["html", "markdown"])
            generate_viz_flag = self.global_evaluation_config.get("generate_visualizations", True)
            
            self.logger.info(f"Report formats from config: {report_formats_list}")
            self.logger.info(f"Calling EvaluationEngine.evaluate() for model_id: {model_artifact_id}, version: {model_version}")
            
            evaluation_results = self.evaluation_engine.evaluate(
                model_id=model_artifact_id,
                model_version=model_version,
                data_source_id=eval_data_source_id, # DataManager will use this and global_data_config
                # data_version can be None to use latest data for that source_id
                metrics=metrics_to_calc,
                compare_benchmarks=compare_benchmarks_flag,
                save_results=save_results_flag, # EvaluationEngine handles saving its own artifacts
                report_formats=report_formats_list,
                generate_visualizations=generate_viz_flag,
                evaluation_name=f"eval_{model_artifact_id}_{model_version or 'latest'}"
            )
            
            self.logger.info("EvaluationEngine.evaluate() completed.")
            
            # Store results in context
            context.set("evaluation_results", evaluation_results.get("metrics")) # Main metrics dict
            context.set("evaluation_reports", evaluation_results.get("reports")) # Dict of report paths/content
            context.set("evaluation_artifacts_summary", evaluation_results) # Store the whole summary

            # Store evaluation metadata (can enhance this based on what EvaluationEngine returns)
            metadata = {
                "evaluated_at": evaluation_results.get("timestamp", datetime.now().isoformat()),
                "model_id_evaluated": model_artifact_id,
                "model_version_evaluated": model_version or results.get("model",{}).get("version"),
                "data_source_evaluated": eval_data_source_id,
                "metrics_computed": list(evaluation_results.get("metrics", {}).keys()),
            }
            context.set("evaluation_metadata", metadata)
            
            self.logger.info(f"Evaluation stage completed. Metrics: {evaluation_results.get('metrics')}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in RL EvaluationStage run: {str(e)}", exc_info=True)
            raise
            
    def teardown(self, context: PipelineContext) -> None:
        """Clean up after evaluation."""
        self.logger.info(f"Tearing down {self.name} stage")
        # Any specific cleanup for EvaluationEngine can go here.

    # Remove old placeholder methods: _make_predictions, _compute_metrics, _evaluate_baseline,
    # _compare_with_baseline, _check_thresholds, _generate_report, _format_key_metrics,
    # _save_evaluation_artifacts as these are now handled by EvaluationEngine.