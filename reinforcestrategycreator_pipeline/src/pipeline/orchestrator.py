from typing import List, Dict, Any, Type
import importlib

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.executor import PipelineExecutor, PipelineExecutionError
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager # Assumes Task 1.2 completed
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger # Changed import

class ModelPipelineError(Exception):
    """Custom exception for ModelPipeline errors.
    
    This exception is raised when there are errors in pipeline configuration,
    loading, or execution that are specific to the ModelPipeline class.
    """
    pass

class ModelPipeline:
    """Main class for defining, configuring, and running a machine learning pipeline.
    
    This class orchestrates the execution of a machine learning pipeline by loading
    pipeline definitions from configuration, instantiating stages, and managing the
    execution flow through the PipelineExecutor.
    
    :param pipeline_name: The name of the pipeline to load from configuration
    :type pipeline_name: str
    :param config_manager: The configuration manager instance to load pipeline definitions
    :type config_manager: ConfigManager
    
    :raises ModelPipelineError: If the pipeline definition cannot be loaded or is invalid
    
    Example:
        >>> config_manager = ConfigManager()
        >>> pipeline = ModelPipeline("training_pipeline", config_manager)
        >>> context = pipeline.run()
    """

    def __init__(self, pipeline_name: str, config_manager: ConfigManager):
        self.pipeline_name = pipeline_name
        self.config_manager = config_manager
        self.logger = get_logger(f"orchestrator.ModelPipeline.{pipeline_name}") # Adjusted usage
        self.stages: List[PipelineStage] = []
        self.executor: PipelineExecutor | None = None
        self.context: PipelineContext = PipelineContext.get_instance()

        self._load_pipeline_definition()

    def _load_pipeline_definition(self) -> None:
        """Load the pipeline definition from the configuration, including its stages.
        
        This method reads the pipeline configuration from the config manager,
        validates the structure, dynamically imports stage classes, and instantiates
        them with their respective configurations.
        
        :raises ModelPipelineError: If the pipeline configuration is missing, invalid,
            or if stage modules/classes cannot be imported
        """
        self.logger.info(f"Loading pipeline definition for '{self.pipeline_name}'...")
        try:
            # Assuming pipeline definitions are under a 'pipelines' key in the main config
            # And each pipeline has a 'name' and a 'stages' list.
            # Example config structure in models.yaml or a dedicated pipeline.yaml:
            # pipelines:
            #   my_training_pipeline:
            #     stages:
            #       - name: "DataIngestion"
            #         module: "reinforcestrategycreator_pipeline.src.data.ingestion_stage" # Example
            #         class: "DataIngestionStage"
            #         config: { ... stage specific config ... }
            #       - name: "FeatureEngineering"
            #         module: "reinforcestrategycreator_pipeline.src.features.engineering_stage"
            #         class: "FeatureEngineeringStage"
            #         config: { ... }
            
            pipeline_configs = self.config_manager.get_config('pipelines')
            if not pipeline_configs:
                raise ModelPipelineError("No 'pipelines' section found in the configuration.")

            pipeline_def = pipeline_configs.get(self.pipeline_name)
            if not pipeline_def:
                raise ModelPipelineError(f"Pipeline definition for '{self.pipeline_name}' not found in configuration.")

            if not isinstance(pipeline_def.get('stages'), list):
                raise ModelPipelineError(f"Pipeline '{self.pipeline_name}' definition is missing a 'stages' list or it's not a list.")

            for stage_config in pipeline_def['stages']:
                stage_name = stage_config.get('name')
                stage_module_path = stage_config.get('module')
                stage_class_name = stage_config.get('class')
                stage_specific_config = stage_config.get('config', {})

                if not all([stage_name, stage_module_path, stage_class_name]):
                    raise ModelPipelineError(
                        f"Stage definition in pipeline '{self.pipeline_name}' is missing 'name', 'module', or 'class'."
                    )
                
                self.logger.debug(f"Loading stage '{stage_name}': module='{stage_module_path}', class='{stage_class_name}'")
                
                try:
                    module = importlib.import_module(stage_module_path)
                    StageClass = getattr(module, stage_class_name)
                except ImportError as e:
                    self.logger.error(f"Failed to import module {stage_module_path} for stage {stage_name}: {e}")
                    raise ModelPipelineError(f"Could not import module for stage {stage_name}: {e}") from e
                except AttributeError as e:
                    self.logger.error(f"Failed to find class {stage_class_name} in module {stage_module_path} for stage {stage_name}: {e}")
                    raise ModelPipelineError(f"Could not find class for stage {stage_name}: {e}") from e

                if not issubclass(StageClass, PipelineStage):
                    raise ModelPipelineError(
                        f"Class {stage_class_name} for stage {stage_name} does not inherit from PipelineStage."
                    )
                
                # Merge global stage configs with stage-specific configs if any
                # global_stage_config = self.config_manager.get_config(f"stages.{stage_name}", {}) # Example
                # final_stage_config = {**global_stage_config, **stage_specific_config}
                
                stage_instance = StageClass(name=stage_name, config=stage_specific_config)
                self.stages.append(stage_instance)
                self.logger.info(f"Successfully loaded and instantiated stage: {stage_name}")

            if not self.stages:
                self.logger.warning(f"Pipeline '{self.pipeline_name}' loaded with no stages.")
            
            self.executor = PipelineExecutor(self.stages)
            self.logger.info(f"Pipeline definition for '{self.pipeline_name}' loaded successfully with {len(self.stages)} stages.")

        except KeyError as e:
            self.logger.error(f"Configuration key error while loading pipeline '{self.pipeline_name}': {e}", exc_info=True)
            raise ModelPipelineError(f"Missing configuration for pipeline '{self.pipeline_name}': {e}") from e
        except Exception as e:
            self.logger.error(f"Failed to load pipeline definition for '{self.pipeline_name}': {e}", exc_info=True)
            # Re-raise as ModelPipelineError or allow specific exceptions
            if not isinstance(e, ModelPipelineError):
                 raise ModelPipelineError(f"An unexpected error occurred while loading pipeline '{self.pipeline_name}': {e}") from e
            raise

    def run(self) -> PipelineContext:
        """Run the entire pipeline.
        
        Executes all stages in the pipeline sequentially through the PipelineExecutor.
        The pipeline context is updated with metadata about the execution status.
        
        :return: The pipeline context containing results and metadata from the execution
        :rtype: PipelineContext
        
        :raises ModelPipelineError: If the pipeline executor is not initialized or if
            a critical error occurs during execution
        """
        if not self.executor:
            self.logger.error("Pipeline executor not initialized. Cannot run pipeline.")
            raise ModelPipelineError("Pipeline executor not initialized. Load pipeline definition first.")

        self.logger.info(f"Starting execution of pipeline: {self.pipeline_name}")
        print(f"DEBUG_ORCH: In ModelPipeline.run for '{self.pipeline_name}', context id={id(self.context)}") # New print
        self.context.set_metadata("pipeline_name", self.pipeline_name)
        print(f"DEBUG_ORCH: Set pipeline_name='{self.pipeline_name}'. Metadata now: {self.context.get_all_metadata()}") # Changed logger to print
        
        try:
            final_context = self.executor.run_pipeline()
            status = final_context.get_metadata("pipeline_status")
            if status == "completed":
                self.logger.info(f"Pipeline '{self.pipeline_name}' executed successfully.")
            else: # failed or other
                error_stage = final_context.get_metadata("error_stage", "Unknown")
                error_msg = final_context.get_metadata("error_message", "Unknown error")
                self.logger.error(f"Pipeline '{self.pipeline_name}' execution failed at stage '{error_stage}'. Error: {error_msg}")
            return final_context
        except PipelineExecutionError as e:
            self.logger.error(f"Pipeline '{self.pipeline_name}' execution failed: {e}", exc_info=True)
            # Context should already have failure details set by the executor
            return self.context # Return context which contains error info
        except Exception as e:
            self.logger.critical(f"An unexpected critical error occurred during pipeline '{self.pipeline_name}' execution: {e}", exc_info=True)
            self.context.set_metadata("pipeline_status", "critical_failure")
            self.context.set_metadata("critical_error_message", str(e))
            raise ModelPipelineError(f"Critical failure in pipeline '{self.pipeline_name}': {e}") from e
            
    def get_stage(self, name: str) -> PipelineStage | None:
        """Retrieve a stage by its name.
        
        :param name: The name of the stage to retrieve
        :type name: str
        
        :return: The stage instance if found, None otherwise
        :rtype: PipelineStage | None
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def __repr__(self) -> str:
        return f"<ModelPipeline(name='{self.pipeline_name}', stages_count={len(self.stages)})>"