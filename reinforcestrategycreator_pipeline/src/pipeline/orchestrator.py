from typing import List, Dict, Any, Type, Optional
import importlib
from pathlib import Path

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.executor import PipelineExecutor, PipelineExecutionError
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger
from reinforcestrategycreator_pipeline.src.monitoring.service import MonitoringService, initialize_monitoring_from_pipeline_config # Added
# Imports for ArtifactStore
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactStore
from reinforcestrategycreator_pipeline.src.artifact_store.local_adapter import LocalFileSystemStore
# from reinforcestrategycreator_pipeline.src.artifact_store.s3_adapter import S3ArtifactStore # Example if S3 needed
from reinforcestrategycreator_pipeline.src.config.models import ArtifactStoreConfig, ArtifactStoreType

# Import DataManager
from reinforcestrategycreator_pipeline.src.data.manager import DataManager
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry


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
        self.context.set("config_manager", self.config_manager) # Make ConfigManager available to stages
        
        self.artifact_store_instance: Optional[ArtifactStore] = None
        self._initialize_artifact_store()
        if self.artifact_store_instance:
            self.context.set("artifact_store", self.artifact_store_instance)

        self.model_registry_instance: Optional[ModelRegistry] = None
        self._initialize_model_registry()
        if self.model_registry_instance:
            self.context.set("model_registry", self.model_registry_instance)

        self.data_manager_instance: Optional[DataManager] = None
        self._initialize_data_manager()
        if self.data_manager_instance:
            self.context.set("data_manager", self.data_manager_instance)
            self._register_configured_data_sources()

        self.monitoring_service_instance: Optional[MonitoringService] = None # Added
        self._initialize_monitoring_service() # Added
        if self.monitoring_service_instance: # Added
            self.context.set("monitoring_service", self.monitoring_service_instance) # Added
 
        self._load_pipeline_definition()
 
    def _initialize_monitoring_service(self) -> None: # Added
        """Initializes the MonitoringService.""" # Added
        self.logger.info("Initializing MonitoringService...") # Added
        try: # Added
            pipeline_cfg = self.config_manager.get_config() # Added
            self.monitoring_service_instance = initialize_monitoring_from_pipeline_config(pipeline_cfg) # Added
            if self.monitoring_service_instance: # Added
                self.logger.info("MonitoringService initialized successfully.") # Added
            else: # Added
                self.logger.warning("MonitoringService could not be initialized from config (returned None).") # Added
        except Exception as e: # Added
            self.logger.error(f"Failed to initialize MonitoringService: {e}", exc_info=True) # Added
            self.monitoring_service_instance = None # Added

    def _initialize_model_registry(self) -> None:
        """Initializes the ModelRegistry."""
        self.logger.info("Initializing ModelRegistry...")
        if not self.artifact_store_instance:
            self.logger.error("ArtifactStore not initialized. Cannot initialize ModelRegistry.")
            self.model_registry_instance = None
            return
        
        try:
            self.model_registry_instance = ModelRegistry(artifact_store=self.artifact_store_instance)
            self.logger.info("ModelRegistry initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelRegistry: {e}", exc_info=True)
            self.model_registry_instance = None

    def _initialize_data_manager(self) -> None:
        """Initializes the DataManager."""
        self.logger.info("Initializing DataManager...")
        if not self.artifact_store_instance:
            self.logger.warning("ArtifactStore not initialized. DataManager might have limited functionality for versioning.")
        
        try:
            # DataManager needs ConfigManager and ArtifactStore
            self.data_manager_instance = DataManager(
                config_manager=self.config_manager,
                artifact_store=self.artifact_store_instance # This can be None
            )
            self.logger.info("DataManager initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize DataManager: {e}", exc_info=True)
            self.data_manager_instance = None

    def _register_configured_data_sources(self) -> None:
        """Registers data sources defined in the global configuration with DataManager."""
        if not self.data_manager_instance:
            self.logger.warning("DataManager not initialized. Cannot register configured data sources.")
            return

        self.logger.info("Registering configured data sources with DataManager...")
        try:
            pipeline_cfg = self.config_manager.get_config()
            # Assuming data sources are defined in a list under pipeline_cfg.data_sources
            # For now, we'll assume the primary data source is defined in pipeline_cfg.data
            # and we register that one.
            # If multiple sources were defined in a list like `data_sources: [...]` in YAML,
            # we would iterate through that list.

            data_config = pipeline_cfg.data # This is a DataConfig object
            if data_config and data_config.source_id and data_config.source_type:
                source_id = data_config.source_id
                source_type = data_config.source_type.value # Get the string value from Enum
                
                # Construct the config dict for the specific source type
                # This needs to align with what each DataSource expects.
                # For YFinanceDataSource, it expects 'tickers', 'period', 'interval', etc.
                # These are now part of DataConfig model.
                source_specific_config = {
                    key: value for key, value in data_config.model_dump().items()
                    if key not in ["source_id", "source_type"] and value is not None
                }
                # Ensure keys match YFinanceDataSource constructor if that's the type
                if source_type == "yfinance":
                    # YFinanceDataSource expects 'tickers', 'period', 'interval', 'start_date', 'end_date'
                    # These should be directly available in data_config now.
                    pass # The model_dump should provide these.

                self.logger.info(f"Registering source '{source_id}' of type '{source_type}' with config: {source_specific_config}")
                self.data_manager_instance.register_source(
                    source_id=source_id,
                    source_type=source_type,
                    config=source_specific_config
                )
                self.logger.info(f"Successfully registered data source '{source_id}' with DataManager.")
            else:
                self.logger.info("No primary data source (data.source_id, data.source_type) found in global config to register.")

        except Exception as e:
            self.logger.error(f"Failed to register configured data sources: {e}", exc_info=True)


    def _initialize_artifact_store(self) -> None:
        """Initializes the artifact store based on configuration."""
        self.logger.info("Initializing ArtifactStore...")
        try:
            pipeline_cfg = self.config_manager.get_config()
            # The field in PipelineConfig is artifact_store, which should be an ArtifactStoreConfig object
            # due to Pydantic validation (especially the @field_validator for it).
            artifact_store_cfg_obj = pipeline_cfg.artifact_store
            
            if not artifact_store_cfg_obj or not isinstance(artifact_store_cfg_obj, ArtifactStoreConfig):
                self.logger.warning(
                    f"No valid 'artifact_store' configuration (ArtifactStoreConfig object) found in PipelineConfig. "
                    f"Type found: {type(artifact_store_cfg_obj)}. ArtifactStore will not be available."
                )
                return

            store_type = artifact_store_cfg_obj.type # This is an Enum: ArtifactStoreType
            root_path = artifact_store_cfg_obj.root_path

            self.logger.info(f"ArtifactStore type from config: {store_type}, root_path: {root_path}")

            if store_type == ArtifactStoreType.LOCAL:
                self.artifact_store_instance = LocalFileSystemStore(root_path=Path(root_path))
                self.logger.info(f"LocalFileSystemStore initialized with root_path: {root_path}")
            # elif store_type == ArtifactStoreType.S3:
            #     # Assuming S3ArtifactStore takes s3_bucket, s3_prefix etc. from artifact_store_cfg_obj.s3_config
            #     s3_config = artifact_store_cfg_obj.s3_config # This field would need to be added to ArtifactStoreConfig
            #     if not s3_config:
            #         raise ModelPipelineError("S3 artifact store selected but no s3_config provided.")
            #     self.artifact_store_instance = S3ArtifactStore(**s3_config.model_dump())
            #     self.logger.info(f"S3ArtifactStore initialized.")
            else:
                self.logger.error(f"Unsupported artifact_store type: {store_type}")
                raise ModelPipelineError(f"Unsupported artifact_store type: {store_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ArtifactStore: {e}", exc_info=True)
            # Decide if this is critical or if pipeline can run without artifact store
            # For now, let it proceed without, stages will check for self.artifact_store
            self.artifact_store_instance = None

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
            
            full_config = self.config_manager.get_config() # Get the PipelineConfig object
            pipeline_configs = full_config.pipelines      # Access the 'pipelines' attribute
            
            if not pipeline_configs: # This will now be a dict from Pydantic, check if it's empty or if the key exists
                raise ModelPipelineError("The 'pipelines' attribute in PipelineConfig is empty or not found.")

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