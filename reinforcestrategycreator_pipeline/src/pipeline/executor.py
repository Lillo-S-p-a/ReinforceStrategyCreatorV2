from typing import List, Any, Dict
from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger # Changed import

class PipelineExecutionError(Exception):
    """Custom exception for errors during pipeline execution.
    
    This exception is raised when a stage fails during execution and
    the pipeline cannot continue. It wraps the original exception with
    additional context about which stage failed.
    """
    pass

class PipelineExecutor:
    """Executes a sequence of pipeline stages, managing their lifecycle and context.
    
    The PipelineExecutor is responsible for orchestrating the execution of pipeline
    stages in sequence. It manages the stage lifecycle (setup, run, teardown),
    handles errors gracefully, and maintains execution metadata in the pipeline context.
    
    The executor ensures that teardown is always attempted for each stage, even in
    case of failures, to prevent resource leaks.
    
    :param stages: List of pipeline stages to execute in order
    :type stages: List[PipelineStage]
    
    :raises ValueError: If initialized with an empty list of stages
    
    Attributes:
        stages: The list of stages to execute
        logger: Logger instance for execution tracking
        context: The shared pipeline context (singleton)
    """

    def __init__(self, stages: List[PipelineStage]):
        if not stages:
            raise ValueError("PipelineExecutor must be initialized with at least one stage.")
        self.stages = stages
        self.logger = get_logger(f"executor.{self.__class__.__name__}") # Adjusted usage
        self.context = PipelineContext.get_instance() # Get the singleton context

    def run_pipeline(self) -> PipelineContext:
        """Execute all stages in the pipeline sequentially.
        
        This method runs each stage in order, managing the full lifecycle:
        1. Setup: Initialize stage resources
        2. Run: Execute stage logic
        3. Teardown: Clean up resources
        
        Execution metadata is tracked in the pipeline context, including:
        - Current stage information
        - Pipeline status
        - Error details if failures occur
        
        The method ensures teardown is attempted for all stages that were set up,
        even if errors occur during execution.

        :return: The final pipeline context after all stages have run
        :rtype: PipelineContext
        
        :raises PipelineExecutionError: If any stage fails during setup or execution
        """
        self.logger.info("Starting pipeline execution...")
        # self.context.reset() # Orchestrator should manage overall context lifecycle.
        # Executor works with the context state as provided (or fresh from get_instance).
        self.context.set_metadata("pipeline_status", "running")
        self.context.set_metadata("total_stages", len(self.stages))

        for i, stage in enumerate(self.stages):
            stage_name = stage.name
            self.context.set_metadata("current_stage_index", i)
            self.context.set_metadata("current_stage_name", stage_name)
            self.logger.info(f"Starting stage: {stage_name} ({i+1}/{len(self.stages)})")

            try:
                self.logger.debug(f"Setting up stage: {stage_name}")
                stage.setup(self.context)
                self.logger.debug(f"Stage setup complete: {stage_name}")

                self.logger.debug(f"Running stage: {stage_name}")
                self.context = stage.run(self.context) # Stage updates and returns context
                self.logger.info(f"Stage completed successfully: {stage_name}")

            except Exception as e:
                self.logger.error(f"Error during execution of stage '{stage_name}': {e}", exc_info=True)
                self.context.set_metadata("pipeline_status", "failed")
                self.context.set_metadata("error_stage", stage_name)
                self.context.set_metadata("error_message", str(e))
                # Attempt to teardown the failed stage if possible
                try:
                    self.logger.info(f"Attempting teardown for failed stage: {stage_name}")
                    stage.teardown(self.context)
                    self.logger.info(f"Teardown for failed stage '{stage_name}' completed.")
                except Exception as td_err:
                    self.logger.error(f"Error during teardown of failed stage '{stage_name}': {td_err}", exc_info=True)
                
                # Propagate the error to stop the pipeline
                raise PipelineExecutionError(f"Stage '{stage_name}' failed: {e}") from e
            finally:
                # Always attempt teardown for the current stage, even if run failed (unless setup failed)
                # If setup fails, run and teardown are not called for that stage by the main try-except.
                # If run fails, teardown is attempted in the except block.
                # If run succeeds, teardown is called here.
                if self.context.get_metadata("error_stage") != stage_name: # only if stage didn't fail
                    try:
                        self.logger.debug(f"Tearing down stage: {stage_name}")
                        stage.teardown(self.context)
                        self.logger.debug(f"Stage teardown complete: {stage_name}")
                    except Exception as e:
                        # Log error during teardown but don't necessarily stop pipeline if stage itself succeeded
                        self.logger.error(f"Error during teardown of stage '{stage_name}': {e}", exc_info=True)
                        # Potentially mark pipeline as having teardown issues
                        self.context.set_metadata(f"teardown_error_{stage_name}", str(e))


        self.logger.info("Pipeline execution finished.")
        if self.context.get_metadata("pipeline_status") != "failed":
            self.context.set_metadata("pipeline_status", "completed")
        
        return self.context

    def __repr__(self) -> str:
        stage_names = [stage.name for stage in self.stages]
        return f"<PipelineExecutor(stages={stage_names})>"