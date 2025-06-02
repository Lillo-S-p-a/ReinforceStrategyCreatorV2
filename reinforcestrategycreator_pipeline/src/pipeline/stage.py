from abc import ABC, abstractmethod
from typing import Any, Dict

from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext # Placeholder for now
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger # Changed import

class PipelineStage(ABC):
    """Abstract base class for a stage in the machine learning pipeline.
    
    This class defines the interface that all pipeline stages must implement.
    Each stage represents a discrete step in the ML pipeline workflow, such as
    data ingestion, feature engineering, model training, or evaluation.
    
    Stages follow a lifecycle pattern: setup -> run -> teardown, ensuring
    proper resource management and clean execution flow.
    
    :param name: The unique name identifier for this stage instance
    :type name: str
    :param config: Configuration dictionary specific to this stage
    :type config: Dict[str, Any]
    
    Attributes:
        name: The stage instance name
        config: Stage-specific configuration
        logger: Logger instance for this stage
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        # Construct a more specific logger name. get_logger prefixes with 'pipeline.'
        logger_name = f"stage.{self.__class__.__name__}.{self.name}"
        self.logger = get_logger(logger_name) # Adjusted usage

    @abstractmethod
    def setup(self, context: PipelineContext) -> None:
        """Set up the stage before execution.
        
        This method is called once before the stage's run method. It should be used
        to perform any initialization tasks such as loading resources, establishing
        connections, validating configuration, or preparing the execution environment.
        
        :param context: The shared pipeline context containing data and metadata
        :type context: PipelineContext
        
        :raises: Stage-specific exceptions if setup fails
        """
        pass

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the main logic of the pipeline stage.
        
        This method contains the core functionality of the stage. It should read
        necessary data from the context, perform its operations, and update the
        context with results for subsequent stages.

        :param context: The shared pipeline context containing input data and metadata
        :type context: PipelineContext

        :return: The updated pipeline context with this stage's outputs
        :rtype: PipelineContext
        
        :raises: Stage-specific exceptions if execution fails
        """
        pass

    @abstractmethod
    def teardown(self, context: PipelineContext) -> None:
        """Clean up resources after the stage has executed.
        
        This method is called after the stage's run method completes (whether
        successfully or with an error). It should release any resources acquired
        during setup or run, such as closing connections, freeing memory, or
        cleaning up temporary files.
        
        :param context: The shared pipeline context
        :type context: PipelineContext
        
        Note:
            This method should be implemented to handle cleanup gracefully even
            if the run method failed or was interrupted.
        """
        pass

    def __repr__(self) -> str:
        return f"<PipelineStage(name='{self.name}')>"