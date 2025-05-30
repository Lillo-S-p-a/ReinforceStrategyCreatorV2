"""Model factory for creating model instances."""

from typing import Any, Dict, Optional, Type
from enum import Enum
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger
import importlib
import inspect
from pathlib import Path

from .base import ModelBase
from ..config.models import ModelType


class ModelFactory:
    """Factory class for creating model instances.
    
    This factory manages model registration and instantiation based on
    configuration. It supports both built-in models and custom model
    implementations.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self._registry: Dict[str, Type[ModelBase]] = {}
        # Get a logger instance for ModelFactory
        self.logger = get_logger(self.__class__.__name__)
        self._register_builtin_models()
    
    def _register_builtin_models(self) -> None:
        """Register built-in model implementations."""
        self.logger.info("Starting registration of built-in models...")
        implementations_path = Path(__file__).parent / "implementations"
        self.logger.info(f"Looking for models in: {implementations_path.resolve()}")

        if not implementations_path.exists():
            self.logger.warning(f"Implementations directory not found: {implementations_path.resolve()}")
            return

        found_model_files = False
        for model_file in implementations_path.glob("*.py"):
            found_model_files = True
            self.logger.debug(f"Processing model file: {model_file.name}")
            if model_file.name.startswith("_"):
                self.logger.debug(f"Skipping private file: {model_file.name}")
                continue
                
            # This block should be at the same indentation level as the 'if' on line 42
            module_name = f".implementations.{model_file.stem}"
            self.logger.debug(f"Attempting to import module: {module_name} with package reinforcestrategycreator_pipeline.src.models")
            try:
                # Ensure the package name matches the actual top-level package visible in sys.path
                module = importlib.import_module(module_name, package="reinforcestrategycreator_pipeline.src.models")
                self.logger.debug(f"Successfully imported module: {module_name}")
                
                found_classes_in_module = False
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, ModelBase) and obj is not ModelBase:
                        found_classes_in_module = True
                        model_type_attr = getattr(obj, "model_type", name) # Use class name as fallback
                        self.logger.info(f"Registering model: type='{model_type_attr}', class='{obj.__name__}' from module {module_name}")
                        self.register_model(model_type_attr, obj)
                if not found_classes_in_module:
                    self.logger.debug(f"No ModelBase subclasses found in {module_name}")

            except ImportError as e:
                self.logger.error(f"Failed to import module {module_name}: {e}", exc_info=True)
            except Exception as e: # Catch other potential errors during registration
                self.logger.error(f"Error processing module {module_name}: {e}", exc_info=True)
    
    def register_model(self, model_type: str, model_class: Type[ModelBase]) -> None:
        """Register a model class with the factory.
        
        Args:
            model_type: Type identifier for the model
            model_class: Model class that inherits from ModelBase
            
        Raises:
            ValueError: If model_type is already registered or model_class
                       doesn't inherit from ModelBase
        """
        if not issubclass(model_class, ModelBase):
            raise ValueError(
                f"Model class {model_class.__name__} must inherit from ModelBase"
            )
        
        if model_type in self._registry:
            print(f"Warning: Overwriting existing model type '{model_type}'")
        
        self._registry[model_type] = model_class
    
    def create_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ModelBase:
        """Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in self._registry:
            available_models = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available models: {available_models}"
            )
        
        model_class = self._registry[model_type]
        
        # Prepare configuration
        if config is None:
            config = {}
        
        # Ensure model_type is in config
        config["model_type"] = model_type
        
        # Create and return model instance
        return model_class(config)
    
    def list_available_models(self) -> list[str]:
        """List all available model types.
        
        Returns:
            List of registered model type names
        """
        return sorted(self._registry.keys())
    
    def get_model_class(self, model_type: str) -> Type[ModelBase]:
        """Get the model class for a given type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in self._registry:
            raise ValueError(f"Unknown model type '{model_type}'")
        return self._registry[model_type]
    
    def is_model_registered(self, model_type: str) -> bool:
        """Check if a model type is registered.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if model is registered, False otherwise
        """
        return model_type in self._registry
    
    def unregister_model(self, model_type: str) -> None:
        """Unregister a model type.
        
        Args:
            model_type: Type of model to unregister
            
        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in self._registry:
            raise ValueError(f"Model type '{model_type}' is not registered")
        del self._registry[model_type]
    
    def create_from_config(self, config: Dict[str, Any]) -> ModelBase:
        """Create a model from a configuration dictionary.
        
        This is a convenience method that extracts the model_type from
        the configuration and creates the appropriate model.
        
        Args:
            config: Configuration dictionary containing 'model_type' key
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If 'model_type' is not in config or is invalid
        """
        if "model_type" not in config:
            raise ValueError("Configuration must contain 'model_type' key")
        
        model_type_value = config["model_type"]
        
        # Ensure model_type is a string (get value if it's an Enum)
        if isinstance(model_type_value, Enum):
            model_type_str = model_type_value.value
        elif isinstance(model_type_value, str):
            model_type_str = model_type_value
        else:
            raise ValueError(f"Invalid model_type format: {type(model_type_value)}. Expected string or Enum.")
            
        return self.create_model(model_type_str, config)
    
    def __repr__(self) -> str:
        """String representation of the factory."""
        model_count = len(self._registry)
        return f"ModelFactory(registered_models={model_count})"


# Global factory instance
_global_factory = ModelFactory()


def get_factory() -> ModelFactory:
    """Get the global model factory instance.
    
    Returns:
        Global ModelFactory instance
    """
    return _global_factory


def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> ModelBase:
    """Create a model using the global factory.
    
    Args:
        model_type: Type of model to create
        config: Model configuration dictionary
        
    Returns:
        Instantiated model object
    """
    return _global_factory.create_model(model_type, config)


def register_model(model_type: str, model_class: Type[ModelBase]) -> None:
    """Register a model with the global factory.
    
    Args:
        model_type: Type identifier for the model
        model_class: Model class that inherits from ModelBase
    """
    _global_factory.register_model(model_type, model_class)


def list_available_models() -> list[str]:
    """List all available model types in the global factory.
    
    Returns:
        List of registered model type names
    """
    return _global_factory.list_available_models()