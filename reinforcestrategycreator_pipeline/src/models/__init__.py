"""Models module for the ML pipeline.

This module provides model management functionality including:
- Base model interface
- Model factory for creating models
- Model registry for versioning and tracking
- Built-in model implementations
"""

from .base import ModelBase
from .factory import (
    ModelFactory,
    get_factory,
    create_model,
    register_model,
    list_available_models
)
from .registry import ModelRegistry

# Import model implementations to register them
from .implementations import DQN, PPO, A2C

__all__ = [
    # Base
    "ModelBase",
    
    # Factory
    "ModelFactory",
    "get_factory",
    "create_model",
    "register_model",
    "list_available_models",
    
    # Registry
    "ModelRegistry",
    
    # Models
    "DQN",
    "PPO",
    "A2C"
]