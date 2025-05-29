"""Training module for model training and management."""

from .callbacks import (
    CallbackBase,
    CallbackList,
    LoggingCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback
)
from .engine import TrainingEngine
from .hpo_optimizer import HPOptimizer

# Optional visualization support
try:
    from .hpo_visualization import HPOVisualizer
    __all__ = [
        "TrainingEngine",
        "HPOptimizer",
        "HPOVisualizer",
        "CallbackBase",
        "CallbackList",
        "LoggingCallback",
        "ModelCheckpointCallback",
        "EarlyStoppingCallback"
    ]
except ImportError:
    # Visualization not available
    __all__ = [
        "TrainingEngine",
        "HPOptimizer",
        "CallbackBase",
        "CallbackList",
        "LoggingCallback",
        "ModelCheckpointCallback",
        "EarlyStoppingCallback"
    ]