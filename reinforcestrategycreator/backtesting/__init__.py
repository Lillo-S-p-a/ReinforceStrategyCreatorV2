"""
Backtesting package for reinforcement learning trading strategies.

This package provides a modular implementation of the backtesting workflow
for reinforcement learning trading strategies. It includes functionality for
data preparation, cross-validation, model training and selection, evaluation,
visualization, reporting, and model export.

For backward compatibility, the original BacktestingWorkflow class is
re-exported from this package.
"""

# Import the main workflow class for backward compatibility
from reinforcestrategycreator.backtesting.workflow import BacktestingWorkflow

# Export the main class for backward compatibility
__all__ = ['BacktestingWorkflow']