"""Evaluation module for model performance assessment.

This module provides comprehensive evaluation capabilities including:
- Multi-metric performance calculation
- Benchmark strategy comparisons
- Report generation in multiple formats
- Result persistence and retrieval
"""

from .engine import EvaluationEngine
from .metrics import MetricsCalculator
from .benchmarks import (
    BenchmarkStrategy,
    BuyAndHoldStrategy,
    SimpleMovingAverageStrategy,
    RandomStrategy,
    BenchmarkEvaluator
)

__all__ = [
    "EvaluationEngine",
    "MetricsCalculator",
    "BenchmarkStrategy",
    "BuyAndHoldStrategy", 
    "SimpleMovingAverageStrategy",
    "RandomStrategy",
    "BenchmarkEvaluator"
]