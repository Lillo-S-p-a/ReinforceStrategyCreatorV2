"""Artifact storage system for the model selection pipeline.

This module provides interfaces and implementations for storing and versioning
artifacts produced by the pipeline (datasets, models, evaluation reports, etc.).
"""

from .base import ArtifactStore, ArtifactMetadata, ArtifactType
from .local_adapter import LocalFileSystemStore

__all__ = [
    "ArtifactStore",
    "ArtifactMetadata", 
    "ArtifactType",
    "LocalFileSystemStore",
]