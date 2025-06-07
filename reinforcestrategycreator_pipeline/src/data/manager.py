"""Data Manager for orchestrating data sources, caching, and versioning."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import pandas as pd
from ..monitoring.logger import get_logger

from ..artifact_store.base import ArtifactStore, ArtifactType
from ..config.manager import ConfigManager
from .base import DataSource, DataSourceMetadata
from .csv_source import CsvDataSource
from .api_source import ApiDataSource
from .yfinance_source import YFinanceDataSource


class DataManager:
    """Manages data sources, caching, versioning, and lineage."""
    
    # Registry of available data source types
    SOURCE_TYPES: Dict[str, Type[DataSource]] = {
        "csv": CsvDataSource,
        "api": ApiDataSource,
        "yfinance": YFinanceDataSource,
    }
    
    def __init__(
        self,
        config_manager: ConfigManager,
        artifact_store: ArtifactStore,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize DataManager.
        
        Args:
            config_manager: Configuration manager instance
            artifact_store: Artifact store for versioning
            cache_dir: Directory for caching data (uses config if not provided)
        """
        self.config_manager = config_manager
        self.artifact_store = artifact_store
        self.logger = get_logger(self.__class__.__name__) # Add logger
        
        # Get data configuration
        pipeline_config = config_manager.get_config()
        self.data_config = pipeline_config.data if hasattr(pipeline_config, 'data') else None
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            default_cache_dir = getattr(self.data_config, 'cache_dir', "./cache/data") if self.data_config else "./cache/data"
            self.cache_dir = Path(default_cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_enabled = self.data_config.cache_enabled if self.data_config else True
        self.cache_ttl_hours = getattr(self.data_config, 'cache_ttl_hours', 24) if self.data_config else 24
        
        # Active data sources
        self.data_sources: Dict[str, DataSource] = {}
        
        # Data lineage tracking
        self.lineage: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_source(
        self,
        source_id: str,
        source_type: str,
        config: Dict[str, Any]
    ) -> DataSource:
        """Register a new data source.
        
        Args:
            source_id: Unique identifier for the data source
            source_type: Type of data source (csv, api, etc.)
            config: Configuration for the data source
            
        Returns:
            Registered DataSource instance
            
        Raises:
            ValueError: If source type is not supported
        """
        if source_type not in self.SOURCE_TYPES:
            raise ValueError(
                f"Unsupported source type: {source_type}. "
                f"Available types: {list(self.SOURCE_TYPES.keys())}"
            )
        
        # Create data source instance
        source_class = self.SOURCE_TYPES[source_type]
        source = source_class(source_id, config)
        
        # Register it
        self.data_sources[source_id] = source
        self.logger.info(f"Registered data source '{source_id}'. Current sources: {list(self.data_sources.keys())}")
        
        # Track lineage
        self._track_lineage(source_id, "register", {
            "source_type": source_type,
            "config": config
        })
        
        return source
    
    def load_data(
        self,
        source_id: str,
        use_cache: Optional[bool] = None,
        version: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from a registered source.
        
        Args:
            source_id: ID of the data source to load from
            use_cache: Whether to use cache (overrides config if provided)
            version: Specific version to load from artifact store
            **kwargs: Additional parameters for the data source
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If source is not registered
        """
        self.logger.info(f"Attempting to load data for source_id: '{source_id}'. Registered sources: {list(self.data_sources.keys())}")
        # Check if loading specific version from artifact store
        if version:
            self.logger.debug(f"Loading version '{version}' for source_id '{source_id}'.")
            return self._load_version(source_id, version)
        
        # Check if source is registered
        if source_id not in self.data_sources:
            self.logger.error(f"Data source '{source_id}' not found in registered sources: {list(self.data_sources.keys())}")
            raise ValueError(f"Data source not registered: {source_id}")
        
        self.logger.debug(f"Source '{source_id}' found in registry.")
        # Determine if we should use cache
        use_cache = self.cache_enabled if use_cache is None else use_cache
        
        # Try to load from cache
        if use_cache:
            cached_data = self._load_from_cache(source_id, kwargs)
            if cached_data is not None:
                self._track_lineage(source_id, "load_from_cache", {
                    "cache_hit": True,
                    "kwargs": kwargs
                })
                return cached_data
        
        # Load from source
        source = self.data_sources[source_id]
        df = source.load_data(**kwargs)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(source_id, df, kwargs)
        
        # Track lineage
        self._track_lineage(source_id, "load_from_source", {
            "rows": len(df),
            "columns": list(df.columns),
            "kwargs": kwargs
        })
        
        return df
    
    def save_version(
        self,
        source_id: str,
        data: pd.DataFrame,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Save a versioned copy of data to the artifact store.
        
        Args:
            source_id: ID of the data source
            data: DataFrame to save
            version: Version string (auto-generated if not provided)
            description: Description of this version
            tags: Tags for categorization
            
        Returns:
            Version string of the saved data
        """
        # Generate version if not provided
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create artifact ID
        artifact_id = f"data_{source_id}"
        
        # Save data to temporary file
        temp_path = self.cache_dir / f"{artifact_id}_{version}.pkl"
        data.to_pickle(temp_path)
        
        # Get source metadata
        source_metadata = {}
        if source_id in self.data_sources:
            source = self.data_sources[source_id]
            source_metadata = source.get_metadata().to_dict()
        
        # Save to artifact store
        metadata = self.artifact_store.save_artifact(
            artifact_id=artifact_id,
            artifact_path=temp_path,
            artifact_type=ArtifactType.DATASET,
            version=version,
            metadata={
                "source_id": source_id,
                "source_metadata": source_metadata,
                "shape": [int(s) for s in data.shape],  # Convert numpy int64 to regular int
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "lineage": self.lineage.get(source_id, [])
            },
            tags=tags or [],
            description=description
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        # Track lineage
        self._track_lineage(source_id, "save_version", {
            "version": version,
            "artifact_id": artifact_id
        })
        
        return version
    
    def list_versions(self, source_id: str) -> List[str]:
        """List all versions of a data source.
        
        Args:
            source_id: ID of the data source
            
        Returns:
            List of version strings
        """
        artifact_id = f"data_{source_id}"
        return self.artifact_store.list_versions(artifact_id)
    
    def get_lineage(self, source_id: str) -> List[Dict[str, Any]]:
        """Get lineage information for a data source.
        
        Args:
            source_id: ID of the data source
            
        Returns:
            List of lineage entries
        """
        lineage = self.lineage.get(source_id, [])
        
        # Also include lineage from the source itself
        if source_id in self.data_sources:
            source = self.data_sources[source_id]
            source_lineage = source.get_metadata().lineage.get("operations", [])
            lineage.extend(source_lineage)
        
        return lineage
    
    def get_source_metadata(self, source_id: str) -> Optional[DataSourceMetadata]:
        """Get metadata for a data source.
        
        Args:
            source_id: ID of the data source
            
        Returns:
            DataSourceMetadata or None if source not found
        """
        if source_id in self.data_sources:
            return self.data_sources[source_id].get_metadata()
        return None
    
    def _load_version(self, source_id: str, version: str) -> pd.DataFrame:
        """Load a specific version from artifact store.
        
        Args:
            source_id: ID of the data source
            version: Version to load
            
        Returns:
            DataFrame containing the versioned data
        """
        artifact_id = f"data_{source_id}"
        
        # Load from artifact store
        artifact_path = self.artifact_store.load_artifact(
            artifact_id=artifact_id,
            version=version
        )
        
        # Load the data
        df = pd.read_pickle(artifact_path)
        
        # Track lineage
        self._track_lineage(source_id, "load_version", {
            "version": version,
            "artifact_id": artifact_id
        })
        
        return df
    
    def _get_cache_key(self, source_id: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for a data request.
        
        Args:
            source_id: ID of the data source
            kwargs: Parameters used for loading
            
        Returns:
            Cache key string
        """
        # Create a stable hash of the parameters
        key_data = {
            "source_id": source_id,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{source_id}_{key_hash}"
    
    def _load_from_cache(
        self,
        source_id: str,
        kwargs: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Try to load data from cache.
        
        Args:
            source_id: ID of the data source
            kwargs: Parameters used for loading
            
        Returns:
            DataFrame if cache hit, None otherwise
        """
        cache_key = self._get_cache_key(source_id, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        meta_file = self.cache_dir / f"{cache_key}.meta"
        
        # Check if cache files exist
        if not cache_file.exists() or not meta_file.exists():
            return None
        
        # Check cache expiry
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            cached_time = datetime.fromisoformat(meta["timestamp"])
            expiry_time = cached_time + timedelta(hours=self.cache_ttl_hours)
            
            if datetime.now() > expiry_time:
                # Cache expired
                cache_file.unlink()
                meta_file.unlink()
                return None
            
            # Load cached data
            return pd.read_pickle(cache_file)
            
        except Exception:
            # Error reading cache, clean up
            if cache_file.exists():
                cache_file.unlink()
            if meta_file.exists():
                meta_file.unlink()
            return None
    
    def _save_to_cache(
        self,
        source_id: str,
        data: pd.DataFrame,
        kwargs: Dict[str, Any]
    ) -> None:
        """Save data to cache.
        
        Args:
            source_id: ID of the data source
            data: DataFrame to cache
            kwargs: Parameters used for loading
        """
        cache_key = self._get_cache_key(source_id, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        meta_file = self.cache_dir / f"{cache_key}.meta"
        
        # Save data
        data.to_pickle(cache_file)
        
        # Save metadata
        meta = {
            "source_id": source_id,
            "timestamp": datetime.now().isoformat(),
            "kwargs": kwargs,
            "shape": list(data.shape)
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
    
    def _track_lineage(
        self,
        source_id: str,
        operation: str,
        details: Dict[str, Any]
    ) -> None:
        """Track lineage information.
        
        Args:
            source_id: ID of the data source
            operation: Operation performed
            details: Details about the operation
        """
        if source_id not in self.lineage:
            self.lineage[source_id] = []
        
        self.lineage[source_id].append({
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    def clear_cache(self, source_id: Optional[str] = None) -> int:
        """Clear cache for a specific source or all sources.
        
        Args:
            source_id: ID of the data source (None to clear all)
            
        Returns:
            Number of cache entries cleared
        """
        count = 0
        
        if source_id:
            # Clear cache for specific source
            pattern = f"{source_id}_*.pkl"
        else:
            # Clear all cache
            pattern = "*.pkl"
        
        for cache_file in self.cache_dir.glob(pattern):
            meta_file = cache_file.with_suffix(".meta")
            
            if cache_file.exists():
                cache_file.unlink()
                count += 1
            
            if meta_file.exists():
                meta_file.unlink()
        
        return count