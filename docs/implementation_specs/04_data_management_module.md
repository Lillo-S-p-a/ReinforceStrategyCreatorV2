# Data Management Module: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Data Management Module of the Trading Model Optimization Pipeline. This module is the foundational component responsible for data acquisition, normalization, feature engineering, dataset management, and providing clean, prepared data for model training and evaluation.

## 2. Component Responsibilities

The Data Management Module is responsible for:

- Loading data from various sources (CSV, APIs, databases)
- Cleaning and normalizing raw market data
- Implementing feature engineering pipelines
- Creating training, validation, and test datasets
- Managing data versioning and reproducibility
- Implementing efficient data loaders for training
- Providing real-time/streaming data capabilities for live evaluation

## 3. Architecture

### 3.1 Overall Architecture

The Data Management Module follows a layered architecture with clear separation of concerns:

```
┌───────────────────────────────────┐
│          Data Interface           │  High-level API for consumers
├───────────────────────────────────┤
│        Pipeline Orchestrator      │  Manages the execution of data pipelines
├───────────────────────────────────┤
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │   Loaders   │ │Transformers │  │  Core components for data operations
│  └─────────────┘ └─────────────┘  │
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │  Features   │ │  Datasets   │  │
│  └─────────────┘ └─────────────┘  │
│                                   │
├───────────────────────────────────┤
│           Data Storage            │  Storage abstraction for data persistence
└───────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── data/
    ├── __init__.py
    ├── interface.py          # High-level interface for other components
    ├── pipeline.py           # Pipeline orchestration and management
    ├── loaders/
    │   ├── __init__.py
    │   ├── base.py           # Abstract base loader class
    │   ├── csv_loader.py     # CSV file loader
    │   ├── api_loader.py     # External API data loader
    │   ├── db_loader.py      # Database loader
    │   ├── streaming.py      # Real-time data streaming
    │   └── factory.py        # Loader factory method pattern
    ├── transformers/
    │   ├── __init__.py
    │   ├── base.py           # Abstract transformer class
    │   ├── cleaner.py        # Data cleaning transformations
    │   ├── normalizer.py     # Data normalization operations
    │   ├── resample.py       # Time series resampling (e.g., 1min to 5min)
    │   └── pipeline.py       # Transformation pipeline builder
    ├── features/
    │   ├── __init__.py
    │   ├── base.py           # Base feature generator
    │   ├── technical.py      # Technical indicators
    │   ├── statistical.py    # Statistical features
    │   ├── calendar.py       # Calendar-based features
    │   └── registry.py       # Feature registry and management
    ├── datasets/
    │   ├── __init__.py
    │   ├── base.py           # Base dataset class
    │   ├── timeseries.py     # Time series dataset implementation
    │   ├── sequential.py     # Sequential data handling
    │   ├── splitter.py       # Train/validation/test splitting logic
    │   └── factory.py        # Dataset factory
    ├── storage/
    │   ├── __init__.py
    │   ├── local.py          # Local file storage implementation
    │   ├── s3.py             # S3/cloud storage implementation
    │   └── registry.py       # Storage registry
    └── utils/
        ├── __init__.py
        ├── validation.py     # Data validation utilities
        ├── visualization.py  # Data visualization helpers
        └── metrics.py        # Data quality metrics
```

## 4. Core Components Design

### 4.1 Data Interface

The high-level interface that other system components will use to interact with the Data Management Module:

```python
# interface.py
from typing import Dict, List, Union, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import torch

from trading_optimization.data.pipeline import DataPipeline
from trading_optimization.data.datasets import DatasetFactory
from trading_optimization.data.features import FeatureRegistry

class DataManager:
    """
    High-level interface for the Data Management Module.
    Acts as the façade for all data operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Data Manager with configuration settings.
        
        Args:
            config: Configuration dictionary with data management settings
        """
        self.config = config
        self.pipeline_cache = {}
        self.feature_registry = FeatureRegistry()
        self.dataset_factory = DatasetFactory(self.feature_registry)
        
    def create_pipeline(self, name: str, pipeline_config: Dict[str, Any]) -> str:
        """
        Create and register a data pipeline.
        
        Args:
            name: Name of the pipeline to create
            pipeline_config: Configuration for the pipeline
            
        Returns:
            ID of the created pipeline
        """
        pipeline = DataPipeline(name, pipeline_config, self.feature_registry)
        pipeline_id = pipeline.id
        self.pipeline_cache[pipeline_id] = pipeline
        return pipeline_id
    
    def get_pipeline(self, pipeline_id: str) -> Optional[DataPipeline]:
        """
        Retrieve a pipeline by ID.
        
        Args:
            pipeline_id: ID of the pipeline to retrieve
            
        Returns:
            The DataPipeline object or None if not found
        """
        return self.pipeline_cache.get(pipeline_id)
    
    def execute_pipeline(
        self, 
        pipeline_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Execute a data pipeline and return the processed data.
        
        Args:
            pipeline_id: ID of the pipeline to execute
            start_date: Optional start date for data range
            end_date: Optional end date for data range
            
        Returns:
            Processed DataFrame from the pipeline
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        
        return pipeline.execute(start_date, end_date)
    
    def create_dataset(
        self,
        data: Union[pd.DataFrame, str],
        dataset_type: str = "timeseries",
        **kwargs
    ) -> Any:
        """
        Create a dataset from processed data or pipeline ID.
        
        Args:
            data: Either a DataFrame or a pipeline_id string
            dataset_type: Type of dataset to create
            **kwargs: Additional dataset parameters
            
        Returns:
            The created dataset object
        """
        if isinstance(data, str):
            # Assume this is a pipeline_id
            data = self.execute_pipeline(data, **kwargs.get('pipeline_args', {}))
        
        return self.dataset_factory.create_dataset(
            data, 
            dataset_type=dataset_type,
            **kwargs
        )
    
    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_method: str = "time",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            split_method: Method for splitting ('time', 'random', 'walk-forward')
            **kwargs: Additional splitting parameters
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames
        """
        from trading_optimization.data.datasets.splitter import DataSplitter
        
        splitter = DataSplitter(
            split_method=split_method,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            **kwargs
        )
        
        return splitter.split(data)
    
    def get_data_loaders(
        self,
        data_splits: Dict[str, pd.DataFrame],
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create PyTorch DataLoader objects for model training.
        
        Args:
            data_splits: Dictionary with 'train', 'val', 'test' DataFrames
            batch_size: Batch size for DataLoaders
            shuffle: Whether to shuffle the data
            **kwargs: Additional DataLoader parameters
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader objects
        """
        loaders = {}
        for split_name, df in data_splits.items():
            dataset = self.dataset_factory.create_dataset(
                df, 
                dataset_type=kwargs.get('dataset_type', 'timeseries'),
                **kwargs
            )
            
            loaders[split_name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(shuffle and split_name == 'train'),
                num_workers=kwargs.get('num_workers', 2),
                pin_memory=kwargs.get('pin_memory', True)
            )
        
        return loaders
    
    def register_custom_feature(self, feature_name: str, feature_fn, feature_type: str = "custom"):
        """
        Register a custom feature generation function.
        
        Args:
            feature_name: Name of the feature
            feature_fn: Function that implements the feature calculation
            feature_type: Type category for the feature
        """
        self.feature_registry.register_feature(feature_name, feature_fn, feature_type)
    
    def save_pipeline_snapshot(self, pipeline_id: str, snapshot_name: str) -> str:
        """
        Save a snapshot of a pipeline's configuration and state.
        
        Args:
            pipeline_id: ID of the pipeline to snapshot
            snapshot_name: Name for the snapshot
            
        Returns:
            Path to saved snapshot
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        
        return pipeline.save_snapshot(snapshot_name)
    
    def load_pipeline_snapshot(self, snapshot_path: str) -> str:
        """
        Load a pipeline from a snapshot.
        
        Args:
            snapshot_path: Path to the snapshot file
            
        Returns:
            ID of the restored pipeline
        """
        pipeline = DataPipeline.from_snapshot(snapshot_path, self.feature_registry)
        pipeline_id = pipeline.id
        self.pipeline_cache[pipeline_id] = pipeline
        return pipeline_id
```

### 4.2 Data Pipeline Orchestrator

The component that manages the execution flow of data operations:

```python
# pipeline.py
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import json
import os

from trading_optimization.data.loaders import LoaderFactory
from trading_optimization.data.transformers.pipeline import TransformerPipeline
from trading_optimization.data.features import FeatureRegistry
from trading_optimization.config import ConfigManager

class DataPipeline:
    """
    Orchestrates the execution of a data processing pipeline.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], feature_registry: FeatureRegistry):
        """
        Initialize a data pipeline.
        
        Args:
            name: Name of the pipeline
            config: Pipeline configuration
            feature_registry: Registry of available features
        """
        self.name = name
        self.config = config
        self.feature_registry = feature_registry
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        
        # Parse config sections
        self.loader_config = config.get('loader', {})
        self.transformer_config = config.get('transformers', [])
        self.feature_config = config.get('features', [])
        self.post_processing_config = config.get('post_processing', [])
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the pipeline components from configuration."""
        # Create data loader
        self.loader = LoaderFactory.create_loader(
            loader_type=self.loader_config.get('type', 'csv'),
            **self.loader_config.get('params', {})
        )
        
        # Create transformer pipeline
        self.transformer = TransformerPipeline(self.transformer_config)
        
        # Extract feature generators from registry
        self.features = []
        for feature_spec in self.feature_config:
            if isinstance(feature_spec, str):
                # Simple feature name
                feature_name = feature_spec
                feature_params = {}
            else:
                # Feature with parameters
                feature_name = feature_spec['name']
                feature_params = feature_spec.get('params', {})
            
            feature_fn = self.feature_registry.get_feature(feature_name)
            if feature_fn:
                # Create a configured feature function with params
                def configured_feature(data, fn=feature_fn, params=feature_params):
                    return fn(data, **params)
                
                self.features.append((feature_name, configured_feature))
            else:
                raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        # Parse post-processing steps
        self.post_processors = []
        for processor in self.post_processing_config:
            processor_type = processor.get('type')
            processor_params = processor.get('params', {})
            
            if processor_type == 'drop_columns':
                self.post_processors.append(
                    lambda df, cols=processor_params.get('columns', []): 
                    df.drop(columns=cols, errors='ignore')
                )
            elif processor_type == 'fillna':
                self.post_processors.append(
                    lambda df, strategy=processor_params.get('strategy', 'ffill'):
                    df.fillna(method=strategy) if strategy in ['ffill', 'bfill'] 
                    else df.fillna(processor_params.get('value', 0))
                )
            elif processor_type == 'clip_outliers':
                def clip_outliers(df, 
                                cols=processor_params.get('columns', df.columns.tolist()),
                                lower=processor_params.get('lower_quantile', 0.001),
                                upper=processor_params.get('upper_quantile', 0.999)):
                    df_out = df.copy()
                    for col in cols:
                        if col in df.columns:
                            lower_val = df[col].quantile(lower)
                            upper_val = df[col].quantile(upper)
                            df_out[col] = df_out[col].clip(lower=lower_val, upper=upper_val)
                    return df_out
                
                self.post_processors.append(clip_outliers)
    
    def execute(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Execute the pipeline to produce processed data.
        
        Args:
            start_date: Optional start date for data range
            end_date: Optional end date for data range
            
        Returns:
            Processed DataFrame
        """
        # 1. Load raw data
        df = self.loader.load_data(start_date, end_date)
        
        # 2. Apply transformations
        df = self.transformer.transform(df)
        
        # 3. Generate features
        for feature_name, feature_fn in self.features:
            try:
                feature_result = feature_fn(df)
                
                # Handle different return types from feature functions
                if isinstance(feature_result, pd.DataFrame):
                    # If returning a DataFrame, merge with original
                    # Avoid duplicate columns
                    new_cols = [col for col in feature_result.columns 
                              if col not in df.columns or col == feature_result.index.name]
                    df = pd.concat([df, feature_result[new_cols]], axis=1)
                elif isinstance(feature_result, pd.Series):
                    # If returning a Series, add as new column
                    df[feature_name] = feature_result
                elif isinstance(feature_result, dict):
                    # If returning a dict, add each item as a column
                    for key, val in feature_result.items():
                        df[f"{feature_name}_{key}"] = val
            except Exception as e:
                print(f"Error generating feature '{feature_name}': {str(e)}")
                # Optionally log this error or re-raise
        
        # 4. Apply post-processing steps
        for post_process_fn in self.post_processors:
            df = post_process_fn(df)
        
        return df
    
    def save_snapshot(self, snapshot_name: str) -> str:
        """
        Save the pipeline configuration and state as a snapshot.
        
        Args:
            snapshot_name: Name for the snapshot
            
        Returns:
            Path to saved snapshot
        """
        config = ConfigManager.instance()
        snapshot_dir = config.get(
            'data.snapshots_path', 
            os.path.join('artifacts', 'pipeline_snapshots')
        )
        
        # Ensure directory exists
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Build snapshot file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = snapshot_name.replace(' ', '_').replace('/', '_')
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(snapshot_dir, filename)
        
        # Create snapshot data
        snapshot = {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'snapshot_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        # Write snapshot to file
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        return filepath
    
    @classmethod
    def from_snapshot(cls, snapshot_path: str, feature_registry: FeatureRegistry) -> 'DataPipeline':
        """
        Load a pipeline from a snapshot.
        
        Args:
            snapshot_path: Path to the snapshot file
            feature_registry: Registry of available features
            
        Returns:
            Restored DataPipeline object
        """
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        
        pipeline = cls(
            name=snapshot['name'],
            config=snapshot['config'],
            feature_registry=feature_registry
        )
        
        # Override the generated ID with the saved one for consistency
        pipeline.id = snapshot['id']
        pipeline.created_at = snapshot['created_at']
        
        return pipeline
```

### 4.3 Data Loaders

The loaders are responsible for fetching data from various sources:

```python
# loaders/base.py
from typing import Optional
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    """
    
    @abstractmethod
    def load_data(self, 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data from the source.
        
        Args:
            start_date: Optional start date for data range
            end_date: Optional end date for data range
            
        Returns:
            DataFrame with loaded data
        """
        pass
```

Example implementation for CSV loader:

```python
# loaders/csv_loader.py
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import glob
import os
import re

from trading_optimization.data.loaders.base import BaseLoader

class CSVLoader(BaseLoader):
    """
    Loader implementation for CSV files.
    """
    
    def __init__(
        self,
        file_path: str,
        date_column: str = 'date',
        date_format: str = '%Y-%m-%d %H:%M:%S',
        recursive: bool = False,
        pattern: str = '*.csv',
        **kwargs
    ):
        """
        Initialize CSV loader.
        
        Args:
            file_path: Path to CSV file or directory containing CSV files
            date_column: Name of the column containing date/time information
            date_format: Format string for parsing date/time column
            recursive: Whether to recursively search for files in subdirectories
            pattern: Glob pattern for finding CSV files
            **kwargs: Additional parameters passed to pd.read_csv()
        """
        self.file_path = file_path
        self.date_column = date_column
        self.date_format = date_format
        self.recursive = recursive
        self.pattern = pattern
        self.kwargs = kwargs
    
    def _list_files(self) -> List[str]:
        """
        Get a list of files to load based on path and pattern.
        
        Returns:
            List of file paths
        """
        if os.path.isdir(self.file_path):
            # Handle directory path
            if self.recursive:
                return glob.glob(os.path.join(self.file_path, '**', self.pattern), recursive=True)
            else:
                return glob.glob(os.path.join(self.file_path, self.pattern))
        else:
            # Handle single file path
            return [self.file_path] if os.path.exists(self.file_path) else []
    
    def _parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Attempt to extract a date from a filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Extracted datetime or None
        """
        patterns = [
            # YYYY-MM-DD
            r'(\d{4}[-_]\d{2}[-_]\d{2})',
            # YYYYMMDD
            r'(\d{8})',
        ]
        
        formats = [
            '%Y-%m-%d',
            '%Y_%m_%d',
            '%Y%m%d'
        ]
        
        basename = os.path.basename(filename)
        
        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                date_str = match.group(1)
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        
        return None
    
    def _should_include_file(
        self, 
        file_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """
        Determine if a file should be included based on date filtering.
        
        Args:
            file_path: Path to the file
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            True if the file should be included, False otherwise
        """
        if start_date is None and end_date is None:
            return True
            
        file_date = self._parse_date_from_filename(file_path)
        if file_date is None:
            # Can't determine date from filename, include it
            return True
            
        if start_date and file_date < start_date:
            return False
            
        if end_date and file_date > end_date:
            return False
            
        return True
    
    def load_data(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from CSV file(s).
        
        Args:
            start_date: Optional start date for data range
            end_date: Optional end date for data range
            
        Returns:
            DataFrame with loaded data
        """
        files = self._list_files()
        
        if not files:
            raise FileNotFoundError(f"No files found at path: {self.file_path}")
            
        # Pre-filter files by date in filename if possible
        files = [f for f in files if self._should_include_file(f, start_date, end_date)]
        
        # Prepare dataframes list
        dataframes = []
        
        # Load each file
        for file_path in files:
            try:
                df = pd.read_csv(file_path, **self.kwargs)
                
                # Parse date column if it exists
                if self.date_column in df.columns:
                    df[self.date_column] = pd.to_datetime(
                        df[self.date_column], 
                        format=self.date_format
                    )
                    
                    # Filter by date range if specified
                    if start_date:
                        df = df[df[self.date_column] >= start_date]
                    if end_date:
                        df = df[df[self.date_column] <= end_date]
                
                if not df.empty:
                    dataframes.append(df)
                    
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
                # Optionally log this error
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combine all dataframes
        result = pd.concat(dataframes, ignore_index=True)
        
        # Set date as index if it exists
        if self.date_column in result.columns:
            result.set_index(self.date_column, inplace=True)
            result.sort_index(inplace=True)
        
        return result
```

Factory for creating loaders:

```python
# loaders/factory.py
from typing import Dict, Any, Optional

from trading_optimization.data.loaders.base import BaseLoader
from trading_optimization.data.loaders.csv_loader import CSVLoader
from trading_optimization.data.loaders.api_loader import APILoader
from trading_optimization.data.loaders.db_loader import DBLoader
from trading_optimization.data.loaders.streaming import StreamingLoader

class LoaderFactory:
    """
    Factory for creating data loader instances.
    """
    
    _loaders = {
        'csv': CSVLoader,
        'api': APILoader,
        'database': DBLoader,
        'streaming': StreamingLoader
    }
    
    @classmethod
    def register_loader(cls, loader_type: str, loader_class):
        """
        Register a new loader type.
        
        Args:
            loader_type: Type identifier for the loader
            loader_class: Class that implements the loader
        """
        cls._loaders[loader_type] = loader_class
    
    @classmethod
    def create_loader(cls, loader_type: str, **kwargs) -> BaseLoader:
        """
        Create a loader instance.
        
        Args:
            loader_type: Type of loader to create
            **kwargs: Parameters for the loader constructor
            
        Returns:
            Instance of a BaseLoader subclass
        """
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        return cls._loaders[loader_type](**kwargs)
```

### 4.4 Transformers

Transformers modify, clean, and prepare the raw data:

```python
# transformers/base.py
import pandas as pd
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.
    """
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        pass
```

Example transformer implementation:

```python
# transformers/normalizer.py
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from trading_optimization.data.transformers.base import BaseTransformer

class Normalizer(BaseTransformer):
    """
    Performs various normalization operations on data.
    """
    
    def __init__(
        self,
        method: str = 'standard',
        columns: Optional[List[str]] = None,
        target_range: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize the normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust', 'log')
            columns: List of columns to normalize (None for all numeric)
            target_range: Target range for scaling (e.g. {'min': 0, 'max': 1})
            **kwargs: Additional parameters
        """
        self.method = method
        self.columns = columns
        self.target_range = target_range or {'min': 0, 'max': 1}
        self.kwargs = kwargs
        self.scalers = {}
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by normalizing selected columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Determine columns to normalize
        cols_to_normalize = self.columns
        if cols_to_normalize is None:
            # Use all numeric columns if none specified
            cols_to_normalize = df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter out columns that don't exist
            cols_to_normalize = [col for col in cols_to_normalize if col in df.columns]
        
        if not cols_to_normalize:
            return df
        
        # Apply normalization based on method
        if self.method == 'standard':
            for col in cols_to_normalize:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
                    self.scalers[col] = {'mean': mean, 'std': std}
        
        elif self.method == 'minmax':
            for col in cols_to_normalize:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    target_min = self.target_range['min']
                    target_max = self.target_range['max']
                    df[col] = ((df[col] - min_val) / (max_val - min_val) * 
                             (target_max - target_min) + target_min)
                    self.scalers[col] = {
                        'min': min_val, 'max': max_val,
                        'target_min': target_min, 'target_max': target_max
                    }
        
        elif self.method == 'robust':
            for col in cols_to_normalize:
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    df[col] = (df[col] - median) / iqr
                    self.scalers[col] = {'median': median, 'iqr': iqr}
        
        elif self.method == 'log':
            for col in cols_to_normalize:
                # Shift data to ensure all values are positive
                min_val = df[col].min()
                shift = 0
                if min_val <= 0:
                    shift = abs(min_val) + 1  # Add 1 to avoid log(0)
                
                df[col] = np.log(df[col] + shift)
                self.scalers[col] = {'shift': shift}
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return df
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized DataFrame
            
        Returns:
            DataFrame in original scale
        """
        df = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                if self.method == 'standard':
                    df[col] = df[col] * scaler['std'] + scaler['mean']
                
                elif self.method == 'minmax':
                    min_val = scaler['min']
                    max_val = scaler['max']
                    target_min = scaler['target_min']
                    target_max = scaler['target_max']
                    
                    df[col] = ((df[col] - target_min) / (target_max - target_min) * 
                             (max_val - min_val) + min_val)
                
                elif self.method == 'robust':
                    df[col] = df[col] * scaler['iqr'] + scaler['median']
                
                elif self.method == 'log':
                    df[col] = np.exp(df[col]) - scaler['shift']
        
        return df
```

Transformation pipeline to chain multiple transformers:

```python
# transformers/pipeline.py
from typing import List, Dict, Any
import pandas as pd

from trading_optimization.data.transformers.base import BaseTransformer
from trading_optimization.data.transformers.cleaner import DataCleaner
from trading_optimization.data.transformers.normalizer import Normalizer
from trading_optimization.data.transformers.resample import Resampler

class TransformerPipeline:
    """
    Pipeline to chain multiple transformers.
    """
    
    def __init__(self, transformer_configs: List[Dict[str, Any]]):
        """
        Initialize the transformer pipeline.
        
        Args:
            transformer_configs: List of transformer configurations
        """
        self.transformers = []
        
        # Map of transformer types to classes
        transformer_map = {
            'cleaner': DataCleaner,
            'normalize': Normalizer,
            'resample': Resampler,
            # Add other transformer types here
        }
        
        # Create transformer instances
        for config in transformer_configs:
            transformer_type = config.get('type')
            params = config.get('params', {})
            
            if transformer_type in transformer_map:
                transformer_cls = transformer_map[transformer_type]
                transformer = transformer_cls(**params)
                self.transformers.append(transformer)
            else:
                raise ValueError(f"Unknown transformer type: {transformer_type}")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers in sequence.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df = data.copy()
        
        for transformer in self.transformers:
            df = transformer.transform(df)
        
        return df
```

### 4.5 Feature Engineering

Feature engineering components create derived features from raw data:

```python
# features/base.py
from typing import Union, Dict, Callable, Any
import pandas as pd

# Type aliases for feature functions
FeatureFunction = Callable[[pd.DataFrame, ...], Union[pd.DataFrame, pd.Series, Dict[str, Any]]]

class BaseFeatureGenerator:
    """
    Base class for feature generators.
    """
    
    def __init__(self, **kwargs):
        """Initialize with optional parameters."""
        self.params = kwargs
    
    def generate(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Generate features from input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Features as DataFrame, Series, or Dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
```

Feature registry to manage and access feature generators:

```python
# features/registry.py
from typing import Dict, Any, Callable, Optional
from collections import defaultdict

class FeatureRegistry:
    """
    Registry for feature generation functions.
    """
    
    def __init__(self):
        """Initialize an empty feature registry."""
        self.features = {}
        self.features_by_type = defaultdict(list)
    
    def register_feature(self, name: str, feature_fn: Callable, feature_type: str = "general"):
        """
        Register a feature generation function.
        
        Args:
            name: Name of the feature
            feature_fn: Function that implements the feature
            feature_type: Type category for the feature
        """
        self.features[name] = feature_fn
        self.features_by_type[feature_type].append(name)
    
    def get_feature(self, name: str) -> Optional[Callable]:
        """
        Get a feature function by name.
        
        Args:
            name: Name of the feature
            
        Returns:
            Feature function or None if not found
        """
        return self.features.get(name)
    
    def get_features_by_type(self, feature_type: str) -> list:
        """
        Get all features of a specific type.
        
        Args:
            feature_type: Type of features to retrieve
            
        Returns:
            List of feature names of that type
        """
        return self.features_by_type.get(feature_type, [])
    
    def list_features(self) -> Dict[str, list]:
        """
        List all registered features grouped by type.
        
        Returns:
            Dictionary mapping feature types to lists of feature names
        """
        return dict(self.features_by_type)
```

Example technical indicators feature module:

```python
# features/technical.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Any

def register_technical_indicators(registry):
    """
    Register all technical indicators in the feature registry.
    
    Args:
        registry: FeatureRegistry instance
    """
    # Register individual feature functions
    registry.register_feature('sma', simple_moving_average, 'technical')
    registry.register_feature('ema', exponential_moving_average, 'technical')
    registry.register_feature('bollinger', bollinger_bands, 'technical')
    registry.register_feature('macd', macd, 'technical')
    registry.register_feature('rsi', relative_strength_index, 'technical')
    registry.register_feature('atr', average_true_range, 'technical')
    registry.register_feature('adx', average_directional_index, 'technical')
    # Register more indicators...

def simple_moving_average(
    data: pd.DataFrame,
    column: str = 'close',
    window: int = 20
) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: DataFrame with price data
        column: Column name to use for calculation
        window: Window size for moving average
        
    Returns:
        Series with SMA values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    return data[column].rolling(window=window).mean()

def exponential_moving_average(
    data: pd.DataFrame,
    column: str = 'close',
    window: int = 20
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: DataFrame with price data
        column: Column name to use for calculation
        window: Window size for moving average
        
    Returns:
        Series with EMA values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    return data[column].ewm(span=window, adjust=False).mean()

def bollinger_bands(
    data: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with price data
        column: Column name to use for calculation
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Dictionary with 'middle', 'upper', and 'lower' band Series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    middle_band = data[column].rolling(window=window).mean()
    std_dev = data[column].rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return {
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band
    }

# Additional technical indicator implementations...
```

### 4.6 Datasets

Dataset classes for model training:

```python
# datasets/base.py
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Abstract base class for all datasets.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        target: Union[str, List[str]] = None,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with features and target
            features: List of feature column names (None for all except target)
            target: Target column name or list of names
            **kwargs: Additional parameters
        """
        self.data = data.copy()
        
        # Identify feature columns
        if features is None:
            # Use all columns except target(s)
            if target is None:
                self.feature_cols = self.data.columns.tolist()
            else:
                if isinstance(target, str):
                    target_cols = [target]
                else:
                    target_cols = target
                self.feature_cols = [col for col in self.data.columns 
                                  if col not in target_cols]
        else:
            self.feature_cols = [col for col in features if col in self.data.columns]
        
        # Identify target column(s)
        self.target_cols = []
        if target is not None:
            if isinstance(target, str):
                if target in self.data.columns:
                    self.target_cols = [target]
            else:
                self.target_cols = [col for col in target if col in self.data.columns]
        
        # Validate we have some features and targets
        if not self.feature_cols:
            raise ValueError("No valid feature columns found")
        
        # Convert data to numpy arrays for efficiency
        self.X = self.data[self.feature_cols].values
        self.y = None if not self.target_cols else self.data[self.target_cols].values
        
        # Store additional parameters
        self.kwargs = kwargs
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (features, target) or just features if no target
        """
        if self.y is not None:
            return self.X[index], self.y[index]
        else:
            return self.X[index]
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature column names.
        
        Returns:
            List of feature column names
        """
        return self.feature_cols
    
    def get_target_names(self) -> List[str]:
        """
        Get the list of target column names.
        
        Returns:
            List of target column names
        """
        return self.target_cols
```

Time series dataset implementation:

```python
# datasets/timeseries.py
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from trading_optimization.data.datasets.base import BaseDataset

class TimeSeriesDataset(BaseDataset):
    """
    Dataset for time series data with sequence-based samples.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        target: Union[str, List[str]] = None,
        sequence_length: int = 60,
        forecast_horizon: int = 1,
        step_size: int = 1,
        **kwargs
    ):
        """
        Initialize the time series dataset.
        
        Args:
            data: DataFrame with features and target
            features: List of feature column names
            target: Target column name or list of names
            sequence_length: Length of input sequences
            forecast_horizon: How many steps ahead to predict
            step_size: Step size for sliding window
            **kwargs: Additional parameters
        """
        super().__init__(data, features, target, **kwargs)
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        
        # Calculate valid indices
        self.valid_indices = []
        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step_size):
            self.valid_indices.append(i)
    
    def __len__(self):
        """Return the number of valid sequences in the dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        """
        Get a sequence from the dataset.
        
        Args:
            index: Index of the sequence
            
        Returns:
            Tuple of (features_sequence, target_sequence) or just features_sequence
        """
        # Get the actual start index for this sequence
        start_idx = self.valid_indices[index]
        
        # Extract the input sequence
        X_seq = self.X[start_idx:start_idx + self.sequence_length]
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq)
        
        if self.y is not None:
            # Extract the target sequence
            target_idx = start_idx + self.sequence_length
            
            if self.forecast_horizon == 1:
                y_seq = self.y[target_idx]
            else:
                y_seq = self.y[target_idx:target_idx + self.forecast_horizon]
                
            y_tensor = torch.FloatTensor(y_seq)
            return X_tensor, y_tensor
        else:
            return X_tensor
    
    def get_dates(self, index):
        """
        Get the dates corresponding to a sequence.
        
        Args:
            index: Index of the sequence
            
        Returns:
            Tuple of (input_dates, target_dates)
        """
        if not hasattr(self.data, 'index') or not isinstance(self.data.index, pd.DatetimeIndex):
            return None
        
        start_idx = self.valid_indices[index]
        
        input_dates = self.data.index[start_idx:start_idx + self.sequence_length]
        target_idx = start_idx + self.sequence_length
        
        if self.forecast_horizon == 1:
            target_dates = self.data.index[target_idx:target_idx + 1]
        else:
            target_dates = self.data.index[target_idx:target_idx + self.forecast_horizon]
            
        return input_dates, target_dates
```

Dataset factory for creating appropriate datasets:

```python
# datasets/factory.py
from typing import Dict, Any, Union
import pandas as pd

from trading_optimization.data.datasets.base import BaseDataset
from trading_optimization.data.datasets.timeseries import TimeSeriesDataset
from trading_optimization.data.datasets.sequential import SequentialDataset
from trading_optimization.data.features import FeatureRegistry

class DatasetFactory:
    """
    Factory for creating dataset instances.
    """
    
    _datasets = {
        'base': BaseDataset,
        'timeseries': TimeSeriesDataset,
        'sequential': SequentialDataset
    }
    
    def __init__(self, feature_registry: FeatureRegistry = None):
        """
        Initialize dataset factory.
        
        Args:
            feature_registry: Optional feature registry for feature generation
        """
        self.feature_registry = feature_registry
    
    @classmethod
    def register_dataset(cls, dataset_type: str, dataset_class):
        """
        Register a new dataset type.
        
        Args:
            dataset_type: Type identifier for the dataset
            dataset_class: Class that implements the dataset
        """
        cls._datasets[dataset_type] = dataset_class
    
    def create_dataset(
        self,
        data: pd.DataFrame,
        dataset_type: str = 'base',
        **kwargs
    ) -> BaseDataset:
        """
        Create a dataset instance.
        
        Args:
            data: DataFrame to use for the dataset
            dataset_type: Type of dataset to create
            **kwargs: Parameters for the dataset constructor
            
        Returns:
            Instance of a BaseDataset subclass
        """
        if dataset_type not in self._datasets:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Apply any feature generation if needed
        if 'generate_features' in kwargs and kwargs['generate_features'] and self.feature_registry:
            features_to_generate = kwargs.pop('generate_features')
            data = self._generate_features(data, features_to_generate)
        
        return self._datasets[dataset_type](data, **kwargs)
    
    def _generate_features(self, data: pd.DataFrame, features_to_generate: Union[list, dict]) -> pd.DataFrame:
        """
        Generate additional features for the dataset.
        
        Args:
            data: Input DataFrame
            features_to_generate: List of feature names or dict of {name: params}
            
        Returns:
            DataFrame with additional features
        """
        if self.feature_registry is None:
            return data
        
        df = data.copy()
        
        # Handle either list or dict format for features
        if isinstance(features_to_generate, list):
            for feature_name in features_to_generate:
                feature_fn = self.feature_registry.get_feature(feature_name)
                if feature_fn:
                    result = feature_fn(df)
                    if isinstance(result, pd.DataFrame):
                        new_cols = [col for col in result.columns if col not in df.columns]
                        df = pd.concat([df, result[new_cols]], axis=1)
                    elif isinstance(result, pd.Series):
                        df[feature_name] = result
                    elif isinstance(result, dict):
                        for key, val in result.items():
                            df[f"{feature_name}_{key}"] = val
        else:
            # Dictionary with parameters
            for feature_name, params in features_to_generate.items():
                feature_fn = self.feature_registry.get_feature(feature_name)
                if feature_fn:
                    result = feature_fn(df, **params)
                    if isinstance(result, pd.DataFrame):
                        new_cols = [col for col in result.columns if col not in df.columns]
                        df = pd.concat([df, result[new_cols]], axis=1)
                    elif isinstance(result, pd.Series):
                        df[feature_name] = result
                    elif isinstance(result, dict):
                        for key, val in result.items():
                            df[f"{feature_name}_{key}"] = val
        
        return df
```

## 5. Configuration

### 5.1 Data Pipeline Configuration Schema

```yaml
# Example data pipeline configuration
name: "btc_daily_pipeline"
loader:
  type: "csv"
  params:
    file_path: "data/crypto/btc_usd_daily.csv"
    date_column: "timestamp"
    date_format: "%Y-%m-%d %H:%M:%S"
    index_col: "timestamp"
    parse_dates: true

transformers:
  - type: "cleaner"
    params:
      drop_na: true
      columns: ["open", "high", "low", "close", "volume"]
  - type: "resample"
    params:
      rule: "1D"
      method: "ohlc"
  - type: "normalize"
    params:
      method: "minmax"
      columns: ["volume"]
      target_range:
        min: 0
        max: 1

features:
  - name: "sma"
    params:
      column: "close"
      window: 14
  - name: "sma"
    params:
      column: "close"
      window: 50
  - name: "bollinger"
    params:
      column: "close"
      window: 20
      num_std: 2.0
  - name: "rsi"
    params:
      column: "close"
      window: 14
  - name: "macd"
    params:
      column: "close"
      fast_period: 12
      slow_period: 26
      signal_period: 9

post_processing:
  - type: "drop_columns"
    params:
      columns: ["adjusted_close"]
  - type: "fillna"
    params:
      strategy: "ffill"
```

### 5.2 Data Storage Configuration Schema

```yaml
# Example data storage configuration
storage:
  type: "local"  # or "s3", etc.
  base_path: "artifacts/data"
  snapshots_path: "artifacts/data_snapshots"
  versioning: true
  metadata:
    enable: true
    store_in_db: true
```

## 6. Usage Examples

### 6.1 Basic Data Pipeline Example

```python
# Example usage of the Data Management Module

from trading_optimization.config import ConfigManager
from trading_optimization.data.interface import DataManager
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load configuration
config = ConfigManager.instance()
data_config = config.get('data', {})

# Create data manager
data_manager = DataManager(data_config)

# Create a data pipeline
pipeline_config = {
    "loader": {
        "type": "csv",
        "params": {
            "file_path": "data/crypto/btc_usd_daily.csv",
            "date_column": "timestamp"
        }
    },
    "transformers": [
        {
            "type": "cleaner",
            "params": {"drop_na": true}
        }
    ],
    "features": [
        {"name": "sma", "params": {"column": "close", "window": 20}},
        {"name": "bollinger", "params": {"column": "close"}}
    ]
}

pipeline_id = data_manager.create_pipeline("btc_analysis", pipeline_config)

# Execute pipeline for a specific date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
df = data_manager.execute_pipeline(pipeline_id, start_date, end_date)

# Create dataset
dataset = data_manager.create_dataset(
    df,
    dataset_type="timeseries",
    features=["close", "sma", "bollinger_upper", "bollinger_lower"],
    target="close",
    sequence_length=30,
    forecast_horizon=5
)

# Split data
data_splits = data_manager.split_data(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    split_method="time"
)

# Create data loaders for PyTorch
loaders = data_manager.get_data_loaders(
    data_splits,
    batch_size=32,
    dataset_type="timeseries",
    sequence_length=30,
    forecast_horizon=5,
    features=["close", "sma", "bollinger_upper", "bollinger_lower"],
    target="close"
)

# Save pipeline snapshot for reproducibility
snapshot_path = data_manager.save_pipeline_snapshot(pipeline_id, "btc_analysis_v1")
```

### 6.2 Feature Engineering Example

```python
# Example of implementing and registering a custom feature

from trading_optimization.data.interface import DataManager
import pandas as pd
import numpy as np

# Define a custom feature function
def momentum_indicator(
    data: pd.DataFrame,
    column: str = 'close',
    period: int = 10
) -> pd.Series:
    """
    Calculate price momentum indicator.
    
    Args:
        data: DataFrame with price data
        column: Column name to use for calculation
        period: Period for momentum calculation
        
    Returns:
        Series with momentum values
    """
    return data[column].diff(period)

# Initialize data manager
data_manager = DataManager(config.get('data', {}))

# Register the custom feature
data_manager.register_custom_feature(
    'momentum',
    momentum_indicator,
    'technical'
)

# Now the feature can be used in pipelines
pipeline_config = {
    # ... loader and transformer config ...
    "features": [
        {"name": "momentum", "params": {"column": "close", "period": 14}}
    ]
}
```

## 7. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. Configuration management system is implemented
3. Testing framework is available
4. Required libraries are installed:
   - pandas
   - numpy
   - torch (PyTorch)
   - matplotlib (for visualization)

## 8. Implementation Sequence

1. Set up the directory structure
2. Implement base classes and interfaces
3. Develop the data loaders for different sources
4. Implement data transformers and transformer pipeline
5. Create feature engineering modules
6. Develop dataset classes for model training
7. Implement the high-level DataManager interface
8. Create utilities for visualization and validation
9. Add comprehensive unit tests and integration tests
10. Create example configurations and usage documentation

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# Example unit tests for data loaders

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from trading_optimization.data.loaders.csv_loader import CSVLoader

class TestCSVLoader(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary CSV file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        volumes = np.random.randint(100, 10000, size=100)
        
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'price': prices,
            'volume': volumes
        })
        
        df.to_csv(self.csv_file, index=False)
    
    def tearDown(self):
        """Remove temporary files."""
        os.remove(self.csv_file)
        os.rmdir(self.temp_dir)
    
    def test_basic_loading(self):
        """Test basic CSV loading functionality."""
        loader = CSVLoader(
            file_path=self.csv_file,
            date_column='date'
        )
        
        data = loader.load_data()
        
        self.assertEqual(len(data), 100)
        self.assertTrue('price' in data.columns)
        self.assertTrue('volume' in data.columns)
    
    def test_date_filtering(self):
        """Test date range filtering."""
        loader = CSVLoader(
            file_path=self.csv_file,
            date_column='date',
            date_format='%Y-%m-%d'
        )
        
        start_date = datetime(2023, 1, 10)
        end_date = datetime(2023, 1, 20)
        
        data = loader.load_data(start_date, end_date)
        
        self.assertGreaterEqual(len(data), 10)
        self.assertLessEqual(len(data), 12)  # Allow for slight variations in filtering
        
        if len(data) > 0:
            min_date = data.index.min()
            max_date = data.index.max()
            self.assertGreaterEqual(min_date, pd.Timestamp(start_date))
            self.assertLessEqual(max_date, pd.Timestamp(end_date))
```

### 9.2 Integration Tests

```python
# Example integration tests for data pipeline

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_optimization.config import ConfigManager
from trading_optimization.data.interface import DataManager
from trading_optimization.data.pipeline import DataPipeline
from trading_optimization.data.features import FeatureRegistry

class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create a minimal configuration for testing
        self.config = {
            'data': {
                'base_path': 'test_artifacts',
                'snapshots_path': 'test_artifacts/snapshots'
            }
        }
        
        # Initialize components
        self.feature_registry = FeatureRegistry()
        
        # Register some test features
        def simple_sma(data, column='close', window=10):
            return data[column].rolling(window=window).mean()
        
        self.feature_registry.register_feature('sma', simple_sma, 'technical')
        
        # Create a test data frame
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        volumes = np.random.randint(100, 10000, size=100)
        
        self.test_df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(100) * 0.5,
            'high': prices + np.random.randn(100) * 1.0 + 1.0,
            'low': prices + np.random.randn(100) * 1.0 - 1.0,
            'close': prices,
            'volume': volumes
        })
        self.test_df.set_index('date', inplace=True)
    
    def test_end_to_end_pipeline(self):
        """Test a complete data pipeline flow."""
        # Create a pipeline configuration
        pipeline_config = {
            "transformers": [
                {
                    "type": "normalize",
                    "params": {
                        "method": "minmax",
                        "columns": ["volume"]
                    }
                }
            ],
            "features": [
                {"name": "sma", "params": {"column": "close", "window": 5}},
                {"name": "sma", "params": {"column": "close", "window": 20}}
            ],
            "post_processing": [
                {
                    "type": "fillna",
                    "params": {"strategy": "ffill"}
                }
            ]
        }
        
        # Create pipeline manually since we're not using CSV loader
        pipeline = DataPipeline(
            name="test_pipeline",
            config=pipeline_config,
            feature_registry=self.feature_registry
        )
        
        # Mock the loader to return our test DataFrame
        pipeline.loader.load_data = lambda *args, **kwargs: self.test_df
        
        # Execute the pipeline
        result_df = pipeline.execute()
        
        # Verify results
        self.assertEqual(len(result_df), len(self.test_df))
        self.assertIn('sma', result_df.columns)  # First SMA with window=5
        self.assertIn('sma_1', result_df.columns)  # Second SMA with window=20
        
        # Check normalization worked on volume
        self.assertGreaterEqual(result_df['volume'].min(), 0)
        self.assertLessEqual(result_df['volume'].max(), 1)
```

## 10. Error Handling

The Data Management Module includes comprehensive error handling:

```python
# Example error handling in data pipeline

def execute(self, start_date=None, end_date=None):
    """Execute the pipeline with error handling and logging."""
    try:
        # Step 1: Load raw data
        df = self.loader.load_data(start_date, end_date)
        
        if df.empty:
            raise ValueError("Loader returned empty DataFrame")
        
        # Log data loading success
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Continue with pipeline execution
        # ...
        
    except FileNotFoundError as e:
        # Handle missing files
        print(f"Error: Data file not found - {str(e)}")
        # Log specific error
        return pd.DataFrame()
        
    except ValueError as e:
        # Handle value errors in configuration or data
        print(f"Error: Invalid value in pipeline execution - {str(e)}")
        # Log specific error
        return pd.DataFrame()
        
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error in pipeline execution: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
```

## 11. Performance Considerations

The Data Management Module is designed with performance in mind:

1. **Memory Efficiency**:
   - Streaming data loading for large datasets
   - Chunked processing for memory-intensive operations

2. **Computation Optimization**:
   - Leveraging vectorized operations in pandas/numpy
   - Caching intermediate results where beneficial

3. **Parallel Processing**:
   - Using multiprocessing for independent feature calculations
   - Batch processing for I/O-bound operations

## 12. Extension Points

The module is designed to be easily extended:

1. **New Data Sources**:
   - Create new loader classes that inherit from BaseLoader
   - Register them with the LoaderFactory

2. **Custom Transformers**:
   - Create new transformer classes that inherit from BaseTransformer
   - Add them to the transformer map in TransformerPipeline

3. **Custom Features**:
   - Register new feature functions with the FeatureRegistry
   - Group related features into feature modules

4. **Custom Datasets**:
   - Create new dataset classes that inherit from BaseDataset
   - Register them with the DatasetFactory