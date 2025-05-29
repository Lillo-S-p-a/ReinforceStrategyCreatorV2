"""Data Splitter for train/validation/test splitting."""

import logging
from typing import Tuple, Optional, Union, List
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSplitter:
    """Utility class for splitting data into train/validation/test sets."""
    
    def __init__(self, method: str = 'time_series', random_seed: int = 42):
        """Initialize the data splitter.
        
        Args:
            method: Splitting method ('time_series', 'random', 'stratified')
            random_seed: Random seed for reproducibility
        """
        self.method = method
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def split(self, 
              data: pd.DataFrame,
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15,
              target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            target_column: Column name for stratified splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
            
        if self.method == 'time_series':
            return self._time_series_split(data, train_ratio, val_ratio, test_ratio)
        elif self.method == 'random':
            return self._random_split(data, train_ratio, val_ratio, test_ratio)
        elif self.method == 'stratified':
            if target_column is None:
                raise ValueError("target_column must be specified for stratified splitting")
            return self._stratified_split(data, train_ratio, val_ratio, test_ratio, target_column)
        else:
            raise ValueError(f"Unknown splitting method: {self.method}")
            
    def _time_series_split(self, 
                          data: pd.DataFrame,
                          train_ratio: float,
                          val_ratio: float,
                          test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data chronologically.
        
        Args:
            data: Input DataFrame (assumed to be sorted by time)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()
        
        logger.info(f"Time series split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _random_split(self,
                     data: pd.DataFrame,
                     train_ratio: float,
                     val_ratio: float,
                     test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data randomly.
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Shuffle indices
        indices = np.random.permutation(len(data))
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = data.iloc[train_indices].copy()
        val_df = data.iloc[val_indices].copy()
        test_df = data.iloc[test_indices].copy()
        
        logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _stratified_split(self,
                         data: pd.DataFrame,
                         train_ratio: float,
                         val_ratio: float,
                         test_ratio: float,
                         target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data with stratification on target variable.
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            target_column: Column to stratify on
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get unique classes
        classes = data[target_column].unique()
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each class proportionally
        for cls in classes:
            cls_indices = data[data[target_column] == cls].index.tolist()
            np.random.shuffle(cls_indices)
            
            n_cls = len(cls_indices)
            train_end = int(n_cls * train_ratio)
            val_end = int(n_cls * (train_ratio + val_ratio))
            
            train_indices.extend(cls_indices[:train_end])
            val_indices.extend(cls_indices[train_end:val_end])
            test_indices.extend(cls_indices[val_end:])
            
        # Shuffle indices within each set
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        train_df = data.loc[train_indices].copy()
        val_df = data.loc[val_indices].copy()
        test_df = data.loc[test_indices].copy()
        
        logger.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_folds(self, 
                    data: pd.DataFrame,
                    n_folds: int = 5,
                    target_column: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create cross-validation folds.
        
        Args:
            data: Input DataFrame
            n_folds: Number of folds
            target_column: Column for stratified folding
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        folds = []
        
        if self.method == 'time_series':
            # Time series cross-validation
            fold_size = len(data) // (n_folds + 1)
            
            for i in range(n_folds):
                train_end = (i + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                train_fold = data.iloc[:train_end].copy()
                val_fold = data.iloc[val_start:val_end].copy()
                
                folds.append((train_fold, val_fold))
                
        else:
            # Standard k-fold or stratified k-fold
            indices = np.arange(len(data))
            
            if self.method == 'stratified' and target_column:
                # Group indices by class
                class_indices = {}
                for cls in data[target_column].unique():
                    class_indices[cls] = data[data[target_column] == cls].index.tolist()
                    
                # Create stratified folds
                for i in range(n_folds):
                    val_indices = []
                    
                    for cls, cls_idx in class_indices.items():
                        np.random.shuffle(cls_idx)
                        fold_size = len(cls_idx) // n_folds
                        start = i * fold_size
                        end = start + fold_size if i < n_folds - 1 else len(cls_idx)
                        val_indices.extend(cls_idx[start:end])
                        
                    train_indices = list(set(indices) - set(val_indices))
                    
                    train_fold = data.iloc[train_indices].copy()
                    val_fold = data.iloc[val_indices].copy()
                    
                    folds.append((train_fold, val_fold))
                    
            else:
                # Standard k-fold
                np.random.shuffle(indices)
                fold_size = len(data) // n_folds
                
                for i in range(n_folds):
                    start = i * fold_size
                    end = start + fold_size if i < n_folds - 1 else len(data)
                    
                    val_indices = indices[start:end]
                    train_indices = np.concatenate([indices[:start], indices[end:]])
                    
                    train_fold = data.iloc[train_indices].copy()
                    val_fold = data.iloc[val_indices].copy()
                    
                    folds.append((train_fold, val_fold))
                    
        logger.info(f"Created {n_folds} cross-validation folds")
        return folds
    
    def get_temporal_splits(self,
                           data: pd.DataFrame,
                           date_column: str,
                           train_end_date: Union[str, datetime],
                           val_end_date: Union[str, datetime],
                           test_end_date: Optional[Union[str, datetime]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data based on specific dates.
        
        Args:
            data: Input DataFrame
            date_column: Name of the date column
            train_end_date: End date for training data
            val_end_date: End date for validation data
            test_end_date: End date for test data (optional, uses all remaining if None)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Convert string dates to datetime if needed
        if isinstance(train_end_date, str):
            train_end_date = pd.to_datetime(train_end_date)
        if isinstance(val_end_date, str):
            val_end_date = pd.to_datetime(val_end_date)
        if test_end_date and isinstance(test_end_date, str):
            test_end_date = pd.to_datetime(test_end_date)
            
        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Split based on dates
        train_df = data[data[date_column] <= train_end_date].copy()
        val_df = data[(data[date_column] > train_end_date) & 
                     (data[date_column] <= val_end_date)].copy()
        
        if test_end_date:
            test_df = data[(data[date_column] > val_end_date) & 
                          (data[date_column] <= test_end_date)].copy()
        else:
            test_df = data[data[date_column] > val_end_date].copy()
            
        logger.info(f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df