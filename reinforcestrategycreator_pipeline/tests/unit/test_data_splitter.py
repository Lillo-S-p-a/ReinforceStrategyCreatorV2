"""Unit tests for data splitting components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from reinforcestrategycreator_pipeline.src.data.splitter import DataSplitter


class TestDataSplitter:
    """Test cases for DataSplitter class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples) * 2 + 5,
            'feature3': np.random.exponential(2, n_samples),
            'target': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
        })
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data for testing."""
        np.random.seed(42)
        n_samples = 365  # One year of daily data
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'value': np.cumsum(np.random.randn(n_samples)) + 100,
            'feature1': np.random.randn(n_samples),
            'feature2': np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.randn(n_samples) * 0.1
        })
    
    def test_init(self):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(method='time_series', random_seed=123)
        
        assert splitter.method == 'time_series'
        assert splitter.random_seed == 123
    
    def test_ratio_validation(self, sample_data):
        """Test that ratios must sum to 1.0."""
        splitter = DataSplitter()
        
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            splitter.split(sample_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)
    
    def test_time_series_split(self, time_series_data):
        """Test time series splitting."""
        splitter = DataSplitter(method='time_series')
        train, val, test = splitter.split(
            time_series_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check sizes
        assert len(train) == int(365 * 0.7)  # 255
        assert len(val) == int(365 * 0.85) - int(365 * 0.7)  # 55
        assert len(test) == 365 - int(365 * 0.85)  # 55
        
        # Check chronological order
        assert train['date'].max() < val['date'].min()
        assert val['date'].max() < test['date'].min()
        
        # Check no overlap
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
    
    def test_random_split(self, sample_data):
        """Test random splitting."""
        splitter = DataSplitter(method='random', random_seed=42)
        train, val, test = splitter.split(
            sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check sizes
        assert len(train) == int(1000 * 0.7)  # 700
        assert len(val) == int(1000 * 0.85) - int(1000 * 0.7)  # 150
        assert len(test) == 1000 - int(1000 * 0.85)  # 150
        
        # Check no overlap
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
        
        # Check that all data is used
        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == len(sample_data)
    
    def test_random_split_reproducibility(self, sample_data):
        """Test that random split is reproducible with same seed."""
        splitter1 = DataSplitter(method='random', random_seed=42)
        train1, val1, test1 = splitter1.split(sample_data)
        
        splitter2 = DataSplitter(method='random', random_seed=42)
        train2, val2, test2 = splitter2.split(sample_data)
        
        # Should get same splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_stratified_split(self, sample_data):
        """Test stratified splitting."""
        splitter = DataSplitter(method='stratified', random_seed=42)
        train, val, test = splitter.split(
            sample_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            target_column='target'
        )
        
        # Check sizes - allow small variations due to stratification
        assert abs(len(train) - int(1000 * 0.7)) <= 5  # Allow ±5 samples
        assert abs(len(val) - (int(1000 * 0.85) - int(1000 * 0.7))) <= 5
        assert abs(len(test) - (1000 - int(1000 * 0.85))) <= 5
        
        # Ensure no data is lost
        assert len(train) + len(val) + len(test) == 1000
        
        # Check stratification - proportions should be similar
        original_proportions = sample_data['target'].value_counts(normalize=True).sort_index()
        train_proportions = train['target'].value_counts(normalize=True).sort_index()
        val_proportions = val['target'].value_counts(normalize=True).sort_index()
        test_proportions = test['target'].value_counts(normalize=True).sort_index()
        
        # Allow some tolerance due to rounding
        for cls in original_proportions.index:
            assert abs(train_proportions[cls] - original_proportions[cls]) < 0.05
            assert abs(val_proportions[cls] - original_proportions[cls]) < 0.05
            assert abs(test_proportions[cls] - original_proportions[cls]) < 0.05
    
    def test_stratified_split_missing_target(self, sample_data):
        """Test stratified split without target column."""
        splitter = DataSplitter(method='stratified')
        
        with pytest.raises(ValueError, match="target_column must be specified"):
            splitter.split(sample_data)
    
    def test_invalid_method(self, sample_data):
        """Test invalid splitting method."""
        splitter = DataSplitter(method='invalid_method')
        
        with pytest.raises(ValueError, match="Unknown splitting method"):
            splitter.split(sample_data)
    
    def test_create_folds_time_series(self, time_series_data):
        """Test creating time series cross-validation folds."""
        splitter = DataSplitter(method='time_series')
        folds = splitter.create_folds(time_series_data, n_folds=5)
        
        assert len(folds) == 5
        
        # Check that each fold has train and validation sets
        for i, (train_fold, val_fold) in enumerate(folds):
            assert len(train_fold) > 0
            assert len(val_fold) > 0
            
            # Training set should grow with each fold
            if i > 0:
                assert len(train_fold) > len(folds[i-1][0])
            
            # Validation set should be after training set
            assert train_fold['date'].max() < val_fold['date'].min()
    
    def test_create_folds_random(self, sample_data):
        """Test creating standard k-fold cross-validation."""
        splitter = DataSplitter(method='random', random_seed=42)
        folds = splitter.create_folds(sample_data, n_folds=5)
        
        assert len(folds) == 5
        
        # Check fold sizes
        val_indices_all = []
        for train_fold, val_fold in folds:
            # Each validation fold should be ~1/5 of data
            assert abs(len(val_fold) - 200) < 10  # Allow small variation
            # Training fold should be ~4/5 of data
            assert abs(len(train_fold) - 800) < 10
            
            # No overlap between train and val
            assert len(set(train_fold.index) & set(val_fold.index)) == 0
            
            val_indices_all.extend(val_fold.index.tolist())
        
        # All indices should be covered exactly once in validation sets
        assert len(set(val_indices_all)) == len(sample_data)
    
    def test_create_folds_stratified(self, sample_data):
        """Test creating stratified k-fold cross-validation."""
        splitter = DataSplitter(method='stratified', random_seed=42)
        folds = splitter.create_folds(sample_data, n_folds=5, target_column='target')
        
        assert len(folds) == 5
        
        # Check stratification in each fold
        original_proportions = sample_data['target'].value_counts(normalize=True).sort_index()
        
        for train_fold, val_fold in folds:
            val_proportions = val_fold['target'].value_counts(normalize=True).sort_index()
            
            # Proportions should be similar
            for cls in original_proportions.index:
                if cls in val_proportions:  # Class might be missing in small folds
                    assert abs(val_proportions[cls] - original_proportions[cls]) < 0.1
    
    def test_create_folds_insufficient_data(self):
        """Test creating folds with insufficient data."""
        small_data = pd.DataFrame({'a': [1, 2, 3]})
        splitter = DataSplitter()
        
        # Should still work but with very small folds
        folds = splitter.create_folds(small_data, n_folds=3)
        assert len(folds) == 3
    
    def test_get_temporal_splits(self, time_series_data):
        """Test temporal splitting with specific dates."""
        splitter = DataSplitter()
        
        train_end = '2023-06-30'
        val_end = '2023-09-30'
        test_end = '2023-12-31'
        
        train, val, test = splitter.get_temporal_splits(
            time_series_data,
            date_column='date',
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end
        )
        
        # Check date ranges
        assert train['date'].max() <= pd.to_datetime(train_end)
        assert val['date'].min() > pd.to_datetime(train_end)
        assert val['date'].max() <= pd.to_datetime(val_end)
        assert test['date'].min() > pd.to_datetime(val_end)
        assert test['date'].max() <= pd.to_datetime(test_end)
        
        # Check sizes (approximately)
        assert len(train) == 181  # Jan 1 to Jun 30
        assert len(val) == 92     # Jul 1 to Sep 30
        assert len(test) == 92    # Oct 1 to Dec 31
    
    def test_get_temporal_splits_no_test_end(self, time_series_data):
        """Test temporal splitting without test end date."""
        splitter = DataSplitter()
        
        train_end = '2023-06-30'
        val_end = '2023-09-30'
        
        train, val, test = splitter.get_temporal_splits(
            time_series_data,
            date_column='date',
            train_end_date=train_end,
            val_end_date=val_end
        )
        
        # Test set should include all data after val_end
        assert test['date'].min() > pd.to_datetime(val_end)
        assert len(test) == len(time_series_data) - len(train) - len(val)
    
    def test_get_temporal_splits_datetime_objects(self, time_series_data):
        """Test temporal splitting with datetime objects instead of strings."""
        splitter = DataSplitter()
        
        train_end = datetime(2023, 6, 30)
        val_end = datetime(2023, 9, 30)
        
        train, val, test = splitter.get_temporal_splits(
            time_series_data,
            date_column='date',
            train_end_date=train_end,
            val_end_date=val_end
        )
        
        # Should work the same as with string dates
        assert train['date'].max() <= pd.to_datetime(train_end)
        assert val['date'].max() <= pd.to_datetime(val_end)
    
    def test_edge_cases_empty_splits(self):
        """Test edge cases that might result in empty splits."""
        # Very small dataset
        tiny_data = pd.DataFrame({'a': [1, 2]})
        splitter = DataSplitter(method='random')
        
        # With only 2 samples, some splits might be empty
        train, val, test = splitter.split(
            tiny_data,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25
        )
        
        # At least train should have data
        assert len(train) > 0
        # Total should equal original
        assert len(train) + len(val) + len(test) == len(tiny_data)
    
    def test_different_split_ratios(self, sample_data):
        """Test various split ratio combinations."""
        splitter = DataSplitter(method='random', random_seed=42)
        
        # 80-10-10 split
        train, val, test = splitter.split(
            sample_data,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100
        
        # 60-20-20 split
        train, val, test = splitter.split(
            sample_data,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        assert len(train) == 600
        assert len(val) == 200
        assert len(test) == 200
        
        # 90-5-5 split
        train, val, test = splitter.split(
            sample_data,
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05
        )
        assert len(train) == 900
        assert len(val) == 50
        assert len(test) == 50
    
    def test_data_integrity_after_split(self, sample_data):
        """Test that data is not modified during splitting."""
        original_data = sample_data.copy()
        splitter = DataSplitter(method='random')
        
        train, val, test = splitter.split(sample_data)
        
        # Original data should be unchanged
        pd.testing.assert_frame_equal(sample_data, original_data)
        
        # Split data should be copies
        train.loc[train.index[0], 'feature1'] = 999
        assert sample_data.loc[train.index[0], 'feature1'] != 999
    
    def test_stratified_with_imbalanced_classes(self):
        """Test stratified splitting with highly imbalanced classes."""
        # Create imbalanced dataset
        np.random.seed(42)
        n_samples = 1000
        # 95% class 0, 4% class 1, 1% class 2
        target = np.concatenate([
            np.zeros(950, dtype=int),
            np.ones(40, dtype=int),
            np.full(10, 2, dtype=int)
        ])
        np.random.shuffle(target)
        
        data = pd.DataFrame({
            'feature': np.random.randn(n_samples),
            'target': target
        })
        
        splitter = DataSplitter(method='stratified', random_seed=42)
        train, val, test = splitter.split(
            data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            target_column='target'
        )
        
        # Check that rare classes are present in all splits
        assert 2 in train['target'].values  # Rare class should be in train
        # Might not be in val/test due to very small numbers, but that's OK
        
        # Check overall distribution is maintained
        for split, split_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            split_dist = split['target'].value_counts(normalize=True)
            # Class 0 should still be dominant
            assert split_dist.get(0, 0) > 0.9
    
    def test_temporal_splits_with_unsorted_data(self):
        """Test temporal splitting with unsorted time series data."""
        # Create unsorted time series data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100)
        })
        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        splitter = DataSplitter()
        train, val, test = splitter.get_temporal_splits(
            data,
            date_column='date',
            train_end_date='2023-02-28',
            val_end_date='2023-03-31'
        )
        
        # Should still split correctly based on dates
        assert train['date'].max() <= pd.to_datetime('2023-02-28')
        assert val['date'].min() > pd.to_datetime('2023-02-28')
        assert val['date'].max() <= pd.to_datetime('2023-03-31')
        assert test['date'].min() > pd.to_datetime('2023-03-31')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])