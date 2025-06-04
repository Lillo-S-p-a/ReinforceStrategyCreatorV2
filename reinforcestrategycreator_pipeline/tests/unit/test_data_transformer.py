"""Unit tests for data transformation components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tempfile
from unittest.mock import Mock, patch

from reinforcestrategycreator_pipeline.src.data.transformer import (
    TransformerBase,
    TechnicalIndicatorTransformer,
    ScalingTransformer,
    DataTransformer
)
from reinforcestrategycreator_pipeline.src.config.models import TransformationConfig


class TestTechnicalIndicatorTransformer:
    """Test cases for TechnicalIndicatorTransformer."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        high_prices = close_prices + np.abs(np.random.randn(100) * 1)
        low_prices = close_prices - np.abs(np.random.randn(100) * 1)
        open_prices = close_prices + np.random.randn(100) * 0.5
        volume = np.random.randint(1000000, 5000000, 100)
        
        return pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
    
    def test_init_default_indicators(self):
        """Test initialization with default indicators."""
        transformer = TechnicalIndicatorTransformer()
        
        assert transformer.indicators == [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands',
            'atr', 'adx', 'aroon', 'historical_volatility'
        ]
        assert transformer.params['rsi_period'] == 14
        assert transformer.params['sma_periods'] == [5, 20, 50]
    
    def test_init_custom_indicators(self):
        """Test initialization with custom indicators."""
        custom_indicators = ['sma', 'rsi']
        transformer = TechnicalIndicatorTransformer(indicators=custom_indicators)
        
        assert transformer.indicators == custom_indicators
    
    def test_sma_calculation(self, sample_ohlcv_data):
        """Test SMA calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['sma'])
        result = transformer.transform(sample_ohlcv_data)
        
        # Check that SMA columns were added
        for period in [5, 20, 50]:
            assert f'sma_{period}' in result.columns
            
        # Verify SMA calculation for period 5
        expected_sma_5 = sample_ohlcv_data['close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(
            result['sma_5'],
            expected_sma_5,
            check_names=False
        )
    
    def test_ema_calculation(self, sample_ohlcv_data):
        """Test EMA calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['ema'])
        result = transformer.transform(sample_ohlcv_data)
        
        # Check that EMA columns were added
        for period in [5, 20]:
            assert f'ema_{period}' in result.columns
    
    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['rsi'])
        result = transformer.transform(sample_ohlcv_data)
        
        assert 'rsi' in result.columns
        # RSI should be between 0 and 100
        assert result['rsi'].dropna().between(0, 100).all()
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['macd'])
        result = transformer.transform(sample_ohlcv_data)
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_diff' in result.columns
    
    def test_bollinger_bands_calculation(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['bollinger_bands'])
        result = transformer.transform(sample_ohlcv_data)
        
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        
        # Upper band should be above middle, middle above lower
        valid_idx = result[['bb_upper', 'bb_middle', 'bb_lower']].dropna().index
        assert (result.loc[valid_idx, 'bb_upper'] >= result.loc[valid_idx, 'bb_middle']).all()
        assert (result.loc[valid_idx, 'bb_middle'] >= result.loc[valid_idx, 'bb_lower']).all()
    
    def test_historical_volatility_calculation(self, sample_ohlcv_data):
        """Test historical volatility calculation."""
        transformer = TechnicalIndicatorTransformer(indicators=['historical_volatility'])
        result = transformer.transform(sample_ohlcv_data)
        
        assert 'hist_volatility' in result.columns
        # Volatility should be non-negative
        assert (result['hist_volatility'].dropna() >= 0).all()
    
    def test_missing_required_columns(self):
        """Test error handling when required columns are missing."""
        # DataFrame without required OHLC columns
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'price': np.random.randn(10)
        })
        
        transformer = TechnicalIndicatorTransformer()
        with pytest.raises(ValueError, match="must contain 'high', 'low', and 'close' columns"):
            transformer.transform(df)
    
    def test_case_insensitive_column_matching(self):
        """Test that column matching is case-insensitive."""
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=50),
            'OPEN': np.random.randn(50) + 100,
            'HIGH': np.random.randn(50) + 101,
            'LOW': np.random.randn(50) + 99,
            'CLOSE': np.random.randn(50) + 100,
            'VOLUME': np.random.randint(1000, 5000, 50)
        })
        
        transformer = TechnicalIndicatorTransformer(indicators=['sma'])
        result = transformer.transform(df)
        
        assert 'sma_5' in result.columns
    
    def test_get_params(self):
        """Test get_params method."""
        indicators = ['sma', 'rsi']
        transformer = TechnicalIndicatorTransformer(indicators=indicators)
        params = transformer.get_params()
        
        assert params['indicators'] == indicators
        assert 'params' in params
        assert params['params']['rsi_period'] == 14


class TestScalingTransformer:
    """Test cases for ScalingTransformer."""
    
    @pytest.fixture
    def sample_numeric_data(self):
        """Create sample numeric data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 50,
            'feature2': np.random.randn(100) * 5 + 20,
            'feature3': np.random.exponential(2, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A']  # Non-numeric column
        })
    
    def test_standard_scaling(self, sample_numeric_data):
        """Test standard scaling."""
        transformer = ScalingTransformer(method='standard')
        result = transformer.transform(sample_numeric_data)
        
        # Check that numeric columns are scaled
        for col in ['feature1', 'feature2', 'feature3']:
            assert np.abs(result[col].mean()) < 1e-10  # Mean should be ~0
            assert np.abs(result[col].std() - 1) < 1e-10  # Std should be ~1
        
        # Non-numeric column should remain unchanged
        assert (result['category'] == sample_numeric_data['category']).all()
    
    def test_minmax_scaling(self, sample_numeric_data):
        """Test min-max scaling."""
        transformer = ScalingTransformer(method='minmax')
        result = transformer.transform(sample_numeric_data)
        
        # Check that numeric columns are scaled to [0, 1]
        for col in ['feature1', 'feature2', 'feature3']:
            assert result[col].min() >= -1e-10  # Min should be ~0
            assert result[col].max() <= 1 + 1e-10  # Max should be ~1
    
    def test_robust_scaling(self, sample_numeric_data):
        """Test robust scaling."""
        transformer = ScalingTransformer(method='robust')
        result = transformer.transform(sample_numeric_data)
        
        # Check that numeric columns are scaled
        for col in ['feature1', 'feature2', 'feature3']:
            assert np.abs(result[col].median()) < 0.1  # Median should be ~0
    
    def test_scaling_specific_columns(self, sample_numeric_data):
        """Test scaling only specific columns."""
        columns_to_scale = ['feature1', 'feature3']
        transformer = ScalingTransformer(method='standard', columns=columns_to_scale)
        result = transformer.transform(sample_numeric_data)
        
        # Check that specified columns are scaled
        for col in columns_to_scale:
            assert np.abs(result[col].mean()) < 1e-10
            assert np.abs(result[col].std() - 1) < 1e-10
        
        # Check that other numeric column is not scaled
        pd.testing.assert_series_equal(
            result['feature2'],
            sample_numeric_data['feature2']
        )
    
    def test_inverse_transform(self, sample_numeric_data):
        """Test inverse transformation."""
        transformer = ScalingTransformer(method='standard')
        scaled = transformer.transform(sample_numeric_data)
        inversed = transformer.inverse_transform(scaled)
        
        # Check that inverse transform recovers original values
        for col in ['feature1', 'feature2', 'feature3']:
            pd.testing.assert_series_equal(
                inversed[col],
                sample_numeric_data[col],
                check_exact=False,
                rtol=1e-10
            )
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance columns."""
        # Create data with a constant column
        df = pd.DataFrame({
            'constant': [5.0] * 100,
            'normal': np.random.randn(100)
        })
        
        transformer = ScalingTransformer(method='standard')
        result = transformer.transform(df)
        
        # Constant column should remain unchanged
        assert (result['constant'] == df['constant']).all()
        # Normal column should be scaled
        assert np.abs(result['normal'].std() - 1) < 1e-10
    
    def test_get_params(self):
        """Test get_params method."""
        transformer = ScalingTransformer(method='minmax', columns=['col1', 'col2'])
        
        # Transform some data to populate scalers
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        transformer.transform(df)
        
        params = transformer.get_params()
        assert params['method'] == 'minmax'
        assert params['columns'] == ['col1', 'col2']
        assert 'scalers' in params
        assert 'col1' in params['scalers']
        assert 'col2' in params['scalers']


class TestDataTransformer:
    """Test cases for DataTransformer orchestrator."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample transformation configuration."""
        return TransformationConfig(
            add_technical_indicators=True,
            technical_indicators=['sma', 'rsi'],
            scaling_method='standard',
            scaling_columns=None
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'date': dates,
            'open': np.random.randn(50) + 100,
            'high': np.random.randn(50) + 101,
            'low': np.random.randn(50) + 99,
            'close': np.random.randn(50) + 100,
            'volume': np.random.randint(1000000, 5000000, 50),
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50) * 10
        })
    
    def test_init_with_config(self, sample_config):
        """Test initialization with configuration."""
        transformer = DataTransformer(config=sample_config)
        
        assert len(transformer.transformers) == 2
        assert isinstance(transformer.transformers[0], TechnicalIndicatorTransformer)
        assert isinstance(transformer.transformers[1], ScalingTransformer)
    
    def test_init_without_config(self):
        """Test initialization without configuration."""
        transformer = DataTransformer()
        
        assert len(transformer.transformers) == 0
        assert not transformer.is_fitted
    
    def test_add_transformer(self):
        """Test adding custom transformer."""
        transformer = DataTransformer()
        custom_transformer = TechnicalIndicatorTransformer(indicators=['sma'])
        
        transformer.add_transformer(custom_transformer)
        
        assert len(transformer.transformers) == 1
        assert transformer.transformers[0] == custom_transformer
    
    def test_fit_transform(self, sample_config, sample_data):
        """Test fit_transform method."""
        transformer = DataTransformer(config=sample_config)
        result = transformer.fit_transform(sample_data)
        
        # Check that transformations were applied
        assert 'sma_5' in result.columns  # Technical indicator added
        assert 'rsi' in result.columns  # Technical indicator added
        
        # Check that numeric columns were scaled
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['date']:  # Skip date-like columns
                # For standard scaling, mean should be ~0 and std ~1
                # But only for columns that had variance > 0
                if result[col].std() > 0:
                    assert np.abs(result[col].mean()) < 0.5  # Relaxed due to indicators
        
        assert transformer.is_fitted
    
    def test_transform_without_fit(self, sample_config, sample_data):
        """Test transform method without prior fit."""
        transformer = DataTransformer(config=sample_config)
        
        # Should call fit_transform internally
        result = transformer.transform(sample_data)
        
        assert 'sma_5' in result.columns
        assert transformer.is_fitted
    
    def test_error_handling(self, sample_config):
        """Test error handling in transformation pipeline."""
        transformer = DataTransformer(config=sample_config)
        
        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError):
            transformer.fit_transform(invalid_data)
    
    def test_save_params(self, sample_config, sample_data):
        """Test saving transformer parameters."""
        transformer = DataTransformer(config=sample_config)
        transformer.fit_transform(sample_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            transformer.save_params(f.name)
            
            # Read and verify saved parameters
            with open(f.name, 'r') as rf:
                saved_params = json.load(rf)
                
            assert 'transformers' in saved_params
            assert len(saved_params['transformers']) == 2
            assert saved_params['is_fitted'] == True
            assert 'timestamp' in saved_params
    
    def test_empty_config_no_transformations(self):
        """Test with empty configuration."""
        config = TransformationConfig(
            add_technical_indicators=False,
            scaling_method=None
        )
        transformer = DataTransformer(config=config)
        
        assert len(transformer.transformers) == 0
        
        # Transform should return data unchanged
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = transformer.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)


class MockTransformer(TransformerBase):
    """Mock transformer for testing abstract base class."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
    
    def get_params(self) -> dict:
        return {'mock': True}


class TestTransformerBase:
    """Test cases for TransformerBase abstract class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            TransformerBase()
    
    def test_mock_implementation(self):
        """Test mock implementation of abstract class."""
        transformer = MockTransformer()
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = transformer.transform(df)
        pd.testing.assert_frame_equal(result, df)
        
        params = transformer.get_params()
        assert params == {'mock': True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])