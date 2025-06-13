"""Data Transformer for feature engineering and preprocessing."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Import technical indicators libraries
import pandas_ta as pta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

from ..config.models import TransformationConfig


logger = logging.getLogger(__name__)


class TransformerBase(ABC):
    """Base class for all data transformers."""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters for reproducibility.
        
        Returns:
            Dictionary of parameters
        """
        pass


class TechnicalIndicatorTransformer(TransformerBase):
    """Transformer for calculating technical indicators."""
    
    def __init__(self, indicators: Optional[List[str]] = None):
        """Initialize the technical indicator transformer.
        
        Args:
            indicators: List of indicators to calculate. If None, calculates all.
                        Can include specific periods like 'sma_20', 'ema_10'.
        """
        self.requested_indicators = indicators or [ # Store what was requested
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', # Default to generic if nothing specific
            'atr', 'adx', 'aroon', 'historical_volatility'
        ]
        
        # Default parameters, can be overridden by specific requests
        self.params = {
            'sma_periods': [], # Will be populated from requested_indicators
            'ema_periods': [], # Will be populated
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20, # For bb_middle, bb_upper, bb_lower
            'bb_std': 2,
            'atr_period': 14,
            'adx_period': 14,
            'aroon_period': 14,
            'hist_vol_period': 20
        }

        self.active_indicators = set() # To track which generic indicators to run

        for ind_request in self.requested_indicators:
            parts = ind_request.split('_')
            base_ind = parts[0]
            
            if base_ind == 'sma' and len(parts) > 1 and parts[1].isdigit():
                self.params['sma_periods'].append(int(parts[1]))
                self.active_indicators.add('sma')
            elif base_ind == 'ema' and len(parts) > 1 and parts[1].isdigit():
                self.params['ema_periods'].append(int(parts[1]))
                self.active_indicators.add('ema')
            elif base_ind == 'bb' and len(parts) > 1 and parts[1] in ['upper', 'middle', 'lower']:
                # bb_period is already set in defaults, this just activates bollinger_bands
                self.active_indicators.add('bollinger_bands') # Generic trigger
            elif base_ind in ['rsi', 'macd', 'atr', 'adx', 'aroon', 'historical_volatility']:
                self.active_indicators.add(base_ind)
            elif base_ind in self.params: # e.g. if 'sma' is passed, use default periods
                 self.active_indicators.add(base_ind)
                 if base_ind == 'sma' and not self.params['sma_periods']: # if 'sma' requested but no specific periods yet
                     self.params['sma_periods'] = [5, 20, 50] # Default periods
                 if base_ind == 'ema' and not self.params['ema_periods']:
                     self.params['ema_periods'] = [5, 20]
            else:
                logger.warning(f"Unknown or unhandled indicator format: {ind_request}")
        
        # Ensure uniqueness if defaults were added
        if 'sma' in self.active_indicators:
            self.params['sma_periods'] = sorted(list(set(self.params['sma_periods'])))
        if 'ema' in self.active_indicators:
            self.params['ema_periods'] = sorted(list(set(self.params['ema_periods'])))

        logger.debug(f"TechnicalIndicatorTransformer initialized. Active indicators: {self.active_indicators}, Params: {self.params}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on the data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added technical indicators
        """
        result = data.copy()
        
        # Find column names case-insensitively
        close_col = self._find_column(result.columns, 'close')
        high_col = self._find_column(result.columns, 'high')
        low_col = self._find_column(result.columns, 'low')
        volume_col = self._find_column(result.columns, 'volume')
        
        if not all([close_col, high_col, low_col]):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")
        
        # Calculate each indicator
        if 'sma' in self.active_indicators:
            for period in self.params['sma_periods']:
                result[f'sma_{period}'] = SMAIndicator(
                    close=result[close_col], window=period
                ).sma_indicator()
                
        if 'ema' in self.active_indicators:
            for period in self.params['ema_periods']:
                result[f'ema_{period}'] = EMAIndicator(
                    close=result[close_col], window=period
                ).ema_indicator()
                
        if 'rsi' in self.active_indicators:
            result['rsi'] = RSIIndicator(
                close=result[close_col], window=self.params['rsi_period']
            ).rsi()
            
        if 'macd' in self.active_indicators:
            macd = MACD(
                close=result[close_col],
                window_slow=self.params['macd_slow'],
                window_fast=self.params['macd_fast'],
                window_sign=self.params['macd_signal']
            )
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
            
        if 'bollinger_bands' in self.active_indicators:
            bb = BollingerBands(
                close=result[close_col],
                window=self.params['bb_period'],
                window_dev=self.params['bb_std']
            )
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_middle'] = bb.bollinger_mavg()
            result['bb_lower'] = bb.bollinger_lband()
            
        if 'atr' in self.active_indicators and high_col and low_col:
            result.ta.atr(
                high=result[high_col],
                low=result[low_col],
                close=result[close_col],
                length=self.params['atr_period'],
                append=True
            )
            
        if 'adx' in self.active_indicators and high_col and low_col:
            result.ta.adx(
                high=result[high_col],
                low=result[low_col],
                close=result[close_col],
                length=self.params['adx_period'],
                append=True
            )
            
        if 'aroon' in self.active_indicators and high_col and low_col:
            result.ta.aroon(
                high=result[high_col],
                low=result[low_col], # This was missing high=, low= before, fixed.
                close=result[close_col], # Added close for aroon as it's often used.
                length=self.params['aroon_period'],
                append=True
            )
            
        if 'historical_volatility' in self.active_indicators:
            returns = result[close_col].pct_change()
            result['hist_volatility'] = returns.rolling(
                window=self.params['hist_vol_period']
            ).std() * np.sqrt(252)  # Annualized
            
        logger.info(f"Calculated {len(self.active_indicators)} types of technical indicators based on {len(self.requested_indicators)} requests.")
        return result
    
    def _find_column(self, columns: pd.Index, target: str) -> Optional[str]:
        """Find column name case-insensitively."""
        for col in columns:
            if isinstance(col, str) and col.lower() == target.lower():
                return col
            elif isinstance(col, tuple) and len(col) > 0 and col[0].lower() == target.lower():
                return col
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            'indicators': self.indicators,
            'params': self.params
        }


class ScalingTransformer(TransformerBase):
    """Transformer for scaling numerical features."""
    
    def __init__(self, method: str = 'standard', columns: Optional[List[str]] = None):
        """Initialize the scaling transformer.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            columns: Columns to scale. If None, scales all numeric columns.
        """
        self.method = method
        self.columns = columns
        self.scalers = {}
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Scaled DataFrame
        """
        result = data.copy()
        
        # Determine columns to scale
        if self.columns is None:
            numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in self.columns if col in result.columns]
            
        for col in numeric_cols:
            if self.method == 'standard':
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
                    self.scalers[col] = {'mean': mean, 'std': std}
            elif self.method == 'minmax':
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
                    self.scalers[col] = {'min': min_val, 'max': max_val}
            elif self.method == 'robust':
                median = result[col].median()
                q75, q25 = np.percentile(result[col].dropna(), [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    result[col] = (result[col] - median) / iqr
                    self.scalers[col] = {'median': median, 'iqr': iqr}
                    
        logger.info(f"Scaled {len(numeric_cols)} columns using {self.method} method")
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled DataFrame
            
        Returns:
            Original scale DataFrame
        """
        result = data.copy()
        
        for col, params in self.scalers.items():
            if col in result.columns:
                if self.method == 'standard':
                    result[col] = result[col] * params['std'] + params['mean']
                elif self.method == 'minmax':
                    result[col] = result[col] * (params['max'] - params['min']) + params['min']
                elif self.method == 'robust':
                    result[col] = result[col] * params['iqr'] + params['median']
                    
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            'method': self.method,
            'columns': self.columns,
            'scalers': self.scalers
        }


class DataTransformer:
    """Main data transformer that orchestrates multiple transformations."""
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        """Initialize the data transformer.
        
        Args:
            config: Transformation configuration
        """
        self.config = config
        self.transformers: List[TransformerBase] = []
        self.is_fitted = False
        
        # Initialize transformers based on config
        if config:
            self._initialize_transformers()
            
    def _initialize_transformers(self):
        """Initialize transformers based on configuration."""
        if self.config.add_technical_indicators:
            self.transformers.append(
                TechnicalIndicatorTransformer(
                    indicators=self.config.technical_indicators
                )
            )
            
        if self.config.scaling_method:
            self.transformers.append(
                ScalingTransformer(
                    method=self.config.scaling_method,
                    columns=self.config.scaling_columns
                )
            )
            
    def add_transformer(self, transformer: TransformerBase):
        """Add a custom transformer to the pipeline.
        
        Args:
            transformer: Transformer instance
        """
        self.transformers.append(transformer)
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers and transform data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        for transformer in self.transformers:
            try:
                result = transformer.transform(result)
            except Exception as e:
                logger.error(f"Error in {transformer.__class__.__name__}: {str(e)}")
                raise
                
        self.is_fitted = True
        logger.info(f"Applied {len(self.transformers)} transformations")
        return result
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            logger.warning("Transformer not fitted, using fit_transform")
            return self.fit_transform(data)
            
        result = data.copy()
        
        for transformer in self.transformers:
            result = transformer.transform(result)
            
        return result
    
    def get_feature_names(self, original_columns: List[str]) -> List[str]:
        """Get the names of all features after transformation.
        
        Args:
            original_columns: Original column names
            
        Returns:
            List of all feature names after transformation
        """
        # This would need to be implemented based on actual transformations
        # For now, return a placeholder
        return original_columns
    
    def save_params(self, filepath: str):
        """Save transformer parameters for reproducibility.
        
        Args:
            filepath: Path to save parameters
        """
        import json
        
        params = {
            'transformers': [
                {
                    'class': transformer.__class__.__name__,
                    'params': transformer.get_params()
                }
                for transformer in self.transformers
            ],
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2, default=str)
            
        logger.info(f"Saved transformer parameters to {filepath}")