"""
Example script demonstrating data transformation and validation.

This script shows how to:
1. Load sample data
2. Validate the data using DataValidator
3. Transform the data using DataTransformer
4. Split the data for training/validation/testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    DataTransformer,
    TechnicalIndicatorTransformer,
    ScalingTransformer,
    DataValidator,
    DataSplitter,
    ValidationStatus
)
from src.config import TransformationConfig, ValidationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_days=500):
    """Generate sample OHLCV data for demonstration."""
    logger.info(f"Generating {n_days} days of sample OHLCV data")
    
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate price data with some realistic patterns
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': close_prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': close_prices * (1 + np.random.uniform(-0.02, 0, n_days)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Add some missing values for demonstration
    missing_indices = np.random.choice(data.index[50:], size=5, replace=False)
    data.loc[missing_indices, 'volume'] = np.nan
    
    # Add an outlier
    data.loc[100, 'high'] = data.loc[100, 'high'] * 3
    
    return data


def main():
    """Main example workflow."""
    
    # 1. Generate sample data
    logger.info("=== Step 1: Generate Sample Data ===")
    data = generate_sample_data()
    logger.info(f"Generated data shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    logger.info(f"\nFirst few rows:\n{data.head()}")
    
    # 2. Configure validation
    logger.info("\n=== Step 2: Configure and Run Data Validation ===")
    validation_config = ValidationConfig(
        check_missing_values=True,
        missing_value_threshold=0.05,  # Allow up to 5% missing
        check_outliers=True,
        outlier_method="iqr",
        outlier_threshold=1.5,
        check_data_types=True,
        expected_types={
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        },
        check_ranges=True,
        value_ranges={
            'open': (0, float('inf')),
            'high': (0, float('inf')),
            'low': (0, float('inf')),
            'close': (0, float('inf')),
            'volume': (0, float('inf'))
        }
    )
    
    # Create validator
    validator = DataValidator(
        check_missing=validation_config.check_missing_values,
        check_outliers=validation_config.check_outliers,
        check_types=validation_config.check_data_types,
        check_ranges=validation_config.check_ranges
    )
    
    # Configure individual validators
    if validation_config.check_missing_values:
        validator.validators['missing'].threshold = validation_config.missing_value_threshold
    
    if validation_config.check_outliers:
        validator.validators['outliers'].method = validation_config.outlier_method
        validator.validators['outliers'].threshold = validation_config.outlier_threshold
    
    if validation_config.check_data_types and validation_config.expected_types:
        validator.validators['types'].expected_types = validation_config.expected_types
    
    if validation_config.check_ranges and validation_config.value_ranges:
        validator.validators['ranges'].value_ranges = validation_config.value_ranges
    
    # Run validation
    validation_result = validator.validate(data)
    
    logger.info(f"Validation Status: {validation_result.status.value}")
    logger.info(f"Valid: {validation_result.is_valid}")
    
    if validation_result.errors:
        logger.warning("Validation Errors:")
        for error in validation_result.errors:
            logger.warning(f"  - {error}")
    
    if validation_result.warnings:
        logger.info("Validation Warnings:")
        for warning in validation_result.warnings:
            logger.info(f"  - {warning}")
    
    # 3. Configure and apply transformations
    logger.info("\n=== Step 3: Configure and Apply Data Transformations ===")
    transformation_config = TransformationConfig(
        add_technical_indicators=True,
        technical_indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
        scaling_method="standard",
        scaling_columns=None  # Scale all numeric columns
    )
    
    # Create transformer
    transformer = DataTransformer()
    
    # Add technical indicator transformer
    if transformation_config.add_technical_indicators:
        tech_transformer = TechnicalIndicatorTransformer(
            indicators=transformation_config.technical_indicators
        )
        transformer.add_transformer('technical_indicators', tech_transformer)
    
    # Add scaling transformer
    if transformation_config.scaling_method:
        scaling_transformer = ScalingTransformer(
            method=transformation_config.scaling_method,
            columns=transformation_config.scaling_columns
        )
        transformer.add_transformer('scaling', scaling_transformer)
    
    # Apply transformations
    logger.info("Applying transformations...")
    transformed_data = transformer.transform(data.copy())
    
    logger.info(f"Transformed data shape: {transformed_data.shape}")
    logger.info(f"New columns added: {set(transformed_data.columns) - set(data.columns)}")
    
    # 4. Split data for training/validation/testing
    logger.info("\n=== Step 4: Split Data for Training/Validation/Testing ===")
    splitter = DataSplitter(method='time_series')
    
    train_data, val_data, test_data = splitter.split(
        transformed_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    
    # 5. Demonstrate saving and loading transformer parameters
    logger.info("\n=== Step 5: Save and Load Transformer Parameters ===")
    
    # Save parameters
    params = transformer.save_params()
    logger.info("Saved transformer parameters")
    
    # Create new transformer and load parameters
    new_transformer = DataTransformer()
    new_transformer.load_params(params)
    logger.info("Loaded parameters into new transformer")
    
    # Verify by transforming a small sample
    sample_data = data.head(10).copy()
    sample_transformed = new_transformer.transform(sample_data)
    logger.info(f"Verified transformation on sample data: {sample_transformed.shape}")
    
    # 6. Summary statistics
    logger.info("\n=== Step 6: Summary Statistics ===")
    logger.info("\nOriginal data statistics:")
    logger.info(data[['open', 'high', 'low', 'close', 'volume']].describe())
    
    logger.info("\nTransformed data statistics (sample columns):")
    sample_cols = ['close', 'sma_20', 'rsi', 'macd_signal'] 
    available_cols = [col for col in sample_cols if col in transformed_data.columns]
    if available_cols:
        logger.info(transformed_data[available_cols].describe())
    
    logger.info("\n=== Example Complete ===")
    logger.info("This example demonstrated:")
    logger.info("1. Data generation and loading")
    logger.info("2. Data validation with multiple checks")
    logger.info("3. Data transformation with technical indicators and scaling")
    logger.info("4. Data splitting for ML pipeline")
    logger.info("5. Saving/loading transformer parameters for reproducibility")


if __name__ == "__main__":
    main()