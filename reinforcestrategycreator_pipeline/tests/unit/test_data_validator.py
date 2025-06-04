"""Unit tests for data validation components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tempfile

from reinforcestrategycreator_pipeline.src.data.validator import (
    ValidationStatus,
    ValidationResult,
    ValidatorBase,
    MissingValueValidator,
    OutlierValidator,
    DataTypeValidator,
    RangeValidator,
    DataValidator
)
from reinforcestrategycreator_pipeline.src.config.models import ValidationConfig


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_init(self):
        """Test ValidationResult initialization."""
        result = ValidationResult("TestValidator")
        
        assert result.validator_name == "TestValidator"
        assert result.status == ValidationStatus.PASSED
        assert result.messages == []
        assert result.details == {}
        assert isinstance(result.timestamp, datetime)
    
    def test_add_error(self):
        """Test adding error to validation result."""
        result = ValidationResult("TestValidator")
        result.add_error("Test error message", column="col1", error_type="missing")
        
        assert result.status == ValidationStatus.FAILED
        assert len(result.messages) == 1
        assert "ERROR: Test error message" in result.messages
        assert result.details["column"] == "col1"
        assert result.details["error_type"] == "missing"
    
    def test_add_warning(self):
        """Test adding warning to validation result."""
        result = ValidationResult("TestValidator")
        result.add_warning("Test warning message", threshold=0.1)
        
        assert result.status == ValidationStatus.WARNING
        assert len(result.messages) == 1
        assert "WARNING: Test warning message" in result.messages
        assert result.details["threshold"] == 0.1
    
    def test_add_warning_after_error(self):
        """Test that error status is not overridden by warning."""
        result = ValidationResult("TestValidator")
        result.add_error("Error message")
        result.add_warning("Warning message")
        
        assert result.status == ValidationStatus.FAILED  # Should remain FAILED
        assert len(result.messages) == 2
    
    def test_add_info(self):
        """Test adding info message."""
        result = ValidationResult("TestValidator")
        result.add_info("Info message", checked_columns=5)
        
        assert result.status == ValidationStatus.PASSED  # Should remain PASSED
        assert len(result.messages) == 1
        assert "INFO: Info message" in result.messages
        assert result.details["checked_columns"] == 5
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult("TestValidator")
        result.add_error("Error message")
        result.add_info("Info message", count=10)
        
        result_dict = result.to_dict()
        
        assert result_dict["validator_name"] == "TestValidator"
        assert result_dict["status"] == "failed"
        assert len(result_dict["messages"]) == 2
        assert result_dict["details"]["count"] == 10
        assert "timestamp" in result_dict


class TestMissingValueValidator:
    """Test cases for MissingValueValidator."""
    
    @pytest.fixture
    def sample_data_no_missing(self):
        """Create sample data without missing values."""
        return pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        return pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],  # 20% missing
            'col2': ['a', None, 'c', None, 'e'],  # 40% missing
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]  # 0% missing
        })
    
    def test_no_missing_values(self, sample_data_no_missing):
        """Test validation with no missing values."""
        validator = MissingValueValidator(threshold=0.1)
        result = validator.validate(sample_data_no_missing)
        
        assert result.status == ValidationStatus.PASSED
        assert any("Checked 3 columns" in msg for msg in result.messages)
    
    def test_missing_values_below_threshold(self, sample_data_with_missing):
        """Test validation with missing values below threshold."""
        validator = MissingValueValidator(threshold=0.5)  # 50% threshold
        result = validator.validate(sample_data_with_missing)
        
        # Should have warnings but not errors
        assert result.status == ValidationStatus.WARNING
        assert any("WARNING" in msg and "col1" in msg for msg in result.messages)
        assert any("WARNING" in msg and "col2" in msg for msg in result.messages)
    
    def test_missing_values_above_threshold(self, sample_data_with_missing):
        """Test validation with missing values above threshold."""
        validator = MissingValueValidator(threshold=0.3)  # 30% threshold
        result = validator.validate(sample_data_with_missing)
        
        # col2 has 40% missing, should fail
        assert result.status == ValidationStatus.FAILED
        assert any("ERROR" in msg and "col2" in msg and "40.00%" in msg for msg in result.messages)
        assert any("WARNING" in msg and "col1" in msg for msg in result.messages)
    
    def test_specific_columns(self, sample_data_with_missing):
        """Test validation on specific columns only."""
        validator = MissingValueValidator(threshold=0.1, columns=['col1', 'col3'])
        result = validator.validate(sample_data_with_missing)
        
        # Should check only col1 and col3, not col2
        assert any("col1" in msg for msg in result.messages)
        assert not any("col2" in msg for msg in result.messages)
        assert any("Checked 2 columns" in msg for msg in result.messages)
    
    def test_all_missing_values(self):
        """Test validation with column having all missing values."""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [1, 2, 3]
        })
        
        validator = MissingValueValidator(threshold=0.5)
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.FAILED
        assert any("100.00%" in msg for msg in result.messages)
    
    def test_get_params(self):
        """Test get_params method."""
        validator = MissingValueValidator(threshold=0.2, columns=['a', 'b'])
        params = validator.get_params()
        
        assert params['threshold'] == 0.2
        assert params['columns'] == ['a', 'b']


class TestOutlierValidator:
    """Test cases for OutlierValidator."""
    
    @pytest.fixture
    def sample_data_normal(self):
        """Create normally distributed data."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal1': np.random.randn(100),
            'normal2': np.random.randn(100) * 2 + 5,
            'category': ['A', 'B', 'C'] * 33 + ['A']
        })
    
    @pytest.fixture
    def sample_data_with_outliers(self):
        """Create data with outliers."""
        np.random.seed(42)
        data = pd.DataFrame({
            'col1': np.concatenate([np.random.randn(95), [10, -10, 15, -15, 20]]),  # 5% outliers
            'col2': np.concatenate([np.random.randn(90), np.array([50] * 10)]),  # 10% outliers
            'col3': np.random.randn(100)  # No outliers
        })
        return data
    
    def test_iqr_method_no_outliers(self, sample_data_normal):
        """Test IQR method with no outliers."""
        validator = OutlierValidator(method='iqr', threshold=1.5)
        result = validator.validate(sample_data_normal)
        
        assert result.status == ValidationStatus.PASSED
        assert any("Checked 2 columns" in msg for msg in result.messages)
    
    def test_iqr_method_with_outliers(self, sample_data_with_outliers):
        """Test IQR method with outliers."""
        validator = OutlierValidator(method='iqr', threshold=1.5)
        result = validator.validate(sample_data_with_outliers)
        
        # Should detect outliers in col1 and col2
        assert result.status == ValidationStatus.WARNING
        assert any("col1" in msg and "outliers" in msg for msg in result.messages)
        assert any("col2" in msg and "outliers" in msg for msg in result.messages)
    
    def test_zscore_method(self, sample_data_with_outliers):
        """Test Z-score method."""
        validator = OutlierValidator(method='zscore', threshold=3)
        result = validator.validate(sample_data_with_outliers)
        
        # Should detect outliers
        assert result.status in [ValidationStatus.WARNING, ValidationStatus.PASSED]
        assert any("zscore" in str(result.details) for msg in result.messages)
    
    def test_specific_columns(self, sample_data_with_outliers):
        """Test outlier detection on specific columns."""
        validator = OutlierValidator(method='iqr', threshold=1.5, columns=['col1'])
        result = validator.validate(sample_data_with_outliers)
        
        # Should check only col1
        assert any("col1" in msg for msg in result.messages if "outliers" in msg)
        assert not any("col2" in msg for msg in result.messages)
        assert any("Checked 1 columns" in msg for msg in result.messages)
    
    def test_high_outlier_fraction_warning(self):
        """Test warning when outlier fraction is high."""
        # Create data with >5% outliers
        data = pd.DataFrame({
            'col1': np.concatenate([np.zeros(90), np.ones(10) * 100])  # 10% outliers
        })
        
        validator = OutlierValidator(method='iqr', threshold=1.5)
        result = validator.validate(data)
        
        assert result.status == ValidationStatus.WARNING
        assert any("WARNING" in msg and "10.00%" in msg for msg in result.messages)
    
    def test_invalid_method(self):
        """Test with invalid outlier detection method."""
        validator = OutlierValidator(method='invalid_method')
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        result = validator.validate(df)
        # Should handle gracefully, likely no outliers detected
        assert result.status == ValidationStatus.PASSED
    
    def test_get_params(self):
        """Test get_params method."""
        validator = OutlierValidator(method='zscore', threshold=2.5, columns=['x', 'y'])
        params = validator.get_params()
        
        assert params['method'] == 'zscore'
        assert params['threshold'] == 2.5
        assert params['columns'] == ['x', 'y']


class TestDataTypeValidator:
    """Test cases for DataTypeValidator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various types."""
        return pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'date_col': pd.date_range('2023-01-01', periods=5),
            'mixed_col': [1, 'a', 2.5, True, None]
        })
    
    def test_correct_types(self, sample_data):
        """Test validation with correct types."""
        expected_types = {
            'int_col': 'int64',
            'float_col': 'float64',
            'str_col': 'object',
            'date_col': 'datetime64[ns]'
        }
        
        validator = DataTypeValidator(expected_types=expected_types)
        result = validator.validate(sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert any("Validated data types for 5 columns" in msg for msg in result.messages)
    
    def test_incorrect_types(self, sample_data):
        """Test validation with incorrect types."""
        expected_types = {
            'int_col': 'float64',  # Wrong: expecting float but got int
            'str_col': 'int64',    # Wrong: expecting int but got object
            'float_col': 'float64'  # Correct
        }
        
        validator = DataTypeValidator(expected_types=expected_types)
        result = validator.validate(sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert any("ERROR" in msg and "int_col" in msg for msg in result.messages)
        assert any("ERROR" in msg and "str_col" in msg for msg in result.messages)
    
    def test_numeric_type_compatibility(self, sample_data):
        """Test numeric type compatibility checking."""
        expected_types = {
            'int_col': 'numeric',
            'float_col': 'numeric',
            'str_col': 'numeric'  # Should fail
        }
        
        validator = DataTypeValidator(expected_types=expected_types)
        result = validator.validate(sample_data)
        
        # int and float should pass for 'numeric', str should fail
        assert result.status == ValidationStatus.FAILED
        assert any("ERROR" in msg and "str_col" in msg for msg in result.messages)
    
    def test_datetime_type_compatibility(self, sample_data):
        """Test datetime type compatibility."""
        expected_types = {
            'date_col': 'datetime',
            'str_col': 'datetime'  # Should fail
        }
        
        validator = DataTypeValidator(expected_types=expected_types)
        result = validator.validate(sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert any("ERROR" in msg and "str_col" in msg for msg in result.messages)
    
    def test_missing_required_columns(self):
        """Test validation when required columns are missing."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        expected_types = {
            'col1': 'int64',
            'col2': 'float64',  # Missing column
            'col3': 'object'    # Missing column
        }
        
        validator = DataTypeValidator(expected_types=expected_types)
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.FAILED
        assert any("Missing required columns" in msg for msg in result.messages)
        assert any("col2" in str(result.details) for msg in result.messages)
        assert any("col3" in str(result.details) for msg in result.messages)
    
    def test_no_expected_types(self):
        """Test validation with no expected types specified."""
        validator = DataTypeValidator()
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.PASSED
        assert any("Validated data types for 1 columns" in msg for msg in result.messages)
    
    def test_get_params(self):
        """Test get_params method."""
        expected_types = {'col1': 'int64', 'col2': 'float64'}
        validator = DataTypeValidator(expected_types=expected_types)
        params = validator.get_params()
        
        assert params['expected_types'] == expected_types


class TestRangeValidator:
    """Test cases for RangeValidator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for range testing."""
        return pd.DataFrame({
            'temperature': [15, 20, 25, 30, 35],  # Celsius
            'humidity': [30, 45, 60, 75, 90],     # Percentage
            'pressure': [1010, 1013, 1015, 1018, 1020],  # hPa
            'invalid': [-10, 0, 50, 100, 150]    # Out of expected range
        })
    
    def test_values_within_range(self, sample_data):
        """Test validation when all values are within range."""
        ranges = {
            'temperature': (10, 40),
            'humidity': (20, 95),
            'pressure': (1000, 1030)
        }
        
        validator = RangeValidator(ranges=ranges)
        result = validator.validate(sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert any("within expected range" in msg for msg in result.messages)
    
    def test_values_outside_range(self, sample_data):
        """Test validation when values are outside range."""
        ranges = {
            'temperature': (20, 30),  # 15 and 35 are outside
            'humidity': (40, 80),     # 30 and 90 are outside
            'invalid': (0, 100)       # -10 and 150 are outside
        }
        
        validator = RangeValidator(ranges=ranges)
        result = validator.validate(sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert any("ERROR" in msg and "temperature" in msg for msg in result.messages)
        assert any("ERROR" in msg and "humidity" in msg for msg in result.messages)
        assert any("ERROR" in msg and "invalid" in msg for msg in result.messages)
    
    def test_minimum_violation(self):
        """Test detection of minimum value violations."""
        df = pd.DataFrame({'col1': [-5, 0, 5, 10]})
        ranges = {'col1': (0, 10)}
        
        validator = RangeValidator(ranges=ranges)
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.FAILED
        assert any("minimum value -5" in msg for msg in result.messages)
    
    def test_maximum_violation(self):
        """Test detection of maximum value violations."""
        df = pd.DataFrame({'col1': [0, 5, 10, 15]})
        ranges = {'col1': (0, 10)}
        
        validator = RangeValidator(ranges=ranges)
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.FAILED
        assert any("maximum value 15" in msg for msg in result.messages)
    
    def test_missing_columns(self):
        """Test validation when specified columns don't exist."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        ranges = {
            'col1': (0, 5),
            'col2': (0, 10)  # Column doesn't exist
        }
        
        validator = RangeValidator(ranges=ranges)
        result = validator.validate(df)
        
        # Should validate col1 but skip col2
        assert any("col1" in msg and "within expected range" in msg for msg in result.messages)
        assert not any("col2" in msg for msg in result.messages)
    
    def test_no_ranges_specified(self):
        """Test validation with no ranges specified."""
        validator = RangeValidator()
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        result = validator.validate(df)
        
        assert result.status == ValidationStatus.PASSED
        assert any("Validated ranges for 0 columns" in msg for msg in result.messages)
    
    def test_get_params(self):
        """Test get_params method."""
        ranges = {'col1': (0, 10), 'col2': (-5, 5)}
        validator = RangeValidator(ranges=ranges)
        params = validator.get_params()
        
        assert params['ranges'] == ranges


class TestDataValidator:
    """Test cases for DataValidator orchestrator."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample validation configuration."""
        return ValidationConfig(
            check_missing_values=True,
            missing_value_threshold=0.1,
            check_outliers=True,
            outlier_method='iqr',
            outlier_threshold=1.5,
            check_data_types=True,
            expected_types={'col1': 'numeric', 'col2': 'object'},
            check_ranges=True,
            value_ranges={'col1': (0, 100)}
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'col1': np.concatenate([np.random.randint(0, 100, 95), [np.nan] * 5]),
            'col2': ['A', 'B', 'C'] * 33 + ['A'],
            'col3': np.random.randn(100)
        })
    
    def test_init_with_config(self, sample_config):
        """Test initialization with configuration."""
        validator = DataValidator(config=sample_config)
        
        assert len(validator.validators) == 4  # All checks enabled
        assert any(isinstance(v, MissingValueValidator) for v in validator.validators)
        assert any(isinstance(v, OutlierValidator) for v in validator.validators)
        assert any(isinstance(v, DataTypeValidator) for v in validator.validators)
        assert any(isinstance(v, RangeValidator) for v in validator.validators)
    
    def test_init_without_config(self):
        """Test initialization without configuration."""
        validator = DataValidator()
        
        assert len(validator.validators) == 0
    
    def test_partial_config(self):
        """Test initialization with partial configuration."""
        config = ValidationConfig(
            check_missing_values=True,
            check_outliers=False,
            check_data_types=False,
            check_ranges=False
        )
        validator = DataValidator(config=config)
        
        assert len(validator.validators) == 1
        assert isinstance(validator.validators[0], MissingValueValidator)
    
    def test_add_validator(self):
        """Test adding custom validator."""
        validator = DataValidator()
        custom_validator = MissingValueValidator(threshold=0.05)
        
        validator.add_validator(custom_validator)
        
        assert len(validator.validators) == 1
        assert validator.validators[0] == custom_validator
    
    def test_validate_all_pass(self, sample_data):
        """Test validation when all checks pass."""
        # Create data without missing values for a true "pass" status
        clean_data = sample_data.dropna()
        
        config = ValidationConfig(
            check_missing_values=True,
            missing_value_threshold=0.1,
            check_outliers=False,  # Disable outlier check as random data may have outliers
            check_data_types=False,
            check_ranges=False
        )
        validator = DataValidator(config=config)
        
        report = validator.validate(clean_data)
        
        assert report['overall_status'] == 'passed'
        assert report['validators_run'] == 1  # Only missing value check
        assert len(report['results']) == 1  # Only one validator result
    
    def test_validate_with_failures(self, sample_data):
        """Test validation with some failures."""
        config = ValidationConfig(
            check_missing_values=True,
            missing_value_threshold=0.01,  # Very low threshold, will fail
            check_outliers=True,
            check_data_types=True,
            expected_types={'col1': 'object'},  # Wrong type, will fail
            check_ranges=False
        )
        validator = DataValidator(config=config)
        
        report = validator.validate(sample_data)
        
        assert report['overall_status'] == 'failed'
        assert report['validators_run'] == 3
        assert any(r['status'] == 'failed' for r in report['results'])
    
    def test_validate_with_warnings(self, sample_data):
        """Test validation with warnings."""
        config = ValidationConfig(
            check_missing_values=True,
            missing_value_threshold=0.1,  # Will generate warning
            check_outliers=False,
            check_data_types=False,
            check_ranges=False
        )
        validator = DataValidator(config=config)
        
        report = validator.validate(sample_data)
        
        assert report['overall_status'] in ['warning', 'passed']
        assert 'timestamp' in report
        assert 'data_shape' in report
        assert report['data_shape'] == (100, 3)
    
    def test_validator_error_handling(self):
        """Test error handling when validator fails."""
        # Create a custom validator that raises an exception
        class FailingValidator(ValidatorBase):
            def validate(self, data):
                raise ValueError("Test error")
            
            def get_params(self):
                return {}
        
        validator = DataValidator()
        validator.add_validator(FailingValidator())
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        report = validator.validate(df)
        
        assert report['overall_status'] == 'failed'
        assert any("Validator error: Test error" in r['messages'][0] for r in report['results'])
    
    def test_save_report(self, sample_config, sample_data):
        """Test saving validation report."""
        validator = DataValidator(config=sample_config)
        report = validator.validate(sample_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            validator.save_report(report, f.name)
            
            # Read and verify saved report
            with open(f.name, 'r') as rf:
                saved_report = json.load(rf)
            
            assert saved_report['overall_status'] == report['overall_status']
            assert saved_report['validators_run'] == report['validators_run']
            assert len(saved_report['results']) == len(report['results'])


class MockValidator(ValidatorBase):
    """Mock validator for testing abstract base class."""
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        result = ValidationResult("MockValidator")
        result.add_info("Mock validation completed")
        return result
    
    def get_params(self) -> dict:
        return {'mock': True}


class TestValidatorBase:
    """Test cases for ValidatorBase abstract class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            ValidatorBase()
    
    def test_mock_implementation(self):
        """Test mock implementation of abstract class."""
        validator = MockValidator()
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = validator.validate(df)
        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.PASSED
        
        params = validator.get_params()
        assert params == {'mock': True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])