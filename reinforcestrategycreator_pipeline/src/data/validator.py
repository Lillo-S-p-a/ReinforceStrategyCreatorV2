"""Data Validator for quality checks and validation."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum

from ..config.models import ValidationConfig


logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, validator_name: str):
        """Initialize validation result.
        
        Args:
            validator_name: Name of the validator
        """
        self.validator_name = validator_name
        self.status = ValidationStatus.PASSED
        self.messages: List[str] = []
        self.details: Dict[str, Any] = {}
        self.timestamp = datetime.now()
        
    def add_error(self, message: str, **details):
        """Add an error to the validation result.
        
        Args:
            message: Error message
            **details: Additional error details
        """
        self.status = ValidationStatus.FAILED
        self.messages.append(f"ERROR: {message}")
        self.details.update(details)
        
    def add_warning(self, message: str, **details):
        """Add a warning to the validation result.
        
        Args:
            message: Warning message
            **details: Additional warning details
        """
        if self.status != ValidationStatus.FAILED:
            self.status = ValidationStatus.WARNING
        self.messages.append(f"WARNING: {message}")
        self.details.update(details)
        
    def add_info(self, message: str, **details):
        """Add an info message to the validation result.
        
        Args:
            message: Info message
            **details: Additional info details
        """
        self.messages.append(f"INFO: {message}")
        self.details.update(details)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'validator_name': self.validator_name,
            'status': self.status.value,
            'messages': self.messages,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class ValidatorBase(ABC):
    """Base class for all data validators."""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ValidationResult object
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass


class MissingValueValidator(ValidatorBase):
    """Validator for checking missing values."""
    
    def __init__(self, threshold: float = 0.1, columns: Optional[List[str]] = None):
        """Initialize missing value validator.
        
        Args:
            threshold: Maximum allowed fraction of missing values (0-1)
            columns: Specific columns to check. If None, checks all columns.
        """
        self.threshold = threshold
        self.columns = columns
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check for missing values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(self.__class__.__name__)
        
        # Determine columns to check
        cols_to_check = self.columns if self.columns else data.columns.tolist()
        
        missing_info = {}
        for col in cols_to_check:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                missing_fraction = missing_count / len(data)
                
                if missing_count > 0:
                    missing_info[col] = {
                        'count': int(missing_count),
                        'fraction': float(missing_fraction)
                    }
                    
                    if missing_fraction > self.threshold:
                        result.add_error(
                            f"Column '{col}' has {missing_fraction:.2%} missing values, "
                            f"exceeding threshold of {self.threshold:.2%}",
                            column=col,
                            missing_count=missing_count,
                            missing_fraction=missing_fraction
                        )
                    else:
                        result.add_warning(
                            f"Column '{col}' has {missing_count} missing values ({missing_fraction:.2%})",
                            column=col,
                            missing_count=missing_count,
                            missing_fraction=missing_fraction
                        )
                        
        result.add_info(
            f"Checked {len(cols_to_check)} columns for missing values",
            missing_summary=missing_info
        )
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters."""
        return {
            'threshold': self.threshold,
            'columns': self.columns
        }


class OutlierValidator(ValidatorBase):
    """Validator for detecting outliers."""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, 
                 columns: Optional[List[str]] = None):
        """Initialize outlier validator.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            columns: Specific columns to check. If None, checks all numeric columns.
        """
        self.method = method
        self.threshold = threshold
        self.columns = columns
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check for outliers in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(self.__class__.__name__)
        
        # Determine columns to check
        if self.columns:
            numeric_cols = [col for col in self.columns if col in data.columns]
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
        outlier_info = {}
        for col in numeric_cols:
            if self.method == 'iqr':
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.threshold * iqr
                upper_bound = q3 + self.threshold * iqr
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            elif self.method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                z_scores = np.abs((data[col] - mean) / std)
                outliers = z_scores > self.threshold
            else:
                continue
                
            outlier_count = outliers.sum()
            outlier_fraction = outlier_count / len(data)
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': int(outlier_count),
                    'fraction': float(outlier_fraction),
                    'method': self.method
                }
                
                if outlier_fraction > 0.05:  # More than 5% outliers
                    result.add_warning(
                        f"Column '{col}' has {outlier_count} outliers ({outlier_fraction:.2%})",
                        column=col,
                        outlier_count=outlier_count,
                        outlier_fraction=outlier_fraction
                    )
                else:
                    result.add_info(
                        f"Column '{col}' has {outlier_count} outliers ({outlier_fraction:.2%})",
                        column=col,
                        outlier_count=outlier_count,
                        outlier_fraction=outlier_fraction
                    )
                    
        result.add_info(
            f"Checked {len(numeric_cols)} columns for outliers using {self.method} method",
            outlier_summary=outlier_info
        )
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'columns': self.columns
        }


class DataTypeValidator(ValidatorBase):
    """Validator for checking data types."""
    
    def __init__(self, expected_types: Optional[Dict[str, str]] = None):
        """Initialize data type validator.
        
        Args:
            expected_types: Dictionary mapping column names to expected data types
        """
        self.expected_types = expected_types or {}
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check data types in the DataFrame.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(self.__class__.__name__)
        
        actual_types = {}
        type_mismatches = {}
        
        for col in data.columns:
            actual_type = str(data[col].dtype)
            actual_types[col] = actual_type
            
            if col in self.expected_types:
                expected_type = self.expected_types[col]
                if not self._types_compatible(actual_type, expected_type):
                    type_mismatches[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
                    result.add_error(
                        f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                        column=col,
                        expected_type=expected_type,
                        actual_type=actual_type
                    )
                    
        # Check for required columns
        missing_columns = set(self.expected_types.keys()) - set(data.columns)
        if missing_columns:
            result.add_error(
                f"Missing required columns: {missing_columns}",
                missing_columns=list(missing_columns)
            )
            
        result.add_info(
            f"Validated data types for {len(data.columns)} columns",
            actual_types=actual_types,
            type_mismatches=type_mismatches
        )
        
        return result
    
    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type.
        
        Args:
            actual: Actual data type
            expected: Expected data type
            
        Returns:
            True if types are compatible
        """
        # Simple compatibility check - can be extended
        if expected == 'numeric':
            return 'int' in actual or 'float' in actual
        elif expected == 'datetime':
            return 'datetime' in actual
        elif expected == 'object':
            return actual == 'object'
        else:
            return actual == expected
    
    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters."""
        return {
            'expected_types': self.expected_types
        }


class RangeValidator(ValidatorBase):
    """Validator for checking value ranges."""
    
    def __init__(self, ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize range validator.
        
        Args:
            ranges: Dictionary mapping column names to (min, max) tuples
        """
        self.ranges = ranges or {}
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Check if values are within expected ranges.
        
        Args:
            data: Input DataFrame
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult(self.__class__.__name__)
        
        range_violations = {}
        
        for col, (min_val, max_val) in self.ranges.items():
            if col in data.columns:
                col_min = data[col].min()
                col_max = data[col].max()
                
                violations = []
                if col_min < min_val:
                    violations.append(f"minimum value {col_min} < {min_val}")
                if col_max > max_val:
                    violations.append(f"maximum value {col_max} > {max_val}")
                    
                if violations:
                    range_violations[col] = {
                        'expected_range': (min_val, max_val),
                        'actual_range': (float(col_min), float(col_max)),
                        'violations': violations
                    }
                    result.add_error(
                        f"Column '{col}' has values outside expected range: {', '.join(violations)}",
                        column=col,
                        expected_min=min_val,
                        expected_max=max_val,
                        actual_min=float(col_min),
                        actual_max=float(col_max)
                    )
                else:
                    result.add_info(
                        f"Column '{col}' values are within expected range [{min_val}, {max_val}]",
                        column=col,
                        actual_range=(float(col_min), float(col_max))
                    )
                    
        result.add_info(
            f"Validated ranges for {len(self.ranges)} columns",
            range_violations=range_violations
        )
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get validator parameters."""
        return {
            'ranges': self.ranges
        }


class DataValidator:
    """Main data validator that orchestrates multiple validation checks."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.validators: List[ValidatorBase] = []
        
        # Initialize validators based on config
        if config:
            self._initialize_validators()
            
    def _initialize_validators(self):
        """Initialize validators based on configuration."""
        if self.config.check_missing_values:
            self.validators.append(
                MissingValueValidator(
                    threshold=self.config.missing_value_threshold
                )
            )
            
        if self.config.check_outliers:
            self.validators.append(
                OutlierValidator(
                    method=self.config.outlier_method,
                    threshold=self.config.outlier_threshold
                )
            )
            
        if self.config.check_data_types and self.config.expected_types:
            self.validators.append(
                DataTypeValidator(
                    expected_types=self.config.expected_types
                )
            )
            
        if self.config.check_ranges and self.config.value_ranges:
            self.validators.append(
                RangeValidator(
                    ranges=self.config.value_ranges
                )
            )
            
    def add_validator(self, validator: ValidatorBase):
        """Add a custom validator to the pipeline.
        
        Args:
            validator: Validator instance
        """
        self.validators.append(validator)
        
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all validators on the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        results = []
        overall_status = ValidationStatus.PASSED
        
        for validator in self.validators:
            try:
                result = validator.validate(data)
                results.append(result.to_dict())
                
                # Update overall status
                if result.status == ValidationStatus.FAILED:
                    overall_status = ValidationStatus.FAILED
                elif result.status == ValidationStatus.WARNING and overall_status != ValidationStatus.FAILED:
                    overall_status = ValidationStatus.WARNING
                    
            except Exception as e:
                logger.error(f"Error in {validator.__class__.__name__}: {str(e)}")
                error_result = ValidationResult(validator.__class__.__name__)
                error_result.add_error(f"Validator error: {str(e)}")
                results.append(error_result.to_dict())
                overall_status = ValidationStatus.FAILED
                
        validation_report = {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'validators_run': len(self.validators),
            'results': results
        }
        
        logger.info(f"Validation completed with status: {overall_status.value}")
        return validation_report
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save validation report to file.
        
        Args:
            report: Validation report
            filepath: Path to save report
        """
        import json
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            """Custom JSON encoder for numpy types."""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
            
        logger.info(f"Saved validation report to {filepath}")