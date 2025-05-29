"""Configuration validator using Pydantic models."""

from typing import Dict, Any, Type, List, Optional, Tuple
from pydantic import BaseModel, ValidationError
from .models import PipelineConfig


class ConfigValidator:
    """Validate configuration using Pydantic models."""
    
    def __init__(self, model_class: Type[BaseModel] = PipelineConfig):
        """
        Initialize the configuration validator.
        
        Args:
            model_class: Pydantic model class to use for validation
        """
        self.model_class = model_class
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, Optional[BaseModel], List[str]]:
        """
        Validate configuration against the Pydantic model.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, validated_model, error_messages)
        """
        try:
            validated_model = self.model_class(**config)
            return True, validated_model, []
        except ValidationError as e:
            error_messages = self._format_validation_errors(e)
            return False, None, error_messages
    
    def validate_partial(self, config: Dict[str, Any], fields: List[str]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate only specific fields of the configuration.
        
        Args:
            config: Configuration dictionary
            fields: List of field names to validate
            
        Returns:
            Tuple of (is_valid, validated_fields, error_messages)
        """
        # Create a partial model for validation
        errors = []
        validated_fields = {}
        
        # Check all requested fields
        for field in fields:
            # First check if field exists in model
            field_info = self.model_class.model_fields.get(field)
            if not field_info:
                errors.append(f"Unknown field: {field}")
                continue
                
            if field not in config:
                # Field was requested but not provided in config
                continue
                
            value = config[field]
            try:
                
                # Validate field value
                if hasattr(field_info.annotation, '__origin__'):
                    # Handle generic types
                    validated_value = value
                else:
                    # Validate using field type
                    validated_value = field_info.annotation(**value) if isinstance(value, dict) else value
                
                validated_fields[field] = validated_value
                
            except (ValidationError, TypeError, ValueError) as e:
                errors.append(f"Field '{field}': {str(e)}")
        
        return len(errors) == 0, validated_fields, errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values from the model.
        
        Returns:
            Dictionary of default values
        """
        # Create instance with minimal required fields
        defaults = {}
        
        for field_name, field_info in self.model_class.model_fields.items():
            if field_info.default is not None:
                defaults[field_name] = field_info.default
            elif field_info.default_factory is not None:
                defaults[field_name] = field_info.default_factory()
        
        return defaults
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields from the model.
        
        Returns:
            List of required field names
        """
        required_fields = []
        
        for field_name, field_info in self.model_class.model_fields.items():
            if field_info.is_required():
                required_fields.append(field_name)
        
        return required_fields
    
    def get_field_info(self, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Dictionary with field information or None if field doesn't exist
        """
        field_info = self.model_class.model_fields.get(field_name)
        
        if not field_info:
            return None
        
        return {
            'type': str(field_info.annotation),
            'required': field_info.is_required(),
            'default': field_info.default,
            'description': field_info.description,
            'constraints': self._get_field_constraints(field_info)
        }
    
    def _format_validation_errors(self, validation_error: ValidationError) -> List[str]:
        """
        Format validation errors into readable messages.
        
        Args:
            validation_error: Pydantic ValidationError
            
        Returns:
            List of formatted error messages
        """
        error_messages = []
        
        for error in validation_error.errors():
            location = ' -> '.join(str(loc) for loc in error['loc'])
            message = error['msg']
            error_type = error['type']
            
            if error_type == 'value_error.missing':
                error_messages.append(f"Missing required field: {location}")
            elif error_type == 'type_error':
                error_messages.append(f"Invalid type for {location}: {message}")
            else:
                error_messages.append(f"{location}: {message}")
        
        return error_messages
    
    def _get_field_constraints(self, field_info) -> Dict[str, Any]:
        """
        Extract constraints from field info.
        
        Args:
            field_info: Pydantic field info
            
        Returns:
            Dictionary of constraints
        """
        constraints = {}
        
        # Extract constraints from field validators
        if hasattr(field_info, 'field_info'):
            field_data = field_info.field_info
            if hasattr(field_data, 'ge'):
                constraints['min'] = field_data.ge
            if hasattr(field_data, 'le'):
                constraints['max'] = field_data.le
            if hasattr(field_data, 'min_length'):
                constraints['min_length'] = field_data.min_length
            if hasattr(field_data, 'max_length'):
                constraints['max_length'] = field_data.max_length
        
        return constraints
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge provided configuration with default values.
        
        Args:
            config: Partial configuration dictionary
            
        Returns:
            Complete configuration with defaults applied
        """
        defaults = self.get_defaults()
        merged = defaults.copy()
        
        # Recursively merge configurations
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(merged, config)