"""Unit tests for ConfigValidator."""

import pytest
from pydantic import BaseModel, Field

from reinforcestrategycreator_pipeline.src.config.validator import ConfigValidator
from reinforcestrategycreator_pipeline.src.config.models import PipelineConfig, ModelType


class TestConfigValidator:
    """Test cases for ConfigValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConfigValidator instance."""
        return ConfigValidator(model_class=PipelineConfig)
    
    def test_validate_valid_config(self, validator):
        """Test validation of a valid configuration."""
        valid_config = {
            'name': 'test_pipeline',
            'version': '1.0.0',
            'environment': 'development',
            'model': {
                'model_type': 'DQN',
                'hyperparameters': {
                    'hidden_layers': [128, 64],
                    'learning_rate': 0.001
                }
            }
        }
        
        is_valid, validated_model, errors = validator.validate(valid_config)
        
        assert is_valid is True
        assert validated_model is not None
        assert errors == []
        assert validated_model.name == 'test_pipeline'
        assert validated_model.model.model_type == ModelType.DQN
    
    def test_validate_invalid_config_missing_required(self, validator):
        """Test validation with missing required fields."""
        invalid_config = {
            'version': '1.0.0',
            # Missing 'name' and 'model' which are required
        }
        
        is_valid, validated_model, errors = validator.validate(invalid_config)
        
        assert is_valid is False
        assert validated_model is None
        assert len(errors) > 0
        assert any('name' in error for error in errors)
        assert any('model' in error for error in errors)
    
    def test_validate_invalid_config_wrong_type(self, validator):
        """Test validation with wrong field types."""
        invalid_config = {
            'name': 'test_pipeline',
            'version': 123,  # Should be string
            'model': {
                'model_type': 'InvalidModel',  # Invalid enum value
                'hyperparameters': {}
            }
        }
        
        is_valid, validated_model, errors = validator.validate(invalid_config)
        
        assert is_valid is False
        assert validated_model is None
        assert len(errors) > 0
    
    def test_validate_partial(self, validator):
        """Test partial validation of specific fields."""
        config = {
            'name': 'test_pipeline',
            'version': '2.0.0',
            'random_seed': 42,
            'invalid_field': 'should_be_ignored'
        }
        
        is_valid, validated_fields, errors = validator.validate_partial(
            config, 
            fields=['name', 'version', 'random_seed', 'unknown_field']
        )
        
        assert is_valid is False  # Because of unknown_field
        assert 'name' in validated_fields
        assert 'version' in validated_fields
        assert 'random_seed' in validated_fields
        assert validated_fields['name'] == 'test_pipeline'
        assert any('unknown_field' in error for error in errors)
    
    def test_get_defaults(self, validator):
        """Test getting default values."""
        defaults = validator.get_defaults()
        
        assert isinstance(defaults, dict)
        assert 'version' in defaults
        assert defaults['version'] == '1.0.0'
        assert 'environment' in defaults
        assert 'random_seed' in defaults
        assert defaults['random_seed'] == 42
    
    def test_get_required_fields(self, validator):
        """Test getting required fields."""
        required_fields = validator.get_required_fields()
        
        assert isinstance(required_fields, list)
        assert 'name' in required_fields
        assert 'model' in required_fields
        assert 'version' not in required_fields  # Has default
    
    def test_get_field_info(self, validator):
        """Test getting field information."""
        # Test existing field
        field_info = validator.get_field_info('name')
        assert field_info is not None
        assert field_info['required'] is True
        assert 'type' in field_info
        assert 'description' in field_info
        
        # Test non-existent field
        field_info = validator.get_field_info('non_existent')
        assert field_info is None
    
    def test_merge_with_defaults(self, validator):
        """Test merging configuration with defaults."""
        partial_config = {
            'name': 'my_pipeline',
            'model': {
                'model_type': 'PPO'
            }
        }
        
        merged = validator.merge_with_defaults(partial_config)
        
        assert merged['name'] == 'my_pipeline'
        assert merged['version'] == '1.0.0'  # From defaults
        assert merged['random_seed'] == 42  # From defaults
        assert merged['model']['model_type'] == 'PPO'
    
    def test_custom_validator_model(self):
        """Test ConfigValidator with a custom model."""
        
        class CustomConfig(BaseModel):
            name: str = Field(..., description="Config name")
            value: int = Field(default=10, ge=0, le=100)
            enabled: bool = Field(default=True)
        
        validator = ConfigValidator(model_class=CustomConfig)
        
        # Test valid config
        valid_config = {'name': 'test', 'value': 50}
        is_valid, model, errors = validator.validate(valid_config)
        assert is_valid is True
        assert model.value == 50
        assert model.enabled is True  # Default
        
        # Test invalid config (value out of range)
        invalid_config = {'name': 'test', 'value': 150}
        is_valid, model, errors = validator.validate(invalid_config)
        assert is_valid is False
        assert any('value' in error for error in errors)
    
    def test_format_validation_errors(self, validator):
        """Test error message formatting."""
        invalid_config = {
            'name': '',  # Empty string
            'model': {
                'model_type': 'INVALID',
                'hyperparameters': 'should_be_dict'  # Wrong type
            }
        }
        
        is_valid, _, errors = validator.validate(invalid_config)
        
        assert is_valid is False
        assert len(errors) > 0
        # Check that errors are formatted as readable strings
        assert all(isinstance(error, str) for error in errors)
        assert any('model_type' in error for error in errors)