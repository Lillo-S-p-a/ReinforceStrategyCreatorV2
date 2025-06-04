"""Unit tests for ConfigLoader."""

import os
import tempfile
from pathlib import Path
import pytest
import yaml

from reinforcestrategycreator_pipeline.src.config.loader import ConfigLoader


class TestConfigLoader:
    """Test cases for ConfigLoader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def loader(self, temp_dir):
        """Create a ConfigLoader instance."""
        return ConfigLoader(base_path=temp_dir)
    
    def test_load_yaml_simple(self, loader, temp_dir):
        """Test loading a simple YAML file."""
        # Create test YAML file
        config_data = {
            'name': 'test_pipeline',
            'version': '1.0.0',
            'data': {
                'source': 'csv',
                'path': '/data/test.csv'
            }
        }
        
        config_file = temp_dir / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and verify
        loaded_config = loader.load_yaml(config_file)
        assert loaded_config == config_data
    
    def test_load_yaml_file_not_found(self, loader):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load_yaml('non_existent.yaml')
    
    def test_load_yaml_invalid_syntax(self, loader, temp_dir):
        """Test loading file with invalid YAML syntax."""
        config_file = temp_dir / 'invalid.yaml'
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: syntax: :")
        
        with pytest.raises(yaml.YAMLError):
            loader.load_yaml(config_file)
    
    def test_environment_variable_substitution(self, loader, temp_dir):
        """Test environment variable substitution."""
        # Set environment variables
        os.environ['TEST_API_KEY'] = 'secret123'
        os.environ['TEST_ENDPOINT'] = 'https://api.test.com'
        
        config_data = {
            'api_key': '${TEST_API_KEY}',
            'endpoint': '${TEST_ENDPOINT}',
            'default_value': '${MISSING_VAR:default}',
            'nested': {
                'key': '${TEST_API_KEY}'
            }
        }
        
        config_file = temp_dir / 'env_test.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and verify substitution
        loaded_config = loader.load_yaml(config_file)
        processed_config = loader._substitute_env_vars(loaded_config)
        
        assert processed_config['api_key'] == 'secret123'
        assert processed_config['endpoint'] == 'https://api.test.com'
        assert processed_config['default_value'] == 'default'
        assert processed_config['nested']['key'] == 'secret123'
        
        # Clean up
        del os.environ['TEST_API_KEY']
        del os.environ['TEST_ENDPOINT']
    
    def test_merge_configs(self, loader):
        """Test configuration merging."""
        base_config = {
            'name': 'base',
            'version': '1.0.0',
            'data': {
                'source': 'csv',
                'path': '/base/path.csv',
                'cache': True
            },
            'model': {
                'type': 'DQN',
                'layers': [128, 64]
            }
        }
        
        override_config = {
            'name': 'override',
            'data': {
                'path': '/override/path.csv',
                'validation': True
            },
            'model': {
                'layers': [256, 128, 64]
            }
        }
        
        merged = loader._merge_configs(base_config, override_config)
        
        assert merged['name'] == 'override'
        assert merged['version'] == '1.0.0'
        assert merged['data']['source'] == 'csv'
        assert merged['data']['path'] == '/override/path.csv'
        assert merged['data']['cache'] is True
        assert merged['data']['validation'] is True
        assert merged['model']['type'] == 'DQN'
        assert merged['model']['layers'] == [256, 128, 64]
    
    def test_load_with_overrides(self, loader, temp_dir):
        """Test loading with environment-specific overrides."""
        # Create base config
        base_dir = temp_dir / 'base'
        base_dir.mkdir()
        base_config = {
            'name': 'pipeline',
            'environment': 'development',
            'data': {'source': 'csv', 'cache': True}
        }
        base_file = base_dir / 'config.yaml'
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment override
        env_dir = temp_dir / 'environments'
        env_dir.mkdir()
        env_config = {
            'environment': 'production',
            'data': {'cache': False, 'validation': True}
        }
        env_file = env_dir / 'production.yaml'
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        # Load with override
        result = loader.load_with_overrides(
            base_config_path=base_file,
            environment='production'
        )
        
        assert result['name'] == 'pipeline'
        assert result['environment'] == 'production'
        assert result['data']['source'] == 'csv'
        assert result['data']['cache'] is False
        assert result['data']['validation'] is True
    
    def test_save_yaml(self, loader, temp_dir):
        """Test saving configuration to YAML."""
        config_data = {
            'name': 'test_save',
            'version': '2.0.0',
            'nested': {
                'key1': 'value1',
                'key2': [1, 2, 3]
            }
        }
        
        save_path = temp_dir / 'saved_config.yaml'
        loader.save_yaml(config_data, save_path)
        
        # Verify file exists and content is correct
        assert save_path.exists()
        
        with open(save_path, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data == config_data