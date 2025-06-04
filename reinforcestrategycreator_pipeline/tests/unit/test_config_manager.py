"""Unit tests for ConfigManager."""

import os
import tempfile
from pathlib import Path
import pytest
import yaml

from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.config.models import PipelineConfig, EnvironmentType


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / 'configs'
            
            # Create directory structure
            (config_dir / 'base').mkdir(parents=True)
            (config_dir / 'environments').mkdir(parents=True)
            
            # Create base config
            base_config = {
                'name': 'test_pipeline',
                'version': '1.0.0',
                'environment': 'development',
                'model': {
                    'model_type': 'DQN',
                    'hyperparameters': {'layers': [128, 64]}
                },
                'training': {
                    'episodes': 100,
                    'batch_size': 32
                }
            }
            
            with open(config_dir / 'base' / 'pipeline.yaml', 'w') as f:
                yaml.dump(base_config, f)
            
            # Create environment configs
            dev_config = {
                'environment': 'development',
                'training': {
                    'episodes': 10
                }
            }
            
            prod_config = {
                'environment': 'production',
                'training': {
                    'episodes': 200,
                    'batch_size': 64
                }
            }
            
            with open(config_dir / 'environments' / 'development.yaml', 'w') as f:
                yaml.dump(dev_config, f)
            
            with open(config_dir / 'environments' / 'production.yaml', 'w') as f:
                yaml.dump(prod_config, f)
            
            yield config_dir
    
    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create a ConfigManager instance."""
        return ConfigManager(config_dir=temp_config_dir)
    
    def test_init_default(self):
        """Test initialization with defaults."""
        manager = ConfigManager()
        assert manager.config_dir == Path('configs')
        assert manager.environment is None
    
    def test_init_with_environment(self):
        """Test initialization with environment."""
        manager = ConfigManager(environment='production')
        assert manager.environment == EnvironmentType.PRODUCTION
        
        # Test with enum
        manager = ConfigManager(environment=EnvironmentType.STAGING)
        assert manager.environment == EnvironmentType.STAGING
    
    def test_init_with_env_var(self):
        """Test initialization from environment variable."""
        os.environ['PIPELINE_ENVIRONMENT'] = 'staging'
        manager = ConfigManager()
        assert manager.environment == EnvironmentType.STAGING
        del os.environ['PIPELINE_ENVIRONMENT']
    
    def test_load_config_default(self, manager):
        """Test loading default configuration."""
        config = manager.load_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.name == 'test_pipeline'
        assert config.version == '1.0.0'
        assert config.model.model_type.value == 'DQN'
    
    def test_load_config_with_environment(self, manager):
        """Test loading configuration with environment override."""
        # Load development config
        dev_config = manager.load_config(environment='development')
        assert dev_config.environment == EnvironmentType.DEVELOPMENT
        assert dev_config.training.episodes == 10  # Overridden
        assert dev_config.training.batch_size == 32  # From base
        
        # Load production config
        prod_config = manager.load_config(environment='production')
        assert prod_config.environment == EnvironmentType.PRODUCTION
        assert prod_config.training.episodes == 200  # Overridden
        assert prod_config.training.batch_size == 64  # Overridden
    
    def test_load_config_validation_error(self, temp_config_dir):
        """Test loading invalid configuration."""
        # Create invalid config
        invalid_config = {
            'version': '1.0.0',
            # Missing required 'name' and 'model'
        }
        
        with open(temp_config_dir / 'base' / 'invalid.yaml', 'w') as f:
            yaml.dump(invalid_config, f)
        
        manager = ConfigManager(config_dir=temp_config_dir)
        
        with pytest.raises(ValueError) as exc_info:
            manager.load_config(config_path='base/invalid.yaml')
        
        assert 'validation failed' in str(exc_info.value).lower()
    
    def test_load_config_without_validation(self, temp_config_dir):
        """Test loading configuration without validation."""
        # Create config with extra fields
        config_with_extras = {
            'name': 'test',
            'model': {'model_type': 'DQN'},
            'extra_field': 'should_not_cause_error'
        }
        
        with open(temp_config_dir / 'base' / 'extras.yaml', 'w') as f:
            yaml.dump(config_with_extras, f)
        
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.load_config(
            config_path='base/extras.yaml',
            validate=False
        )
        
        assert config.name == 'test'
    
    def test_get_config(self, manager):
        """Test getting current configuration."""
        # Should raise error before loading
        with pytest.raises(RuntimeError):
            manager.get_config()
        
        # Load config
        manager.load_config()
        
        # Should return config after loading
        config = manager.get_config()
        assert isinstance(config, PipelineConfig)
        assert config.name == 'test_pipeline'
    
    def test_get_raw_config(self, manager):
        """Test getting raw configuration dictionary."""
        # Should raise error before loading
        with pytest.raises(RuntimeError):
            manager.get_raw_config()
        
        # Load config
        manager.load_config()
        
        # Should return dict after loading
        raw_config = manager.get_raw_config()
        assert isinstance(raw_config, dict)
        assert raw_config['name'] == 'test_pipeline'
    
    def test_reload_config(self, manager):
        """Test reloading configuration."""
        # Should raise error before initial load
        with pytest.raises(RuntimeError):
            manager.reload_config()
        
        # Load config
        config1 = manager.load_config()
        
        # Reload should work
        config2 = manager.reload_config()
        assert config2.name == config1.name
    
    def test_save_config(self, manager, temp_config_dir):
        """Test saving configuration."""
        # Load config first
        manager.load_config()
        
        # Save to new location
        save_path = temp_config_dir / 'saved_config.yaml'
        manager.save_config(path=save_path)
        
        assert save_path.exists()
        
        # Verify saved content
        with open(save_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['name'] == 'test_pipeline'
    
    def test_update_config(self, manager):
        """Test updating configuration."""
        # Load initial config
        manager.load_config()
        original_episodes = manager.get_config().training.episodes
        
        # Update config
        updates = {
            'training.episodes': 500,
            'training.learning_rate': 0.0001
        }
        
        updated_config = manager.update_config(updates)
        
        assert updated_config.training.episodes == 500
        assert updated_config.training.learning_rate == 0.0001
        assert updated_config.training.batch_size == 32  # Unchanged
    
    def test_update_config_validation_error(self, manager):
        """Test updating with invalid values."""
        manager.load_config()
        
        # Try to update with invalid value
        updates = {
            'model.model_type': 'INVALID_MODEL'
        }
        
        with pytest.raises(ValueError) as exc_info:
            manager.update_config(updates)
        
        assert 'validation failed' in str(exc_info.value).lower()
    
    def test_validate_config(self, manager):
        """Test configuration validation."""
        valid_config = {
            'name': 'test',
            'model': {'model_type': 'PPO'}
        }
        
        is_valid, errors = manager.validate_config(valid_config)
        assert is_valid is True
        assert errors == []
        
        invalid_config = {
            'name': 'test'
            # Missing required 'model'
        }
        
        is_valid, errors = manager.validate_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_from_file(self, temp_config_dir):
        """Test creating manager from specific file."""
        config_file = temp_config_dir / 'base' / 'pipeline.yaml'
        
        manager = ConfigManager.from_file(
            config_file,
            environment='production'
        )
        
        config = manager.get_config()
        assert config.name == 'test_pipeline'
        assert config.environment == EnvironmentType.PRODUCTION
        assert config.training.episodes == 200  # From production override
    
    def test_environment_variable_substitution(self, temp_config_dir):
        """Test environment variable substitution in config."""
        # Create config with env vars
        config_with_env = {
            'name': 'env_test',
            'model': {'model_type': 'DQN'},
            'api_key': '${TEST_API_KEY:default_key}',
            'endpoint': '${TEST_ENDPOINT}'
        }
        
        with open(temp_config_dir / 'base' / 'env_test.yaml', 'w') as f:
            yaml.dump(config_with_env, f)
        
        # Set env var
        os.environ['TEST_API_KEY'] = 'secret123'
        
        manager = ConfigManager(config_dir=temp_config_dir)
        manager.load_config(config_path='base/env_test.yaml', validate=False)
        
        raw_config = manager.get_raw_config()
        assert raw_config['api_key'] == 'secret123'
        assert raw_config['endpoint'] == '${TEST_ENDPOINT}'  # Not substituted
        
        # Clean up
        del os.environ['TEST_API_KEY']