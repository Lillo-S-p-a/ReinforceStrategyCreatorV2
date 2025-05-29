"""Integration tests for ConfigManager focusing on its interaction with collaborators."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.config.models import PipelineConfig, EnvironmentType, ModelConfig, TrainingConfig # Added TrainingConfig
from reinforcestrategycreator_pipeline.src.config.loader import ConfigLoader
from reinforcestrategycreator_pipeline.src.config.validator import ConfigValidator


@pytest.fixture
def mock_loader():
    """Fixture for a mocked ConfigLoader."""
    loader = MagicMock(spec=ConfigLoader)
    # Configure default return values for common scenarios
    loader.load_with_overrides.return_value = {
        'name': 'test_pipeline_loaded',
        'version': '1.0.0',
        'environment': 'development',
        'model': {'model_type': 'DQN', 'hyperparameters': {'layers': [64, 32]}},
        'training': {'episodes': 50, 'batch_size': 16}
    }
    return loader

@pytest.fixture
def mock_validator():
    """Fixture for a mocked ConfigValidator."""
    validator = MagicMock(spec=ConfigValidator)
    # Configure default return values for common scenarios
    # Successful validation
    mock_pipeline_config = PipelineConfig(
        name='test_pipeline_validated',
        version='1.0.0',
        environment=EnvironmentType.DEVELOPMENT,
        model=ModelConfig(model_type='DQN', hyperparameters={'layers': [64, 32]}),
        training=TrainingConfig(episodes=50, batch_size=16) # Added TrainingConfig
    )
    validator.validate.return_value = (True, mock_pipeline_config, [])
    return validator

@pytest.fixture
def manager_with_mocks(mock_loader, mock_validator):
    """Fixture for ConfigManager with mocked loader and validator."""
    # Patch the __init__ of ConfigManager to inject mocks
    # This is a bit more involved if ConfigManager instantiates them directly.
    # A cleaner way would be if ConfigManager accepted loader/validator in __init__.
    # For now, let's assume we can patch where they are used or instantiated.
    # Or, we can pass them if the constructor allows.
    # ConfigManager's __init__ creates its own loader and validator.
    # We need to patch these instances after ConfigManager is created.

    manager = ConfigManager(config_dir="dummy_configs")
    manager.loader = mock_loader
    manager.validator = mock_validator
    return manager


class TestConfigManagerIntegration:
    """Test cases for ConfigManager integration."""

    def test_load_config_successful_validation(self, manager_with_mocks, mock_loader, mock_validator):
        """Test load_config calls loader and validator correctly on success."""
        base_config_path = "base/pipeline.yaml"
        environment = EnvironmentType.DEVELOPMENT

        # Expected raw config from loader
        raw_config_from_loader = {
            'name': 'test_pipeline_loaded',
            'version': '1.0.0',
            'environment': 'development',
            'model': {'model_type': 'DQN', 'hyperparameters': {'layers': [64, 32]}},
            'training': {'episodes': 50, 'batch_size': 16}
        }
        mock_loader.load_with_overrides.return_value = raw_config_from_loader

        # Expected validated config from validator
        validated_config_model = PipelineConfig(
            name='test_pipeline_validated',
            version='1.0.0',
            environment=EnvironmentType.DEVELOPMENT,
            model=ModelConfig(model_type='DQN', hyperparameters={'layers': [64, 32]}),
            training=TrainingConfig(episodes=50, batch_size=16) # Added TrainingConfig
        )
        mock_validator.validate.return_value = (True, validated_config_model, [])

        loaded_config = manager_with_mocks.load_config(
            config_path=base_config_path,
            environment=environment
        )

        # Assert loader was called correctly
        mock_loader.load_with_overrides.assert_called_once_with(
            base_config_path=base_config_path,
            environment=environment.value
        )

        # Assert validator was called correctly
        # The raw_config passed to validator might have 'environment' added by ConfigManager
        expected_raw_config_for_validator = raw_config_from_loader.copy()
        # ConfigManager adds environment if not present, which it is in this mock
        mock_validator.validate.assert_called_once_with(expected_raw_config_for_validator)

        # Assert the final loaded config is the one from the validator
        assert loaded_config == validated_config_model
        assert loaded_config.name == 'test_pipeline_validated'

    def test_load_config_validation_fails(self, manager_with_mocks, mock_loader, mock_validator):
        """Test load_config handles validation failure from validator."""
        base_config_path = "base/pipeline.yaml"
        environment = EnvironmentType.PRODUCTION

        raw_config_from_loader = {'name': 'bad_config'}
        mock_loader.load_with_overrides.return_value = raw_config_from_loader

        validation_errors = ["Missing field 'version'", "Invalid model_type"]
        mock_validator.validate.return_value = (False, None, validation_errors)

        with pytest.raises(ValueError) as exc_info:
            manager_with_mocks.load_config(
                config_path=base_config_path,
                environment=environment
            )

        mock_loader.load_with_overrides.assert_called_once_with(
            base_config_path=base_config_path,
            environment=environment.value
        )
        # ConfigManager adds environment if not present
        expected_raw_config_for_validator = raw_config_from_loader.copy()
        expected_raw_config_for_validator['environment'] = environment.value
        mock_validator.validate.assert_called_once_with(expected_raw_config_for_validator)

        assert "Configuration validation failed" in str(exc_info.value)
        for error in validation_errors:
            assert error in str(exc_info.value)

    def test_save_config_uses_loader(self, manager_with_mocks, mock_loader):
        """Test save_config calls loader.save_yaml correctly."""
        # First, ensure there's a raw_config to save by calling load_config
        # We can let load_config use its default mock behaviors for this setup
        manager_with_mocks.load_config() # This will populate _raw_config

        config_to_save = manager_with_mocks.get_raw_config() # Use the loaded raw config
        save_path = "output/my_config.yaml"

        manager_with_mocks.save_config(path=save_path)

        mock_loader.save_yaml.assert_called_once_with(
            config_to_save,
            save_path
        )

    def test_save_config_with_provided_config_dict(self, manager_with_mocks, mock_loader):
        """Test save_config with an explicit config dictionary."""
        config_dict_to_save = {"key": "value", "setting": 123}
        save_path = "output/another_config.yaml"

        # No need to call load_config if we provide the config directly
        manager_with_mocks.save_config(config=config_dict_to_save, path=save_path)

        mock_loader.save_yaml.assert_called_once_with(
            config_dict_to_save,
            save_path
        )

    def test_save_config_with_provided_pipeline_config_object(self, manager_with_mocks, mock_loader):
        """Test save_config with an explicit PipelineConfig object."""
        pipeline_config_to_save = PipelineConfig(
            name='saved_pipeline',
            version='2.0',
            environment=EnvironmentType.STAGING,
            model=ModelConfig(model_type='PPO', hyperparameters={'actor_lr': 0.001}),
            training=TrainingConfig(episodes=1000, batch_size=128) # Added TrainingConfig
        )
        save_path = "output/pipeline_object_config.yaml"

        manager_with_mocks.save_config(config=pipeline_config_to_save, path=save_path)

        # ConfigManager calls .dict() on the PipelineConfig object
        mock_loader.save_yaml.assert_called_once_with(
            pipeline_config_to_save.model_dump(),
            save_path
        )

    def test_update_config_successful_validation(self, manager_with_mocks, mock_validator):
        """Test update_config calls validator correctly on success."""
        # Load initial config (uses mocks for loader and validator)
        initial_config = manager_with_mocks.load_config()
        
        # Setup validator mock for the update call
        updated_raw_config_after_apply = manager_with_mocks.get_raw_config().copy() # Get a copy of raw config
        # Simulate applying updates to this copy for validator expectation
        updates_to_apply = {'training.episodes': 75, 'model.hyperparameters.layers': [128, 128]}
        
        # Manually apply updates to the copied raw_config to set expectation for validator
        updated_raw_config_after_apply['training']['episodes'] = 75
        updated_raw_config_after_apply['model']['hyperparameters']['layers'] = [128, 128]

        expected_validated_config_after_update = PipelineConfig(
            name=initial_config.name, # Assuming name doesn't change
            version=initial_config.version,
            environment=initial_config.environment,
            model=ModelConfig(model_type=initial_config.model.model_type, hyperparameters={'layers': [128, 128]}),
            training=TrainingConfig(episodes=75, batch_size=initial_config.training.batch_size)
        )
        mock_validator.validate.return_value = (True, expected_validated_config_after_update, [])

        updated_config = manager_with_mocks.update_config(updates_to_apply)

        # Validator should be called with the raw config *after* updates are applied internally
        mock_validator.validate.assert_called_with(updated_raw_config_after_apply)
        assert updated_config == expected_validated_config_after_update
        assert updated_config.training.episodes == 75
        assert updated_config.model.hyperparameters['layers'] == [128, 128]

    def test_update_config_validation_fails(self, manager_with_mocks, mock_validator):
        """Test update_config handles validation failure after update."""
        manager_with_mocks.load_config() # Load initial

        updates_to_apply = {'model.model_type': 'INVALID_TYPE'}
        
        # Simulate raw config after internal update by ConfigManager
        raw_config_after_internal_update = manager_with_mocks.get_raw_config().copy()
        raw_config_after_internal_update['model']['model_type'] = 'INVALID_TYPE'

        validation_errors = ["Invalid model_type 'INVALID_TYPE'"]
        mock_validator.validate.return_value = (False, None, validation_errors)

        with pytest.raises(ValueError) as exc_info:
            manager_with_mocks.update_config(updates_to_apply)

        mock_validator.validate.assert_called_with(raw_config_after_internal_update)
        assert "Configuration validation failed after update" in str(exc_info.value)
        assert validation_errors[0] in str(exc_info.value)