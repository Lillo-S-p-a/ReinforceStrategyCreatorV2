"""Main configuration manager that orchestrates loading, validation, and access."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from .loader import ConfigLoader
from .validator import ConfigValidator
from .models import PipelineConfig, EnvironmentType


class ConfigManager:
    """Central configuration manager for the pipeline."""
    
    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        environment: Optional[Union[str, EnvironmentType]] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Base directory for configuration files. This will be used as the
                        project root for resolving relative config paths by the ConfigLoader.
                        If None, ConfigLoader defaults to CWD.
            environment: Environment to use (development, staging, production)
        """
        self.config_dir = Path(config_dir).resolve() if config_dir else Path("configs").resolve() # Keep self.config_dir for now, might be used elsewhere
        self.environment = self._resolve_environment(environment)
        
        # If config_dir is provided, it acts as the root for config loading for the loader.
        # Otherwise, ConfigLoader will default its project_root to Path.cwd().
        loader_project_root = Path(config_dir).resolve() if config_dir else None
        self.loader = ConfigLoader(project_root=loader_project_root)
        
        self.validator = ConfigValidator(model_class=PipelineConfig)
        self._config: Optional[PipelineConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment: Optional[Union[str, EnvironmentType]] = None,
        validate: bool = True
    ) -> PipelineConfig:
        """
        Load and validate configuration.
        
        Args:
            config_path: Path to configuration file (relative to config_dir)
            environment: Environment override
            validate: Whether to validate the configuration
            
        Returns:
            Validated PipelineConfig object
            
        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        # Use provided environment or fall back to instance environment
        env = self._resolve_environment(environment) or self.environment
        
        # Default config path if not provided
        if not config_path:
            config_path = "base/pipeline.yaml"
        
        # Load configuration with environment overrides
        self._raw_config = self.loader.load_with_overrides(
            base_config_path=config_path,
            environment=env.value if env else None
        )
        
        # Add environment to config if not present
        if 'environment' not in self._raw_config and env:
            self._raw_config['environment'] = env.value
        
        # Validate configuration
        if validate:
            is_valid, validated_config, errors = self.validator.validate(self._raw_config)
            
            if not is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(errors)
                raise ValueError(error_msg)
            
            self._config = validated_config
        else:
            # Create config object without validation
            self._config = PipelineConfig(**self._raw_config)
        
        return self._config
    
    def get_config(self) -> PipelineConfig:
        """
        Get the current configuration.
        
        Returns:
            Current PipelineConfig object
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Get the raw configuration dictionary.
        
        Returns:
            Raw configuration dictionary
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if not self._raw_config:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._raw_config
    
    def reload_config(self) -> PipelineConfig:
        """
        Reload the configuration from disk.
        
        Returns:
            Reloaded PipelineConfig object
        """
        if not self._config:
            raise RuntimeError("No configuration to reload. Call load_config() first.")
        
        # Reload using the same parameters
        return self.load_config(validate=True)
    
    def save_config(
        self,
        config: Optional[Union[Dict[str, Any], PipelineConfig]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if not provided)
            path: Path to save to (uses original path if not provided)
        """
        # Determine what to save
        if config is None:
            if self._raw_config:
                config_dict = self._raw_config
            else:
                raise RuntimeError("No configuration to save")
        elif isinstance(config, PipelineConfig):
            config_dict = config.model_dump()
        else:
            config_dict = config
        
        # Determine where to save
        if path is None:
            path = "base/pipeline.yaml"
        
        # Save configuration
        self.loader.save_yaml(config_dict, path)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid, _, errors = self.validator.validate(config)
        return is_valid, errors
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required configuration fields.
        
        Returns:
            List of required field names
        """
        return self.validator.get_required_fields()
    
    def get_field_info(self, field_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a configuration field.
        
        Args:
            field_path: Dot-separated path to field (e.g., "training.batch_size")
            
        Returns:
            Field information dictionary or None
        """
        parts = field_path.split('.')
        
        if len(parts) == 1:
            return self.validator.get_field_info(parts[0])
        
        # For nested fields, we need to traverse the model structure
        # This is a simplified implementation
        return None
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> PipelineConfig:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
            validate: Whether to validate after update
            
        Returns:
            Updated PipelineConfig object
            
        Raises:
            RuntimeError: If no configuration is loaded
            ValueError: If validation fails
        """
        if not self._raw_config:
            raise RuntimeError("No configuration loaded to update")
        
        # Apply updates to raw config
        self._apply_updates(self._raw_config, updates)
        
        # Validate if requested
        if validate:
            is_valid, validated_config, errors = self.validator.validate(self._raw_config)
            
            if not is_valid:
                error_msg = "Configuration validation failed after update:\n" + "\n".join(errors)
                raise ValueError(error_msg)
            
            self._config = validated_config
        else:
            self._config = PipelineConfig(**self._raw_config)
        
        return self._config
    
    def _resolve_environment(
        self,
        environment: Optional[Union[str, EnvironmentType]]
    ) -> Optional[EnvironmentType]:
        """Resolve environment from string or enum."""
        if environment is None:
            # Try to get from environment variable
            env_str = os.environ.get('PIPELINE_ENVIRONMENT')
            if env_str:
                try:
                    return EnvironmentType(env_str.lower())
                except ValueError:
                    pass
            return None
        
        if isinstance(environment, EnvironmentType):
            return environment
        
        try:
            return EnvironmentType(environment.lower())
        except ValueError:
            raise ValueError(f"Invalid environment: {environment}")
    
    def _apply_updates(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively apply updates to configuration."""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys
                parts = key.split('.', 1)
                if parts[0] not in config:
                    config[parts[0]] = {}
                self._apply_updates(config[parts[0]], {parts[1]: value})
            elif key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Merge nested dictionaries
                self._apply_updates(config[key], value)
            else:
                # Direct assignment
                config[key] = value
    
    @classmethod
    def from_file(
        cls,
        config_file: Union[str, Path],
        environment: Optional[Union[str, EnvironmentType]] = None
    ) -> 'ConfigManager':
        """
        Create ConfigManager instance and load configuration from file.
        
        Args:
            config_file: Path to configuration file
            environment: Environment to use
            
        Returns:
            ConfigManager instance with loaded configuration
        """
        config_path = Path(config_file)
        config_dir = config_path.parent.parent  # Assume configs/base/file.yaml structure
        
        manager = cls(config_dir=config_dir, environment=environment)
        manager.load_config(config_path=config_path.relative_to(config_dir))
        
        return manager