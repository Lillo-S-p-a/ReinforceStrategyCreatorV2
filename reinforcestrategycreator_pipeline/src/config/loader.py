"""Configuration loader with YAML support and environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
from copy import deepcopy


class ConfigLoader:
    """Load and process configuration from YAML files."""
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            base_path: Base path for configuration files. Defaults to current directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
        """
        path = self._resolve_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")
    
    def load_with_overrides(
        self,
        base_config_path: Union[str, Path],
        override_config_path: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load base configuration and apply overrides.
        
        Args:
            base_config_path: Path to base configuration file
            override_config_path: Path to override configuration file
            environment: Environment name for automatic override lookup
            
        Returns:
            Merged configuration dictionary
        """
        # Load base configuration
        config = self.load_yaml(base_config_path)
        
        # Apply environment-specific overrides if specified
        if environment and not override_config_path:
            env_config_path = self._get_environment_config_path(base_config_path, environment)
            resolved_env_path = self._resolve_path(env_config_path)
            if resolved_env_path.exists():
                override_config_path = env_config_path
        
        # Apply overrides if available
        if override_config_path:
            override_config = self.load_yaml(override_config_path)
            config = self._merge_configs(config, override_config)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        return config
    
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path relative to base path."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        return path
    
    def _get_environment_config_path(
        self,
        base_config_path: Union[str, Path],
        environment: str
    ) -> Path:
        """
        Get the path for environment-specific configuration.
        
        Args:
            base_config_path: Path to base configuration
            environment: Environment name
            
        Returns:
            Path to environment-specific configuration
        """
        base_path = Path(base_config_path)
        base_dir = base_path.parent
        
        # Check for environment-specific file in environments subdirectory
        env_dir = base_dir.parent / "environments"
        env_file = env_dir / f"{environment}.yaml"
        
        return env_file
    
    def _merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge override configuration into base configuration.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        merged = deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override the value
                merged[key] = deepcopy(value)
        
        return merged
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Environment variables are specified as ${VAR_NAME} or ${VAR_NAME:default_value}.
        
        Args:
            config: Configuration data (can be dict, list, or scalar)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {
                key: self._substitute_env_vars(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_env_vars(config)
        else:
            return config
    
    def _substitute_string_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in a string value.
        
        Args:
            value: String value potentially containing env var references
            
        Returns:
            String with environment variables substituted
        """
        def replace_env_var(match):
            env_var_spec = match.group(1)
            
            # Check for default value syntax (VAR_NAME:default_value)
            if ':' in env_var_spec:
                var_name, default_value = env_var_spec.split(':', 1)
            else:
                var_name = env_var_spec
                default_value = None
            
            # Get environment variable value
            env_value = os.environ.get(var_name.strip())
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Keep the original placeholder if no value found
                return match.group(0)
        
        return self.ENV_VAR_PATTERN.sub(replace_env_var, value)
    
    def save_yaml(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            file_path: Path to save the YAML file
        """
        path = self._resolve_path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)