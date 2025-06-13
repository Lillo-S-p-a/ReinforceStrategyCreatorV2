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
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            project_root: The root directory for resolving top-level relative config paths.
                          Defaults to current working directory.
        """
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    def load_yaml(self, file_path_str: Union[str, Path],
                  current_file_processing_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from a YAML file, processing includes.
        
        Args:
            file_path_str: Path to the YAML file (string or Path).
            current_file_processing_dir: The directory of the YAML file currently being processed.
                                         Used to resolve relative paths in 'includes' directives.
                                         If None, `file_path_str` is resolved against `self.project_root`.
                                         
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
        """
        path_to_load_obj = Path(file_path_str)

        if path_to_load_obj.is_absolute():
            resolved_path = path_to_load_obj.resolve()
        elif current_file_processing_dir:
            # This is an include, resolve relative to the directory of the file that included it
            resolved_path = (current_file_processing_dir / path_to_load_obj).resolve()
        else:
            # This is a top-level config file, resolve relative to project_root
            resolved_path = (self.project_root / path_to_load_obj).resolve()
            
        if not resolved_path.exists():
            error_msg = f"Configuration file not found: {resolved_path}. " \
                        f"Original path: '{file_path_str}'"
            if current_file_processing_dir:
                error_msg += f", relative to: '{current_file_processing_dir}'"
            else:
                error_msg += f", relative to project root: '{self.project_root}'"
            raise FileNotFoundError(error_msg)
        
        try:
            with open(resolved_path, 'r') as f:
                current_config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {resolved_path}: {e}")

        # Process includes
        final_config = {} # Start with an empty dict for merging includes
        
        # The pop operation modifies current_config_data, so copy its items for iteration if needed,
        # or ensure "includes" is processed before other keys if order matters for direct values.
        # Here, we pop "includes" and then merge the rest of current_config_data, which is fine.
        includes_list = current_config_data.pop("includes", [])
        
        for include_entry_str in includes_list:
            # Includes are resolved relative to the directory of the current file (resolved_path.parent)
            included_config = self.load_yaml(include_entry_str, current_file_processing_dir=resolved_path.parent)
            # Merge included_config into final_config.
            # The new content from included_config will overwrite existing keys in final_config if they conflict.
            final_config = self._merge_configs(final_config, included_config)

        # Merge current file's data (its specific keys will override anything from includes with the same name at the same level)
        final_config = self._merge_configs(final_config, current_config_data)
        
        return final_config

    def load_with_overrides(
        self,
        base_config_path: Union[str, Path], # This path is relative to project_root or absolute
        override_config_path: Optional[Union[str, Path]] = None, # Also relative to project_root or absolute
        environment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load base configuration and apply overrides.
        
        Args:
            base_config_path: Path to base configuration file (relative to project_root or absolute)
            override_config_path: Path to override configuration file (relative to project_root or absolute)
            environment: Environment name for automatic override lookup
            
        Returns:
            Merged configuration dictionary
        """
        # Load base configuration. current_file_processing_dir is None for top-level files.
        config = self.load_yaml(base_config_path, current_file_processing_dir=None)
        
        # Apply environment-specific overrides if specified
        if environment and not override_config_path:
            # _get_environment_config_path constructs a path relative to the base_config_path's location
            # For example, if base_config_path is "configs/dev/main.yaml",
            # it looks for "configs/environments/prod.yaml"
            env_config_path_str = self._get_environment_config_path(base_config_path, environment)
            
            # Check existence of this constructed path.
            # Since _get_environment_config_path might return a path like ../environments/env.yaml
            # we resolve it relative to the original base_config_path's directory.
            original_base_path_obj = Path(base_config_path)
            anchor_for_env_override = (self.project_root / original_base_path_obj.parent).resolve() if not original_base_path_obj.is_absolute() else original_base_path_obj.parent.resolve()
            
            check_path = (anchor_for_env_override / env_config_path_str).resolve()
            
            # Fallback: if env_config_path_str was simple (e.g. "environments/prod.yaml")
            # and not found via anchor, try relative to project_root directly.
            if not check_path.exists() and not Path(env_config_path_str).is_absolute() and ".." not in str(env_config_path_str):
                 simple_project_relative_path = (self.project_root / env_config_path_str).resolve()
                 if simple_project_relative_path.exists():
                     check_path = simple_project_relative_path


            if check_path.exists():
                # If found, use the original string path for load_yaml, which will resolve it
                # relative to project_root (as current_file_processing_dir will be None)
                override_config_path = env_config_path_str
        
        # Apply overrides if available
        if override_config_path:
            # current_file_processing_dir is None for top-level override files.
            override_config = self.load_yaml(override_config_path, current_file_processing_dir=None)
            config = self._merge_configs(config, override_config)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        return config
    
    # _resolve_path method is removed as its logic is now integrated into load_yaml

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