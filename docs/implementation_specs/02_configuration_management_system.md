# Configuration Management System: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Configuration Management System of the Trading Model Optimization Pipeline. It defines how configurations are structured, validated, loaded, and accessed throughout the system to ensure consistency, type safety, and reproducibility of experiments.

## 2. Component Responsibilities

The Configuration Management System is responsible for:

- Loading configuration from various sources (files, environment variables, command line)
- Validating configuration against schema definitions
- Providing typed access to configuration parameters
- Managing configuration versioning
- Supporting environment-specific configurations (development, testing, production)
- Enabling dynamic updates to certain configuration parameters
- Linking experiment runs with their specific configurations

## 3. Class Structure

```
trading_optimization/
└── config/
    ├── __init__.py
    ├── manager.py
    ├── validator.py
    ├── schema.py
    ├── loader.py
    ├── utils.py
    └── exceptions.py
```

### 3.1 Class Definitions

#### 3.1.1 ConfigManager

```python
class ConfigManager:
    """
    Central configuration manager that handles loading, validation, and access to configurations.
    
    This is the main entry point for the configuration system and should be used by other components
    to access configuration values.
    """
    
    def __init__(self, config_path: Optional[str] = None, env: str = "development"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration file. If None, uses default locations.
            env: Environment to use (development, testing, production)
        """
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], env: str = "development") -> "ConfigManager":
        """Create a ConfigManager instance from a dictionary."""
        
    def load(self) -> None:
        """Load configuration from all sources and validate it."""
        
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its path.
        
        Args:
            path: Dot-notation path to the configuration value (e.g., "model.training.learning_rate")
            default: Default value to return if the path doesn't exist
            
        Returns:
            The configuration value
        """
        
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            path: Dot-notation path to the configuration value
            value: New value to set
        """
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the current configuration to a dictionary."""
        
    def save(self, path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path where to save the configuration
        """
        
    def validate(self) -> None:
        """Validate the current configuration against its schema."""
        
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the current configuration."""
        
    def create_experiment_config(self) -> str:
        """
        Create a versioned snapshot of the current configuration for an experiment.
        
        Returns:
            The ID of the created configuration snapshot
        """
```

#### 3.1.2 ConfigValidator

```python
class ConfigValidator:
    """
    Validates configuration against JSON Schema definitions.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the validator with a schema.
        
        Args:
            schema_path: Path to the JSON Schema file. If None, uses default schema.
        """
        
    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "ConfigValidator":
        """Create a validator from a schema dictionary."""
        
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            A tuple of (is_valid, error_messages)
        """
        
    def validate_section(self, section: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a specific section of the configuration.
        
        Args:
            section: Name of the section to validate
            config: Configuration dictionary to validate
            
        Returns:
            A tuple of (is_valid, error_messages)
        """
```

#### 3.1.3 ConfigLoader

```python
class ConfigLoader:
    """
    Loads configuration from various sources and merges them according to precedence.
    """
    
    def __init__(self, env: str = "development"):
        """
        Initialize the config loader.
        
        Args:
            env: Environment to use (development, testing, production)
        """
        
    def load_file(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            The loaded configuration as a dictionary
        """
        
    def load_env_vars(self, prefix: str = "TRADING_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables to consider
            
        Returns:
            The loaded configuration as a dictionary
        """
        
    def load_default(self) -> Dict[str, Any]:
        """
        Load the default configuration.
        
        Returns:
            The default configuration as a dictionary
        """
        
    def load_all(self, custom_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from all sources and merge them.
        
        Precedence (highest to lowest):
        1. Command-line arguments (if applicable)
        2. Environment variables
        3. Custom configuration file (if provided)
        4. Environment-specific configuration file
        5. Default configuration file
        
        Args:
            custom_path: Path to a custom configuration file
            
        Returns:
            The merged configuration as a dictionary
        """
```

#### 3.1.4 ExperimentConfig

```python
class ExperimentConfig:
    """
    Represents a versioned configuration snapshot for an experiment.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_id: str):
        """
        Initialize an experiment configuration.
        
        Args:
            config: The configuration dictionary
            experiment_id: ID of the experiment
        """
        
    @property
    def id(self) -> str:
        """Get the unique ID of this configuration snapshot."""
        
    def save(self) -> None:
        """Save the configuration to the database."""
        
    @classmethod
    def load(cls, config_id: str) -> "ExperimentConfig":
        """
        Load an experiment configuration from the database.
        
        Args:
            config_id: ID of the configuration to load
            
        Returns:
            The loaded configuration
        """
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
```

## 4. Configuration Structure

The configuration system uses YAML as the primary format for configuration files. The structure is organized into logical sections corresponding to different components of the system.

### 4.1 Base Configuration Structure

```yaml
# Base configuration structure
version: "1.0"

# Experiment information
experiment:
  name: "default"
  description: "Default configuration"
  tags: []

# Data management configuration
data:
  # Data sources
  sources:
    - type: "yfinance"
      tickers: ["SPY"]
      start_date: "2015-01-01"
      end_date: "2023-12-31"
      timeframe: "1d"
    
  # Data processing settings
  processing:
    fill_missing: "ffill"
    normalization: "z_score"
    
  # Feature engineering
  features:
    - name: "rsi"
      params:
        window: 14
    - name: "macd"
      params:
        fast: 12
        slow: 26
        signal: 9
    - name: "bollinger_bands"
      params:
        window: 20
        num_std: 2

# Model configuration
model:
  type: "dqn"  # Type of model to use
  
  # Network architecture
  network:
    architecture: "mlp"
    hidden_layers: [128, 64]
    activation: "relu"
    dropout: 0.2
  
  # Training parameters
  training:
    batch_size: 64
    learning_rate: 0.0001
    optimizer: "adam"
    episodes: 1000
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 0.995

# Hyperparameter tuning configuration
hyperparameter_tuning:
  algorithm: "bayesian"  # Optimization algorithm
  metric: "sharpe_ratio"  # Metric to optimize
  num_trials: 50
  parallel_trials: 4
  
  # Search space
  search_space:
    learning_rate:
      range: [0.00001, 0.01]
      distribution: "log_uniform"
    hidden_layers:
      values: [[64, 32], [128, 64], [256, 128, 64]]
    dropout:
      range: [0.0, 0.5]
      distribution: "uniform"
  
  # Scheduler
  scheduler:
    type: "asha"  # Asynchronous Successive Halving Algorithm
    max_iterations: 100
    grace_period: 10

# Evaluation configuration
evaluation:
  metrics: ["sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate"]
  
  # Walk-forward analysis
  walk_forward:
    window_size: 90
    step_size: 30
    windows: 5
  
  # Out-of-sample testing
  out_of_sample:
    test_size: 0.2
    recent_period: "1Y"

# Risk management configuration
risk_management:
  max_drawdown_limit: 0.15
  position_sizing: "kelly"
  stop_loss_pct: 0.05
  max_leverage: 2.0

# Monitoring configuration
monitoring:
  comparison_frequency: "daily"
  alert_threshold: 0.1
  metrics_to_monitor: ["pnl", "drawdown", "win_rate"]

# Paper trading configuration
paper_trading:
  capital: 100000
  simulation_days: 30
  evaluation_criteria:
    min_sharpe: 1.0
    max_drawdown: 0.1
    min_trades: 20

# Live trading configuration
live_trading:
  capital: 100000
  max_position_pct: 0.1
  api:
    provider: "alpaca"
    key_id: ${ALPACA_KEY_ID}
    secret_key: ${ALPACA_SECRET_KEY}

# Database configuration
database:
  engine: "postgresql"
  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  name: ${DB_NAME:-trading_optimization}
  username: ${DB_USERNAME:-postgres}
  password: ${DB_PASSWORD}

# Logging configuration
logging:
  level: ${LOG_LEVEL:-info}
  format: "${timestamp} [${level}] ${message}"
  output:
    - type: "console"
    - type: "file"
      path: "logs/trading_optimization.log"
      rotation: "daily"
```

### 4.2 Environment-Specific Configuration

Configuration files can be environment-specific, following the naming pattern `config.{env}.yaml`:

- `config.development.yaml`: Development environment settings
- `config.testing.yaml`: Test environment settings
- `config.production.yaml`: Production environment settings

Environment-specific configurations are merged with the base configuration, with the environment-specific values taking precedence.

## 5. Configuration Validation

### 5.1 JSON Schema

The configuration is validated against a JSON Schema definition to ensure type safety and required values. The schema is defined in `config/schema/config_schema.json`.

Example schema excerpt:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "experiment", "data", "model"],
  "properties": {
    "version": {
      "type": "string",
      "description": "Configuration schema version"
    },
    "experiment": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the experiment"
        },
        "description": {
          "type": "string",
          "description": "Description of the experiment"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags associated with the experiment"
        }
      }
    },
    "data": {
      "type": "object",
      "required": ["sources"],
      "properties": {
        "sources": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["type", "tickers"],
            "properties": {
              "type": {
                "type": "string",
                "enum": ["yfinance", "alpaca", "csv", "database"],
                "description": "Type of data source"
              },
              "tickers": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of ticker symbols"
              }
            }
          }
        }
      }
    }
  }
}
```

## 6. Configuration Loading and Access

### 6.1 Loading Process

1. Default configuration is loaded from `config/default.yaml`
2. Environment-specific configuration is loaded from `config/{env}.yaml`
3. Custom configuration file is loaded if specified
4. Environment variables override file-based configuration
5. Command-line arguments override all other sources
6. The merged configuration is validated against the schema
7. The validated configuration is made available through the `ConfigManager` singleton

### 6.2 Access Patterns

Configuration is accessed through the `ConfigManager` instance:

```python
from trading_optimization.config import ConfigManager

# Get the singleton instance
config = ConfigManager.instance()

# Access configuration values using dot notation
learning_rate = config.get("model.training.learning_rate")
batch_size = config.get("model.training.batch_size")

# Provide defaults for optional values
max_episodes = config.get("model.training.max_episodes", 1000)

# Set configuration values (if dynamic updates are allowed)
config.set("model.training.learning_rate", 0.0002)
```

## 7. Configuration Versioning

### 7.1 Experiment Configuration Versioning

Each experiment run is associated with a versioned configuration snapshot:

1. When an experiment is started, a deep copy of the current configuration is created
2. The configuration is assigned a unique ID (UUID)
3. The configuration is stored in the database with metadata (timestamp, experiment ID)
4. All components reference this configuration ID for reproducibility
5. Previous configuration versions can be loaded to reproduce experiments exactly

### 7.2 Schema Versioning

The configuration schema itself is versioned to track evolution:

```python
def validate_config_version(config: Dict[str, Any]) -> bool:
    """
    Check if the configuration version is compatible with the current schema.
    
    Args:
        config: The configuration to check
        
    Returns:
        True if compatible, False otherwise
    """
    config_version = config.get("version", "1.0")
    # Semantic versioning comparison
    return semver.compare(config_version, "1.0") >= 0 and semver.compare(config_version, "2.0") < 0
```

## 8. Dynamic Configuration Updates

Some parts of the configuration can be updated at runtime, while others are immutable after initialization:

### 8.1 Immutable Configuration Sections

- `experiment`: To ensure experiment consistency
- `data.sources`: To ensure data consistency
- `model.network`: To ensure model architecture consistency

### 8.2 Mutable Configuration Sections

- `model.training`: Training hyperparameters can be adjusted
- `evaluation`: Evaluation settings can be modified
- `monitoring`: Monitoring thresholds can be updated
- `logging`: Logging levels can be changed

## 9. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. YAML parsing library is available (PyYAML)
3. JSON Schema validation library is available (jsonschema)

## 10. Implementation Sequence

1. Create basic configuration loader
2. Implement JSON Schema validator 
3. Define base configuration schema
4. Implement ConfigManager class
5. Add environment-specific configuration support
6. Implement configuration versioning
7. Add dynamic configuration update capabilities
8. Create configuration documentation generators