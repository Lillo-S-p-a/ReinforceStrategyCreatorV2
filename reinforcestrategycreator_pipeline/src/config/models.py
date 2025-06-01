"""Pydantic models for configuration validation."""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class EnvironmentType(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DataSourceType(str, Enum):
    """Supported data source types."""
    CSV = "csv"
    API = "api"
    DATABASE = "database"
    YFINANCE = "yfinance"


class ModelType(str, Enum):
    """Supported model types."""
    DQN = "DQN"
    PPO = "PPO"
    A2C = "A2C"
    RAINBOW = "Rainbow"
    SAC = "SAC"


class DataConfig(BaseModel):
    """Configuration for data management."""

    source_id: str = Field(
        default="default_data_source", # Provide a default ID
        description="Unique identifier for this data configuration/source"
    )
    
    source_type: DataSourceType = Field(
        default=DataSourceType.CSV,
        description="Type of data source"
    )
    
    source_path: Optional[str] = Field(
        default=None,
        description="Path to data source (for CSV)"
    )
    
    api_endpoint: Optional[str] = Field(
        default=None,
        description="API endpoint (for API source)"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key (supports env var substitution)"
    )
    
    symbols: Optional[List[str]] = Field( # Made optional
        default_factory=list,
        description="List of symbols to fetch (used by some sources, not yfinance)"
    )

    tickers: Optional[Union[str, List[str]]] = Field( # Added for yfinance
        default=None,
        description="Ticker(s) for yfinance source (e.g., 'SPY' or ['SPY', 'AAPL'])"
    )

    period: Optional[str] = Field( # Added for yfinance
        default=None,
        description="Data period for yfinance (e.g., '1y', '5d'). Overridden by start_date/end_date."
    )

    interval: Optional[str] = Field( # Added for yfinance
        default="1d", # Default to daily as per yfinance_source
        description="Data interval for yfinance (e.g., '1d', '1wk', '1m')"
    )
    
    start_date: Optional[str] = Field(
        default=None,
        description="Start date for data (YYYY-MM-DD)"
    )
    
    end_date: Optional[str] = Field(
        default=None,
        description="End date for data (YYYY-MM-DD)"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable data caching"
    )
    
    cache_dir: str = Field(
        default="./cache/data",
        description="Directory for data cache"
    )
    
    validation_enabled: bool = Field(
        default=True,
        description="Enable data validation"
    )
    
    transformation: Optional[TransformationConfig] = Field(
        default=None,
        description="Data transformation configuration"
    )
    
    validation: Optional[ValidationConfig] = Field(
        default=None,
        description="Data validation configuration"
    )
    
    @field_validator("source_path", "api_endpoint", "tickers", "period", "start_date", "end_date")
    def validate_source_specific_fields(cls, v, info):
        """Validate that appropriate fields are provided based on source_type."""
        source_type = info.data.get("source_type")
        field_name = info.field_name

        if source_type == DataSourceType.CSV:
            if field_name == "source_path" and not v:
                raise ValueError("source_path is required for CSV source_type.")
        elif source_type == DataSourceType.API:
            if field_name == "api_endpoint" and not v:
                raise ValueError("api_endpoint is required for API source_type.")
        elif source_type == DataSourceType.YFINANCE:
            # For YFinance, 'tickers' is essential.
            # 'period' or ('start_date' and 'end_date') should be present.
            # This validator runs per field, so we check 'tickers' when it's the current field.
            # A more holistic validation might be needed in a model-level validator if complex interdependencies exist.
            if field_name == "tickers" and not v:
                raise ValueError("tickers are required for YFINANCE source_type.")
            # yfinance library itself handles if period or start/end is missing,
            # but we can add a basic check if 'period' is None and 'start_date' is also None.
            if field_name == "period" and v is None and info.data.get("start_date") is None:
                # This check is a bit tricky in a field validator.
                # A root validator might be better for cross-field dependencies.
                # For now, we'll rely on yfinance's own error handling if insufficient period/date info is given.
                pass
        # No specific validation for DATABASE type here for these fields.
        return v


class ModelConfig(BaseModel):
    """Configuration for model management."""
    
    model_type: ModelType = Field(
        description="Type of model to use"
    )
    
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model hyperparameters"
    )
    
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory for model checkpoints"
    )
    
    save_frequency: int = Field(
        default=10,
        description="Save checkpoint every N episodes"
    )
    
    load_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to checkpoint to load"
    )


class TrainingConfig(BaseModel):
    """Configuration for training."""
    
    episodes: int = Field(
        default=100,
        description="Number of training episodes"
    )
    
    batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate"
    )
    
    gamma: float = Field(
        default=0.99,
        description="Discount factor"
    )
    
    epsilon_start: float = Field(
        default=1.0,
        description="Starting epsilon for exploration"
    )
    
    epsilon_end: float = Field(
        default=0.01,
        description="Ending epsilon for exploration"
    )
    
    epsilon_decay: float = Field(
        default=0.995,
        description="Epsilon decay rate"
    )
    
    replay_buffer_size: int = Field(
        default=10000,
        description="Size of replay buffer"
    )
    
    target_update_frequency: int = Field(
        default=100,
        description="Update target network every N steps"
    )
    
    validation_split: float = Field(
        default=0.2,
        description="Validation data split ratio"
    )
    
    early_stopping_patience: int = Field(
        default=10,
        description="Early stopping patience"
    )
    
    use_tensorboard: bool = Field(
        default=True,
        description="Enable TensorBoard logging"
    )
    
    log_dir: str = Field(
        default="./logs",
        description="Directory for logs"
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    metrics: List[str] = Field(
        default_factory=lambda: ["sharpe_ratio", "total_return", "max_drawdown"],
        description="Metrics to calculate"
    )
    
    benchmark_symbols: List[str] = Field(
        default_factory=lambda: ["SPY"],
        description="Benchmark symbols for comparison"
    )
    
    test_episodes: int = Field(
        default=10,
        description="Number of test episodes"
    )
    
    save_results: bool = Field(
        default=True,
        description="Save evaluation results"
    )
    
    results_dir: str = Field(
        default="./results",
        description="Directory for results"
    )
    
    generate_plots: bool = Field(
        default=True,
        description="Generate visualization plots"
    )


class DeploymentConfig(BaseModel):
    """Configuration for deployment."""
    
    mode: str = Field(
        default="paper_trading",
        description="Deployment mode (paper_trading, live)"
    )
    
    api_endpoint: Optional[str] = Field(
        default=None,
        description="Trading API endpoint"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="Trading API key"
    )
    
    max_positions: int = Field(
        default=10,
        description="Maximum number of positions"
    )
    
    position_size: float = Field(
        default=0.1,
        description="Position size as fraction of portfolio"
    )
    
    risk_limit: float = Field(
        default=0.02,
        description="Maximum risk per trade"
    )
    
    update_frequency: str = Field(
        default="1h",
        description="Model update frequency"
    )


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""
    
    enabled: bool = Field(
        default=True,
        description="Enable monitoring"
    )
    
    datadog_api_key: Optional[str] = Field(
        default=None,
        description="Datadog API key"
    )
    
    datadog_app_key: Optional[str] = Field(
        default=None,
        description="Datadog app key"
    )
    
    metrics_prefix: str = Field(
        default="model_pipeline",
        description="Prefix for metrics"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    alert_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Alert thresholds for metrics"
    )
    
    data_drift: Optional[DataDriftConfig] = Field(
        default=None,
        description="Configuration for data drift detection"
    )
    
    model_drift: Optional[ModelDriftConfig] = Field(
        default=None,
        description="Configuration for model drift detection"
    )

    alert_manager: Optional[AlertManagerConfig] = Field(
        default=None,
        description="Configuration for the alert manager"
    )


class DataDriftDetectionMethod(str, Enum):
    """Supported data drift detection methods."""
    PSI = "psi"  # Population Stability Index
    KS = "ks"    # Kolmogorov-Smirnov
    CHI2 = "chi2" # Chi-Squared


class DataDriftConfig(BaseModel):
    """Configuration for data drift detection."""
    enabled: bool = Field(default=True, description="Enable data drift detection")
    method: DataDriftDetectionMethod = Field(
        default=DataDriftDetectionMethod.PSI,
        description="Method for data drift detection"
    )
    threshold: float = Field(
        default=0.2,
        description="Threshold for triggering a drift alert (method-specific)"
    )
    features_to_monitor: Optional[List[str]] = Field(
        default=None,
        description="Specific features to monitor for drift (None means all)"
    )
    reference_data_window_size: int = Field(
        default=1000,
        description="Number of samples for the reference data window"
    )
    current_data_window_size: int = Field(
        default=100,
        description="Number of samples for the current data window"
    )
    check_frequency_seconds: int = Field(
        default=3600, # 1 hour
        description="How often to check for data drift in seconds"
    )


class ModelDriftDetectionMethod(str, Enum):
    """Supported model drift detection methods."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PREDICTION_CONFIDENCE = "prediction_confidence"


class ModelDriftConfig(BaseModel):
    """Configuration for model drift detection."""
    enabled: bool = Field(default=True, description="Enable model drift detection")
    method: ModelDriftDetectionMethod = Field(
        default=ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION,
        description="Method for model drift detection"
    )
    performance_metric: Optional[str] = Field(
        default="accuracy",
        description="Performance metric to monitor for degradation (e.g., accuracy, f1_score)"
    )
    degradation_threshold: Optional[float] = Field(
        default=0.1,
        description="Relative degradation threshold (e.g., 0.1 for 10% drop)"
    )
    confidence_threshold: Optional[float] = Field(
        default=0.7,
        description="Minimum average prediction confidence"
    )
    check_frequency_seconds: int = Field(
        default=86400, # 24 hours
        description="How often to check for model drift in seconds"
    )


class AlertChannelType(str, Enum):
    """Supported alert channel types."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    DATADOG_EVENT = "datadog_event" # For internal Datadog events/monitors


class AlertChannelConfig(BaseModel):
    """Configuration for a single alert channel."""
    type: AlertChannelType = Field(description="Type of alert channel")
    name: str = Field(description="Unique name for this channel configuration")
    enabled: bool = Field(default=True, description="Enable this alert channel")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Channel-specific configuration (e.g., email_to, slack_webhook_url, pagerduty_service_key)"
    )


class AlertRuleConfig(BaseModel):
    """Configuration for an alert rule."""
    name: str = Field(description="Name of the alert rule")
    description: Optional[str] = Field(default=None, description="Description of the alert rule")
    enabled: bool = Field(default=True, description="Enable this alert rule")
    event_type: str = Field(description="Event type to trigger on (e.g., 'data_drift_detected', 'model_drift_detected', 'high_error_rate')")
    severity: str = Field(default="warning", description="Severity of the alert (e.g., info, warning, error, critical)")
    conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional conditions for the rule (e.g., {'drift_score_gt': 0.3})"
    )
    channels: List[str] = Field(
        description="List of alert channel names to notify"
    )
    deduplication_window_seconds: int = Field(
        default=300, # 5 minutes
        description="Time window in seconds to deduplicate similar alerts"
    )


class AlertManagerConfig(BaseModel):
    """Configuration for the alert manager."""
    enabled: bool = Field(default=True, description="Enable the alert manager")
    channels: List[AlertChannelConfig] = Field(
        default_factory=list,
        description="List of configured alert channels"
    )
    rules: List[AlertRuleConfig] = Field(
        default_factory=list,
        description="List of configured alert rules"
    )


class ArtifactStoreType(str, Enum):
    """Supported artifact store types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class MetadataBackend(str, Enum):
    """Supported metadata backends."""
    JSON = "json"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class CleanupPolicyConfig(BaseModel):
    """Configuration for artifact cleanup policy."""
    
    enabled: bool = Field(
        default=False,
        description="Enable cleanup policy"
    )
    
    max_versions_per_artifact: int = Field(
        default=10,
        description="Maximum versions to keep per artifact"
    )
    
    max_age_days: int = Field(
        default=90,
        description="Maximum age in days before cleanup"
    )


class ArtifactStoreConfig(BaseModel):
    """Configuration for artifact storage."""
    
    type: ArtifactStoreType = Field(
        default=ArtifactStoreType.LOCAL,
        description="Type of artifact store"
    )
    
    root_path: str = Field(
        default="./artifacts",
        description="Root path for artifact storage"
    )
    
    versioning_enabled: bool = Field(
        default=True,
        description="Enable artifact versioning"
    )
    
    metadata_backend: MetadataBackend = Field(
        default=MetadataBackend.JSON,
        description="Backend for storing metadata"
    )
    
    cleanup_policy: CleanupPolicyConfig = Field(
        default_factory=CleanupPolicyConfig,
        description="Cleanup policy configuration"
    )


class TransformationConfig(BaseModel):
    """Configuration for data transformation."""
    
    add_technical_indicators: bool = Field(
        default=True,
        description="Whether to add technical indicators"
    )
    
    technical_indicators: Optional[List[str]] = Field(
        default=None,
        description="List of technical indicators to calculate"
    )
    
    scaling_method: Optional[str] = Field(
        default="standard",
        description="Scaling method (standard, minmax, robust)"
    )
    
    scaling_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to scale (None means all numeric)"
    )
    
    custom_transformations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom transformation configurations"
    )


class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    
    check_missing_values: bool = Field(
        default=True,
        description="Check for missing values"
    )
    
    missing_value_threshold: float = Field(
        default=0.1,
        description="Maximum allowed fraction of missing values"
    )
    
    check_outliers: bool = Field(
        default=True,
        description="Check for outliers"
    )
    
    outlier_method: str = Field(
        default="iqr",
        description="Outlier detection method (iqr, zscore)"
    )
    
    outlier_threshold: float = Field(
        default=1.5,
        description="Threshold for outlier detection"
    )
    
    check_data_types: bool = Field(
        default=True,
        description="Check data types"
    )
    
    expected_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Expected data types for columns"
    )
    
    check_ranges: bool = Field(
        default=False,
        description="Check value ranges"
    )
    
    value_ranges: Optional[Dict[str, tuple]] = Field(
        default=None,
        description="Expected value ranges for columns"
    )


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    
    name: str = Field(
        description="Pipeline name"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Pipeline version"
    )
    
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Environment type"
    )
    
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data configuration"
    )
    
    model: ModelConfig = Field(
        description="Model configuration"
    )
    
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration"
    )
    
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration"
    )
    
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig,
        description="Deployment configuration"
    )
    
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    
    artifact_store: Union[str, ArtifactStoreConfig] = Field(
        default_factory=lambda: ArtifactStoreConfig(),
        description="Artifact store configuration"
    )
    
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    pipelines: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Definitions of executable pipelines and their stages"
    )
    
    @field_validator("name")
    def validate_name(cls, v):
        """Validate pipeline name."""
        if not v or not v.strip():
            raise ValueError("Pipeline name cannot be empty")
        return v
    
    @field_validator("artifact_store", mode="before")
    def validate_artifact_store(cls, v):
        """Convert string artifact_store to ArtifactStoreConfig."""
        if isinstance(v, str):
            # Convert simple string path to full config
            return ArtifactStoreConfig(root_path=v)
        return v