"""Main monitoring service that integrates logging and metrics."""

from typing import Optional, Dict, Any, List
import os
from pathlib import Path

from ..config.models import (
    MonitoringConfig,
    PipelineConfig,
    DataDriftConfig,
    ModelDriftConfig,
    ModelDriftDetectionMethod,
    DataDriftDetectionMethod, # Added import
    AlertManagerConfig,
    AlertRuleConfig,
    AlertChannelConfig,
    AlertChannelType
)
from .logger import configure_logging, get_logger, log_with_context
from .datadog_client import configure_datadog, datadog_client, track_pipeline_event
from .drift_detection import DataDriftDetector, ModelDriftDetector
from .alerting import AlertManager


class MonitoringService:
    """Service for managing logging and monitoring across the pipeline."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None, deployment_manager: Optional[Any] = None):
        """
        Initialize the monitoring service.
        
        Args:
            config: Monitoring configuration
            deployment_manager: Optional deployment manager for tracking deployed models
        """
        self.logger = get_logger("monitoring.service")
        self.config = config
        self._initialized = False
        self.data_drift_config: Optional[DataDriftConfig] = None
        self.data_drift_detector: Optional[DataDriftDetector] = None # Uncommented
        self.model_drift_config: Optional[ModelDriftConfig] = None
        self.model_drift_detector: Optional[ModelDriftDetector] = None
        self.alert_manager_config: Optional[AlertManagerConfig] = None
        self.alert_manager: Optional[AlertManager] = None
        self.deployment_manager = deployment_manager
        
        if config:
            self.initialize(config)
    
    def initialize(self, config: MonitoringConfig) -> None:
        """
        Initialize monitoring with the provided configuration.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        
        if not config.enabled:
            self.logger.info("Monitoring is disabled")
            return
        
        # Configure logging
        self._configure_logging()
        
        # Configure Datadog
        self._configure_datadog()

        # Configure Data Drift Detection
        self._configure_data_drift_detection()

        # Configure Model Drift Detection
        self._configure_model_drift_detection()

        # Configure Alert Manager
        self._configure_alert_manager()
        
        self._initialized = True
        self.logger.info("Monitoring service initialized with all components")
        
        # Send initialization event
        track_pipeline_event(
            "monitoring_initialized",
            "Monitoring service has been initialized",
            alert_type="success"
        )
    
    def _configure_logging(self) -> None:
        """Configure the logging system."""
        if not self.config:
            return
        
        # Determine log file path
        log_file = None
        if hasattr(self.config, 'log_file') and self.config.log_file:
            log_file = self.config.log_file
        else:
            # Default to logs directory
            log_dir = Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / "pipeline.log")
        
        # Configure logging
        configure_logging(
            log_level=self.config.log_level,
            log_file=log_file,
            enable_console=True,
            enable_json=True
        )
        
        self.logger.info(
            f"Logging configured with level: {self.config.log_level}"
        )
    
    def _configure_datadog(self) -> None:
        """Configure Datadog integration."""
        if not self.config:
            return
        
        # Get API keys (support environment variable substitution)
        api_key = self._resolve_env_var(self.config.datadog_api_key)
        app_key = self._resolve_env_var(self.config.datadog_app_key)
        
        # Configure Datadog
        configure_datadog(
            api_key=api_key,
            app_key=app_key,
            metrics_prefix=self.config.metrics_prefix,
            enabled=self.config.enabled
        )
        
        if api_key:
            self.logger.info("Datadog integration configured")
        else:
            self.logger.warning("Datadog API key not provided")
    
    def _resolve_env_var(self, value: Optional[str]) -> Optional[str]:
        """
        Resolve environment variable references in configuration values.
        
        Args:
            value: Configuration value that may contain ${VAR_NAME}
            
        Returns:
            Resolved value
        """
        if not value:
            return value
        
        # Check if it's an environment variable reference
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return os.getenv(var_name)
        
        return value
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        tags: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        environment: Optional[str] = None
    ) -> None:
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric (gauge, increment, histogram)
            tags: Optional tags
            model_id: Optional model ID for deployment tracking
            environment: Optional environment for deployment tracking
        """
        if not self._initialized:
            return
        
        # Enrich tags with deployment info if available
        enriched_tags = tags.copy() if tags else []
        if model_id and environment and self.deployment_manager:
            try:
                current_deployment = self.deployment_manager.get_current_deployment(
                    model_id=model_id,
                    target_environment=environment
                )
                if current_deployment:
                    enriched_tags.extend([
                        f"model_version:{current_deployment.get('model_version')}",
                        f"deployment_id:{current_deployment.get('deployment_id')}"
                    ])
            except Exception as e:
                self.logger.debug(f"Could not enrich metric with deployment info: {e}")
        
        # Log to structured logs
        log_with_context(
            "info",
            f"Metric: {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            metric_type=metric_type,
            tags=enriched_tags
        )
        
        # Send to Datadog
        if metric_type == "gauge":
            datadog_client.gauge(metric_name, value, tags=enriched_tags)
        elif metric_type == "increment":
            datadog_client.increment(metric_name, int(value), tags=enriched_tags)
        elif metric_type == "histogram":
            datadog_client.histogram(metric_name, value, tags=enriched_tags)
    
    def log_event(
        self,
        event_type: str,
        description: str,
        level: str = "info",
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an event.
        
        Args:
            event_type: Type of event
            description: Event description
            level: Log level (info, warning, error)
            tags: Optional tags
            context: Optional additional context
        """
        if not self._initialized:
            return
        
        # Log to structured logs
        log_context = {
            "event_type": event_type,
            "tags": tags
        }
        if context:
            log_context.update(context)
        
        log_with_context(level, description, **log_context)
        
        # Send to Datadog
        alert_type = "info" if level == "info" else level
        track_pipeline_event(event_type, description, alert_type=alert_type, tags=tags)
    
    def check_alert_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check if any metrics violate alert thresholds.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            List of alerts (if any)
        """
        alerts = []
        
        if not self.config or not self.config.alert_thresholds:
            return alerts
        
        for metric_name, value in metrics.items():
            # Check for minimum thresholds
            min_key = f"{metric_name}_min"
            if min_key in self.config.alert_thresholds:
                threshold = self.config.alert_thresholds[min_key]
                if value < threshold:
                    alert = {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "type": "below_minimum",
                        "message": f"{metric_name} ({value}) is below minimum threshold ({threshold})"
                    }
                    alerts.append(alert)
                    self.log_event(
                        "alert_threshold_violated",
                        alert["message"],
                        level="warning",
                        tags=[f"metric:{metric_name}", "threshold:minimum"]
                    )
            
            # Check for maximum thresholds
            max_key = f"{metric_name}_max"
            if max_key in self.config.alert_thresholds:
                threshold = self.config.alert_thresholds[max_key]
                if value > threshold:
                    alert = {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "type": "above_maximum",
                        "message": f"{metric_name} ({value}) is above maximum threshold ({threshold})"
                    }
                    alerts.append(alert)
                    self.log_event(
                        "alert_threshold_violated",
                        alert["message"],
                        level="warning",
                        tags=[f"metric:{metric_name}", "threshold:maximum"]
                    )
        
        return alerts
    
    def create_health_check(self) -> Dict[str, Any]:
        """
        Create a health check status.
        
        Returns:
            Health check status dictionary
        """
        health_status = {
            "status": "healthy" if self._initialized else "not_initialized",
            "monitoring_enabled": self.config.enabled if self.config else False,
            "logging_configured": self._initialized,
            "datadog_configured": datadog_client.enabled if self._initialized else False
        }
        
        if self.config:
            health_status["log_level"] = self.config.log_level
            health_status["metrics_prefix"] = self.config.metrics_prefix
        
        health_status["data_drift_detection_enabled"] = self.data_drift_config.enabled if self.data_drift_config else False
        health_status["model_drift_detection_enabled"] = self.model_drift_config.enabled if self.model_drift_config else False
        health_status["alert_manager_enabled"] = self.alert_manager_config.enabled if self.alert_manager_config else False
        health_status["deployment_tracking_enabled"] = self.deployment_manager is not None
        
        return health_status

    def _configure_data_drift_detection(self) -> None:
        """Configure data drift detection components."""
        if not self.config or not self.config.data_drift:
            self.logger.info("Data drift detection not configured or disabled.")
            return
        
        self.data_drift_config = self.config.data_drift
        if self.data_drift_config.enabled:
            self.data_drift_detector = DataDriftDetector(self.data_drift_config) # Initialize actual
            self.logger.info(f"Data drift detection configured: {self.data_drift_config.method.value}")
        else:
            self.logger.info("Data drift detection is disabled in configuration.")
            self.data_drift_detector = None # Ensure it's None if disabled

    def _configure_model_drift_detection(self) -> None:
        """Configure model drift detection components."""
        if not self.config or not self.config.model_drift:
            self.logger.info("Model drift detection not configured or disabled.")
            return

        self.model_drift_config = self.config.model_drift
        if self.model_drift_config.enabled:
            self.model_drift_detector = ModelDriftDetector(self.model_drift_config) # Initialize actual
            self.logger.info(f"Model drift detection configured: {self.model_drift_config.method.value}")
        else:
            self.logger.info("Model drift detection is disabled in configuration.")
            self.model_drift_detector = None # Ensure it's None if disabled

    def _configure_alert_manager(self) -> None:
        """Configure the alert manager."""
        if not self.config or not self.config.alert_manager:
            self.logger.info("Alert manager not configured or disabled.")
            return

        self.alert_manager_config = self.config.alert_manager
        if self.alert_manager_config and self.alert_manager_config.enabled: # Check if alert_manager_config exists
            self.alert_manager = AlertManager(self.alert_manager_config)
            self.logger.info("Alert manager configured.")
        else:
            self.logger.info("Alert manager not configured or is disabled in configuration.")
            self.alert_manager = None

    def check_data_drift(self, current_data: Any, reference_data: Any, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform data drift check.
        Actual implementation will call self.data_drift_detector.
        
        Args:
            current_data: The current batch of data.
            reference_data: The reference dataset.
            model_version: Optional version of the model being monitored.
            
        Returns:
        Drift detection result or None if not enabled/configured.
    """
        if not self.data_drift_detector or not self.data_drift_config or not self.data_drift_config.enabled:
            self.logger.debug("Data drift detection skipped (disabled or not configured).")
            return None

        self.logger.info(f"Performing data drift check for model: {model_version or 'N/A'}")
        drift_result = self.data_drift_detector.detect(current_data, reference_data) # Use actual detector
        
        tags = [f"model_version:{model_version}"] if model_version else []
        tags.append(f"drift_method:{self.data_drift_config.method.value}")

        if drift_result and drift_result.get("drift_detected"):
            self.log_event(
                event_type="data_drift_detected",
                description=f"Data drift detected. Score: {drift_result.get('score')}",
                level="warning",
                tags=tags + ["status:drift_detected"],
                context=drift_result
            )
            self.process_alert("data_drift_detected", drift_result, severity="warning", tags=tags) # Uncommented
        
        # Send service checks for each feature
        if drift_result and drift_result.get("details"):
            feature_details = drift_result["details"].get("feature_scores") or \
                              drift_result["details"].get("feature_results")
            
            # Attempt to get environment tag, default to "unknown"
            environment_tag = "unknown"
            if self.config and hasattr(self.config, 'environment_tag'): # Placeholder for actual config attribute
                environment_tag = self.config.environment_tag
            elif os.getenv("PIPELINE_ENVIRONMENT"): # Or from an environment variable
                environment_tag = os.getenv("PIPELINE_ENVIRONMENT")


            if feature_details:
                for feature_name, feature_data in feature_details.items():
                    feature_tags = [
                        f"feature:{feature_name}",
                        f"model_version:{model_version or 'N/A'}",
                        f"environment:{environment_tag}" # Added environment tag
                    ]
                    
                    status = 3 # UNKNOWN by default
                    message = f"Drift status for feature {feature_name}."

                    # Determine status based on drift method and score/p_value
                    # This logic mirrors how drift_detected is set in DataDriftDetector
                    # For PSI, higher score means drift.
                    # For KS/Chi2, lower p-value means drift.
                    feature_score = 0.0
                    is_psi = self.data_drift_config.method == DataDriftDetectionMethod.PSI
                    
                    if isinstance(feature_data, (float, int)): # PSI score
                        feature_score = feature_data
                        if feature_score > self.data_drift_config.threshold:
                            status = 1 # WARNING
                            message = f"Data drift WARNING for {feature_name}. Score: {feature_score:.4f} > Threshold: {self.data_drift_config.threshold}"
                        else:
                            status = 0 # OK
                            message = f"Data drift OK for {feature_name}. Score: {feature_score:.4f} <= Threshold: {self.data_drift_config.threshold}"
                    elif isinstance(feature_data, dict): # KS or Chi2 result
                        p_value = feature_data.get("p_value")
                        if p_value is not None:
                            feature_score = p_value
                            if feature_score < self.data_drift_config.threshold:
                                status = 1 # WARNING
                                message = f"Data drift WARNING for {feature_name}. P-value: {feature_score:.4f} < Threshold: {self.data_drift_config.threshold}"
                            else:
                                status = 0 # OK
                                message = f"Data drift OK for {feature_name}. P-value: {feature_score:.4f} >= Threshold: {self.data_drift_config.threshold}"
                        else: # If p_value is not in dict, status remains UNKNOWN
                             message = f"Could not determine drift status for {feature_name}, p_value missing."
                    
                    datadog_client.send_service_check(
                        check_name="drift_check", # This is the 'model_pipeline.drift_check' from dashboard
                        status=status,
                        tags=feature_tags,
                        message=message
                    )

        if not (drift_result and drift_result.get("drift_detected")): # Log overall score if no drift
            if drift_result: # Ensure drift_result is not None
                self.log_metric("data_drift_score", drift_result.get("score", 0.0), tags=tags)
                self.logger.info(f"No significant overall data drift detected. Score: {drift_result.get('score')}")
            else: # Should not happen if detector is present, but good practice
                self.logger.warning("Data drift detection returned None unexpectedly.")
            
        return drift_result

    def check_model_drift(self, model_predictions: Any, ground_truth: Any, model_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Perform model drift check.
        Actual implementation will call self.model_drift_detector.

        Args:
            model_predictions: Predictions from the model.
            ground_truth: Actual outcomes.
            model_version: Optional version of the model being monitored.

        Returns:
            Drift detection result or None if not enabled/configured.
        """
        if not self.model_drift_detector or not self.model_drift_config or not self.model_drift_config.enabled:
            self.logger.debug("Model drift detection skipped (disabled or not configured).")
            return None

        self.logger.info(f"Performing model drift check for model: {model_version or 'N/A'}")
        # Pass confidence if available and method requires it
        confidence_data = None
        if self.model_drift_config.method == ModelDriftDetectionMethod.PREDICTION_CONFIDENCE:
            # Assuming model_predictions might contain confidence or it's passed separately
            # This part needs to be adapted based on how confidence is actually provided
            # For now, let's assume it's part of `model_predictions` if that's a dict, or a separate arg
            if isinstance(model_predictions, dict) and 'confidence' in model_predictions:
                confidence_data = model_predictions.get('confidence')
            # else: # Or if it's passed as a separate argument to this function in the future
            #    confidence_data = passed_confidence_argument

        drift_result = self.model_drift_detector.detect(model_predictions, ground_truth, model_confidence=confidence_data) # Use actual detector

        tags = [f"model_version:{model_version}"] if model_version else []
        tags.append(f"drift_method:{self.model_drift_config.method.value}")

        if drift_result and drift_result.get("drift_detected"):
            self.log_event(
                event_type="model_drift_detected",
                description=f"Model drift detected. Metric: {self.model_drift_config.performance_metric}, Value: {drift_result.get('metric_value')}",
                level="warning",
                tags=tags + ["status:drift_detected"],
                context=drift_result
            )
            self.process_alert("model_drift_detected", drift_result, severity="warning", tags=tags) # Uncommented
        else:
            if drift_result: # Ensure drift_result is not None
                metric_name_suffix = self.model_drift_config.performance_metric if self.model_drift_config.method == ModelDriftDetectionMethod.PERFORMANCE_DEGRADATION else "confidence"
                self.log_metric(f"model_drift_{metric_name_suffix}", drift_result.get("metric_value", 0.0), tags=tags)
                self.logger.info(f"No significant model drift detected. Metric: {metric_name_suffix}, Value: {drift_result.get('metric_value')}")
            else: # Should not happen
                self.logger.warning("Model drift detection returned None unexpectedly.")

        return drift_result

    def process_alert(self, event_type: str, event_data: Dict[str, Any], severity: str = "info", tags: Optional[List[str]] = None) -> None:
        """
        Process an event through the alert manager.
        Actual implementation will call self.alert_manager.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Data associated with the event.
            severity: Severity of the event.
            tags: Optional tags for the event.
        """
        if not self.alert_manager: # Check if alert_manager instance exists
            self.logger.debug(f"Alert processing skipped for event '{event_type}' (alert manager not initialized or disabled).")
            return

        self.logger.info(f"Handing off event to AlertManager: {event_type}, severity: {severity}")
        self.alert_manager.handle_event(event_type, event_data, severity, tags)
        
        # The AlertManager itself will log dispatch actions.
        # If specific Datadog events are desired *from MonitoringService* for alerts,
        # that logic could remain or be triggered by AlertManager callbacks if needed.
        # For now, primary alert dispatch logging is in AlertManager.

    def track_deployment(self, model_id: str, model_version: str, environment: str, deployment_id: str) -> None:
        """
        Track a model deployment event.
        
        Args:
            model_id: ID of the deployed model
            model_version: Version of the deployed model
            environment: Target environment (e.g., production, staging)
            deployment_id: Unique deployment identifier
        """
        if not self._initialized:
            return
            
        tags = [
            f"model_id:{model_id}",
            f"model_version:{model_version}",
            f"environment:{environment}",
            f"deployment_id:{deployment_id}"
        ]
        
        # Log deployment event
        self.log_event(
            event_type="model_deployed",
            description=f"Model {model_id} version {model_version} deployed to {environment}",
            level="info",
            tags=tags,
            context={
                "model_id": model_id,
                "model_version": model_version,
                "environment": environment,
                "deployment_id": deployment_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Track deployment metric
        self.log_metric("model_deployment", 1, metric_type="increment", tags=tags)
        
    def get_deployed_model_info(self, environment: str) -> Optional[Dict[str, Any]]:
        """
        Get information about currently deployed models in an environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Deployment information if deployment manager is available
        """
        if not self.deployment_manager:
            self.logger.debug("Deployment manager not available")
            return None
            
        try:
            # Get all deployments for the environment
            deployments = self.deployment_manager.list_deployments(
                target_environment=environment,
                status="deployed"
            )
            
            # Get current deployments grouped by model
            current_deployments = {}
            for deployment in deployments:
                model_id = deployment.get("model_id")
                if model_id:
                    current_deployments[model_id] = {
                        "model_version": deployment.get("model_version"),
                        "deployment_id": deployment.get("deployment_id"),
                        "deployed_at": deployment.get("deployed_at"),
                        "deployment_path": deployment.get("deployment_path")
                    }
                    
            return current_deployments
            
        except Exception as e:
            self.logger.error(f"Error getting deployment info: {e}")
            return None
            
    def enrich_metrics_with_deployment_info(self, metrics: Dict[str, Any], model_id: str, environment: str) -> Dict[str, Any]:
        """
        Enrich metrics with deployment information.
        
        Args:
            metrics: Base metrics dictionary
            model_id: Model ID
            environment: Environment
            
        Returns:
            Enriched metrics with deployment tags
        """
        if not self.deployment_manager:
            return metrics
            
        try:
            # Get current deployment for this model
            current_deployment = self.deployment_manager.get_current_deployment(
                model_id=model_id,
                target_environment=environment
            )
            
            if current_deployment:
                # Add deployment tags to metrics
                deployment_tags = {
                    "model_version": current_deployment.get("model_version"),
                    "deployment_id": current_deployment.get("deployment_id"),
                    "deployed_at": current_deployment.get("deployed_at")
                }
                
                # Merge with existing metrics
                enriched_metrics = metrics.copy()
                enriched_metrics["deployment_info"] = deployment_tags
                
                return enriched_metrics
                
        except Exception as e:
            self.logger.error(f"Error enriching metrics with deployment info: {e}")
            
        return metrics

# Singleton instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service(config: Optional[MonitoringConfig] = None, deployment_manager: Optional[Any] = None) -> MonitoringService:
    """
    Get or create the monitoring service instance.
    
    Args:
        config: Optional monitoring configuration
        deployment_manager: Optional deployment manager for tracking deployed models
        
    Returns:
        Monitoring service instance
    """
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(config, deployment_manager)
    elif config and not _monitoring_service._initialized:
        _monitoring_service.initialize(config)
    
    # Update deployment manager if provided
    if deployment_manager and _monitoring_service:
        _monitoring_service.deployment_manager = deployment_manager
    
    return _monitoring_service


def initialize_monitoring_from_pipeline_config(pipeline_config: PipelineConfig) -> MonitoringService:
    """
    Initialize monitoring from a pipeline configuration.
    
    Args:
        pipeline_config: Pipeline configuration
        
    Returns:
        Initialized monitoring service
    """
    return get_monitoring_service(pipeline_config.monitoring)