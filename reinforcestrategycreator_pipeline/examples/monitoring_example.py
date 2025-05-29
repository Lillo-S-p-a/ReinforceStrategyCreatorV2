"""Example demonstrating how to use the monitoring utilities."""

import time
import random
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.manager import ConfigManager
from src.monitoring import (
    get_logger,
    log_step,
    log_execution_time,
    track_metric,
    track_model_metrics,
    track_pipeline_event,
    initialize_monitoring_from_pipeline_config
)


@log_execution_time
def simulate_data_loading():
    """Simulate data loading with logging."""
    logger = get_logger("example.data")
    logger.info("Starting data loading...")
    
    # Simulate some work
    time.sleep(random.uniform(0.5, 1.5))
    
    logger.info("Data loading completed")
    return 1000  # Return number of samples


@log_step("Model Training")
def train_model(epochs: int = 5):
    """Simulate model training with step logging."""
    logger = get_logger("example.training")
    
    for epoch in range(epochs):
        # Simulate training
        time.sleep(random.uniform(0.2, 0.5))
        
        # Generate fake metrics
        metrics = {
            "loss": random.uniform(0.1, 0.5),
            "accuracy": random.uniform(0.7, 0.95),
            "learning_rate": 0.001 * (0.95 ** epoch)
        }
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1}/{epochs} - Metrics: {metrics}")
        
        # Track metrics to Datadog
        track_model_metrics("example_model", metrics, epoch=epoch + 1)
    
    return metrics


@track_metric("histogram")
def evaluate_model():
    """Simulate model evaluation with metric tracking."""
    logger = get_logger("example.evaluation")
    logger.info("Evaluating model...")
    
    # Simulate evaluation
    time.sleep(random.uniform(0.3, 0.7))
    
    # Return a score (will be tracked as histogram)
    score = random.uniform(0.8, 0.95)
    logger.info(f"Evaluation score: {score}")
    
    return score


def main():
    """Main example function."""
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Load configuration
    try:
        pipeline_config = config_manager.load_config("development")
        print(f"Loaded configuration for environment: {pipeline_config.environment}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        print("Using default configuration...")
        
        # Create a minimal config for the example
        from src.config.models import PipelineConfig, ModelConfig, ModelType
        pipeline_config = PipelineConfig(
            name="example_pipeline",
            model=ModelConfig(model_type=ModelType.DQN)
        )
    
    # Initialize monitoring
    monitoring_service = initialize_monitoring_from_pipeline_config(pipeline_config)
    logger = get_logger("example.main")
    
    # Log pipeline start event
    track_pipeline_event(
        "pipeline_started",
        "Example pipeline execution started",
        alert_type="info",
        tags=["example", "demo"]
    )
    
    try:
        # Step 1: Load data
        logger.info("=== Step 1: Data Loading ===")
        num_samples = simulate_data_loading()
        monitoring_service.log_metric("data.samples_loaded", num_samples, tags=["step:data_loading"])
        
        # Step 2: Train model
        logger.info("=== Step 2: Model Training ===")
        final_metrics = train_model(epochs=3)
        
        # Check alert thresholds
        alerts = monitoring_service.check_alert_thresholds(final_metrics)
        if alerts:
            logger.warning(f"Alert thresholds violated: {alerts}")
        
        # Step 3: Evaluate model
        logger.info("=== Step 3: Model Evaluation ===")
        score = evaluate_model()
        
        # Log final results
        monitoring_service.log_event(
            "pipeline_completed",
            f"Pipeline completed successfully with score: {score:.3f}",
            level="info",
            context={
                "final_score": score,
                "final_metrics": final_metrics,
                "num_samples": num_samples
            }
        )
        
        # Track completion event
        track_pipeline_event(
            "pipeline_completed",
            f"Example pipeline completed successfully",
            alert_type="success",
            tags=["example", "demo", f"score:{score:.3f}"]
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        # Track failure event
        track_pipeline_event(
            "pipeline_failed",
            f"Example pipeline failed: {str(e)}",
            alert_type="error",
            tags=["example", "demo", "error"]
        )
        raise
    
    finally:
        # Log health check
        health_status = monitoring_service.create_health_check()
        logger.info(f"Health check: {health_status}")


if __name__ == "__main__":
    main()