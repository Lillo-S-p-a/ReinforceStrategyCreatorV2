"""Feature engineering stage implementation."""

from typing import Any, Dict, Optional
import pandas as pd

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
# Assuming TransformationConfig is defined in config.models
from reinforcestrategycreator_pipeline.src.config.models import TransformationConfig
from reinforcestrategycreator_pipeline.src.monitoring.service import MonitoringService # Added

class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""
    pass

class FeatureEngineeringStage(PipelineStage):
    """
    Stage responsible for transforming raw data into features for model training.
    """
    
    def __init__(self, name: str = "feature_engineering", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering stage.
        
        Args:
            name: Stage name.
            config: Stage-specific configuration. Can override global transformation settings.
        """
        super().__init__(name, config or {})
        self.transformation_config: Optional[TransformationConfig] = None
        self.monitoring_service: Optional[MonitoringService] = None # Added
 
    def setup(self, context: PipelineContext) -> None:
        """Set up the feature engineering stage."""
        self.logger.info(f"Setting up {self.name} stage.")
        
        config_manager = context.get("config_manager")
        if config_manager:
            global_pipeline_config = config_manager.get_config()
            # Prioritize stage-specific config for transformations, then global.
            # The self.config here is from the pipeline definition for this stage.
            # global_pipeline_config.data.transformation is the global one.
            
            # For now, let's assume TransformationConfig can be directly in self.config
            # or we construct it from global_pipeline_config.data.transformation
            if "transformation" in self.config and isinstance(self.config["transformation"], dict):
                 self.transformation_config = TransformationConfig(**self.config["transformation"])
            elif global_pipeline_config.data and global_pipeline_config.data.transformation:
                self.transformation_config = global_pipeline_config.data.transformation
            else:
                self.logger.info("No specific transformation config found. Using default behavior.")
                # Create a default TransformationConfig if needed, or handle None
                self.transformation_config = TransformationConfig() # Default if none specified
        else:
            self.logger.warning("ConfigManager not found in context. Using default transformation behavior.")
            self.transformation_config = TransformationConfig()

        # Example: Log what transformations will be applied
        if self.transformation_config:
            self.logger.info(f"Transformation config: Add technical indicators: {self.transformation_config.add_technical_indicators}, Indicators: {self.transformation_config.technical_indicators}")

        # Get monitoring service from context
        self.monitoring_service = context.get("monitoring_service") # Added
        if self.monitoring_service: # Added
            self.logger.info("MonitoringService retrieved from context.") # Added
        else: # Added
            self.logger.warning("MonitoringService not found in context. Monitoring will be disabled for this stage.") # Added
 
 
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the feature engineering stage."""
        self.logger.info(f"Running {self.name} stage.")

        if self.monitoring_service: # Added
            self.monitoring_service.log_event(event_type=f"{self.name}.started", description=f"Stage {self.name} started.") # Added
        
        initial_features_count = 0 # Added
        try: # Added
            raw_data = context.get("raw_data")
            if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
                raise FeatureEngineeringError("Raw data (non-empty DataFrame) not found in context.")
            initial_features_count = len(raw_data.columns) # Added
            
            self.logger.info(f"Raw data received with {len(raw_data)} rows and {initial_features_count} columns.") # Modified
            
            # --- Placeholder Feature Engineering ---
            # In a real scenario, use self.transformation_config to guide this.
            # For now, just create a simple moving average as an example.
            processed_df = raw_data.copy()
            
            if 'Close' in processed_df.columns:
                if self.transformation_config and self.transformation_config.add_technical_indicators:
                    # A very basic example, actual indicators would be more complex
                    # and driven by self.transformation_config.technical_indicators
                    if not self.transformation_config.technical_indicators or "SMA_10" in self.transformation_config.technical_indicators:
                        processed_df['SMA_10'] = processed_df['Close'].rolling(window=10).mean()
                        self.logger.info("Calculated SMA_10.")
                    
                    # Add more indicators based on config...
                    # Example: if "RSI_14" in self.transformation_config.technical_indicators:
                    #   processed_df['RSI_14'] = calculate_rsi(processed_df['Close'], window=14)

                    # Drop NaNs created by rolling windows
                    processed_df.dropna(inplace=True)
                    self.logger.info(f"DataFrame shape after adding indicators and dropping NaN: {processed_df.shape}")
                else:
                    self.logger.info("Skipping technical indicators as per configuration.")
            else:
                self.logger.warning("'Close' column not found in raw data. Skipping SMA calculation.")
                
            if processed_df.empty and not raw_data.empty:
                 self.logger.warning("Processed DataFrame is empty after transformations (e.g., due to dropna). Check window sizes and data length.")
            
            # --- End Placeholder ---
            
            context.set("processed_features", processed_df)

            final_features_count = len(processed_df.columns) if isinstance(processed_df, pd.DataFrame) else 0 # Added
            features_created = final_features_count - initial_features_count # Added
            # Note: features_dropped would be more complex to calculate if columns are selectively dropped.
            # For now, assuming new columns are mostly added.

            if self.monitoring_service: # Added
                self.monitoring_service.log_metric(f"{self.name}.initial_feature_count", initial_features_count) # Added
                self.monitoring_service.log_metric(f"{self.name}.final_feature_count", final_features_count) # Added
                self.monitoring_service.log_metric(f"{self.name}.features_created_count", features_created if features_created > 0 else 0) # Added
            
            # Create dummy labels for now to allow TrainingStage to proceed
            if not processed_df.empty:
                dummy_labels = pd.Series(0, index=processed_df.index) # Series of 0s
                context.set("labels", dummy_labels)
                self.logger.info(f"Set dummy 'labels' in context with shape {dummy_labels.shape}")
            elif not raw_data.empty and processed_df.empty: # If dropna made it empty
                 dummy_labels = pd.Series(0, index=raw_data.index[:0]) # Empty series with correct index type but 0 length
                 context.set("labels", dummy_labels)
                 self.logger.info(f"Set empty dummy 'labels' in context due to empty processed_df after dropna.")
            else: # if raw_data was empty
                context.set("labels", pd.Series(dtype='int64')) # Empty series
                self.logger.info("Raw data was empty, setting empty 'labels' in context.")

            self.logger.info(f"Feature engineering complete. Processed features stored in context. Shape: {processed_df.shape}")

            if self.monitoring_service: # Added
                self.monitoring_service.log_event(event_type=f"{self.name}.completed", description=f"Stage {self.name} completed successfully.", level="info") # Added
            
            return context
        # Added error handling block
        except Exception as e: # Added
            self.logger.error(f"Error in {self.name} stage: {str(e)}", exc_info=True) # Added
            if self.monitoring_service: # Added
                self.monitoring_service.log_event(event_type=f"{self.name}.failed", description=f"Stage {self.name} failed: {str(e)}", level="error", context={"error_details": str(e)}) # Added
            raise # Added

    def teardown(self, context: PipelineContext) -> None:
        """Clean up after feature engineering."""
        self.logger.info(f"Tearing down {self.name} stage.")