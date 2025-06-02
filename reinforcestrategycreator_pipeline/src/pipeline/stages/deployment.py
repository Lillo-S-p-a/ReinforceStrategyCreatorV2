from typing import Any, Dict

from reinforcestrategycreator_pipeline.src.pipeline.stage import PipelineStage
from reinforcestrategycreator_pipeline.src.pipeline.context import PipelineContext
from reinforcestrategycreator_pipeline.src.config.manager import ConfigManager
from reinforcestrategycreator_pipeline.src.artifact_store.base import ArtifactStore
from reinforcestrategycreator_pipeline.src.models.registry import ModelRegistry
from reinforcestrategycreator_pipeline.src.data.manager import DataManager
from reinforcestrategycreator_pipeline.src.monitoring.service import MonitoringService # Added


class DeploymentStage(PipelineStage):
    """
    Pipeline stage for deploying a trained model, currently supporting paper trading.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.deployment_config: Dict[str, Any] = {}
        self.model_artifact_id: str = ""
        self.model_version: str = "" # Optional, could be part of artifact_id or separate
        self.model: Any = None # Placeholder for the loaded model
        self.portfolio: Dict[str, Any] = {} # For paper trading
        self.monitoring_service: Optional[MonitoringService] = None # Added
 
    def setup(self, context: PipelineContext) -> None:
        """
        Set up the deployment stage.
        Loads deployment configuration and identifies the model to be deployed.
        """
        self.logger.info(f"Setting up stage: {self.name}")

        config_manager: Optional[ConfigManager] = context.get('config_manager')
        artifact_store: Optional[ArtifactStore] = context.get('artifact_store')

        if not config_manager:
            self.logger.error("ConfigManager not found in PipelineContext.")
            raise ValueError("ConfigManager not found in PipelineContext during DeploymentStage setup.")
        
        # Load global deployment configuration
        pipeline_config = config_manager.get_config()
        self.deployment_config = pipeline_config.deployment
        if not self.deployment_config: # This check might be redundant if DeploymentConfig has defaults or is non-optional
            self.logger.warning("No 'deployment' configuration found in pipeline_config.deployment.")
            # Decide if this is an error or if stage can proceed with defaults/no-op
            # For now, allow proceeding, run() will check mode.
        
        self.logger.info(f"Deployment configuration loaded: {self.deployment_config}")

        # Load trained model artifact ID and version from context
        # These keys are assumed to be set by a preceding TrainingStage or similar
        self.model_artifact_id = context.get('trained_model_artifact_id')
        self.model_version = context.get('trained_model_version') # Optional

        if not self.model_artifact_id:
            self.logger.warning("Trained model artifact ID not found in PipelineContext. Deployment may not be possible.")
            # Depending on requirements, this could be an error.
            # For paper trading, a pre-defined model might be used if not found, or it's an error.
        else:
            self.logger.info(f"Model artifact ID for deployment: {self.model_artifact_id}")
            if self.model_version:
                self.logger.info(f"Model version for deployment: {self.model_version}")
        
        # Store artifact_store for use in run() if needed for direct model loading
        # self.artifact_store = artifact_store # If direct loading is planned

        # Get monitoring service from context
        self.monitoring_service = context.get("monitoring_service") # Added
        if self.monitoring_service: # Added
            self.logger.info("MonitoringService retrieved from context.") # Added
        else: # Added
            self.logger.warning("MonitoringService not found in context. Monitoring will be disabled for this stage.") # Added
 
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the deployment stage.
        Loads the model and performs deployment actions (e.g., paper trading).
        """
        self.logger.info(f"Running stage: {self.name}")

        if self.monitoring_service: # Added
            print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.started'")
            self.monitoring_service.log_event(event_type=f"{self.name}.started", description=f"Stage {self.name} started.") # Added
        
        try: # Added
            # --- Load Model ---
            if not self.model_artifact_id:
                self.logger.error("No model_artifact_id found in setup. Cannot proceed with deployment.")
                # Potentially update context with an error status
                context.set_metadata(f"{self.name}_status", "error_no_model_id")
                if self.monitoring_service: # Added
                    print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.failed' (no model_id)")
                    self.monitoring_service.log_event(event_type=f"{self.name}.failed", description="No model_artifact_id found.", level="error") # Added
                return context
        
            model_registry: Optional[ModelRegistry] = context.get('model_registry')
            if not model_registry:
                self.logger.error("ModelRegistry not found in PipelineContext.")
                # Potentially update context with an error status
                context.set_metadata(f"{self.name}_status", "error_no_model_registry")
                if self.monitoring_service: # Added
                    print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.failed' (no model_registry)")
                    self.monitoring_service.log_event(event_type=f"{self.name}.failed", description="ModelRegistry not found.", level="error") # Added
                return context
            
            try:
                self.logger.info(f"Loading model with artifact ID: {self.model_artifact_id}, version: {self.model_version}")
                self.model = model_registry.load_model(
                    model_id=self.model_artifact_id,
                    version=self.model_version # Pass None if not specified, registry handles latest
                )
                self.logger.info(f"Model '{self.model_artifact_id}' loaded successfully: {type(self.model)}")
            except Exception as e_load: # Added specific exception variable
                self.logger.error(f"Failed to load model '{self.model_artifact_id}': {e_load}")
                context.set_metadata(f"{self.name}_status", f"error_model_load_failed: {e_load}")
                if self.monitoring_service: # Added
                    print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.model_load.failed'")
                    self.monitoring_service.log_event(event_type=f"{self.name}.model_load.failed", description=f"Failed to load model: {e_load}", level="error", context={"model_id": self.model_artifact_id, "error_details": str(e_load)}) # Added
                return context
        
            # --- Paper Trading Logic --- # Indented
            if self.deployment_config and self.deployment_config.mode == "paper_trading": # Indented
                self.logger.info("Paper trading mode activated.") # Indented
                
                # Initialize virtual portfolio # Indented
                initial_cash = self.deployment_config.initial_cash if hasattr(self.deployment_config, 'initial_cash') and self.deployment_config.initial_cash is not None else 100000.0 # Indented
                self.portfolio = { # Indented
                    "cash": initial_cash, # Indented
                    "holdings": {}, # e.g., {"AAPL": {"shares": 10, "avg_price": 150.0}} # Indented
                    "trades": [], # Indented
                    "pnl": 0.0, # Indented
                    "portfolio_value_history": [] # Indented
                } # Indented
                self.logger.info(f"Initial portfolio: {self.portfolio}") # Indented
 
                # Fetch data for paper trading # Indented
                # This is a simplified example; real paper trading might need continuous data feed. # Indented
                # For now, let's assume we use evaluation data or a specific dataset defined in config. # Indented
                data_manager: Optional[DataManager] = context.get('data_manager') # Indented
                if not data_manager: # Indented
                    self.logger.error("DataManager not found in PipelineContext. Cannot fetch data for paper trading.") # Indented
                    context.set_metadata(f"{self.name}_status", "error_no_data_manager") # Indented
                    return context # Indented
 
                # Example: Use data from a source specified in deployment_config or context # Indented
                # This part needs to be adapted based on how data is made available for deployment # Indented
                # For now, let's assume 'evaluation_data' is in context or a specific source is defined. # Indented
                paper_trading_data_source_id = self.deployment_config.paper_trading_data_source_id if hasattr(self.deployment_config, 'paper_trading_data_source_id') else None # Indented
                
                # Try to get data from context first (e.g., output of evaluation stage) # Indented
                # This key 'evaluation_data_output' is an assumption. # Indented
                data_for_paper_trading = context.get("evaluation_data_output") # Indented
 
                if data_for_paper_trading is None and paper_trading_data_source_id: # Indented
                    self.logger.info(f"Fetching data from source '{paper_trading_data_source_id}' for paper trading.") # Indented
                    try: # Indented
                        # Additional params for load_data might be needed from config # Indented
                        data_params = self.deployment_config.paper_trading_data_params if hasattr(self.deployment_config, 'paper_trading_data_params') else {} # Indented
                        data_for_paper_trading = data_manager.load_data(paper_trading_data_source_id, **data_params) # Indented
                    except Exception as e: # Indented
                        self.logger.error(f"Failed to load data for paper trading from source '{paper_trading_data_source_id}': {e}") # Indented
                        context.set_metadata(f"{self.name}_status", f"error_data_load_failed: {e}") # Indented
                        return context # Indented
                elif data_for_paper_trading is None: # Indented
                     self.logger.error("No data available for paper trading (neither in context via 'evaluation_data_output' nor via 'paper_trading_data_source_id').") # Indented
                     context.set_metadata(f"{self.name}_status", "error_no_paper_trading_data") # Indented
                     return context # Indented
 
 
                self.logger.info(f"Data for paper trading (shape: {data_for_paper_trading.shape if data_for_paper_trading is not None else 'N/A'}) loaded.") # Indented
 
                # Simulate trading loop (simplified) # Indented
                # This would typically iterate through time steps in the data # Indented
                if data_for_paper_trading is not None and hasattr(self.model, 'predict'): # Indented
                    # Assuming data_for_paper_trading is a DataFrame and model.predict takes it # Indented
                    # The actual prediction and signal generation logic depends heavily on the model type # Indented
                    try: # Indented
                        predictions = self.model.predict(data_for_paper_trading) # This is a placeholder # Indented
                        self.logger.info(f"Generated {len(predictions)} predictions/signals for paper trading.") # Indented
                        
                        # Example: Iterate through data and predictions to simulate trades # Indented
                        # This is highly simplified. Real logic would involve: # Indented
                        # - Iterating row by row (or by time step) # Indented
                        # - Getting current price # Indented
                        # - Applying risk management, position sizing # Indented
                        # - Executing virtual trades # Indented
                        # - Updating portfolio # Indented
                        
                        # Placeholder: Log a summary action # Indented
                        self.logger.info("Simulating trade decisions based on model predictions...") # Indented
                        # Example: if predictions suggest BUY for a step, log a buy. # Indented
                        # This needs to be fleshed out based on model output and trading strategy. # Indented
                        
                        # For now, just log a few dummy trades for demonstration # Indented
                        if len(predictions) > 0: # Indented
                            self.portfolio["trades"].append({ # Indented
                                "timestamp": str(data_for_paper_trading.index[0]) if hasattr(data_for_paper_trading, 'index') else "N/A", # Indented
                                "action": "BUY", "asset": "DUMMY_ASSET", "shares": 10, "price": 100.0 # Indented
                            }) # Indented
                            self.portfolio["cash"] -= 10 * 100.0 # Indented
                            self.portfolio["holdings"]["DUMMY_ASSET"] = {"shares": 10, "avg_price": 100.0} # Indented
                            self.logger.info(f"Simulated BUY trade. Portfolio: {self.portfolio}") # Indented
                            if self.monitoring_service: # Added # Indented
                                print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.paper_trade.executed'") # Added # Indented
                                self.monitoring_service.log_event( # Added # Indented
                                    event_type=f"{self.name}.paper_trade.executed", # Added # Indented
                                    description="Simulated BUY trade executed for DUMMY_ASSET.", # Added # Indented
                                    level="info", # Added # Indented
                                    context={"asset": "DUMMY_ASSET", "action": "BUY", "shares": 10, "price": 100.0} # Added # Indented
                                ) # Added # Indented
                                self.monitoring_service.log_metric(f"{self.name}.paper_trade.cash", self.portfolio["cash"]) # Added # Indented
                                self.monitoring_service.log_metric(f"{self.name}.paper_trade.holdings.DUMMY_ASSET.shares", self.portfolio["holdings"]["DUMMY_ASSET"]["shares"]) # Added # Indented
                                # P&L and portfolio value would be updated more realistically in a full simulation loop # Indented
                                self.monitoring_service.log_metric(f"{self.name}.paper_trade.pnl", self.portfolio.get("pnl", 0.0)) # Added # Indented
                                # Example: Calculate portfolio value # Indented
                                current_portfolio_value = self.portfolio["cash"] + self.portfolio["holdings"].get("DUMMY_ASSET", {}).get("shares", 0) * 100.0 # Assuming current price is 100 for dummy # Indented
                                self.monitoring_service.log_metric(f"{self.name}.paper_trade.portfolio_value", current_portfolio_value) # Added # Indented
 
                    except Exception as e_sim: # Added specific exception variable # Indented
                        self.logger.error(f"Error during prediction or trade simulation: {e_sim}") # Indented
                        context.set_metadata(f"{self.name}_status", f"error_prediction_simulation: {e_sim}") # Indented
                        if self.monitoring_service: # Added # Indented
                            print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.simulation.failed'") # Added # Indented
                            self.monitoring_service.log_event(event_type=f"{self.name}.simulation.failed", description=f"Prediction or trade simulation failed: {e_sim}", level="error", context={"error_details": str(e_sim)}) # Added # Indented
                        # Continue or return context based on severity # Indented
        
                else: # Indented
                    self.logger.warning("Model does not have a predict method or no data for paper trading. Skipping simulation loop.") # Indented
 
                self.logger.info("Paper trading simulation complete.") # Indented
                self.logger.info(f"Final portfolio status: {self.portfolio}") # Indented
                context.set(f"{self.name}_paper_trading_portfolio", self.portfolio) # Save portfolio to context # Indented
                if self.monitoring_service: # Added # Indented
                    self.monitoring_service.log_metric(f"{self.name}.paper_trading.final_cash", self.portfolio["cash"]) # Added # Indented
                    self.monitoring_service.log_metric(f"{self.name}.paper_trading.final_pnl", self.portfolio.get("pnl", 0.0)) # Added # Indented
                    # Log final number of trades # Indented
                    self.monitoring_service.log_metric(f"{self.name}.paper_trading.trade_count", len(self.portfolio.get("trades", []))) # Added # Indented
 
 
            elif self.deployment_config and self.deployment_config.mode == "live_trading": # Indented, Changed .get("mode") to .mode
                self.logger.warning("Live trading mode is configured but not yet implemented in this stage. Skipping.") # Indented
                # Placeholder for future live trading integration # Indented
                # - Connect to broker API # Indented
                # - Manage live orders, positions # Indented
                # - Handle real-time data feeds # Indented
            else: # Indented
                self.logger.warning( # Indented
                    f"Deployment mode '{self.deployment_config.mode if self.deployment_config else 'N/A'}' is not " # Indented, Changed .get("mode") to .mode
                    f"recognized or not yet implemented for this stage. Skipping." # Indented
                ) # Indented
            
            context.set_metadata(f"{self.name}_status", "completed") # Indented
            if self.monitoring_service: # Added # Indented
                print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.completed'") # Added # Indented
                self.monitoring_service.log_event(event_type=f"{self.name}.completed", description=f"Stage {self.name} completed successfully.", level="info") # Added # Indented
            return context # Indented
        # Added general exception handling for the run method
        except Exception as e_run: # Added
            self.logger.error(f"Error in {self.name} stage run: {str(e_run)}", exc_info=True) # Added
            context.set_metadata(f"{self.name}_status", f"error_stage_run: {str(e_run)}") # Added
            if self.monitoring_service: # Added
                print(f"DEBUG_DEPLOYMENT: {self.name} - Calling monitoring_service.log_event for '.failed' (in except e_run)") # Added
                self.monitoring_service.log_event(event_type=f"{self.name}.failed", description=f"Stage {self.name} failed: {str(e_run)}", level="error", context={"error_details": str(e_run)}) # Added
            # Unlike specific errors above, this might re-raise if it's a critical failure not caught by inner try-excepts
            # Re-raise the original exception to ensure it's propagated
            raise e_run # Explicitly raise the caught exception object
        
    def teardown(self, context: PipelineContext) -> None:
        """
        Clean up resources after the deployment stage has executed.
        """
        self.logger.info(f"Tearing down stage: {self.name}")
        # TODO: Implement teardown logic if any cleanup is needed
        pass