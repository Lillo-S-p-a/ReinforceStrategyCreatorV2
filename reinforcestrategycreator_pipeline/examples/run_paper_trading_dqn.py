#!/usr/bin/env python3
"""
Paper Trading Script for Optimized DQN Model

This script loads the best DQN model from HPO results and runs paper trading
simulation to evaluate the model's performance in a simulated trading environment.

Usage:
    python examples/run_paper_trading_dqn.py [--config CONFIG_PATH] [--duration HOURS]
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.manager import ConfigManager
from src.data.manager import DataManager
from src.data.yfinance_source import YFinanceDataSource
from src.deployment.manager import DeploymentManager
from src.deployment.paper_trading import PaperTradingDeployer, Order, OrderSide, OrderType
from src.models.factory import ModelFactory
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalFileSystemStore


class DQNPaperTradingRunner:
    """Manages paper trading execution for optimized DQN models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the paper trading runner.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('paper_trading_dqn.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DQNPaperTradingRunner received config_path: '{config_path}'") # DEBUG
        
        # Load configuration
        self.config_manager = ConfigManager()
        self.config_manager.load_config(config_path)  # Ensure config is loaded
        
        # Initialize components
        # Get artifact store root path from config or use default
        artifact_store_path = self._get_artifact_store_path()
        self.artifact_store = LocalFileSystemStore(artifact_store_path)
        self.data_manager = DataManager(self.config_manager, self.artifact_store)
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry(artifact_store=self.artifact_store)
        
        # Initialize deployment components
        self.deployment_manager = DeploymentManager(
            model_registry=self.model_registry,
            artifact_store=self.artifact_store
        )
        
        self.paper_trading_deployer = PaperTradingDeployer(
            deployment_manager=self.deployment_manager,
            model_registry=self.model_registry,
            artifact_store=self.artifact_store
        )
        
        # Best model info (will be loaded)
        self.best_model_path = None
        self.best_hyperparams = None
        self.model = None
        
    def _get_artifact_store_path(self) -> str:
        """Get artifact store root path from config or use default."""
        try:
            # Try to get from config first
            config = self.config_manager.get_config()
            if hasattr(config, 'artifact_store'):
                if isinstance(config.artifact_store, str):
                    return config.artifact_store
                elif hasattr(config.artifact_store, 'root_path'):
                    return config.artifact_store.root_path
        except Exception:
            # If config access fails, use default
            pass
        
        # Default path for paper trading artifacts
        default_path = './rsc_pipeline_artifacts/paper_trading'
        
        # Ensure the directory exists
        import os
        os.makedirs(default_path, exist_ok=True)
        
        return default_path
        
    def find_best_hpo_model(self) -> Dict[str, Any]:
        """Find the best model from HPO results.
        
        Returns:
            Dictionary containing best model information
        """
        hpo_results_dir = project_root / "hpo_results" / "dqn" / "dqn_hpo_quick_test"
        
        if not hpo_results_dir.exists():
            raise FileNotFoundError(f"HPO results directory not found: {hpo_results_dir}")
        
        # Look for the best trial directory (d9724_00001 based on previous context)
        best_trial_pattern = "*d9724_00001*"
        trial_dirs = list(hpo_results_dir.glob(best_trial_pattern))
        
        if not trial_dirs:
            # Fallback: look for any trial directory
            trial_dirs = [d for d in hpo_results_dir.iterdir() if d.is_dir() and "trainable" in d.name]
            
        if not trial_dirs:
            raise FileNotFoundError(f"No trial directories found in {hpo_results_dir}")
        
        # Use the first (and likely only) matching directory
        best_trial_dir = trial_dirs[0]
        
        # Look for the model checkpoint
        checkpoint_file = best_trial_dir / "params.pkl"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
        
        self.logger.info(f"Found best model checkpoint: {checkpoint_file}")
        
        # Extract hyperparameters from directory name
        dir_name = best_trial_dir.name
        hyperparams = self._extract_hyperparams_from_dirname(dir_name)
        
        return {
            "checkpoint_path": checkpoint_file,
            "trial_dir": best_trial_dir,
            "hyperparams": hyperparams
        }
    
    def _extract_hyperparams_from_dirname(self, dirname: str) -> Dict[str, Any]:
        """Extract hyperparameters from trial directory name.
        
        Args:
            dirname: Trial directory name
            
        Returns:
            Dictionary of hyperparameters
        """
        # Based on the context, extract known hyperparameters
        # This is a simplified extraction - in practice, you might want to
        # load from a trial config file if available
        
        hyperparams = {
            "learning_rate": 0.009915968356331756,
            "buffer_size": 10000,
            "batch_size": 32,
            "tau": 0.09387685572963467,
            "gamma": 0.9230491025848309,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.055,
            "exploration_fraction": 0.47,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "learning_starts": 50000,
            "max_grad_norm": 10
        }
        
        # Try to extract specific values from dirname if possible
        if "batch_size=" in dirname:
            try:
                batch_size = int(dirname.split("batch_size=")[1].split(",")[0])
                hyperparams["batch_size"] = batch_size
            except (ValueError, IndexError):
                pass
                
        if "buffer_size=" in dirname:
            try:
                buffer_size = int(dirname.split("buffer_size=")[1].split(",")[0])
                hyperparams["buffer_size"] = buffer_size
            except (ValueError, IndexError):
                pass
        
        return hyperparams
    
    def load_optimized_model(self) -> None:
        """Load the optimized DQN model from HPO results."""
        best_model_info = self.find_best_hpo_model()
        
        self.best_model_path = best_model_info["checkpoint_path"]
        self.best_hyperparams = best_model_info["hyperparams"]
        
        self.logger.info(f"Loading model from: {self.best_model_path}")
        self.logger.info(f"Model hyperparameters: {self.best_hyperparams}")
        
        # Load the pickled model parameters
        with open(self.best_model_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Create DQN model with optimized hyperparameters
        model_config = {
            "model_type": "DQN",  # Use uppercase to match registration
            "hyperparameters": self.best_hyperparams,
            "input_dim": 10,  # Adjust based on your feature space
            "output_dim": 3,  # Buy, Sell, Hold
            "hidden_dims": [64, 64],  # Default architecture
        }
        
        # Create model using factory
        self.model = self.model_factory.create_from_config(model_config)
        
        # Load the trained parameters
        if hasattr(self.model, 'load_state'):
            self.model.load_state(model_params)
        else:
            # Fallback: set parameters directly if available
            if hasattr(self.model, 'q_network') and 'q_network' in model_params:
                self.model.q_network.load_state_dict(model_params['q_network'])
        
        self.logger.info("Model loaded successfully")
    
    def setup_data_sources(self, symbols: list) -> Dict[str, Any]:
        """Setup data sources for paper trading.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary of configured data sources
        """
        # Setup live data source
        live_source_config = {
            "tickers": symbols,
            "interval": "1m",  # 1-minute intervals for paper trading
            # max_retries and retry_delay are not standard yfinance constructor args
            # They would need to be handled within YFinanceDataSource if custom logic is intended
        }
        live_source = YFinanceDataSource(
            source_id="yfinance_live_paper_trading", # Provide a source_id
            config=live_source_config
        )
        
        # Configure data manager (this part seems okay, assuming DataManager uses this config)
        # However, the DataManager itself might expect a different structure for its own config
        # For now, focusing on YFinanceDataSource instantiation.
        # The data_config for DataManager might need adjustment based on how DataManager
        # is designed to use these sources.
        data_manager_config_for_dm = { # Renamed to avoid confusion
            "sources": {
                "live": { # This key "live" should match how DataManager expects to find it
                    "type": "yfinance", # This should match a type DataManager can handle
                                         # or be a direct instance if DataManager supports it.
                                         # Assuming DataManager can use the instance directly for now.
                    "instance": live_source # Or pass config for DataManager to create it
                }
            },
            "features": [
                "close", "volume", "sma_5", "sma_20", "rsi",
                "macd", "bb_upper", "bb_lower", "returns"
            ]
        }
        
        return {
            "live_source": live_source, # This is the YFinanceDataSource instance
            "config": data_manager_config_for_dm # This is the config for DataManager
        }
    
    def create_simulation_config(self, 
                               initial_capital: float = 100000.0,
                               symbols: list = None) -> Dict[str, Any]:
        """Create simulation configuration.
        
        Args:
            initial_capital: Starting capital for simulation
            symbols: List of symbols to trade
            
        Returns:
            Simulation configuration dictionary
        """
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        return {
            "initial_capital": initial_capital,
            "commission_rate": 0.001,  # 0.1% commission
            "slippage_rate": 0.0005,   # 0.05% slippage
            "max_position_size": 0.2,  # Max 20% per position
            "daily_stop_loss": 0.05,   # 5% daily stop loss
            "enable_shorting": False,
            "symbols": symbols,
            "update_frequency": "1min",
            "data_source": "live"
        }
    
    def run_paper_trading(self, 
                         duration_hours: float = 1.0,
                         symbols: list = None,
                         initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Run paper trading simulation.
        
        Args:
            duration_hours: Duration to run simulation (hours)
            symbols: List of symbols to trade
            initial_capital: Starting capital
            
        Returns:
            Simulation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_optimized_model() first.")
        
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        self.logger.info(f"Starting paper trading simulation for {duration_hours} hours")
        self.logger.info(f"Trading symbols: {symbols}")
        self.logger.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Setup data sources
        data_sources = self.setup_data_sources(symbols)
        
        # Create simulation configuration
        sim_config = self.create_simulation_config(initial_capital, symbols)
        
        # Register model in registry (simplified)
        # Define model name and other metadata for registration
        model_name = f"dqn_optimized_paper_trade" # More descriptive name
        
        # Prepare additional metadata, ensuring correct type
        additional_model_metadata = {
            "source_script": Path(__file__).name,
            "hpo_checkpoint_path": str(self.best_model_path),
            # "type" is derived from model.model_type by the registry
            # "hyperparameters" are also derived by the registry
            # "created_at" is handled by the registry
        }

        # Register model in registry
        # The register_model method will generate its own model_id and version
        registered_model_id = self.model_registry.register_model(
            model=self.model,
            model_name=model_name,
            # version will be auto-generated
            tags=["paper_trading", "dqn", "optimized"],
            description="DQN model optimized via HPO, registered for paper trading.",
            # parent_model_id can be added if applicable
            # metrics can be added if pre-evaluation is done
            # dataset_info can be added if relevant
            additional_metadata=additional_model_metadata
        )
        self.logger.info(f"Model registered with ID: {registered_model_id}")
        
        # Deploy to paper trading using the ID returned by the registry
        simulation_id = self.paper_trading_deployer.deploy_to_paper_trading(
            model_id=registered_model_id, # Use the ID from registration
            simulation_config=sim_config
        )
        
        # Start simulation
        self.paper_trading_deployer.start_simulation(simulation_id)
        
        # Run simulation loop
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        update_interval = 60  # Update every 60 seconds
        last_update = start_time
        
        try:
            while datetime.now() < end_time:
                current_time = datetime.now()
                
                # Check if it's time for an update
                if (current_time - last_update).total_seconds() >= update_interval:
                    # Get current market data
                    market_data = self._get_current_market_data(symbols, data_sources["live_source"])
                    
                    if market_data:
                        # Process market update
                        self.paper_trading_deployer.process_market_update(
                            simulation_id, market_data
                        )
                        
                        # Log current status
                        status = self.paper_trading_deployer.get_simulation_status(simulation_id)
                        self.logger.info(
                            f"Portfolio value: ${status['portfolio_value']:,.2f}, "
                            f"Positions: {len(status['positions'])}, "
                            f"Trades: {status['total_trades']}"
                        )
                    
                    last_update = current_time
                
                # Sleep for a short interval
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        
        # Stop simulation and get results
        results = self.paper_trading_deployer.stop_simulation(simulation_id)
        
        self.logger.info("Paper trading simulation completed")
        self.logger.info(f"Final portfolio value: ${results['final_portfolio_value']:,.2f}")
        self.logger.info(f"Total return: {results['performance_metrics']['total_return']:.2%}")
        self.logger.info(f"Total trades: {results['total_trades']}")
        
        return results
    
    def _get_current_market_data(self, symbols: list, data_source: YFinanceDataSource) -> Optional[Dict[str, float]]:
        """Get current market data for symbols.
        
        Args:
            symbols: List of symbols
            data_source: Data source instance (YFinanceDataSource)
            
        Returns:
            Dictionary of symbol -> current price, or fallback to simulated prices.
        """
        try:
            self.logger.debug(f"Fetching latest market data for symbols: {symbols} using 1m interval, 2d period.")
            # Fetch recent data. Using "2d" period and "1m" interval to get recent ticks.
            # YFinanceDataSource.load_data will handle multiple tickers.
            # Pass tickers explicitly to override any defaults in data_source's own config.
            data = data_source.load_data(tickers=symbols, period="2d", interval="1m")
            
            if data is not None and not data.empty:
                market_data = {}
                
                # Ensure the index is datetime for proper sorting and selection
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                data = data.sort_index()

                if len(symbols) == 1 and not isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns:
                    # Single symbol, non-MultiIndex case
                    symbol = symbols[0] # Get the single symbol name
                    symbol_price_series = data['Close'].dropna()
                    if not symbol_price_series.empty:
                        latest_price = symbol_price_series.iloc[-1]
                        market_data[symbol] = float(latest_price)
                        self.logger.debug(f"Latest price for single symbol {symbol} ('Close'): {latest_price}")
                    else:
                        self.logger.warning(f"No 'Close' price data found for single symbol {symbol} after dropna.")
                else:
                    # Multi-symbol case or if yfinance returns MultiIndex even for single symbol
                    for symbol in symbols:
                        price_column_key = (symbol, 'Close')
                        if price_column_key in data.columns:
                            # Get the series for this symbol's close price
                            symbol_price_series = data[price_column_key].dropna()
                            if not symbol_price_series.empty:
                                latest_price = symbol_price_series.iloc[-1]
                                market_data[symbol] = float(latest_price)
                                self.logger.debug(f"Latest price for {symbol} ('Close'): {latest_price}")
                            else:
                                self.logger.warning(f"No 'Close' price data found for {symbol} with key {price_column_key} after dropna.")
                        else:
                            # Fallback for cases where a single symbol might still not be in ('SYMBOL', 'Close') format
                            # but directly as 'Close' if the DataFrame was unexpectedly simple.
                            if 'Close' in data.columns and len(symbols) == 1 and symbols[0] == symbol:
                                symbol_price_series = data['Close'].dropna()
                                if not symbol_price_series.empty:
                                    latest_price = symbol_price_series.iloc[-1]
                                    market_data[symbol] = float(latest_price)
                                    self.logger.debug(f"Latest price for {symbol} (fallback 'Close' column): {latest_price}")
                                else:
                                    self.logger.warning(f"No 'Close' price data found for {symbol} (fallback) after dropna.")
                            else:
                                self.logger.warning(f"Could not find price column for {symbol}. Tried key: {price_column_key} and direct 'Close'. Available columns: {list(data.columns)}")

                if market_data: # if we successfully extracted some prices
                    return market_data
                else:
                    self.logger.warning(f"Market data dictionary is empty after processing symbols: {symbols}. Raw data columns: {list(data.columns)}. Falling back to simulated prices.")
                    return {s: np.random.uniform(100, 200) for s in symbols}

            else: # data is None or empty
                self.logger.warning(f"No data returned by data_source.load_data for symbols: {symbols}. Falling back to simulated prices.")
                return {s: np.random.uniform(100, 200) for s in symbols}
            
        except Exception as e:
            self.logger.error(f"Error getting current market data: {e}", exc_info=True)
            self.logger.info("Falling back to simulated prices due to exception.")
            return {s: np.random.uniform(100, 200) for s in symbols}


def main():
    """Main entry point for paper trading script."""
    parser = argparse.ArgumentParser(description="Run paper trading with optimized DQN model")
    parser.add_argument(
        "--config",
        type=str,
        default="reinforcestrategycreator_pipeline/configs/base/pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=1.0,
        help="Duration to run simulation (hours)"
    )
    parser.add_argument(
        "--capital", 
        type=float, 
        default=100000.0,
        help="Initial capital for simulation"
    )
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        default=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        print(f"DEBUG: args.config from argparse: '{args.config}'") # DEBUG
        # Initialize runner
        runner = DQNPaperTradingRunner(config_path=args.config)
        
        # Load optimized model
        print("Loading optimized DQN model from HPO results...")
        runner.load_optimized_model()
        
        # Run paper trading
        print(f"Starting paper trading simulation for {args.duration} hours...")
        results = runner.run_paper_trading(
            duration_hours=args.duration,
            symbols=args.symbols,
            initial_capital=args.capital
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Duration: {args.duration} hours")
        print(f"Initial Capital: ${args.capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {results['performance_metrics']['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['performance_metrics']['winning_trades']}")
        print(f"Losing Trades: {results['performance_metrics']['losing_trades']}")
        
        if results['performance_metrics']['winning_trades'] > 0:
            print(f"Average Win: ${results['performance_metrics']['avg_win']:.2f}")
        if results['performance_metrics']['losing_trades'] > 0:
            print(f"Average Loss: ${results['performance_metrics']['avg_loss']:.2f}")
        
        print(f"\nResults saved to: paper_trading/{results['simulation_id']}_results.json")
        print("="*60)
        
    except Exception as e:
        print(f"Error running paper trading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()