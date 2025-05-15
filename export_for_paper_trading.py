#!/usr/bin/env python3
"""
Model Export for Paper Trading with Interactive Brokers

This script:
1. Loads the best trained model from hyperparameter optimization
2. Exports it in a format suitable for inference
3. Creates a configuration file for the paper trading system
4. Sets up the basic structure for Interactive Brokers integration
"""

import os
import json
import datetime
import logging
import shutil
import torch
import numpy as np
from typing import Dict, Any, Optional
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.registry import register_env

# Import project-specific modules
from reinforcestrategycreator.trading_environment import TradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("export_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("export_model")

# Constants
BEST_MODELS_DIR = "./best_models"
EXPORT_DIR = "./paper_trading"
IB_CONFIG_DIR = os.path.join(EXPORT_DIR, "ib_config")
MODEL_EXPORT_DIR = os.path.join(EXPORT_DIR, "models")

# Create directories if they don't exist
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(IB_CONFIG_DIR, exist_ok=True)
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

def load_best_model() -> tuple:
    """
    Load the best model from hyperparameter optimization.
    
    Returns:
        Tuple of (algorithm, config)
    """
    logger.info("Loading best model...")
    
    # Load best configuration
    config_path = os.path.join(BEST_MODELS_DIR, "best_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Best model configuration not found at {config_path}")
    
    with open(config_path, "r") as f:
        best_config = json.load(f)
    
    # Find the latest checkpoint
    checkpoints_dir = os.path.join(BEST_MODELS_DIR, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_dir}")
    
    checkpoint_dirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
    
    # Sort by modification time (latest first)
    checkpoint_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(checkpoints_dir, d)), reverse=True)
    latest_checkpoint_dir = os.path.join(checkpoints_dir, checkpoint_dirs[0])
    
    # Find the latest checkpoint file
    checkpoint_files = [f for f in os.listdir(latest_checkpoint_dir) if f.startswith("checkpoint-")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {latest_checkpoint_dir}")
    
    # Sort by checkpoint number (latest first)
    checkpoint_files.sort(key=lambda f: int(f.split("-")[1]), reverse=True)
    latest_checkpoint = os.path.join(latest_checkpoint_dir, checkpoint_files[0])
    
    logger.info(f"Loading model from checkpoint: {latest_checkpoint}")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Register the environment
    register_env("TradingEnv-v0", lambda config: TradingEnv(config))
    
    # Create algorithm configuration
    algo_config = (
        DQNConfig()
        .environment(
            env="TradingEnv-v0",
            env_config=best_config["env_config"]
        )
        .training(
            gamma=best_config["training_config"]["gamma"],
            lr=best_config["training_config"]["lr"],
            train_batch_size=best_config["training_config"]["train_batch_size"],
            target_network_update_freq=best_config["training_config"]["target_network_update_freq"],
            n_step=best_config["training_config"]["n_step"],
            grad_clip=best_config["training_config"]["grad_clip"]
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.01,  # Low epsilon for inference
                "final_epsilon": 0.01,
                "epsilon_timesteps": 1000
            }
        )
        .resources(
            num_gpus=0
        )
        .rollouts(
            num_rollout_workers=0,  # Single worker for inference
            rollout_fragment_length=best_config["training_config"]["rollout_fragment_length"]
        )
        .framework("torch")
        .debugging(log_level="ERROR")
    )
    
    # Set model configuration
    algo_config.training(model={
        "fcnet_hiddens": best_config["model_config"]["fcnet_hiddens"],
        "fcnet_activation": best_config["model_config"]["fcnet_activation"]
    })
    
    # Build the algorithm
    algo = algo_config.build()
    
    # Load the checkpoint
    algo.restore(latest_checkpoint)
    
def export_model_for_inference(algo: DQN, config: Dict[str, Any]) -> str:
    """
    Export the model in a format suitable for inference.
    
    Args:
        algo: The trained RLlib algorithm
        config: Model configuration
        
    Returns:
        Path to the exported model
    """
    logger.info("Exporting model for inference...")
    
    # Create a timestamp for the export
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create export directory
    export_path = os.path.join(MODEL_EXPORT_DIR, f"model_{timestamp}")
    os.makedirs(export_path, exist_ok=True)
    
    # Export the model weights
    try:
        # Get the policy
        policy = algo.get_policy()
        
        # Get the model
        model = policy.model
        
        # Export the model weights using PyTorch
        model_path = os.path.join(export_path, "model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model weights saved to {model_path}")
        
        # Export model architecture
        model_config = {
            "fcnet_hiddens": config["model_config"]["fcnet_hiddens"],
            "fcnet_activation": config["model_config"]["fcnet_activation"],
            "observation_space": algo.get_policy().observation_space.shape,
            "action_space": algo.get_policy().action_space.n
        }
        
        with open(os.path.join(export_path, "model_architecture.json"), "w") as f:
            json.dump(model_config, f, indent=4)
        
        # Export environment configuration
        env_config = config["env_config"].copy()
        
        # Remove the DataFrame reference as it's not serializable
        if "df" in env_config:
            del env_config["df"]
        
        with open(os.path.join(export_path, "env_config.json"), "w") as f:
            json.dump(env_config, f, indent=4)
        
        # Export training configuration
        with open(os.path.join(export_path, "training_config.json"), "w") as f:
            json.dump(config["training_config"], f, indent=4)
        
        # Create a metadata file
        metadata = {
            "export_timestamp": timestamp,
            "model_type": "DQN",
            "framework": "PyTorch",
            "observation_space": str(algo.get_policy().observation_space),
            "action_space": str(algo.get_policy().action_space)
        }
        
        with open(os.path.join(export_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Copy the checkpoint for reference
        shutil.copy(algo.save(), os.path.join(export_path, "checkpoint"))
        
        logger.info(f"Model successfully exported to {export_path}")
        
        return export_path
    
    except Exception as e:
        logger.error(f"Error exporting model: {e}", exc_info=True)
        raise

def create_inference_module(export_path: str) -> None:
    """
    Create a Python module for model inference.
    
    Args:
        export_path: Path to the exported model
    """
    logger.info("Creating inference module...")
    
    # Create inference.py
    inference_code = """
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class TradingModelInference:
    def __init__(self, model_dir: str):
        \"\"\"
        Initialize the inference model.
        
        Args:
            model_dir: Directory containing the exported model
        \"\"\"
        self.model_dir = model_dir
        
        # Load model architecture
        with open(os.path.join(model_dir, "model_architecture.json"), "r") as f:
            self.model_architecture = json.load(f)
        
        # Load environment configuration
        with open(os.path.join(model_dir, "env_config.json"), "r") as f:
            self.env_config = json.load(f)
        
        # Load model weights
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        self.model.eval()  # Set to evaluation mode
        
        # Load metadata
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
    
    def _build_model(self) -> torch.nn.Module:
        \"\"\"
        Build the PyTorch model based on the architecture.
        
        Returns:
            PyTorch model
        \"\"\"
        # Extract architecture parameters
        fcnet_hiddens = self.model_architecture["fcnet_hiddens"]
        fcnet_activation = self.model_architecture["fcnet_activation"]
        input_dim = self.model_architecture["observation_space"][0]
        output_dim = self.model_architecture["action_space"]
        
        # Map activation function
        activation_map = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid
        }
        activation_fn = activation_map.get(fcnet_activation, torch.nn.ReLU)
        
        # Build layers
        layers = []
        prev_layer_size = input_dim
        
        for size in fcnet_hiddens:
            layers.append(torch.nn.Linear(prev_layer_size, size))
            layers.append(activation_fn())
            prev_layer_size = size
        
        # Output layer
        layers.append(torch.nn.Linear(prev_layer_size, output_dim))
        
        # Create sequential model
        return torch.nn.Sequential(*layers)
    
    def preprocess_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        \"\"\"
        Preprocess the observation for model input.
        
        Args:
            observation: Raw observation dictionary
            
        Returns:
            Preprocessed observation tensor
        \"\"\"
        # Convert to numpy array
        obs_array = np.array(list(observation.values()), dtype=np.float32)
        
        # Convert to PyTorch tensor
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        return obs_tensor
    
    def predict(self, observation: Dict[str, Any]) -> int:
        \"\"\"
        Predict action for the given observation.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Predicted action (0: hold, 1: buy, 2: sell)
        \"\"\"
        # Preprocess observation
        obs_tensor = self.preprocess_observation(observation)
        
        # Get model prediction
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def get_action_probabilities(self, observation: Dict[str, Any]) -> List[float]:
        \"\"\"
        Get action probabilities for the given observation.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            List of action probabilities [p_hold, p_buy, p_sell]
        \"\"\"
        # Preprocess observation
        obs_tensor = self.preprocess_observation(observation)
        
        # Get model prediction
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            # Convert to probabilities using softmax
            probs = torch.nn.functional.softmax(q_values, dim=1).squeeze().tolist()
        
        return probs
"""
    
    with open(os.path.join(export_path, "inference.py"), "w") as f:
        f.write(inference_code.strip())
    
    logger.info(f"Inference module created at {os.path.join(export_path, 'inference.py')}")
    return algo, best_config
def create_ib_config_files() -> None:
    """Create configuration files for Interactive Brokers integration."""
    logger.info("Creating Interactive Brokers configuration files...")
    
    # Create paper trading configuration
    paper_trading_config = {
        "interactive_brokers": {
            "host": "127.0.0.1",
            "port": 7497,  # 7497 for TWS Paper Trading, 4002 for IB Gateway Paper Trading
            "client_id": 1,
            "account_id": "DU123456",  # Replace with your paper trading account ID
            "timeout": 60
        },
        "trading": {
            "symbol": "SPY",
            "exchange": "SMART",
            "currency": "USD",
            "historical_data_duration": "1 D",
            "bar_size": "5 mins",
            "risk_per_trade": 0.02,  # 2% of account value per trade
            "max_position_size": 0.2,  # 20% of account value
            "stop_loss_pct": 0.05,  # 5% stop loss
            "take_profit_pct": 0.1,  # 10% take profit
            "trading_hours": {
                "start": "09:30:00",
                "end": "16:00:00",
                "timezone": "America/New_York"
            },
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "paper_trading.log",
            "trade_log_file": "trades.log"
        },
        "monitoring": {
            "email_alerts": False,
            "email_address": "",
            "performance_report_frequency": "daily"
        }
    }
    
    # Save configuration
    with open(os.path.join(EXPORT_DIR, "paper_trading_config.json"), "w") as f:
        json.dump(paper_trading_config, f, indent=4)
    
    logger.info(f"Paper trading configuration saved to {os.path.join(EXPORT_DIR, 'paper_trading_config.json')}")

def create_paper_trading_script() -> None:
    """Create the main paper trading script."""
    logger.info("Creating paper trading script...")
    
    # Create paper_trading.py
    paper_trading_code = """
#!/usr/bin/env python3
\"\"\"
Paper Trading Script for Interactive Brokers

This script:
1. Loads the trained model
2. Connects to Interactive Brokers
3. Fetches market data
4. Makes trading decisions based on model predictions
5. Executes trades in paper trading account
6. Logs performance
\"\"\"

import os
import json
import logging
import datetime
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trading")

# Import IB client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ib_config.ib_client import connect_to_ib, IBClient

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Add technical indicators to the dataframe.\"\"\"
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    
    # Add SMA indicators
    df['sma_5'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    
    # Add EMA indicators
    df['ema_5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    # Add MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Add RSI
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    
    # Add Bollinger Bands
    bollinger = BollingerBands(close=df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    
    # Add Stochastic Oscillator
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Add OBV
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

class PaperTradingSystem:
    def __init__(self, model_dir: str, config_file: str = "paper_trading_config.json"):
        \"\"\"
        Initialize the paper trading system.
        
        Args:
            model_dir: Directory containing the exported model
            config_file: Path to the configuration file
        \"\"\"
        self.model_dir = model_dir
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Import the inference module dynamically
        sys.path.append(model_dir)
        from inference import TradingModelInference
        
        # Initialize the model
        self.model = TradingModelInference(model_dir)
        
        # Initialize IB client
        self.ib_client = None
        
        # Initialize trading state
        self.current_position = 0
        self.last_trade_time = None
        self.last_trade_price = None
        self.account_value = 0
        
        logger.info(f"Paper trading system initialized with model from {model_dir}")
    
    def connect_to_ib(self) -> None:
        \"\"\"Connect to Interactive Brokers.\"\"\"
        ib_config = self.config["interactive_brokers"]
        self.ib_client = connect_to_ib(
            host=ib_config["host"],
            port=ib_config["port"],
            client_id=ib_config["client_id"]
        )
    
    def disconnect_from_ib(self) -> None:
        \"\"\"Disconnect from Interactive Brokers.\"\"\"
        if self.ib_client:
            self.ib_client.disconnect()
            logger.info("Disconnected from IB")
    
    def get_market_data(self) -> pd.DataFrame:
        \"\"\"
        Get market data from Interactive Brokers.
        
        Returns:
            DataFrame containing market data
        \"\"\"
        # Get trading parameters
        symbol = self.config["trading"]["symbol"]
        duration = self.config["trading"]["historical_data_duration"]
        bar_size = self.config["trading"]["bar_size"]
        
        # Create contract
        contract = self.ib_client.create_stock_contract(symbol)
        
        # Request historical data
        reqId = self.ib_client.request_historical_data(
            contract=contract,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES"
        )
        
        # Wait for data
        self.ib_client.wait_for_historical_data(reqId)
        
        # Get data as DataFrame
        df = self.ib_client.get_historical_data_as_dataframe(reqId)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        return df
    
    def get_account_info(self) -> None:
        \"\"\"Get account information from Interactive Brokers.\"\"\"
        # Request account summary
        reqId = self.ib_client.request_account_summary()
        
        # Wait for data
        self.ib_client.wait_for_account_summary()
        
        # Get account value
        account_id = self.config["interactive_brokers"]["account_id"]
        if account_id in self.ib_client.account_summary:
            if "NetLiquidation" in self.ib_client.account_summary[account_id]:
                self.account_value = float(self.ib_client.account_summary[account_id]["NetLiquidation"]["value"])
                logger.info(f"Account value: ${self.account_value:.2f}")
        
        # Request positions
        self.ib_client.request_positions()
        
        # Wait for data
        self.ib_client.wait_for_positions()
        
        # Get current position
        symbol = self.config["trading"]["symbol"]
        self.current_position = 0
        
        for account_id, positions in self.ib_client.positions.items():
            if symbol in positions:
                self.current_position = positions[symbol]["position"]
                logger.info(f"Current position in {symbol}: {self.current_position} shares")
    
    def prepare_observation(self, df: pd.DataFrame) -> Dict[str, float]:
        \"\"\"
        Prepare observation for model input.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Observation dictionary
        \"\"\"
        # Get the latest row
        latest = df.iloc[-1]
        
        # Create observation dictionary
        observation = {}
        
        # Add price data
        observation["close"] = latest["close"]
        observation["open"] = latest["open"]
        observation["high"] = latest["high"]
        observation["low"] = latest["low"]
        observation["volume"] = latest["volume"]
        
        # Add technical indicators
        for col in df.columns:
            if col not in ["date", "open", "high", "low", "close", "volume", "wap", "count"]:
                observation[col] = latest[col]
        
        return observation
    
    def execute_trade(self, action: int) -> None:
        \"\"\"
        Execute a trade based on the model's action.
        
        Args:
            action: Model action (0: hold, 1: buy, 2: sell)
        \"\"\"
        symbol = self.config["trading"]["symbol"]
        
        # Get trading parameters
        risk_per_trade = self.config["trading"]["risk_per_trade"]
        max_position_size = self.config["trading"]["max_position_size"]
        
        # Create contract
        contract = self.ib_client.create_stock_contract(symbol)
        
        # Get current price
        reqId = self.ib_client.request_historical_data(
            contract=contract,
            duration="60 S",
            bar_size="1 min",
            what_to_show="TRADES"
        )
        
        self.ib_client.wait_for_historical_data(reqId)
        df = self.ib_client.get_historical_data_as_dataframe(reqId)
        
        if df.empty:
            logger.warning("Could not get current price. Skipping trade execution.")
            return
        
        current_price = df["close"].iloc[-1]
        
        # Calculate position size based on risk
        max_shares = int(self.account_value * max_position_size / current_price)
        risk_shares = int(self.account_value * risk_per_trade / current_price)
        
        # Execute trade based on action
        if action == 1:  # Buy
            if self.current_position >= max_shares:
                logger.info(f"Already at maximum position size ({self.current_position} shares). Skipping buy.")
                return
            
            # Calculate shares to buy
            shares_to_buy = min(risk_shares, max_shares - self.current_position)
            
            if shares_to_buy <= 0:
                logger.info("No shares to buy. Skipping buy.")
                return
            
            # Create order
            order = self.ib_client.create_market_order("BUY", shares_to_buy)
            
            # Place order
            orderId = self.ib_client.place_order(contract, order)
            
            logger.info(f"Placed buy order {orderId} for {shares_to_buy} shares of {symbol} at market price")
            
            # Update state
            self.last_trade_time = datetime.datetime.now()
            self.last_trade_price = current_price
            
        elif action == 2:  # Sell
            if self.current_position <= 0:
                logger.info("No position to sell. Skipping sell.")
                return
            
            # Calculate shares to sell
            shares_to_sell = min(risk_shares, self.current_position)
            
            if shares_to_sell <= 0:
                logger.info("No shares to sell. Skipping sell.")
                return
            
            # Create order
            order = self.ib_client.create_market_order("SELL", shares_to_sell)
            
            # Place order
            orderId = self.ib_client.place_order(contract, order)
            
            logger.info(f"Placed sell order {orderId} for {shares_to_sell} shares of {symbol} at market price")
            
            # Update state
            self.last_trade_time = datetime.datetime.now()
            self.last_trade_price = current_price
    
    def run_trading_loop(self, interval_seconds: int = 300) -> None:
        \"\"\"
        Run the trading loop.
        
        Args:
            interval_seconds: Interval between trading decisions in seconds
        \"\"\"
        logger.info(f"Starting trading loop with interval {interval_seconds} seconds")
        
        try:
            while True:
                # Check if within trading hours
                now = datetime.datetime.now()
                trading_hours = self.config["trading"]["trading_hours"]
                start_time = datetime.datetime.strptime(trading_hours["start"], "%H:%M:%S").time()
                end_time = datetime.datetime.strptime(trading_hours["end"], "%H:%M:%S").time()
                
                if now.time() < start_time or now.time() > end_time:
                    logger.info(f"Outside trading hours ({start_time} - {end_time}). Waiting...")
                    time.sleep(60)
                    continue
                
                # Check if trading day
                trading_days = self.config["trading"]["trading_days"]
                if now.strftime("%A") not in trading_days:
                    logger.info(f"Not a trading day ({now.strftime('%A')}). Waiting...")
                    time.sleep(60)
                    continue
                
                # Get account info
                self.get_account_info()
                
                # Get market data
                df = self.get_market_data()
                
                if df.empty:
                    logger.warning("Empty market data. Skipping trading decision.")
                    time.sleep(interval_seconds)
                    continue
                
                # Prepare observation
                observation = self.prepare_observation(df)
                
                # Get model prediction
                action = self.model.predict(observation)
                action_probs = self.model.get_action_probabilities(observation)
                
                # Log prediction
                action_names = ["HOLD", "BUY", "SELL"]
                logger.info(f"Model prediction: {action_names[action]} (Probabilities: {[f'{p:.4f}' for p in action_probs]})")
                
                # Execute trade
                self.execute_trade(action)
                
                # Wait for next interval
                logger.info(f"Waiting {interval_seconds} seconds until next trading decision...")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user.")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
        finally:
            self.disconnect_from_ib()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Paper Trading with Interactive Brokers")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing the exported model")
    parser.add_argument("--config", type=str, default="paper_trading_config.json", help="Path to configuration file")
    parser.add_argument("--interval", type=int, default=300, help="Trading interval in seconds")
    args = parser.parse_args()
    
    # Initialize paper trading system
    system = PaperTradingSystem(args.model_dir, args.config)
    
    # Connect to IB
    system.connect_to_ib()
    
    # Run trading loop
    system.run_trading_loop(args.interval)

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(EXPORT_DIR, "paper_trading.py"), "w") as f:
        f.write(paper_trading_code.strip())
    
    logger.info(f"Paper trading script created at {os.path.join(EXPORT_DIR, 'paper_trading.py')}")

def main() -> None:
    """Main function."""
    try:
        # Load best model
        algo, config = load_best_model()
        
        # Export model for inference
        export_path = export_model_for_inference(algo, config)
        
        # Create inference module
        create_inference_module(export_path)
        
        # Create IB configuration files
        create_ib_config_files()
        
        # Create paper trading script
        create_paper_trading_script()
        
        # Clean up
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Model export for paper trading completed successfully.")
        logger.info(f"Exported model: {export_path}")
        logger.info(f"Paper trading configuration: {os.path.join(EXPORT_DIR, 'paper_trading_config.json')}")
        logger.info(f"Paper trading script: {os.path.join(EXPORT_DIR, 'paper_trading.py')}")
        logger.info("To start paper trading, run:")
        logger.info(f"python {os.path.join(EXPORT_DIR, 'paper_trading.py')} --model-dir {export_path}")
    
    except Exception as e:
        logger.error(f"Error exporting model for paper trading: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()