#!/usr/bin/env python3
"""
Hyperparameter Optimization Framework for RLlib Trading Strategy

This script implements a systematic approach to optimize the trading model by:
1. Running multiple training runs with different hyperparameter configurations
2. Evaluating each model based on key metrics (PnL, Sharpe ratio, drawdown, win rate)
3. Selecting the best configuration for further refinement
4. Providing visualization of the optimization results
"""

import os
import json
import datetime
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.dqn import DQNConfig
from sqlalchemy import func, desc

# Import project-specific modules
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode, TrainingRun, Step
from reinforcestrategycreator.trading_environment import TradingEnv
from reinforcestrategycreator.callbacks import DatabaseLoggingCallbacks

# Configure logging
import logging
logger = logging.getLogger("hyperparameter_optimization")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("hyperparameter_optimization.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
NUM_TRAINING_ITERATIONS = 50  # Increased from 10 to allow for better convergence
RESULTS_DIR = "./optimization_results"
BEST_MODELS_DIR = "./best_models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data for the specified ticker and date range."""
    import yfinance as yf
    
    logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to lowercase
    df.columns = [col.lower() for col in df.columns]
    logger.info(f"Data fetched successfully: {len(df)} rows")
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    
    logger.info("Adding technical indicators...")
    
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
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Dropped {nan_count} NaN values after indicator calculation.")
        df.dropna(inplace=True)
    
    logger.info("Technical indicators added successfully.")
    return df

def train_evaluate(config: Dict[str, Any], checkpoint_dir: Optional[str] = None) -> None:
    """
    Training function for Ray Tune.
    
    Args:
        config: Hyperparameters for the training
        checkpoint_dir: Directory where checkpoints are stored
    """
    # Extract hyperparameters from config
    env_config = config["env_config"]
    model_config = config["model_config"]
    training_config = config["training_config"]
    
    # Generate a unique run ID
    run_id = f"TUNE-{TICKER}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    
    # Create callback config
    callback_config = {
        "run_id": run_id,
        "num_training_iterations": NUM_TRAINING_ITERATIONS
    }
    
    # Create RLlib configuration
    algo_config = (
        DQNConfig()
        .environment(
            env="TradingEnv-v0",
            env_config=env_config
        )
        .training(
            gamma=training_config["gamma"],
            lr=training_config["lr"],
            train_batch_size=training_config["train_batch_size"],
            target_network_update_freq=training_config["target_network_update_freq"],
            n_step=training_config["n_step"],
            grad_clip=training_config["grad_clip"]
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": training_config["initial_epsilon"],
                "final_epsilon": training_config["final_epsilon"],
                "epsilon_timesteps": training_config["epsilon_timesteps"]
            }
        )
        .resources(
            num_gpus=0
        )
        .rollouts(
            num_rollout_workers=training_config["num_rollout_workers"],
            rollout_fragment_length=training_config["rollout_fragment_length"]
        )
        .callbacks(DatabaseLoggingCallbacks)
        .framework("torch")
        .debugging(log_level="ERROR")
    )
    
    # Set model configuration
    algo_config.training(model={
        "fcnet_hiddens": model_config["fcnet_hiddens"],
        "fcnet_activation": model_config["fcnet_activation"]
    })
    
    # Set callbacks configuration
    algo_config.callbacks_config = callback_config
    
    # Build and train the algorithm
    algo = algo_config.build()
    
    # Training loop
    for i in range(NUM_TRAINING_ITERATIONS):
        result = algo.train()
        
        # Report metrics to Ray Tune
        metrics = get_episode_metrics(run_id)
        
        # Add metrics to the result
        result.update(metrics)
        
        # Report result to Ray Tune
        tune.report(
            episode_reward_mean=result.get("episode_return_mean", 0),
            pnl_mean=metrics.get("pnl_mean", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            win_rate=metrics.get("win_rate", 0),
            training_iteration=i
        )
    
    # Save the final model
    checkpoint_path = algo.save()
    logger.info(f"Final model saved at {checkpoint_path}")
    
    # Clean up
    algo.stop()

def get_episode_metrics(run_id: str) -> Dict[str, float]:
    """
    Calculate metrics for episodes in a specific training run.
    
    Args:
        run_id: The ID of the training run
        
    Returns:
        Dictionary containing metrics (pnl_mean, sharpe_ratio, max_drawdown, win_rate)
    """
    metrics = {
        "pnl_mean": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0
    }
    
    try:
        with get_db_session() as db:
            # Get all completed episodes for this run
            episodes = db.query(Episode).filter(
                Episode.run_id == run_id,
                Episode.status == "completed"
            ).all()
            
            if not episodes:
                logger.warning(f"No completed episodes found for run_id {run_id}")
                return metrics
            
            # Calculate PnL mean
            pnl_values = [episode.pnl for episode in episodes if episode.pnl is not None]
            if pnl_values:
                metrics["pnl_mean"] = sum(pnl_values) / len(pnl_values)
            
            # Calculate win rate (episodes with positive PnL)
            if pnl_values:
                win_count = sum(1 for pnl in pnl_values if pnl > 0)
                metrics["win_rate"] = win_count / len(pnl_values)
            
            # Calculate Sharpe ratio (simplified)
            if pnl_values and len(pnl_values) > 1:
                returns = np.array(pnl_values)
                metrics["sharpe_ratio"] = np.mean(returns) / (np.std(returns) + 1e-6)
            
            # Calculate max drawdown
            # For this, we need the portfolio value over time for each episode
            # We'll take the average max drawdown across episodes
            max_drawdowns = []
            for episode in episodes:
                # Get all steps for this episode
                steps = db.query(Step).filter(
                    Step.episode_id == episode.episode_id
                ).order_by(Step.step_number).all()
                
                if steps:
                    # Extract portfolio values
                    portfolio_values = [step.portfolio_value for step in steps if step.portfolio_value is not None]
                    
                    if portfolio_values:
                        # Calculate drawdown
                        max_value = portfolio_values[0]
                        max_drawdown = 0
                        
                        for value in portfolio_values:
                            if value > max_value:
                                max_value = value
                            drawdown = (max_value - value) / max_value if max_value > 0 else 0
                            max_drawdown = max(max_drawdown, drawdown)
                        
                        max_drawdowns.append(max_drawdown)
            
            if max_drawdowns:
                metrics["max_drawdown"] = sum(max_drawdowns) / len(max_drawdowns)
    
    except Exception as e:
        logger.error(f"Error calculating metrics for run_id {run_id}: {e}", exc_info=True)
    
    return metrics

def hyperparameter_search() -> None:
    """Run hyperparameter search using Ray Tune."""
    # Fetch and prepare data
    df = fetch_historical_data(TICKER, START_DATE, END_DATE)
    df = add_technical_indicators(df)
    
    # Put the data in Ray object store
    ray.init()
    df_ref = ray.put(df)
    
    # Define hyperparameter search space
    search_space = {
        "env_config": {
            "df": df_ref,
            "initial_balance": 10000.0,
            "transaction_fee_percent": tune.uniform(0.0005, 0.002),
            "window_size": tune.choice([5, 10, 20]),
            "sharpe_window_size": tune.choice([50, 100, 200]),
            "use_sharpe_ratio": tune.choice([True, False]),
            "trading_frequency_penalty": tune.uniform(0.001, 0.005),
            "drawdown_penalty": tune.uniform(0.001, 0.01),
            "risk_fraction": tune.uniform(0.05, 0.2),
            "stop_loss_pct": tune.uniform(2.0, 10.0),
            "normalization_window_size": tune.choice([10, 20, 50])
        },
        "model_config": {
            "fcnet_hiddens": tune.choice([
                [64, 64],
                [128, 128],
                [256, 256],
                [128, 64],
                [256, 128],
                [64, 64, 64],
                [128, 128, 64]
            ]),
            "fcnet_activation": tune.choice(["relu", "tanh", "sigmoid"])
        },
        "training_config": {
            "gamma": tune.uniform(0.8, 0.99),
            "lr": tune.loguniform(1e-5, 1e-3),
            "train_batch_size": tune.choice([32, 64, 128, 256]),
            "target_network_update_freq": tune.choice([100, 500, 1000]),
            "n_step": tune.choice([1, 3, 5]),
            "grad_clip": tune.uniform(10.0, 50.0),
            "initial_epsilon": 1.0,
            "final_epsilon": tune.uniform(0.01, 0.1),
            "epsilon_timesteps": tune.choice([5000, 10000, 20000]),
            "num_rollout_workers": 4,
            "rollout_fragment_length": tune.choice([50, 100, 200])
        }
    }
    
    # Define scheduler
    scheduler = ASHAScheduler(
        metric="sharpe_ratio",
        mode="max",
        max_t=NUM_TRAINING_ITERATIONS,
        grace_period=10,
        reduction_factor=2
    )
    
    # Run hyperparameter search
    logger.info("Starting hyperparameter search...")
    
    analysis = tune.run(
        train_evaluate,
        config=search_space,
        scheduler=scheduler,
        num_samples=20,  # Number of different hyperparameter combinations to try
        resources_per_trial={"cpu": 4, "gpu": 0},
        local_dir=RESULTS_DIR,
        verbose=1,
        progress_reporter=tune.CLIReporter(
            metric_columns=["training_iteration", "episode_reward_mean", "pnl_mean", "sharpe_ratio", "max_drawdown", "win_rate"]
        )
    )
    
    # Get best configuration
    best_config = analysis.get_best_config(metric="sharpe_ratio", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(metric="sharpe_ratio", mode="max")
    
    # Save best configuration
    with open(os.path.join(BEST_MODELS_DIR, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=4)
    
    logger.info(f"Best configuration: {best_config}")
    logger.info(f"Best checkpoint: {best_checkpoint}")
    
    # Generate visualization of the results
    visualize_results(analysis)
    
    # Clean up
    ray.shutdown()

def visualize_results(analysis: tune.ExperimentAnalysis) -> None:
    """
    Visualize the results of the hyperparameter search.
    
    Args:
        analysis: Ray Tune ExperimentAnalysis object
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    
    # Get dataframe with all results
    df_results = analysis.results_df
    
    # Plot metrics over iterations for best trial
    best_trial = analysis.get_best_trial(metric="sharpe_ratio", mode="max")
    best_df = analysis.trial_dataframes[best_trial.trial_id]
    
    plt.figure(figsize=(15, 10))
    
    # Plot PnL
    plt.subplot(2, 2, 1)
    plt.plot(best_df["training_iteration"], best_df["pnl_mean"])
    plt.title("PnL Mean over Training Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("PnL Mean")
    plt.grid(True)
    
    # Plot Sharpe Ratio
    plt.subplot(2, 2, 2)
    plt.plot(best_df["training_iteration"], best_df["sharpe_ratio"])
    plt.title("Sharpe Ratio over Training Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    
    # Plot Max Drawdown
    plt.subplot(2, 2, 3)
    plt.plot(best_df["training_iteration"], best_df["max_drawdown"])
    plt.title("Max Drawdown over Training Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Max Drawdown")
    plt.grid(True)
    
    # Plot Win Rate
    plt.subplot(2, 2, 4)
    plt.plot(best_df["training_iteration"], best_df["win_rate"])
    plt.title("Win Rate over Training Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Win Rate")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "best_trial_metrics.png"))
    
    # Plot correlation between hyperparameters and metrics
    if len(df_results) > 5:  # Only if we have enough trials
        # Select relevant columns
        param_cols = [col for col in df_results.columns if "config" in col and not col.endswith("df")]
        metric_cols = ["pnl_mean", "sharpe_ratio", "max_drawdown", "win_rate"]
        
        # Create correlation matrix
        corr_df = df_results[param_cols + metric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_df.loc[metric_cols, param_cols], annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation between Hyperparameters and Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plots", "hyperparameter_correlation.png"))
    
    # Plot parallel coordinates plot for top trials
    if len(df_results) > 5:
        # Get top 10 trials by Sharpe ratio
        top_df = df_results.sort_values("sharpe_ratio", ascending=False).head(10)
        
        # Select relevant columns
        param_cols = [col for col in top_df.columns if "config" in col and not col.endswith("df")]
        
        # Normalize parameters for parallel coordinates plot
        for col in param_cols:
            if top_df[col].dtype != object:
                top_df[col] = (top_df[col] - top_df[col].min()) / (top_df[col].max() - top_df[col].min() + 1e-10)
        
        # Create parallel coordinates plot
        plt.figure(figsize=(20, 10))
        pd.plotting.parallel_coordinates(
            top_df, "trial_id", 
            cols=param_cols,
            colormap=plt.cm.viridis
        )
        plt.title("Parallel Coordinates Plot for Top 10 Trials")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plots", "parallel_coordinates.png"))

def export_best_model() -> None:
    """Export the best model for use in paper trading."""
    # Load best configuration
    with open(os.path.join(BEST_MODELS_DIR, "best_config.json"), "r") as f:
        best_config = json.load(f)
    
    # TODO: Implement model export for paper trading
    # This will depend on the specific requirements of the paper trading system
    # For now, we'll just log a message
    logger.info("Model export for paper trading not yet implemented.")

if __name__ == "__main__":
    # Run hyperparameter search
    hyperparameter_search()
    
    # Export best model
    export_best_model()