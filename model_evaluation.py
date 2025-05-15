#!/usr/bin/env python3
"""
Model Evaluation Script for Trading Strategy

This script evaluates a trained RLlib model on a test dataset to assess:
1. Performance metrics (PnL, Sharpe ratio, drawdown, win rate)
2. Trade analysis (frequency, size, timing)
3. Performance visualization
4. Comparison against benchmark strategies

The evaluation helps determine if the model is ready for paper trading.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import ray
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env

# Import project-specific modules
from reinforcestrategycreator.trading_environment import TradingEnv
from reinforcestrategycreator.callbacks import DatabaseLoggingCallbacks
from reinforcestrategycreator.technical_analyzer import add_indicators

# Configure logging
import logging
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("model_evaluation.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants
TICKER = "SPY"
TEST_START_DATE = "2024-01-01"  # Use recent data for testing
TEST_END_DATE = "2024-04-30"
BEST_MODELS_DIR = "./best_models"
EVALUATION_RESULTS_DIR = "./evaluation_results"

# Create directories if they don't exist
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)

def fetch_test_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch test data for the specified ticker and date range."""
    import yfinance as yf
    
    logger.info(f"Fetching test data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns to lowercase
    df.columns = [col.lower() for col in df.columns]
    logger.info(f"Test data fetched successfully: {len(df)} rows")
    
    return df

def prepare_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare test data by adding technical indicators."""
    logger.info("Adding technical indicators to test data...")
    df = add_indicators(df)
    
    # Drop NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Dropped {nan_count} NaN values after indicator calculation.")
        df.dropna(inplace=True)
    
    logger.info("Technical indicators added successfully to test data.")
    return df

def load_best_model() -> Tuple[DQN, Dict[str, Any]]:
    """Load the best model and its configuration."""
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
                "initial_epsilon": 0.01,  # Low epsilon for evaluation
                "final_epsilon": 0.01,
                "epsilon_timesteps": 1000
            }
        )
        .resources(
            num_gpus=0
        )
        .rollouts(
            num_rollout_workers=0,  # Single worker for evaluation
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
    
    return algo, best_config

def evaluate_model(algo: DQN, test_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the model on test data.
    
    Args:
        algo: The trained RLlib algorithm
        test_data: Test dataset with technical indicators
        config: Model configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    # Create environment configuration for evaluation
    env_config = config["env_config"].copy()
    env_config["df"] = test_data
    
    # Create evaluation environment
    env = TradingEnv(env_config)
    
    # Initialize metrics
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_pnls": [],
        "final_portfolio_values": [],
        "trades": [],
        "portfolio_values": [],
        "actions": [],
        "observations": [],
        "rewards": []
    }
    
    # Run multiple evaluation episodes
    num_episodes = 10
    logger.info(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        logger.info(f"Starting evaluation episode {episode+1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_trades = []
        episode_portfolio_values = [env.portfolio_value]
        episode_actions = []
        episode_observations = [obs]
        episode_rewards = []
        
        # Run episode
        while not done:
            # Get action from model
            action = algo.compute_single_action(obs, explore=False)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            episode_portfolio_values.append(env.portfolio_value)
            episode_actions.append(action)
            episode_observations.append(next_obs)
            episode_rewards.append(reward)
            
            # Record trade if one occurred
            if "trade" in info and info["trade"]:
                episode_trades.append({
                    "step": episode_length,
                    "action": action,
                    "price": info["price"],
                    "size": info["size"],
                    "type": info["operation_type"],
                    "portfolio_value": env.portfolio_value
                })
            
            # Update observation
            obs = next_obs
        
        # Record episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["episode_pnls"].append(env.portfolio_value - env_config["initial_balance"])
        metrics["final_portfolio_values"].append(env.portfolio_value)
        metrics["trades"].append(episode_trades)
        metrics["portfolio_values"].append(episode_portfolio_values)
        metrics["actions"].append(episode_actions)
        metrics["observations"].append(episode_observations)
        metrics["rewards"].append(episode_rewards)
        
        logger.info(f"Episode {episode+1} completed: Reward={episode_reward:.2f}, PnL=${env.portfolio_value - env_config['initial_balance']:.2f}, Trades={len(episode_trades)}")
    
    # Calculate aggregate metrics
    metrics["mean_reward"] = np.mean(metrics["episode_rewards"])
    metrics["mean_pnl"] = np.mean(metrics["episode_pnls"])
    metrics["win_rate"] = np.mean([pnl > 0 for pnl in metrics["episode_pnls"]])
    
    # Calculate Sharpe ratio
    if len(metrics["episode_pnls"]) > 1:
        metrics["sharpe_ratio"] = np.mean(metrics["episode_pnls"]) / (np.std(metrics["episode_pnls"]) + 1e-6)
    else:
        metrics["sharpe_ratio"] = 0
    
    # Calculate max drawdown
    max_drawdowns = []
    for portfolio_values in metrics["portfolio_values"]:
        max_value = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > max_value:
                max_value = value
            drawdown = (max_value - value) / max_value if max_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdowns.append(max_drawdown)
    
    metrics["max_drawdown"] = np.mean(max_drawdowns)
    
    # Calculate trade metrics
    all_trades = [trade for episode_trades in metrics["trades"] for trade in episode_trades]
    metrics["total_trades"] = len(all_trades)
    metrics["avg_trades_per_episode"] = len(all_trades) / num_episodes
    
    if all_trades:
        buy_trades = [trade for trade in all_trades if trade["type"] == "buy"]
        sell_trades = [trade for trade in all_trades if trade["type"] == "sell"]
        
        metrics["buy_trades"] = len(buy_trades)
        metrics["sell_trades"] = len(sell_trades)
        
        if buy_trades:
            metrics["avg_buy_size"] = np.mean([trade["size"] for trade in buy_trades])
        else:
            metrics["avg_buy_size"] = 0
            
        if sell_trades:
            metrics["avg_sell_size"] = np.mean([trade["size"] for trade in sell_trades])
        else:
            metrics["avg_sell_size"] = 0
    else:
        metrics["buy_trades"] = 0
        metrics["sell_trades"] = 0
        metrics["avg_buy_size"] = 0
        metrics["avg_sell_size"] = 0
    
    logger.info("Model evaluation completed.")
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f}")
    logger.info(f"Mean PnL: ${metrics['mean_pnl']:.2f}")
    logger.info(f"Win rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Total trades: {metrics['total_trades']}")
    logger.info(f"Avg trades per episode: {metrics['avg_trades_per_episode']:.2f}")
    
    return metrics

def compare_with_benchmarks(test_data: pd.DataFrame, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare model performance with benchmark strategies.
    
    Args:
        test_data: Test dataset
        metrics: Model evaluation metrics
        config: Model configuration
        
    Returns:
        Dictionary containing benchmark comparison metrics
    """
    logger.info("Comparing model performance with benchmarks...")
    
    # Initialize benchmark metrics
    benchmark_metrics = {
        "buy_and_hold": {},
        "moving_average_crossover": {}
    }
    
    # Buy and Hold strategy
    initial_balance = config["env_config"]["initial_balance"]
    initial_price = test_data["close"].iloc[0]
    final_price = test_data["close"].iloc[-1]
    
    # Calculate number of shares that could be bought with initial balance
    shares = initial_balance / initial_price
    
    # Calculate final portfolio value
    final_portfolio_value = shares * final_price
    
    # Calculate PnL
    pnl = final_portfolio_value - initial_balance
    
    # Calculate return
    returns = final_price / initial_price - 1
    
    # Store metrics
    benchmark_metrics["buy_and_hold"]["pnl"] = pnl
    benchmark_metrics["buy_and_hold"]["return"] = returns
    benchmark_metrics["buy_and_hold"]["final_portfolio_value"] = final_portfolio_value
    
    logger.info(f"Buy and Hold PnL: ${pnl:.2f}")
    logger.info(f"Buy and Hold Return: {returns*100:.2f}%")
    
    # Moving Average Crossover strategy
    # Use 5-day and 20-day moving averages
    test_data["ma_short"] = test_data["close"].rolling(window=5).mean()
    test_data["ma_long"] = test_data["close"].rolling(window=20).mean()
    
    # Drop NaN values
    test_data_ma = test_data.dropna()
    
    # Initialize portfolio and position
    portfolio_value = initial_balance
    position = 0
    portfolio_values = [portfolio_value]
    trades = []
    
    # Simulate trading
    for i in range(1, len(test_data_ma)):
        # Get current and previous signals
        prev_short_ma = test_data_ma["ma_short"].iloc[i-1]
        prev_long_ma = test_data_ma["ma_long"].iloc[i-1]
        curr_short_ma = test_data_ma["ma_short"].iloc[i]
        curr_long_ma = test_data_ma["ma_long"].iloc[i]
        
        # Get current price
        price = test_data_ma["close"].iloc[i]
        
        # Check for crossover
        prev_signal = prev_short_ma > prev_long_ma
        curr_signal = curr_short_ma > curr_long_ma
        
        # Buy signal: short MA crosses above long MA
        if not prev_signal and curr_signal and position == 0:
            # Calculate number of shares to buy
            shares_to_buy = portfolio_value / price
            position = shares_to_buy
            
            # Record trade
            trades.append({
                "type": "buy",
                "price": price,
                "size": shares_to_buy,
                "portfolio_value": portfolio_value
            })
        
        # Sell signal: short MA crosses below long MA
        elif prev_signal and not curr_signal and position > 0:
            # Calculate portfolio value after selling
            portfolio_value = position * price
            position = 0
            
            # Record trade
            trades.append({
                "type": "sell",
                "price": price,
                "size": position,
                "portfolio_value": portfolio_value
            })
        
        # Update portfolio value
        if position > 0:
            portfolio_value = position * price
        
        portfolio_values.append(portfolio_value)
    
    # Calculate final metrics
    final_portfolio_value = portfolio_values[-1]
    pnl = final_portfolio_value - initial_balance
    
    # Calculate max drawdown
    max_value = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > max_value:
            max_value = value
        drawdown = (max_value - value) / max_value if max_value > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Store metrics
    benchmark_metrics["moving_average_crossover"]["pnl"] = pnl
    benchmark_metrics["moving_average_crossover"]["return"] = pnl / initial_balance
    benchmark_metrics["moving_average_crossover"]["final_portfolio_value"] = final_portfolio_value
    benchmark_metrics["moving_average_crossover"]["max_drawdown"] = max_drawdown
    benchmark_metrics["moving_average_crossover"]["trades"] = len(trades)
    
    logger.info(f"Moving Average Crossover PnL: ${pnl:.2f}")
    logger.info(f"Moving Average Crossover Return: {(pnl/initial_balance)*100:.2f}%")
    logger.info(f"Moving Average Crossover Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"Moving Average Crossover Trades: {len(trades)}")
    
    # Compare model with benchmarks
    comparison = {
        "model_vs_buy_and_hold": metrics["mean_pnl"] - benchmark_metrics["buy_and_hold"]["pnl"],
        "model_vs_moving_average": metrics["mean_pnl"] - benchmark_metrics["moving_average_crossover"]["pnl"]
    }
    
    logger.info(f"Model vs Buy and Hold: ${comparison['model_vs_buy_and_hold']:.2f}")
    logger.info(f"Model vs Moving Average Crossover: ${comparison['model_vs_moving_average']:.2f}")
    
    return {
        "benchmarks": benchmark_metrics,
        "comparison": comparison
    }

def visualize_results(test_data: pd.DataFrame, metrics: Dict[str, Any], benchmark_metrics: Dict[str, Any]) -> None:
    """
    Visualize evaluation results.
    
    Args:
        test_data: Test dataset
        metrics: Model evaluation metrics
        benchmark_metrics: Benchmark comparison metrics
    """
    logger.info("Visualizing evaluation results...")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(EVALUATION_RESULTS_DIR, "plots"), exist_ok=True)
    
    # Plot portfolio value over time for each episode
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio values for each episode
    for i, portfolio_values in enumerate(metrics["portfolio_values"]):
        plt.plot(portfolio_values, label=f"Episode {i+1}")
    
    # Add benchmark portfolio values
    # (Note: This is simplified and may not align perfectly with episode timelines)
    plt.axhline(y=benchmark_metrics["benchmarks"]["buy_and_hold"]["final_portfolio_value"], 
                color='r', linestyle='--', label="Buy and Hold")
    
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(EVALUATION_RESULTS_DIR, "plots", "portfolio_values.png"))
    
    # Plot reward distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(metrics["episode_rewards"], bins=10)
    plt.title("Episode Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Plot PnL distribution
    plt.subplot(2, 2, 2)
    plt.hist(metrics["episode_pnls"], bins=10)
    plt.title("Episode PnL Distribution")
    plt.xlabel("PnL ($)")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Plot trade count distribution
    trade_counts = [len(trades) for trades in metrics["trades"]]
    plt.subplot(2, 2, 3)
    plt.hist(trade_counts, bins=10)
    plt.title("Trade Count Distribution")
    plt.xlabel("Number of Trades")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # Plot action distribution
    all_actions = [action for episode_actions in metrics["actions"] for action in episode_actions]
    plt.subplot(2, 2, 4)
    plt.hist(all_actions, bins=3)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_RESULTS_DIR, "plots", "distributions.png"))
    
    # Plot comparison with benchmarks
    plt.figure(figsize=(10, 6))
    
    strategies = ["Model", "Buy and Hold", "MA Crossover"]
    pnls = [
        metrics["mean_pnl"],
        benchmark_metrics["benchmarks"]["buy_and_hold"]["pnl"],
        benchmark_metrics["benchmarks"]["moving_average_crossover"]["pnl"]
    ]
    
    plt.bar(strategies, pnls)
    plt.title("Strategy Comparison - PnL")
    plt.xlabel("Strategy")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.savefig(os.path.join(EVALUATION_RESULTS_DIR, "plots", "strategy_comparison.png"))
    
    logger.info("Visualization completed.")

def save_evaluation_results(metrics: Dict[str, Any], benchmark_metrics: Dict[str, Any]) -> None:
    """
    Save evaluation results to a file.
    
    Args:
        metrics: Model evaluation metrics
        benchmark_metrics: Benchmark comparison metrics
    """
    logger.info("Saving evaluation results...")
    
    # Create results object
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_metrics": {
            "mean_reward": metrics["mean_reward"],
            "mean_pnl": metrics["mean_pnl"],
            "win_rate": metrics["win_rate"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "total_trades": metrics["total_trades"],
            "avg_trades_per_episode": metrics["avg_trades_per_episode"],
            "buy_trades": metrics.get("buy_trades", 0),
            "sell_trades": metrics.get("sell_trades", 0),
            "avg_buy_size": metrics.get("avg_buy_size", 0),
            "avg_sell_size": metrics.get("avg_sell_size", 0)
        },
        "benchmark_metrics": benchmark_metrics["benchmarks"],
        "comparison": benchmark_metrics["comparison"]
    }
    
    # Save to file
    with open(os.path.join(EVALUATION_RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("Evaluation results saved.")

def main() -> None:
    """Main function to run the evaluation."""
    try:
        # Fetch and prepare test data
        test_data = fetch_test_data(TICKER, TEST_START_DATE, TEST_END_DATE)
        test_data = prepare_test_data(test_data)
        
        # Load best model
        algo, config = load_best_model()
        
        # Evaluate model
        metrics = evaluate_model(algo, test_data, config)
        
        # Compare with benchmarks
        benchmark_metrics = compare_with_benchmarks(test_data, metrics, config)
        
        # Visualize results
        visualize_results(test_data, metrics, benchmark_metrics)
        
        # Save evaluation results
        save_evaluation_results(metrics, benchmark_metrics)
        
        # Clean up
        algo.stop()
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Evaluation completed successfully.")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()