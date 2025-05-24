"""
Hyperparameter Optimization module for backtesting.

This module provides functionality for optimizing hyperparameters
for reinforcement learning trading strategies using Ray Tune.
"""

import os
import logging
import pandas as pd
import numpy as np
import ray
import time
from typing import Dict, List, Any, Optional, Tuple
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from reinforcestrategycreator.trading_environment import TradingEnv as TradingEnvironment
from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent
from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for reinforcement learning trading strategies.
    
    This class uses Ray Tune to perform hyperparameter optimization,
    evaluating different hyperparameter configurations and selecting the best one.
    """
    
    def __init__(self, 
                 train_data: pd.DataFrame,
                 config: Dict[str, Any],
                 models_dir: str = "models",
                 random_seed: int = 42,
                 num_samples: int = 10,
                 max_concurrent_trials: int = 4) -> None:
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            train_data: Training data for hyperparameter optimization
            config: Dictionary containing configuration parameters
            models_dir: Directory to save models
            random_seed: Random seed for reproducibility
            num_samples: Number of hyperparameter configurations to try
            max_concurrent_trials: Maximum number of concurrent trials
        """
        self.train_data = train_data
        self.config = config
        self.models_dir = models_dir
        self.random_seed = random_seed
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize containers for results
        self.hpo_results = []
        self.best_params = None
        self.best_score = None
        
    def _trainable(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None) -> None:
        """
        Trainable function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration to evaluate
            checkpoint_dir: Directory for checkpoints
        """
        # Extract hyperparameters from config
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        layers = config["layers"]
        gamma = config.get("gamma", self.config.get("gamma", 0.99))
        epsilon = config.get("epsilon", self.config.get("epsilon", 1.0))
        epsilon_decay = config.get("epsilon_decay", self.config.get("epsilon_decay", 0.995))
        epsilon_min = config.get("epsilon_min", self.config.get("epsilon_min", 0.01))
        
        # Create environment config
        env_config = {
            "df": ray.put(self.train_data),  # Put data in Ray object store
            "initial_balance": self.config.get("initial_balance", 10000),
            "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
            "window_size": self.config.get("window_size", 10),
            "sharpe_window_size": self.config.get("sharpe_window_size", 100),
            "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
            "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
            "trading_incentive": self.config.get("trading_incentive", 0.002),
            "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
            "risk_fraction": self.config.get("risk_fraction", 0.1),
            "normalization_window_size": self.config.get("normalization_window_size", 20)
        }
        
        # Create environment
        env = TradingEnvironment(env_config=env_config)
        
        # Create agent with the hyperparameters to evaluate
        agent = RLAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )
        
        # Train agent
        episodes = self.config.get("episodes", 100)
        validation_metrics = []
        
        for episode in range(episodes):
            # Handle Gymnasium API which returns (observation, info)
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result  # Unpack observation and info
            else:
                state = reset_result  # Fallback for older gym API
                
            done = False
            
            while not done:
                action = agent.select_action(state)
                step_result = env.step(action)
                
                # Handle Gymnasium API which returns (observation, reward, terminated, truncated, info)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # Fallback for older gym API
                    next_state, reward, done, info = step_result
                    
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # Train on batch if enough samples
                if len(agent.memory) > batch_size:
                    agent.learn()
            
            # Evaluate on validation data every 10 episodes
            if (episode + 1) % 10 == 0 or episode == episodes - 1:
                metrics = self._evaluate_agent(agent)
                validation_metrics.append(metrics)
                
                # Calculate combined score
                combined_score = float(self._calculate_combined_score(metrics))
                
                # Report metrics to Ray Tune using a dictionary
                # This is compatible with newer Ray versions
                tune.report({"score": combined_score})
        
        # Save model
        # Generate a unique filename using timestamp instead of relying on Ray Tune API
        import time
        timestamp = int(time.time())
        model_path = os.path.join(self.models_dir, f"hpo_model_{timestamp}.h5")
        agent.save_model(model_path)
    
    def _evaluate_agent(self, agent: RLAgent) -> Dict[str, float]:
        """
        Evaluate agent on validation data.
        
        Args:
            agent: Trained RL agent
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use a portion of the training data as validation
        val_size = int(len(self.train_data) * 0.2)
        val_data = self.train_data.iloc[-val_size:].copy()
        
        # Create validation environment
        val_env_config = {
            "df": ray.put(val_data),
            "initial_balance": self.config.get("initial_balance", 10000),
            "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
            "window_size": self.config.get("window_size", 10),
            "sharpe_window_size": self.config.get("sharpe_window_size", 100),
            "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
            "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
            "trading_incentive": self.config.get("trading_incentive", 0.002),
            "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
            "risk_fraction": self.config.get("risk_fraction", 0.1),
            "normalization_window_size": self.config.get("normalization_window_size", 20)
        }
        
        val_env = TradingEnvironment(env_config=val_env_config)
        
        # Run evaluation episode
        # Handle Gymnasium API which returns (observation, info)
        reset_result = val_env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # Unpack observation and info
        else:
            state = reset_result  # Fallback for older gym API
            
        done = False
        
        while not done:
            action = agent.select_action(state)
            step_result = val_env.step(action)
            
            # Handle Gymnasium API which returns (observation, reward, terminated, truncated, info)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Fallback for older gym API
                next_state, reward, done, info = step_result
                
            state = next_state
        
        # Calculate metrics
        metrics = self.metrics_calculator.get_episode_metrics(val_env)
        
        return metrics
    
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate combined score from multiple metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Combined score
        """
        # Get metric weights from config or use defaults
        weights = self.config.get("cross_validation", {}).get("metric_weights", {
            "sharpe_ratio": 0.4,
            "pnl": 0.3,
            "win_rate": 0.2,
            "max_drawdown": 0.1
        })
        
        # Calculate normalized PnL (as percentage of initial balance)
        initial_balance = self.config.get("initial_balance", 10000)
        pnl_pct = metrics["pnl"] / initial_balance if initial_balance > 0 else 0
        
        # Calculate combined score
        score = (
            weights["sharpe_ratio"] * metrics["sharpe_ratio"] +
            weights["pnl"] * pnl_pct +
            weights["win_rate"] * metrics["win_rate"] -
            weights["max_drawdown"] * metrics["max_drawdown"]
        )
        
        return score
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary containing best hyperparameters
        """
        start_time = time.time()
        logger.info("Starting hyperparameter optimization")
        
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=True)
            logger.info("Ray initialized for hyperparameter optimization")
        
        # Define search space from config
        hyperparams = self.config.get("hyperparameters", {})
        
        search_space = {
            "learning_rate": tune.choice(hyperparams.get("learning_rate", [0.001, 0.0001])),
            "batch_size": tune.choice(hyperparams.get("batch_size", [32, 64])),
            "layers": tune.choice(hyperparams.get("layers", [[64, 32], [128, 64]])),
            "gamma": tune.uniform(0.95, 0.99),
            "epsilon_decay": tune.uniform(0.99, 0.999),
            "epsilon_min": tune.uniform(0.01, 0.1)
        }
        
        # Define scheduler
        scheduler = ASHAScheduler(
            metric="score",  # Use the score key from the metrics dictionary
            mode="max",
            max_t=self.config.get("episodes", 100),
            grace_period=10,
            reduction_factor=2
        )
        
        # Run hyperparameter optimization
        analysis = tune.run(
            self._trainable,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1, "gpu": 0},
            verbose=1,
            progress_reporter=tune.CLIReporter(
                metric_columns=["training_iteration", "pnl", "sharpe_ratio", "max_drawdown", "win_rate", "combined_score"]
            )
        )
        
        # Get best configuration
        best_trial = analysis.get_best_trial(metric="score", mode="max")
        best_config = best_trial.config
        best_result = best_trial.last_result
        
        # Store best parameters
        self.best_params = {
            "learning_rate": best_config["learning_rate"],
            "batch_size": best_config["batch_size"],
            "layers": best_config["layers"],
            "gamma": best_config["gamma"],
            "epsilon_decay": best_config["epsilon_decay"],
            "epsilon_min": best_config["epsilon_min"],
            # We can't reference the exact model file by trial ID anymore since we're using timestamps
            # Instead, we'll need to find the best model based on metrics after training
            "model_path": os.path.join(self.models_dir, "best_hpo_model.h5")
        }
        
        self.best_score = best_result["score"]
        
        # Store all results
        self.hpo_results = analysis.results
        
        elapsed_time = time.time() - start_time
        logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        # Log additional metrics if available
        if 'sharpe_ratio' in best_result and 'pnl' in best_result and 'win_rate' in best_result and 'max_drawdown' in best_result:
            logger.info(f"Best metrics - Sharpe: {best_result['sharpe_ratio']:.4f}, "
                       f"PnL: ${best_result['pnl']:.2f}, "
                       f"Win Rate: {best_result['win_rate']*100:.2f}%, "
                       f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
        else:
            logger.info("Detailed metrics not available in result")
        
        return self.best_params
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters.
        
        Returns:
            Dictionary containing best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No best parameters available. Run optimize_hyperparameters() first.")
        
        return self.best_params
    
    def get_hpo_results(self) -> List[Dict[str, Any]]:
        """
        Get all hyperparameter optimization results.
        
        Returns:
            List of dictionaries containing results for each trial
        """
        if not self.hpo_results:
            raise ValueError("No HPO results available. Run optimize_hyperparameters() first.")
        
        return self.hpo_results
    
    def generate_hpo_report(self) -> str:
        """
        Generate a report of hyperparameter optimization results.
        
        Returns:
            String containing the report
        """
        if not self.hpo_results:
            raise ValueError("No HPO results available. Run optimize_hyperparameters() first.")
        
        report = "Hyperparameter Optimization Results\n"
        report += "=" * 40 + "\n\n"
        
        report += "Best Hyperparameters:\n"
        for param, value in self.best_params.items():
            report += f"  {param}: {value}\n"
        
        report += f"\nBest Score: {self.best_score:.4f}\n\n"
        
        report += "Top 5 Configurations:\n"
        # Sort results by score
        sorted_results = sorted(self.hpo_results.values(), key=lambda x: x.get("score", 0), reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):
            report += f"Configuration {i+1}:\n"
            report += f"  Score: {result.get('score', 0):.4f}\n"
            
            # Add detailed metrics if available
            if 'sharpe_ratio' in result:
                report += f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}\n"
            if 'pnl' in result:
                report += f"  PnL: ${result.get('pnl', 0):.2f}\n"
            if 'win_rate' in result:
                report += f"  Win Rate: {result.get('win_rate', 0)*100:.2f}%\n"
            if 'max_drawdown' in result:
                report += f"  Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%\n"
                
            report += f"  Parameters: {result.get('config', {})}\n\n"
        
        return report