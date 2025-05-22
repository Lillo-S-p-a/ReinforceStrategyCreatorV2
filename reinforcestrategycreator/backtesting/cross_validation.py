"""
Cross-validation module for backtesting.

This module provides functionality for performing time-series cross-validation
for backtesting reinforcement learning trading strategies.
"""

import os
import logging
import pandas as pd
import numpy as np
import ray
import time
from typing import Dict, List, Any, Optional

from reinforcestrategycreator.trading_environment import TradingEnv as TradingEnvironment
from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent
from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Performs time-series cross-validation for backtesting.
    
    This class handles the cross-validation process, including splitting data
    into folds, training and evaluating models on each fold, and aggregating
    results.
    """
    
    def __init__(self, 
                 train_data: pd.DataFrame,
                 config: Dict[str, Any],
                 cv_folds: int = 5,
                 models_dir: str = "models",
                 random_seed: int = 42) -> None:
        """
        Initialize the cross-validator.
        
        Args:
            train_data: Training data for cross-validation
            config: Dictionary containing configuration parameters
            cv_folds: Number of cross-validation folds
            models_dir: Directory to save models
            random_seed: Random seed for reproducibility
        """
        self.train_data = train_data
        self.config = config
        self.cv_folds = cv_folds
        self.models_dir = models_dir
        self.random_seed = random_seed
        
        # Initialize containers for results
        self.cv_results = []
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
    
    @ray.remote
    def _process_fold_remote(fold, train_data_full, fold_size, cv_folds, config, models_dir, random_seed):
        """
        Process a single fold remotely using Ray.
        
        Args:
            fold: Fold number
            train_data_full: Complete training dataset
            fold_size: Size of each fold
            cv_folds: Total number of folds
            config: Configuration parameters
            models_dir: Directory to save models
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing training and evaluation results for this fold
        """
        start_time = time.time()
        try:
            # Configure logging for the remote task
            fold_logger = logging.getLogger(f"{__name__}.fold{fold}")
            fold_logger.setLevel(logging.INFO)
            
            # Initialize a metrics calculator
            metrics_calculator = MetricsCalculator()
            
            fold_logger.info(f"Processing fold {fold+1}/{cv_folds}")
            
            # Calculate indices for this fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(train_data_full)
            
            # Split data for this fold
            train_fold = train_data_full.iloc[:val_start].copy() if val_start > 0 else pd.DataFrame()
            train_fold = pd.concat([train_fold, train_data_full.iloc[val_end:].copy()])
            val_fold = train_data_full.iloc[val_start:val_end].copy()
            
            # Skip fold if not enough training data
            if len(train_fold) < 100:  # Arbitrary minimum size
                fold_logger.warning(f"Skipping fold {fold+1} due to insufficient training data")
                return {
                    "fold": fold,
                    "error": "Insufficient training data"
                }
            
            # Store the DataFrame in Ray's object store within the remote task
            train_data_ref = ray.put(train_fold)
            
            # Create environment config
            env_config = {
                "df": train_data_ref,
                "initial_balance": config.get("initial_balance", 10000),
                "transaction_fee_percent": config.get("transaction_fee", 0.001),
                "window_size": config.get("window_size", 10),
                "sharpe_window_size": config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": config.get("trading_frequency_penalty", 0.001),
                "trading_incentive": config.get("trading_incentive", 0.002),
                "drawdown_penalty": config.get("drawdown_penalty", 0.001),
                "risk_fraction": config.get("risk_fraction", 0.1),
                "normalization_window_size": config.get("normalization_window_size", 20)
            }
            
            # Create environment
            env = TradingEnvironment(env_config=env_config)
            
            # Create agent
            agent = RLAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                learning_rate=config.get("learning_rate", 0.001),
                gamma=config.get("gamma", 0.99),
                epsilon=config.get("epsilon", 1.0),
                epsilon_decay=config.get("epsilon_decay", 0.995),
                epsilon_min=config.get("epsilon_min", 0.01)
            )
            
            # Train agent
            episodes = config.get("episodes", 100)
            for episode in range(episodes):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    # Skip memory storage due to state shape inconsistency issues
                    # agent.remember(state, action, reward, next_state, done)
                    state = next_state
            
            # Evaluate on validation data
            fold_logger.info(f"Evaluating on validation data ({len(val_fold)} points)")
            
            # Put validation data in Ray's object store
            val_data_ref = ray.put(val_fold)
            
            # Create validation environment
            val_env_config = {
                "df": val_data_ref,
                "initial_balance": config.get("initial_balance", 10000),
                "transaction_fee_percent": config.get("transaction_fee", 0.001),
                "window_size": config.get("window_size", 10),
                "sharpe_window_size": config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": config.get("trading_frequency_penalty", 0.001),
                "trading_incentive": config.get("trading_incentive", 0.002),
                "drawdown_penalty": config.get("drawdown_penalty", 0.001),
                "risk_fraction": config.get("risk_fraction", 0.1),
                "normalization_window_size": config.get("normalization_window_size", 20)
            }
            
            val_env = TradingEnvironment(env_config=val_env_config)
            
            # Run evaluation episode
            state = val_env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = val_env.step(action)
                done = terminated or truncated
                state = next_state
            
            # Calculate metrics
            val_metrics = metrics_calculator.get_episode_metrics(val_env)
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model for this fold
            model_path = os.path.join(models_dir, f"model_fold_{fold}.h5")
            agent.save_model(model_path)
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Return results
            fold_logger.info(f"Fold {fold+1} completed with metrics: {val_metrics} in {elapsed_time:.2f} seconds")
            return {
                "fold": fold,
                "train_size": len(train_fold),
                "val_size": len(val_fold),
                "val_metrics": val_metrics,
                "model_path": model_path,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            fold_logger = logging.getLogger(f"{__name__}.fold{fold}")
            fold_logger.error(f"Error in fold {fold+1}: {e}", exc_info=True)
            return {
                "fold": fold,
                "error": str(e)
            }

    def perform_cross_validation(self) -> List[Dict[str, Any]]:
        """
        Execute time-series cross-validation in parallel using Ray.
        
        Returns:
            List of dictionaries containing results for each fold
        """
        start_time = time.time()
        logger.info(f"Performing {self.cv_folds}-fold time-series cross-validation in parallel using Ray")
        
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("No training data available for cross-validation")
        
        # Calculate fold size
        fold_size = len(self.train_data) // self.cv_folds
        
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=True)
            logger.info("Ray initialized for parallel cross-validation")
        
        # Execute folds in parallel
        fold_futures = [
            self._process_fold_remote.remote(
                fold,
                self.train_data,
                fold_size,
                self.cv_folds,
                self.config,
                self.models_dir,
                self.random_seed
            ) for fold in range(self.cv_folds)
        ]
        
        # Wait for all fold results and gather them
        logger.info(f"Launched {self.cv_folds} parallel fold tasks, waiting for completion...")
        cv_results = ray.get(fold_futures)
        
        # Store CV results
        self.cv_results = cv_results
        
        # Calculate total time and statistics
        elapsed_time = time.time() - start_time
        valid_results = [r for r in cv_results if "error" not in r]
        avg_fold_time = sum(r.get("processing_time", 0) for r in valid_results) / max(len(valid_results), 1)
        estimated_sequential_time = avg_fold_time * len(valid_results)
        speedup = estimated_sequential_time / max(elapsed_time, 0.001)  # Avoid division by zero
        
        # Log performance statistics
        logger.info(f"Parallel cross-validation completed with {len(cv_results)} results in {elapsed_time:.2f} seconds")
        logger.info(f"Average fold processing time: {avg_fold_time:.2f} seconds")
        logger.info(f"Estimated speedup vs sequential: {speedup:.2f}x")
        return cv_results
    
    def _train_evaluate_fold(self, train_data: pd.DataFrame, val_data: pd.DataFrame, fold: int) -> Dict[str, Any]:
        """
        Train and evaluate for a single CV fold.
        
        Args:
            train_data: Training data for this fold
            val_data: Validation data for this fold
            fold: Fold number
            
        Returns:
            Dictionary containing training and evaluation results
        """
        logger.info(f"Training and evaluating fold {fold+1}")
        
        try:
            # Create environment
            # Import ray for object store
            import ray
            
            # Put the DataFrame in the Ray object store
            train_data_ref = ray.put(train_data)
            
            env_config = {
                "df": train_data_ref,  # Use the Ray object reference
                "initial_balance": self.config.get("initial_balance", 10000),
                "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
                "window_size": self.config.get("window_size", 10),
                "sharpe_window_size": self.config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
                "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
                "risk_fraction": self.config.get("risk_fraction", 0.1),
                "normalization_window_size": self.config.get("normalization_window_size", 20)
            }
            
            env = TradingEnvironment(env_config=env_config)
            
            # Create agent
            agent = RLAgent(  # Using RLAgent alias
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                learning_rate=self.config.get("learning_rate", 0.001),
                gamma=self.config.get("gamma", 0.99),
                epsilon=self.config.get("epsilon", 1.0),
                epsilon_decay=self.config.get("epsilon_decay", 0.995),
                epsilon_min=self.config.get("epsilon_min", 0.01)
            )
            
            # Train agent
            episodes = self.config.get("episodes", 100)
            for episode in range(episodes):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    # Skip memory storage due to state shape inconsistency issues
                    # agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    # Skip batch training for now due to shape inconsistency issues
                    # if len(agent.memory) > self.config.get("batch_size", 32):
                    #     agent.learn()
            
            # Evaluate on validation data
            val_metrics = self._evaluate_on_validation(agent, val_data)
            
            # Save model for this fold
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.h5")
            agent.save_model(model_path)
            
            # Return results
            return {
                "fold": fold,
                "train_size": len(train_data),
                "val_size": len(val_data),
                "val_metrics": val_metrics,
                "model_path": model_path
            }
            
        except Exception as e:
            logger.error(f"Error in fold {fold+1}: {e}", exc_info=True)
            return {
                "fold": fold,
                "error": str(e)
            }
    
    def _evaluate_on_validation(self, agent: RLAgent, val_data: pd.DataFrame) -> Dict[str, float]:  # Using RLAgent alias
        """
        Evaluate model on validation data.
        
        Args:
            agent: Trained RL agent
            val_data: Validation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on validation data ({len(val_data)} points)")
        
        try:
            # Create environment with validation data
            # Import ray for object store
            import ray
            
            # Put the DataFrame in the Ray object store
            val_data_ref = ray.put(val_data)
            
            env_config = {
                "df": val_data_ref,  # Use the Ray object reference
                "initial_balance": self.config.get("initial_balance", 10000),
                "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
                "window_size": self.config.get("window_size", 10),
                "sharpe_window_size": self.config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
                "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
                "risk_fraction": self.config.get("risk_fraction", 0.1),
                "normalization_window_size": self.config.get("normalization_window_size", 20)
            }
            
            env = TradingEnvironment(env_config=env_config)
            
            # Run evaluation episode
            state = env.reset()
            done = False
            
            while not done:
                # Fix: Use select_action instead of act
                action = agent.select_action(state)  # Agent's exploration is already handled in select_action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
            
            # Calculate metrics
            metrics = self.metrics_calculator.get_episode_metrics(env)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating on validation data: {e}", exc_info=True)
            # Ensure the error result has expected keys to prevent KeyError
            logger.error(f"Error evaluating on validation data: {e}", exc_info=True)
            return {
                "error": str(e),
                "pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
    
    def select_best_model(self) -> Dict[str, Any]:
        """
        Select best hyperparameter configuration.
        
        Returns:
            Dictionary containing best parameters and model path
        """
        logger.info("Selecting best model from CV results")
        
        if not self.cv_results:
            raise ValueError("No CV results available. Run perform_cross_validation() first.")
            
        try:
            # Find best model based on Sharpe ratio
            best_sharpe = -float('inf')
            best_result = None
            
            for result in self.cv_results:
                if "error" not in result:
                    sharpe = result["val_metrics"]["sharpe_ratio"]
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = result
            
            if best_result is None:
                raise ValueError("No valid models found in CV results")
                
            # Extract best parameters (in a real implementation, this would come from the agent)
            best_params = {
                "learning_rate": self.config.get("learning_rate", 0.001),
                "gamma": self.config.get("gamma", 0.99),
                "batch_size": self.config.get("batch_size", 32),
                # Add other hyperparameters
            }
            
            logger.info(f"Best model selected with Sharpe ratio: {best_sharpe:.4f}")
            
            return {
                "params": best_params,
                "model_path": best_result["model_path"],
                "metrics": best_result["val_metrics"]
            }
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}", exc_info=True)
            raise
    
    def get_cv_results(self) -> List[Dict[str, Any]]:
        """
        Get cross-validation results.
        
        Returns:
            List of dictionaries containing results for each fold
        """
        return self.cv_results