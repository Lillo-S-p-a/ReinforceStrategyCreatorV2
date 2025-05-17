"""
Cross-validation module for backtesting.

This module provides functionality for performing time-series cross-validation
for backtesting reinforcement learning trading strategies.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from reinforcestrategycreator.trading_environment import TradingEnvironment
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
    
    def perform_cross_validation(self) -> List[Dict[str, Any]]:
        """
        Execute time-series cross-validation.
        
        Returns:
            List of dictionaries containing results for each fold
        """
        logger.info(f"Performing {self.cv_folds}-fold time-series cross-validation")
        
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("No training data available for cross-validation")
        
        # Calculate fold size
        fold_size = len(self.train_data) // self.cv_folds
        
        # Store results for each fold
        cv_results = []
        
        for fold in range(self.cv_folds):
            logger.info(f"Processing fold {fold+1}/{self.cv_folds}")
            
            # Calculate indices for this fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.cv_folds - 1 else len(self.train_data)
            
            # Split data for this fold
            train_fold = self.train_data.iloc[:val_start].copy() if val_start > 0 else pd.DataFrame()
            train_fold = pd.concat([train_fold, self.train_data.iloc[val_end:].copy()])
            val_fold = self.train_data.iloc[val_start:val_end].copy()
            
            # Skip fold if not enough training data
            if len(train_fold) < 100:  # Arbitrary minimum size
                logger.warning(f"Skipping fold {fold+1} due to insufficient training data")
                continue
                
            # Train and evaluate on this fold
            fold_results = self._train_evaluate_fold(train_fold, val_fold, fold)
            cv_results.append(fold_results)
        
        # Store CV results
        self.cv_results = cv_results
        
        logger.info("Cross-validation completed")
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
            env = TradingEnvironment(
                data=train_data,
                initial_balance=self.config.get("initial_balance", 10000),
                transaction_fee=self.config.get("transaction_fee", 0.001)
            )
            
            # Create agent
            agent = RLAgent(  # Using RLAgent alias
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
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
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    # Batch training
                    if len(agent.memory) > self.config.get("batch_size", 32):
                        agent.replay()
            
            # Evaluate on validation data
            val_metrics = self._evaluate_on_validation(agent, val_data)
            
            # Save model for this fold
            model_path = os.path.join(self.models_dir, f"model_fold_{fold}.h5")
            agent.save(model_path)
            
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
            env = TradingEnvironment(
                data=val_data,
                initial_balance=self.config.get("initial_balance", 10000),
                transaction_fee=self.config.get("transaction_fee", 0.001)
            )
            
            # Run evaluation episode
            state = env.reset()
            done = False
            
            while not done:
                action = agent.act(state, evaluate=True)  # No exploration
                next_state, reward, done, _ = env.step(action)
                state = next_state
            
            # Calculate metrics
            metrics = self.metrics_calculator.get_episode_metrics(env)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating on validation data: {e}", exc_info=True)
            return {"error": str(e)}
    
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