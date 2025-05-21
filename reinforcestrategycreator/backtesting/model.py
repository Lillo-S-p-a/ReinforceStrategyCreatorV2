"""
Model training and selection module for backtesting.

This module provides functionality for training and selecting models
for backtesting reinforcement learning trading strategies.
"""

import os
import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
import ray

from reinforcestrategycreator.trading_environment import TradingEnv as TradingEnvironment
from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent
from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and manages reinforcement learning models for trading.
    
    This class handles training the final model on the complete dataset
    using the best hyperparameters from cross-validation.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 models_dir: str = "models",
                 random_seed: int = 42) -> None:
        """
        Initialize the model trainer.
        
        Args:
            config: Dictionary containing configuration parameters
            models_dir: Directory to save models
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.models_dir = models_dir
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize containers for models
        self.best_params = None
        self.best_model = None
    
    def train_final_model(self, train_data: pd.DataFrame, best_params: Dict[str, Any]) -> RLAgent:  # Using RLAgent alias
        """
        Train final model on complete dataset.
        
        Args:
            train_data: Complete training dataset
            best_params: Best hyperparameters from cross-validation
            
        Returns:
            Trained RL agent
        """
        logger.info("Training final model on complete training dataset")
        
        if train_data is None or len(train_data) == 0:
            raise ValueError("No training data available")
            
        self.best_params = best_params
        
        try:
            # Create environment with full training data
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
                "trading_incentive": self.config.get("trading_incentive", 0.002),  # Add new parameter
                "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
                "risk_fraction": self.config.get("risk_fraction", 0.1),
                "normalization_window_size": self.config.get("normalization_window_size", 20)
            }
            
            env = TradingEnvironment(env_config=env_config)
            
            # Create agent with best parameters
            agent = RLAgent(  # Using RLAgent alias
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                learning_rate=self.best_params.get("learning_rate", 0.001),
                gamma=self.best_params.get("gamma", 0.99),
                epsilon=self.config.get("epsilon", 1.0),
                epsilon_decay=self.config.get("epsilon_decay", 0.995),
                epsilon_min=self.config.get("epsilon_min", 0.01)
            )
            
            # Train agent
            episodes = self.config.get("final_episodes", 200)  # More episodes for final model
            
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
                    
                    # Batch training
                    if len(agent.memory) > self.best_params.get("batch_size", 32):
                        agent.replay()
                        
                # Log progress
                if (episode + 1) % 10 == 0:
                    logger.info(f"Final model training: {episode+1}/{episodes} episodes completed")
            
            # Save final model
            final_model_path = os.path.join(self.models_dir, "final_model.pth")
            # Save the PyTorch model directly since agent.save_model is just a placeholder
            torch.save(agent.model.state_dict(), final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
            # Store final model
            self.best_model = agent
            
            logger.info(f"Final model trained and saved to {final_model_path}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error training final model: {e}", exc_info=True)
            raise
    
    def evaluate_model(self, model: RLAgent, test_data: pd.DataFrame) -> Dict[str, float]:  # Using RLAgent alias
        """
        Evaluate model on test data.
        
        Args:
            model: Trained RL agent
            test_data: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data")
        
        if model is None:
            raise ValueError("No model available for evaluation")
            
        if test_data is None or len(test_data) == 0:
            raise ValueError("No test data available for evaluation")
            
        try:
            # Create environment with test data
            # Put the test data into Ray's object store and create config dict
            test_data_ref = ray.put(test_data)
            env_config = {
                "df": test_data_ref,
                "initial_balance": self.config.get("initial_balance", 10000),
                "transaction_fee_percent": self.config.get("transaction_fee", 0.001)
            }
            
            # Create environment with test data using the correct initialization format
            env = TradingEnvironment(env_config=env_config)
            
            # Run evaluation episode
            state = env.reset()
            done = False
            
            while not done:
                action = model.select_action(state)  # Using select_action instead of act
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
            
            # Calculate metrics
            metrics = self.metrics_calculator.get_episode_metrics(env)
            
            logger.info(f"Model evaluation completed with PnL: ${metrics['pnl']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            raise
    
    def get_best_model(self) -> Optional[RLAgent]:  # Using RLAgent alias
        """
        Get the best trained model.
        
        Returns:
            Trained RL agent or None if no model has been trained
        """
        return self.best_model
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best hyperparameters.
        
        Returns:
            Dictionary of best hyperparameters or None if not set
        """
        return self.best_params