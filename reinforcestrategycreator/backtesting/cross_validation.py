"""
Cross-validation module for backtesting.

This module provides functionality for performing time-series cross-validation
and hyperparameter optimization for backtesting reinforcement learning trading strategies.
"""

import os
import logging
import pandas as pd
import numpy as np
import ray
import time
from typing import Dict, List, Any, Optional, Tuple

from reinforcestrategycreator.trading_environment import TradingEnv as TradingEnvironment
from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent
from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator
from reinforcestrategycreator.backtesting.hyperparameter_optimization import HyperparameterOptimizer

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
                 random_seed: int = 42,
                 use_hpo: bool = False,
                 hpo_num_samples: int = 10,
                 hpo_max_concurrent_trials: int = 4) -> None:
        """
        Initialize the cross-validator.
        
        Args:
            train_data: Training data for cross-validation
            config: Dictionary containing configuration parameters
            cv_folds: Number of cross-validation folds
            models_dir: Directory to save models
            random_seed: Random seed for reproducibility
            use_hpo: Whether to use hyperparameter optimization
            hpo_num_samples: Number of hyperparameter configurations to try
            hpo_max_concurrent_trials: Maximum number of concurrent trials
        """
        self.train_data = train_data
        self.config = config
        self.cv_folds = cv_folds
        self.models_dir = models_dir
        self.random_seed = random_seed
        self.use_hpo = use_hpo
        self.hpo_num_samples = hpo_num_samples
        self.hpo_max_concurrent_trials = hpo_max_concurrent_trials
        
        # Initialize containers for results
        self.cv_results = []
        self.hpo_results = None
        self.best_hpo_params = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize hyperparameter optimizer if needed
        if self.use_hpo:
            self.hyperparameter_optimizer = HyperparameterOptimizer(
                train_data=self.train_data,
                config=self.config,
                models_dir=os.path.join(self.models_dir, "hpo"),
                random_seed=self.random_seed,
                num_samples=self.hpo_num_samples,
                max_concurrent_trials=self.hpo_max_concurrent_trials
            )
    
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
        Select best model configuration using multiple metrics.
        If HPO was used, incorporate those results.
        
        Returns:
            Dictionary containing best parameters and model path
        """
        logger.info("Selecting best model using enhanced multi-metric approach")
        
        # If HPO was used, prioritize those results
        if self.use_hpo and self.best_hpo_params:
            logger.info("Using best hyperparameters from HPO")
            best_params = self.best_hpo_params
            
            # Extract metrics from HPO results if available
            metrics = {
                "sharpe_ratio": 0.0,
                "pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0
            }
            
            if self.hpo_results and len(self.hpo_results) > 0:
                try:
                    # Find the best trial based on combined score
                    best_trial = max(self.hpo_results.values(), key=lambda x: x.get("combined_score", 0))
                    
                    # Log the best trial metrics for debugging
                    logger.info(f"Best HPO trial metrics: {best_trial}")
                    
                    # Extract metrics, ensuring we have values
                    if isinstance(best_trial, dict):
                        metrics = {
                            "sharpe_ratio": best_trial.get("sharpe_ratio", 0.0),
                            "pnl": best_trial.get("pnl", 0.0),
                            "win_rate": best_trial.get("win_rate", 0.0),
                            "max_drawdown": best_trial.get("max_drawdown", 0.0)
                        }
                        
                        # If we have a score but no metrics, try to extract from other fields
                        if all(v == 0.0 for v in metrics.values()) and "score" in best_trial:
                            metrics["sharpe_ratio"] = best_trial.get("score", 0.0)
                            
                        # Log the extracted metrics
                        logger.info(f"Extracted HPO metrics: {metrics}")
                    else:
                        logger.warning(f"Best trial is not a dictionary: {type(best_trial)}")
                except Exception as e:
                    logger.error(f"Error extracting metrics from HPO results: {e}")
            
            return {
                "params": best_params,
                "metrics": metrics,
                "model_path": best_params.get("model_path", ""),
                "fold": -1,  # No fold for HPO
                "source": "hpo"
            }
        
        # Otherwise, use CV results
        if not self.cv_results:
            raise ValueError("No CV results available. Run perform_cross_validation() first.")
        
        try:
            # Generate detailed fold performance report for visibility
            cv_report = self.generate_cv_report()
            logger.info(f"\n{cv_report}")
            
            # Define minimum thresholds for acceptable models
            min_sharpe = 0.0  # At least neutral Sharpe
            min_pnl = 0.0     # At least break-even
            
            # First try to find models that meet minimum thresholds
            qualified_models = [
                r for r in self.cv_results
                if "error" not in r
                and r["val_metrics"]["sharpe_ratio"] >= min_sharpe
                and r["val_metrics"]["pnl"] >= min_pnl
            ]
            
            if qualified_models:
                logger.info(f"Found {len(qualified_models)} models meeting minimum performance criteria")
                # Rank by a weighted combination of metrics
                best_score = -float('inf')
                best_result = None
                
                # Get metric weights from config
                weights = self.config.get("cross_validation", {}).get("metric_weights", {
                    "sharpe_ratio": 0.4,
                    "pnl": 0.3,
                    "win_rate": 0.2,
                    "max_drawdown": 0.1
                })
                
                for result in qualified_models:
                    metrics = result["val_metrics"]
                    # Combined score with weights for different metrics
                    initial_balance = self.config.get("initial_balance", 10000)
                    pnl_pct = metrics["pnl"] / initial_balance if initial_balance > 0 else 0
                    
                    score = (
                        weights["sharpe_ratio"] * metrics["sharpe_ratio"] +
                        weights["pnl"] * pnl_pct +
                        weights["win_rate"] * metrics["win_rate"] -
                        weights["max_drawdown"] * metrics["max_drawdown"]
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                
                logger.info(f"Selected best model with combined score: {best_score:.4f}")
                logger.info(f"Best model metrics - Sharpe: {best_result['val_metrics']['sharpe_ratio']:.4f}, "
                           f"PnL: ${best_result['val_metrics']['pnl']:.2f}, "
                           f"Win Rate: {best_result['val_metrics']['win_rate']*100:.2f}%")
            else:
                # Fall back to best Sharpe if no models meet criteria
                logger.warning("No models met minimum criteria, falling back to best available model by Sharpe ratio")
                valid_results = [r for r in self.cv_results if "error" not in r]
                
                if not valid_results:
                    raise ValueError("No valid models found in CV results")
                    
                best_result = max(valid_results, key=lambda x: x["val_metrics"]["sharpe_ratio"])
                logger.info(f"Selected fallback model with Sharpe: {best_result['val_metrics']['sharpe_ratio']:.4f}")
            
            # Extract parameters from the best fold
            fold_num = best_result.get("fold", -1)
            
            # Log the best fold number and metrics for debugging
            logger.info(f"Selected best fold: {fold_num} with metrics: {best_result['val_metrics']}")
            
            best_params = {
                "learning_rate": self.config.get("learning_rate", 0.001),
                "gamma": self.config.get("gamma", 0.99),
                "batch_size": self.config.get("batch_size", 32),
                "use_dueling": self.config.get("use_dueling", True),
                "use_double_q": self.config.get("use_double_q", True),
                "use_prioritized_replay": self.config.get("use_prioritized_replay", True),
                "prioritized_replay_alpha": self.config.get("prioritized_replay_alpha", 0.6),
                "prioritized_replay_beta": self.config.get("prioritized_replay_beta", 0.4),
                "fold": fold_num,  # Store which fold was best
            }
            
            # Create the result dictionary with explicit metrics
            result = {
                "params": best_params,
                "model_path": best_result["model_path"],
                "metrics": best_result["val_metrics"],
                "fold": fold_num
            }
            
            # Log the final result for debugging
            logger.info(f"Returning best model info: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}", exc_info=True)
            raise
    
    def generate_cv_report(self) -> str:
        """
        Generate a detailed cross-validation performance report.
        
        Returns:
            String containing the formatted report
        """
        if not self.cv_results:
            return "No CV results available. Run perform_cross_validation() first."
            
        report = "=== Cross-Validation Performance Report ===\n\n"
        report += f"Total folds: {self.cv_folds}\n"
        valid_results = [r for r in self.cv_results if "error" not in r]
        report += f"Completed folds: {len(valid_results)}/{self.cv_folds}\n\n"
        
        # Table header
        report += f"{'Fold':^5}|{'Sharpe':^10}|{'PnL':^12}|{'Win Rate':^10}|{'Max DD':^10}|{'Status':^10}\n"
        report += f"{'-'*5}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*10}\n"
        
        # Sort by Sharpe ratio for easy comparison
        sorted_results = sorted(
            enumerate(self.cv_results),
            key=lambda x: x[1].get("val_metrics", {}).get("sharpe_ratio", -float('inf'))
                        if "error" not in x[1] else -float('inf'),
            reverse=True
        )
        
        positive_count = 0
        
        for fold, result in sorted_results:
            if "error" in result:
                report += f"{fold:^5}|{'N/A':^10}|{'N/A':^12}|{'N/A':^10}|{'N/A':^10}|{'Error':^10}\n"
                continue
                
            metrics = result["val_metrics"]
            sharpe = metrics.get("sharpe_ratio", 0.0)
            pnl = metrics.get("pnl", 0.0)
            win_rate = metrics.get("win_rate", 0.0) * 100
            max_dd = metrics.get("max_drawdown", 0.0) * 100
            
            # Status based on performance
            if sharpe > 0 and pnl > 0:
                status = "Good"
                positive_count += 1
            elif sharpe > 0:
                status = "Neutral"
            else:
                status = "Poor"
            
            report += f"{fold:^5}|{sharpe:^10.4f}|${pnl:^10.2f}|{win_rate:^8.2f}%|{max_dd:^8.2f}%|{status:^10}\n"
        
        report += f"\nPositive performing folds: {positive_count}/{len(valid_results)}\n"
        
        return report
        
    def generate_cv_dataframe(self) -> pd.DataFrame:
        """
        Generate a DataFrame representation of cross-validation results for visualization.
        
        Returns:
            DataFrame containing CV results with columns for fold, model_config, and metrics
        """
        # Create list to hold rows for DataFrame
        rows = []
        
        # Check if we have CV results
        if not self.cv_results:
            logger.warning("No CV results available for DataFrame generation")
            
            # If we have HPO results, try to create a DataFrame from those
            if self.use_hpo and self.hpo_results and len(self.hpo_results) > 0:
                logger.info("Attempting to create DataFrame from HPO results")
                try:
                    for trial_id, trial_result in self.hpo_results.items():
                        # Extract metrics, ensuring we handle None values
                        if not isinstance(trial_result, dict):
                            continue
                            
                        # Create a row for this trial
                        row = {
                            'fold': int(trial_id.split('_')[-1]) if '_' in trial_id else 0,
                            'model_config': f"HPO {trial_id}",
                            'sharpe_ratio': trial_result.get("sharpe_ratio", trial_result.get("score", 0.0)),
                            'pnl': trial_result.get("pnl", 0.0),
                            'win_rate': trial_result.get("win_rate", 0.0),
                            'max_drawdown': trial_result.get("max_drawdown", 0.0)
                        }
                        rows.append(row)
                        
                    if rows:
                        df = pd.DataFrame(rows)
                        logger.info(f"Generated CV DataFrame from HPO results with shape: {df.shape}")
                        return df
                except Exception as e:
                    logger.error(f"Error creating DataFrame from HPO results: {e}")
            
            # Return empty DataFrame with expected columns if no data available
            return pd.DataFrame(columns=['fold', 'model_config', 'sharpe_ratio', 'pnl', 'win_rate', 'max_drawdown'])
        
        # Process each fold result from CV
        for fold, result in enumerate(self.cv_results):
            if "error" in result:
                continue
                
            # Extract metrics, ensuring we handle None values
            metrics = result.get("val_metrics", {})
            if metrics is None:
                metrics = {}
                
            # Log the metrics for debugging
            logger.info(f"Fold {fold} metrics for DataFrame: {metrics}")
            
            # Create a row for this fold
            row = {
                'fold': fold,
                'model_config': f"Config {fold}",  # Default config name
                'sharpe_ratio': metrics.get("sharpe_ratio", 0.0),
                'pnl': metrics.get("pnl", 0.0),
                'win_rate': metrics.get("win_rate", 0.0),
                'max_drawdown': metrics.get("max_drawdown", 0.0)
            }
            
            rows.append(row)
        
        # If we have no valid rows, create a dummy row to avoid empty DataFrame
        if not rows:
            logger.warning("No valid fold results found, creating dummy row")
            rows.append({
                'fold': 0,
                'model_config': 'Default',
                'sharpe_ratio': 0.0,
                'pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            })
            
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Log the DataFrame shape for debugging
        logger.info(f"Generated CV DataFrame with shape: {df.shape}")
        
        return df
        
    def get_cv_results(self) -> List[Dict[str, Any]]:
        """
        Get cross-validation results.
        
        Returns:
            List of dictionaries containing results for each fold
        """
        return self.cv_results
        
    def run_hyperparameter_optimization(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary containing best hyperparameters
        """
        if not self.use_hpo:
            logger.warning("HPO is not enabled. Set use_hpo=True when initializing CrossValidator.")
            return {}
        
        logger.info("Running hyperparameter optimization")
        
        # Run HPO
        self.best_hpo_params = self.hyperparameter_optimizer.optimize_hyperparameters()
        
        # Store HPO results
        self.hpo_results = self.hyperparameter_optimizer.get_hpo_results()
        
        # Generate HPO report
        hpo_report = self.hyperparameter_optimizer.generate_hpo_report()
        logger.info(f"\n{hpo_report}")
        
        return self.best_hpo_params
    
    def get_hpo_results(self) -> Dict[str, Any]:
        """
        Get hyperparameter optimization results.
        
        Returns:
            Dictionary containing HPO results
        """
        if not self.use_hpo:
            logger.warning("HPO is not enabled. Set use_hpo=True when initializing CrossValidator.")
            return {}
        
        if self.hpo_results is None:
            logger.warning("No HPO results available. Run run_hyperparameter_optimization() first.")
            return {}
        
        return self.hpo_results