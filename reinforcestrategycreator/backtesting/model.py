"""
Model training and selection module for backtesting.

This module provides functionality for training and selecting models
for backtesting reinforcement learning trading strategies.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import torch
import ray
import time

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
    Includes parallel training capabilities using Ray.
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
    
    @ray.remote
    def _train_episode_batch(
        batch_id: int,
        batch_size: int,
        start_episode: int,
        train_data_ref: ray.ObjectRef,
        env_config_base: Dict[str, Any],
        state_size: int,
        action_size: int,
        agent_params: Dict[str, Any],
        random_seed: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Train a batch of episodes remotely and collect experiences.
        
        Args:
            batch_id: Batch identification number
            batch_size: Number of episodes in this batch
            start_episode: Starting episode number
            train_data_ref: Reference to training data in Ray object store
            env_config_base: Base environment configuration
            state_size: Agent state size
            action_size: Agent action size
            agent_params: Agent parameters
            random_seed: Random seed for reproducibility
            
        Returns:
            List of experience tuples collected during training
        """
        batch_logger = logging.getLogger(f"{__name__}.batch{batch_id}")
        batch_logger.info(f"Starting training batch {batch_id}, episodes {start_episode}-{start_episode+batch_size-1}")
        
        # Set random seed for reproducibility (unique per batch)
        batch_seed = random_seed + batch_id
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        
        try:
            # Create environment config
            env_config = env_config_base.copy()
            env_config["df"] = train_data_ref
            
            # Create environment
            env = TradingEnvironment(env_config=env_config)
            
            # Create agent with specified parameters and enhanced DQN features
            agent = RLAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=agent_params.get("learning_rate", 0.001),
                gamma=agent_params.get("gamma", 0.99),
                epsilon=agent_params.get("epsilon", 1.0),
                epsilon_decay=agent_params.get("epsilon_decay", 0.995),
                epsilon_min=agent_params.get("epsilon_min", 0.01),
                # Enhanced DQN features
                use_dueling=env_config_base.get("use_dueling", True),
                use_double_q=env_config_base.get("use_double_q", True),
                use_prioritized_replay=env_config_base.get("use_prioritized_replay", True),
                prioritized_replay_alpha=env_config_base.get("prioritized_replay_alpha", 0.6),
                prioritized_replay_beta=env_config_base.get("prioritized_replay_beta", 0.4),
                prioritized_replay_beta_annealing=True
            )
            
            # Apply epsilon decay for starting episode (to match sequential training)
            for i in range(start_episode):
                agent.epsilon = max(
                    agent.epsilon_min,
                    agent.epsilon * agent.epsilon_decay
                )
            
            # Initialize collection of experiences
            all_experiences = []
            
            # Run episodes in this batch
            for i in range(batch_size):
                episode = start_episode + i
                experiences = []
                
                state = env.reset()
                done = False
                
                while not done:
                    # Handle tuple states properly
                    if isinstance(state, tuple):
                        # For tuples, extract the first element if it's the observation
                        # or flatten the tuple structure if needed
                        if len(state) > 0 and isinstance(state[0], (np.ndarray, list)):
                            state = state[0]  # Use just the observation part
                        else:
                            # Try to convert the tuple elements to a flat list
                            try:
                                flat_state = []
                                for item in state:
                                    if isinstance(item, (list, np.ndarray)):
                                        flat_state.extend(item)
                                    else:
                                        flat_state.append(item)
                                state = np.array(flat_state, dtype=np.float32)
                            except:
                                # If conversion fails, log and use a random action
                                batch_logger.warning(f"Could not convert state tuple to array, using random action")
                                action = np.random.randint(action_size)
                                continue
                    # Get action and confidence from agent
                    action, confidence = agent.select_action(state, return_confidence=True)
                    # Pass confidence to environment for dynamic position sizing
                    next_state, reward, terminated, truncated, info = env.step(action, confidence)
                    done = terminated or truncated
                    
                    # Store experience tuple
                    experience = (state, action, reward, next_state, done)
                    experiences.append(experience)
                    
                    state = next_state
                
                # Update epsilon for next episode
                agent.epsilon = max(
                    agent.epsilon_min,
                    agent.epsilon * agent.epsilon_decay
                )
                
                # Collect experiences from this episode
                all_experiences.extend(experiences)
                
                if (i + 1) % 5 == 0 or i == batch_size - 1:
                    batch_logger.info(f"Batch {batch_id}: {i+1}/{batch_size} episodes completed")
            
            batch_logger.info(f"Batch {batch_id} completed, collected {len(all_experiences)} experiences")
            return all_experiences
            
        except Exception as e:
            batch_logger.error(f"Error in training batch {batch_id}: {e}", exc_info=True)
            return []  # Return empty experiences list on error
    
    def train_final_model(self, train_data: pd.DataFrame, best_params: Dict[str, Any]) -> RLAgent:  # Using RLAgent alias
        """
        Train final model on complete dataset using parallel processing.
        
        Args:
            train_data: Complete training dataset
            best_params: Best hyperparameters from cross-validation
            
        Returns:
            Trained RL agent
        """
        start_time = time.time()
        logger.info("Training final model on complete training dataset using parallel processing")
        
        if train_data is None or len(train_data) == 0:
            raise ValueError("No training data available")
            
        # Update configuration to include enhanced DQN features
        self.config["use_dueling"] = True  # Task 3.1
        self.config["use_double_q"] = True  # Task 3.1
        self.config["use_prioritized_replay"] = True  # Task 3.2
        self.config["prioritized_replay_alpha"] = 0.6  # Task 3.2
        self.config["prioritized_replay_beta"] = 0.4  # Task 3.2
        self.config["learning_rate"] = 2e-4  # Task 3.3
            
        self.best_params = best_params
        
        try:
            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=True)
                logger.info("Ray initialized for parallel model training")
                
            # Put the DataFrame in the Ray object store
            train_data_ref = ray.put(train_data)
            
            # Base environment configuration
            env_config_base = {
                "initial_balance": self.config.get("initial_balance", 10000),
                "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
                "window_size": self.config.get("window_size", 10),
                "sharpe_window_size": self.config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
                "trading_incentive": self.config.get("trading_incentive", 0.002),
                "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
                "risk_fraction": self.config.get("risk_fraction", 0.1),
                "normalization_window_size": self.config.get("normalization_window_size", 20),
                # Dynamic position sizing parameters
                "use_dynamic_sizing": self.config.get("use_dynamic_sizing", False),
                "min_risk_fraction": self.config.get("min_risk_fraction", 0.05),
                "max_risk_fraction": self.config.get("max_risk_fraction", 0.20)
            }
            
            # Create a temporary environment to determine state and action sizes
            # First, create a regular environment without Ray to get shapes
            direct_env_config = env_config_base.copy()
            direct_env_config["df"] = train_data  # Use DataFrame directly, not Ray ref
            
            # Create a non-Ray environment for shape determination only
            class_name = TradingEnvironment.__name__
            logger.info(f"Creating temporary {class_name} to determine state/action sizes")
            tmp_env = TradingEnvironment(env_config=direct_env_config)
            state_size = tmp_env.observation_space.shape[0]
            action_size = tmp_env.action_space.n
            logger.info(f"State size: {state_size}, Action size: {action_size}")
            
            # Create main agent with best parameters and enhanced DQN features
            agent = RLAgent(  # Using RLAgent alias
                state_size=state_size,
                action_size=action_size,
                learning_rate=self.config.get("learning_rate", 2e-4),  # Use reduced learning rate from config
                gamma=self.best_params.get("gamma", 0.99),
                epsilon=self.config.get("epsilon", 1.0),
                epsilon_decay=self.config.get("epsilon_decay", 0.995),
                epsilon_min=self.config.get("epsilon_min", 0.01),
                # Enhanced DQN features from config
                use_dueling=self.config.get("use_dueling", True),
                use_double_q=self.config.get("use_double_q", True),
                use_prioritized_replay=self.config.get("use_prioritized_replay", True),
                prioritized_replay_alpha=self.config.get("prioritized_replay_alpha", 0.6),
                prioritized_replay_beta=self.config.get("prioritized_replay_beta", 0.4),
                prioritized_replay_beta_annealing=True  # Enable beta annealing
            )
            
            # Agent parameters for consistent initialization across workers
            agent_params = {
                "learning_rate": agent.learning_rate,
                "gamma": agent.gamma,
                "epsilon": self.config.get("epsilon", 1.0),
                "epsilon_decay": self.config.get("epsilon_decay", 0.995),
                "epsilon_min": self.config.get("epsilon_min", 0.01)
            }
            
            # Configure training parameters
            total_episodes = self.config.get("final_episodes", 200)  # Total number of episodes
            # Get all available CPUs while reserving a few for system operations
            available_cpus = ray.available_resources().get("CPU", 2)
            num_cpus = int(max(available_cpus - 4, 4))  # Reserve 4 CPUs for main thread and system, minimum 4
            num_batches = int(min(num_cpus * 4, total_episodes))  # Create 4 batches per CPU (increased from 2) but don't exceed total_episodes
            batch_size = max(total_episodes // num_batches, 1)  # Ensure at least 1 episode per batch
            
            logger.info(f"Parallel training setup: {total_episodes} episodes across {num_batches} batches "
                       f"({batch_size} episodes per batch) using {num_cpus} CPUs")
            
            # Train in batches
            batch_futures = []
            for batch_id in range(num_batches):
                start_episode = batch_id * batch_size
                # Adjust batch size for the last batch to account for any remainder
                current_batch_size = min(batch_size, total_episodes - start_episode)
                
                # Skip if no episodes left
                if current_batch_size <= 0:
                    continue
                
                # Launch the remote batch training
                batch_future = self._train_episode_batch.remote(
                    batch_id,
                    current_batch_size,
                    start_episode,
                    train_data_ref,
                    env_config_base,
                    state_size,
                    action_size,
                    agent_params,
                    self.random_seed  # Base random seed
                )
                batch_futures.append(batch_future)
                
            # Collect experiences from all batches
            logger.info(f"Launched {len(batch_futures)} training batches, waiting for completion...")
            batch_results = ray.get(batch_futures)
            
            # Aggregate all experiences
            all_experiences = []
            for experiences in batch_results:
                all_experiences.extend(experiences)
                
            logger.info(f"Collected {len(all_experiences)} experiences from all batches")
            
            # Store experiences in agent's memory
            # Reset memory first to avoid potential duplication
            agent.memory = []
            for experience in all_experiences:
                state, action, reward, next_state, done = experience
                agent.remember(state, action, reward, next_state, done)
            
            # Train agent on collected experiences
            batch_size = self.best_params.get("batch_size", 32)
            num_batches_to_train = len(agent.memory) // batch_size
            logger.info(f"Training main agent on {len(agent.memory)} experiences ({num_batches_to_train} batches)")
            
            # Initialize metrics tracking for PER
            per_loss_values = []
            priority_mean_values = []
            
            for i in range(num_batches_to_train):
                result = agent.learn(return_stats=True)  # Get training stats
                
                # Extract and track PER metrics if available
                if isinstance(result, dict) and 'td_error' in result:
                    per_loss_values.append(result.get('td_error', 0))
                    priority_mean_values.append(result.get('mean_priority', 0))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Training progress: {i+1}/{num_batches_to_train} batches completed")
                    if per_loss_values and priority_mean_values:
                        logger.info(f"Recent PER metrics - Loss: {per_loss_values[-1]:.4f}, Priority Mean: {priority_mean_values[-1]:.4f}")
            
            # Calculate and log average PER metrics if available
            avg_per_loss = sum(per_loss_values) / len(per_loss_values) if per_loss_values else 0.0
            avg_priority_mean = sum(priority_mean_values) / len(priority_mean_values) if priority_mean_values else 0.0
            logger.info(f"Training completed with avg PER Loss: {avg_per_loss:.4f}, avg Priority Mean: {avg_priority_mean:.4f}")
            
            # Save PER metrics to agent
            agent.per_metrics = {
                'td_error': avg_per_loss,
                'mean_priority': avg_priority_mean
            }
            
            # Add method to retrieve PER metrics
            def get_per_metrics(self):
                return getattr(self, 'per_metrics', {'td_error': 0.0, 'mean_priority': 0.0})
            
            # Add the method to the agent instance
            import types
            agent.get_per_metrics = types.MethodType(get_per_metrics, agent)
            
            # Save final model
            final_model_path = os.path.join(self.models_dir, "final_model.pth")
            # Save the PyTorch model directly since agent.save_model is just a placeholder
            torch.save(agent.model.state_dict(), final_model_path)
            
            # Store final model
            self.best_model = agent
            
            elapsed_time = time.time() - start_time
            logger.info(f"Final model trained and saved to {final_model_path} ({elapsed_time:.2f} seconds)")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error training final model: {e}", exc_info=True)
            raise
    
    @ray.remote
    def _evaluate_episode_remotely(
        episode_id: int,
        test_data_ref: ray.ObjectRef,
        model_state_dict: Dict,
        env_config_base: Dict[str, Any],
        state_size: int,
        action_size: int,
        agent_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evaluate model in a single episode remotely.
        
        Args:
            episode_id: Episode identification number
            test_data_ref: Reference to test data in Ray object store
            model_state_dict: PyTorch model state dictionary
            env_config_base: Base environment configuration
            state_size: Agent state size
            action_size: Agent action size
            agent_params: Agent parameters
            
        Returns:
            Dictionary of evaluation metrics for this episode
        """
        # Create metrics calculator for this worker
        metrics_calculator = MetricsCalculator()
        
        eval_logger = logging.getLogger(f"{__name__}.eval{episode_id}")
        eval_logger.info(f"Starting evaluation episode {episode_id}")
        
        try:
            # Create environment config
            env_config = env_config_base.copy()
            env_config["df"] = test_data_ref
            
            # Create environment
            env = TradingEnvironment(env_config=env_config)
            
            # Create agent with enhanced DQN features
            agent = RLAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=agent_params.get("learning_rate", 0.001),
                gamma=agent_params.get("gamma", 0.99),
                epsilon=agent_params.get("epsilon_min", 0.01),  # Use minimal exploration for evaluation
                epsilon_decay=1.0,  # No decay during evaluation
                epsilon_min=agent_params.get("epsilon_min", 0.01),
                # Enhanced DQN features
                use_dueling=env_config_base.get("use_dueling", True),
                use_double_q=env_config_base.get("use_double_q", True),
                use_prioritized_replay=env_config_base.get("use_prioritized_replay", True),
                prioritized_replay_alpha=env_config_base.get("prioritized_replay_alpha", 0.6),
                prioritized_replay_beta=env_config_base.get("prioritized_replay_beta", 0.4),
                prioritized_replay_beta_annealing=False  # No annealing during evaluation
            )
            
            # Load model state dictionary
            agent.model.load_state_dict(model_state_dict)
            agent.model.eval()  # Set to evaluation mode
            
            # Run evaluation episode
            state = env.reset()
            done = False
            
            while not done:
                # Handle tuple states properly
                if isinstance(state, tuple):
                    # For tuples, extract the first element if it's the observation
                    # or flatten the tuple structure if needed
                    if len(state) > 0 and isinstance(state[0], (np.ndarray, list)):
                        state = state[0]  # Use just the observation part
                    else:
                        # Try to convert the tuple elements to a flat list
                        try:
                            flat_state = []
                            for item in state:
                                if isinstance(item, (list, np.ndarray)):
                                    flat_state.extend(item)
                                else:
                                    flat_state.append(item)
                            state = np.array(flat_state, dtype=np.float32)
                        except:
                            # If conversion fails, log and use a random action
                            eval_logger.warning(f"Could not convert state tuple to array, using random action")
                            action = np.random.randint(action_size)
                            continue
                # Get action and confidence from agent
                action, confidence = agent.select_action(state, return_confidence=True)
                # Pass confidence to environment for dynamic position sizing
                next_state, reward, terminated, truncated, info = env.step(action, confidence)
                done = terminated or truncated
                state = next_state
            
            # Calculate metrics
            metrics = metrics_calculator.get_episode_metrics(env)
            eval_logger.info(f"Episode {episode_id} evaluation completed with PnL: ${metrics['pnl']:.2f}")
            
            return metrics
            
        except Exception as e:
            eval_logger.error(f"Error in evaluation episode {episode_id}: {e}", exc_info=True)
            return {
                "error": str(e),
                "pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }

    def evaluate_model(self, model: RLAgent, test_data: pd.DataFrame) -> Dict[str, float]:  # Using RLAgent alias
        """
        Evaluate model on test data using multiple parallel evaluation episodes.
        
        Args:
            model: Trained RL agent
            test_data: Test dataset
            
        Returns:
            Dictionary of evaluation metrics (averaged across multiple evaluation episodes)
        """
        start_time = time.time()
        logger.info("Evaluating model on test data using parallel processing")
        
        if model is None:
            raise ValueError("No model available for evaluation")
            
        if test_data is None or len(test_data) == 0:
            raise ValueError("No test data available for evaluation")
            
        try:
            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=True)
                logger.info("Ray initialized for parallel model evaluation")
            
            # Put the test data into Ray's object store
            test_data_ref = ray.put(test_data)
            
            # Base environment configuration
            env_config_base = {
                "initial_balance": self.config.get("initial_balance", 10000),
                "transaction_fee_percent": self.config.get("transaction_fee", 0.001),
                "window_size": self.config.get("window_size", 10),
                "sharpe_window_size": self.config.get("sharpe_window_size", 100),
                "use_sharpe_ratio": self.config.get("use_sharpe_ratio", True),
                "trading_frequency_penalty": self.config.get("trading_frequency_penalty", 0.001),
                "trading_incentive": self.config.get("trading_incentive", 0.002),
                "drawdown_penalty": self.config.get("drawdown_penalty", 0.001),
                "risk_fraction": self.config.get("risk_fraction", 0.1),
                "normalization_window_size": self.config.get("normalization_window_size", 20),
                # Dynamic position sizing parameters
                "use_dynamic_sizing": self.config.get("use_dynamic_sizing", False),
                "min_risk_fraction": self.config.get("min_risk_fraction", 0.05),
                "max_risk_fraction": self.config.get("max_risk_fraction", 0.20)
            }
            
            # Create a temporary environment to determine state and action sizes
            tmp_env_config = env_config_base.copy()
            tmp_env_config["df"] = test_data_ref
            tmp_env = TradingEnvironment(env_config=tmp_env_config)
            state_size = tmp_env.observation_space.shape[0]
            action_size = tmp_env.action_space.n
            
            # Get model state dictionary for distribution to workers
            model_state_dict = model.model.state_dict()
            
            # Agent parameters for consistent initialization
            agent_params = {
                "learning_rate": model.learning_rate,
                "gamma": model.gamma,
                "epsilon": model.epsilon_min,  # Use minimal epsilon for evaluation
                "epsilon_decay": 1.0,
                "epsilon_min": model.epsilon_min
            }
            
            # Perform multiple evaluation episodes in parallel for more stable metrics
            # Configure evaluation with more parallel episodes
            max_available_cpus = ray.available_resources().get("CPU", 2) - 4  # Reserve 4 CPUs for main threads
            num_eval_episodes = int(min(max(max_available_cpus, 10), 30))  # Between 10-30 episodes based on available CPUs
            logger.info(f"Running {num_eval_episodes} parallel evaluation episodes")
            
            # Launch parallel evaluation episodes
            episode_futures = [
                self._evaluate_episode_remotely.remote(
                    i,  # episode_id
                    test_data_ref,
                    model_state_dict,
                    env_config_base,
                    state_size,
                    action_size,
                    agent_params
                )
                for i in range(num_eval_episodes)
            ]
            
            # Wait for all episodes to complete
            logger.info(f"Launched {num_eval_episodes} evaluation episodes, waiting for completion...")
            episode_metrics = ray.get(episode_futures)
            
            # Filter out episodes with errors
            valid_metrics = [m for m in episode_metrics if "error" not in m]
            if not valid_metrics:
                raise ValueError("All evaluation episodes encountered errors")
            
            # Average the metrics across all episodes
            # Get PER metrics from model if available
            per_loss = 0.0
            priority_mean = 0.0
            
            # Try to extract PER metrics if the model has them
            if hasattr(model, 'get_per_metrics'):
                per_metrics = model.get_per_metrics()
                per_loss = per_metrics.get('td_error', 0.0)
                priority_mean = per_metrics.get('mean_priority', 0.0)
                
            aggregated_metrics = {
                # Add PER metrics to the aggregated metrics
                "per_loss": per_loss,
                "priority_mean": priority_mean,
            }
            for key in ["pnl", "sharpe_ratio", "max_drawdown", "win_rate", "trades"]:
                values = [m.get(key, 0) for m in valid_metrics]
                aggregated_metrics[key] = sum(values) / len(values) if values else 0
            
            # Add derived metrics
            aggregated_metrics['pnl_percentage'] = (aggregated_metrics['pnl'] / self.config.get('initial_balance', 10000)) * 100
            
            elapsed_time = time.time() - start_time
            logger.info(f"Parallel model evaluation completed with avg PnL: ${aggregated_metrics['pnl']:.2f} ({elapsed_time:.2f} seconds)")
            
            return aggregated_metrics
            
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