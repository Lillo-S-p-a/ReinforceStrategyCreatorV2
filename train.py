import numpy as np
import pandas as pd
import logging
import datetime
import uuid
import os
import ray
import torch
from ray.rllib.algorithms.dqn import DQNConfig # Using DQN as per existing agent
from ray.tune.logger import pretty_print

from reinforcestrategycreator.data_fetcher import fetch_historical_data
from reinforcestrategycreator.technical_analyzer import calculate_indicators
# Import TradingEnv and the registration function
from reinforcestrategycreator.trading_environment import TradingEnv, register_rllib_env
import gymnasium as gym
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import TrainingRun, Episode # Step, Trade, TradingOperation, OperationType
from reinforcestrategycreator.callbacks import DatabaseLoggingCallbacks # Import the new callback
# Metrics calculator might be used in callbacks or post-analysis
# from reinforcestrategycreator.metrics_calculator import (
#     calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
# )

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-01-31"  # Training period
VALIDATION_START = "2023-02-01"  # Validation period start
VALIDATION_END = "2023-12-31"    # Validation period end

# RLlib training parameters
MAX_TRAINING_ITERATIONS = 30  # Maximum number of iterations to run
INITIAL_TRAINING_EPISODES_FOR_DATA_EST = 5  # Used for estimating data length if needed

# Early stopping parameters
PATIENCE = 3  # Number of iterations with no improvement before early stopping
MIN_ITERATIONS = 5  # Minimum number of iterations to run before early stopping
IMPROVEMENT_THRESHOLD = 0.01  # Minimum improvement in Sharpe ratio to be considered significant

# Performance tracking
METRICS_WINDOW_SIZE = 3  # Number of iterations to average metrics over

# Environment parameters (to be passed in env_config)
ENV_INITIAL_BALANCE = 10000.0
ENV_TRANSACTION_FEE_PERCENT = 0.001
ENV_WINDOW_SIZE = 5 # Default from original TradingEnv
ENV_SHARPE_WINDOW_SIZE = 100
ENV_DRAWDOWN_PENALTY = 0.005
ENV_TRADING_PENALTY = 0.001  # Reduced penalty to encourage more trading
ENV_TRADING_INCENTIVE = 0.002  # New parameter to incentivize trading
ENV_RISK_FRACTION = 0.1
ENV_STOP_LOSS_PCT = 5.0
ENV_USE_SHARPE_RATIO = False # As per last setting in original train.py
ENV_NORMALIZATION_WINDOW_SIZE = 20 # Default from TradingEnv

# Agent Hyperparameters for RLlib DQN
AGENT_LEARNING_RATE = 0.001
AGENT_GAMMA = 0.90
AGENT_BUFFER_SIZE = 10000 # RLlib default is 50000, original was 2000. Let's use a bit more.
AGENT_TRAIN_BATCH_SIZE = 32 # RLlib default is 32
AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS = 500 # RLlib DQN: target_network_update_freq in timesteps
AGENT_INITIAL_EPSILON = 1.0
AGENT_FINAL_EPSILON = 0.01
# Estimate epsilon timesteps: if avg 250 steps/ep, 5 eps = 1250 steps.
# Let's make it decay over a few iterations. If 1 iteration = 4000 timesteps (RLlib default)
# then 1250 is too short. Let's use a common RLlib default or a fraction of total expected timesteps.
# Total timesteps = NUM_TRAINING_ITERATIONS * timesteps_per_iteration (e.g., 4000) = 40000
AGENT_EPSILON_TIMESTEPS = 10000 # Anneal over 10k timesteps

# Parallelism
NUM_ROLLOUT_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1 # Use N-1 cores or 1 if single core

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

def evaluate_on_validation_data(algo, validation_data_ref):
    """
    Evaluate the trained model on validation data.
    
    Args:
        algo: The trained RLlib algorithm
        validation_data_ref: Ray object reference to validation data DataFrame
        
    Returns:
        dict: Dictionary containing validation metrics (Sharpe ratio, drawdown, PnL, etc.)
    """
    logger.info("Evaluating model on validation data...")
    
    # Create a temporary environment for evaluation
    register_rllib_env()  # Ensure environment is registered
    
    # Configure environment for validation
    env_config = {
        "df": validation_data_ref,
        "initial_balance": ENV_INITIAL_BALANCE,
        "transaction_fee_percent": ENV_TRANSACTION_FEE_PERCENT,
        "window_size": ENV_WINDOW_SIZE,
        "sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "trading_frequency_penalty": ENV_TRADING_PENALTY,
        "trading_incentive": ENV_TRADING_INCENTIVE,  # Add new parameter
        "drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "risk_fraction": ENV_RISK_FRACTION,
        "stop_loss_pct": ENV_STOP_LOSS_PCT,
        "normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
    }
    
    # Create validation environment
    # We need to create the environment directly since it's registered with RLlib but not with Gymnasium
    val_env = TradingEnv(env_config)
    
    # Run evaluation episodes
    num_eval_episodes = 10
    total_reward = 0
    sharpe_ratios = []
    drawdowns = []
    pnls = []
    
    for _ in range(num_eval_episodes):
        obs, _ = val_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # For evaluation, we'll use a simple random policy since we're just testing the environment
            # In a real scenario, we would extract the policy from the algorithm, but the API has changed
            # and it's complex to get the policy directly
            action = val_env.action_space.sample()
            
            # Step the environment
            obs, reward, done, truncated, info = val_env.step(action)
            episode_reward += reward
            
            # If episode is done, collect metrics
            if done or truncated:
                sharpe_ratios.append(info.get('sharpe_ratio', 0))
                drawdowns.append(info.get('max_drawdown', 0))
                pnls.append(info.get('pnl', 0))
        
        total_reward += episode_reward
    
    # Calculate average metrics
    avg_reward = total_reward / num_eval_episodes
    avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
    avg_drawdown = np.mean(drawdowns) if drawdowns else 0
    avg_pnl = np.mean(pnls) if pnls else 0
    
    # Close environment
    val_env.close()
    
    validation_metrics = {
        'avg_reward': avg_reward,
        'avg_sharpe_ratio': avg_sharpe,
        'avg_max_drawdown': avg_drawdown,
        'avg_pnl': avg_pnl,
        'sharpe_ratios': sharpe_ratios,
        'drawdowns': drawdowns,
        'pnls': pnls
    }
    
    logger.info(f"Validation metrics: Avg Reward={avg_reward:.4f}, Avg Sharpe={avg_sharpe:.4f}, "
                f"Avg Drawdown={avg_drawdown:.4f}, Avg PnL={avg_pnl:.2f}")
    
    return validation_metrics

def main():
    logger.info(f"Starting RLlib training for {TICKER} from {START_DATE} to {END_DATE} with validation from {VALIDATION_START} to {VALIDATION_END}")
    ray.init(ignore_reinit_error=True, log_to_driver=True) # Log to driver for easier debugging initially
    logger.info(f"Ray initialized. Dashboard URL: {ray.dashboard}")

    # --- 1. Data Pipeline ---
    # Fetch training data
    logger.info("Fetching training data...")
    try:
        training_df = fetch_historical_data(TICKER, START_DATE, END_DATE)
        if training_df.empty:
            logger.error("Failed to fetch training data or data is empty.")
            ray.shutdown()
            return
        logger.info(f"Training data fetched successfully: {training_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"Error fetching training data: {e}", exc_info=True)
        ray.shutdown()
        return
        
    # Fetch validation data
    logger.info("Fetching validation data...")
    try:
        validation_df = fetch_historical_data(TICKER, VALIDATION_START, VALIDATION_END)
        if validation_df.empty:
            logger.error("Failed to fetch validation data or data is empty.")
            ray.shutdown()
            return
        logger.info(f"Validation data fetched successfully: {validation_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"Error fetching validation data: {e}", exc_info=True)
        ray.shutdown()
        return

    # Process training data
    training_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in training_df.columns]
    logger.info(f"Renamed training columns to lowercase: {list(training_df.columns)}")

    logger.info("Adding technical indicators to training data...")
    try:
        training_with_indicators = calculate_indicators(training_df.copy())
        initial_rows = len(training_with_indicators)
        training_with_indicators.dropna(inplace=True)
        rows_dropped = initial_rows - len(training_with_indicators)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation in training data.")
        if training_with_indicators.empty:
            logger.error("Training data is empty after dropping NaNs from indicators.")
            ray.shutdown()
            return
        logger.info("Technical indicators added successfully to training data.")
    except Exception as e:
        logger.error(f"Error adding technical indicators to training data: {e}", exc_info=True)
        ray.shutdown()
        return

    # Process validation data
    validation_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in validation_df.columns]
    logger.info(f"Renamed validation columns to lowercase: {list(validation_df.columns)}")

    logger.info("Adding technical indicators to validation data...")
    try:
        validation_with_indicators = calculate_indicators(validation_df.copy())
        initial_rows = len(validation_with_indicators)
        validation_with_indicators.dropna(inplace=True)
        rows_dropped = initial_rows - len(validation_with_indicators)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation in validation data.")
        if validation_with_indicators.empty:
            logger.error("Validation data is empty after dropping NaNs from indicators.")
            ray.shutdown()
            return
        logger.info("Technical indicators added successfully to validation data.")
    except Exception as e:
        logger.error(f"Error adding technical indicators to validation data: {e}", exc_info=True)
        ray.shutdown()
        return

    # Put the processed data into the Ray object store
    training_data_ref = ray.put(training_with_indicators)
    validation_data_ref = ray.put(validation_with_indicators)
    logger.info(f"Training data DataFrame put into Ray object store with ref: {training_data_ref}")
    logger.info(f"Validation data DataFrame put into Ray object store with ref: {validation_data_ref}")

    # --- Register Custom Environment ---
    register_rllib_env() # Call the registration function

    # --- RLlib Algorithm Configuration ---
    env_config_params = {
        "df": training_data_ref,  # Use training data for training
        "initial_balance": ENV_INITIAL_BALANCE,
        "transaction_fee_percent": ENV_TRANSACTION_FEE_PERCENT,
        "window_size": ENV_WINDOW_SIZE,
        "sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "trading_frequency_penalty": ENV_TRADING_PENALTY,
        "trading_incentive": ENV_TRADING_INCENTIVE,  # Add new parameter
        "drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "risk_fraction": ENV_RISK_FRACTION,
        "stop_loss_pct": ENV_STOP_LOSS_PCT,
        "normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
        # Add any other parameters TradingEnv expects
    }
    
    # --- Database Logging Setup (run_id needs to be defined before config) ---
    run_id = f"RLlibRUN-{TICKER}-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated Run ID: {run_id}")
    
    # Create the callback config using the run_id and MAX_TRAINING_ITERATIONS
    callback_config = {
        "run_id": run_id,
        "num_training_iterations": MAX_TRAINING_ITERATIONS
    }
    logger.info(f"Created callback_config with run_id: {run_id} and num_training_iterations: {MAX_TRAINING_ITERATIONS}")
    
    # Define NUM_TRAINING_ITERATIONS for backward compatibility with callbacks
    NUM_TRAINING_ITERATIONS = MAX_TRAINING_ITERATIONS

    config = (
        DQNConfig()
        # New API stack is enabled by default, so removing the explicit disabling.
        .environment(env="TradingEnv-v0", env_config=env_config_params)
        .framework("torch") # Switch to PyTorch
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length=50, # Explicitly set rollout_fragment_length
            exploration_config={} # Use empty dict for default exploration with new API
        )
        .training(
            lr=AGENT_LEARNING_RATE,
            gamma=AGENT_GAMMA,
            train_batch_size=AGENT_TRAIN_BATCH_SIZE,
            n_step=1, # Changed from 0 to 1 for consistency with replay buffer
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 50000,
                "n_step": 1, # Changed from 0 to 1 to fix empty reward list issue
                "worker_side_prioritization": False,
                # 'store_n_step_transitions' is intentionally removed/False
            },
            target_network_update_freq=AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS
            # Model config now under .rl_module()
        )
        .rl_module( # Model config under .rl_module() for new API stack
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        # Top-level .exploration() removed
        .resources(
            num_gpus=1 if ray.is_initialized() and ray.cluster_resources().get("GPU", 0) > 0 else 0
        )
        # Initialize DatabaseLoggingCallbacks with a dictionary containing run_id
        .callbacks(lambda: DatabaseLoggingCallbacks(callback_config)) # Use callback_config here
        .reporting(min_time_s_per_iteration=10) # Report metrics at least every 10s
    ) # End of RLlib AlgorithmConfig fluent interface chain
    config.log_level = "ERROR" # Set RLlib's algorithm log level to ERROR to suppress WARNINGs
    
    # For debugging the config
    # logger.info(f"RLlib Config: {pretty_print(config.to_dict())}")

    run_params = {
        "ticker": TICKER,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "rllib_max_iterations": MAX_TRAINING_ITERATIONS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_min_iterations": MIN_ITERATIONS,
        "early_stopping_improvement_threshold": IMPROVEMENT_THRESHOLD,
        "rllib_num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "agent_learning_rate": AGENT_LEARNING_RATE, "agent_gamma": AGENT_GAMMA,
        "agent_buffer_size": AGENT_BUFFER_SIZE, "agent_train_batch_size": AGENT_TRAIN_BATCH_SIZE,
        "agent_target_network_update_freq": AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS,
        "agent_initial_epsilon": AGENT_INITIAL_EPSILON, "agent_final_epsilon": AGENT_FINAL_EPSILON,
        "agent_epsilon_timesteps": AGENT_EPSILON_TIMESTEPS,
        # Env params
        "env_initial_balance": ENV_INITIAL_BALANCE,
        "env_transaction_fee": ENV_TRANSACTION_FEE_PERCENT,
        "env_window_size": ENV_WINDOW_SIZE,
        "env_sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "env_drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "env_trading_penalty": ENV_TRADING_PENALTY,
        "env_risk_fraction": ENV_RISK_FRACTION,
        "env_stop_loss_pct": ENV_STOP_LOSS_PCT,
        "env_use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "env_normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
    }

    training_run_db_id = None
    with get_db_session() as db:
        try:
            training_run_record = TrainingRun(
                run_id=run_id,
                start_time=datetime.datetime.now(datetime.UTC),
                parameters=run_params,
                status='starting_rllib'
            )
            db.add(training_run_record)
            db.commit()
            db.refresh(training_run_record) # Refresh is good practice, though PK is manually set
            training_run_db_id = training_run_record.run_id # Correct attribute is run_id
            logger.info(f"TrainingRun record created in DB with run_id: {training_run_db_id}")
        except Exception as e:
            logger.error(f"Failed to create TrainingRun DB record: {e}", exc_info=True)
            ray.shutdown()
            return
    
    # Both set callbacks_config (for algorithm config) and directly pass run_id to the callback via lambda
    config.callbacks_config = callback_config
    # Directly set the config value to ensure it's available
    logger.info(f"Set config.callbacks_config to: {config.callbacks_config}")
    
    # Pass environment variables to worker processes to disable warnings
    config.extra_python_environs_for_driver = {
        "RAY_DISABLE_DEPRECATION_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning",
        "RAY_DEDUP_LOGS": "0",
        "RLLIB_DISABLE_API_STACK_WARNING": "1",
        "RAY_LOGGING_LEVEL": "ERROR"
    }
    
    # Also pass to workers
    config.extra_python_environs_for_worker = {
        "RAY_DISABLE_DEPRECATION_WARNINGS": "1",
        "PYTHONWARNINGS": "ignore::DeprecationWarning",
        "RAY_DEDUP_LOGS": "0",
        "RLLIB_DISABLE_API_STACK_WARNING": "1",
        "RAY_LOGGING_LEVEL": "ERROR"
    }

    # --- Build and Train Algorithm ---
    try:
        algo = config.build_algo()
        logger.info("RLlib DQN Algorithm built successfully.")

        # Initialize variables for early stopping
        best_validation_sharpe = -float('inf')
        best_iteration = -1
        best_checkpoint_dir = None
        patience_counter = 0
        validation_metrics_history = []
        training_metrics_history = []
        
        # Training loop with early stopping
        for i in range(MAX_TRAINING_ITERATIONS):
            # Train for one iteration
            result = algo.train()
            current_iteration = i + 1
            logger.info(f"Iteration {current_iteration}/{MAX_TRAINING_ITERATIONS}:")
            logger.info(pretty_print(result))
            
            # Extract training metrics
            training_metrics = {
                'iteration': current_iteration,
                'episode_reward_mean': result.get('episode_reward_mean', 0),
                'episode_reward_min': result.get('episode_reward_min', 0),
                'episode_reward_max': result.get('episode_reward_max', 0),
                'episode_len_mean': result.get('episode_len_mean', 0),
                'episodes_this_iter': result.get('episodes_this_iter', 0),
                'episodes_total': result.get('episodes_total', 0),
            }
            training_metrics_history.append(training_metrics)
            
            # Log training metrics
            logger.info(f"Training metrics: Reward Mean={training_metrics['episode_reward_mean']:.4f}, "
                        f"Episodes={training_metrics['episodes_this_iter']}, "
                        f"Total Episodes={training_metrics['episodes_total']}")
            
            # Evaluate on validation data
            validation_metrics = evaluate_on_validation_data(algo, validation_data_ref)
            validation_metrics['iteration'] = current_iteration
            validation_metrics_history.append(validation_metrics)
            
            # Check for improvement
            current_validation_sharpe = validation_metrics['avg_sharpe_ratio']
            is_improvement = current_validation_sharpe > (best_validation_sharpe + IMPROVEMENT_THRESHOLD)
            
            # Save checkpoint
            checkpoint_dir = algo.save()
            logger.info(f"Checkpoint saved at iteration {current_iteration} in {checkpoint_dir}")
            
            if is_improvement:
                logger.info(f"Validation Sharpe improved from {best_validation_sharpe:.4f} to {current_validation_sharpe:.4f}")
                best_validation_sharpe = current_validation_sharpe
                best_iteration = current_iteration
                best_checkpoint_dir = checkpoint_dir
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement in validation Sharpe. Patience: {patience_counter}/{PATIENCE}")
            
            # Check early stopping conditions
            if current_iteration >= MIN_ITERATIONS and patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {current_iteration} iterations. "
                           f"Best validation Sharpe: {best_validation_sharpe:.4f} at iteration {best_iteration}")
                break
        
        # If we completed all iterations without early stopping
        if i == MAX_TRAINING_ITERATIONS - 1:
            logger.info(f"Completed all {MAX_TRAINING_ITERATIONS} iterations. "
                       f"Best validation Sharpe: {best_validation_sharpe:.4f} at iteration {best_iteration}")
        
        # Use the best checkpoint as the final model
        final_checkpoint_dir = best_checkpoint_dir if best_checkpoint_dir else checkpoint_dir
        logger.info(f"Final model checkpoint (best validation performance) saved in {final_checkpoint_dir}")
        
        # Analyze convergence and performance
        logger.info("Analyzing training convergence and performance...")
        
        # Calculate metrics for final report
        total_episodes = training_metrics_history[-1]['episodes_total']
        avg_episodes_per_iteration = total_episodes / len(training_metrics_history)
        
        # Calculate metrics for diminishing returns analysis
        if len(validation_metrics_history) >= 2:
            sharpe_improvements = []
            for i in range(1, len(validation_metrics_history)):
                prev_sharpe = validation_metrics_history[i-1]['avg_sharpe_ratio']
                curr_sharpe = validation_metrics_history[i]['avg_sharpe_ratio']
                improvement = curr_sharpe - prev_sharpe
                sharpe_improvements.append(improvement)
            
            # Calculate average improvement in the first half vs second half of training
            mid_point = len(sharpe_improvements) // 2
            early_improvements = sharpe_improvements[:mid_point]
            late_improvements = sharpe_improvements[mid_point:]
            avg_early_improvement = sum(early_improvements) / len(early_improvements) if early_improvements else 0
            avg_late_improvement = sum(late_improvements) / len(late_improvements) if late_improvements else 0
            
            logger.info(f"Average Sharpe improvement in early iterations: {avg_early_improvement:.6f}")
            logger.info(f"Average Sharpe improvement in late iterations: {avg_late_improvement:.6f}")
            logger.info(f"Diminishing returns ratio: {avg_late_improvement/avg_early_improvement if avg_early_improvement != 0 else 0:.6f}")
        
        # Log final performance metrics
        logger.info(f"Total episodes completed: {total_episodes}")
        logger.info(f"Average episodes per iteration: {avg_episodes_per_iteration:.2f}")
        logger.info(f"Best validation Sharpe ratio: {best_validation_sharpe:.4f} at iteration {best_iteration}")
        logger.info(f"Final validation metrics: Avg PnL={validation_metrics_history[-1]['avg_pnl']:.2f}, "
                   f"Avg Drawdown={validation_metrics_history[-1]['avg_max_drawdown']:.4f}")

        # Update TrainingRun status to completed with metrics
        if training_run_db_id: # training_run_db_id stores the string run_id
            with get_db_session() as db:
                run_to_update = db.query(TrainingRun).filter(TrainingRun.run_id == training_run_db_id).first() # Corrected to use run_id
                if run_to_update:
                    run_to_update.end_time = datetime.datetime.now(datetime.UTC)
                    run_to_update.status = 'completed_rllib'
                    
                    # Add final metrics to TrainingRun
                    final_metrics = {
                        'best_validation_sharpe': float(best_validation_sharpe),
                        'best_iteration': best_iteration,
                        'total_episodes': int(total_episodes),
                        'avg_episodes_per_iteration': float(avg_episodes_per_iteration),
                        'early_stopping_triggered': i < MAX_TRAINING_ITERATIONS - 1,
                        'iterations_completed': len(training_metrics_history),
                        'final_validation_metrics': validation_metrics_history[-1],
                        'training_metrics_history': training_metrics_history,
                        'validation_metrics_history': validation_metrics_history,
                        'best_checkpoint_dir': final_checkpoint_dir
                    }
                    
                    # Store metrics in the database
                    run_to_update.final_metrics = final_metrics
                    db.commit()
                    logger.info(f"TrainingRun {run_id} status updated to completed_rllib with final metrics.")

    except Exception as e:
        logger.error(f"Error during RLlib training: {e}", exc_info=True)
        if training_run_db_id:
            with get_db_session() as db:
                run_to_update = db.query(TrainingRun).filter(TrainingRun.run_id == training_run_db_id).first() # Corrected to use run_id
                if run_to_update:
                    run_to_update.end_time = datetime.datetime.now(datetime.UTC)
                    run_to_update.status = 'failed_rllib'
                    db.commit()
                    logger.info(f"TrainingRun {run_id} status updated to failed_rllib due to error.")
    finally:
        # Ensure algo is defined before calling stop()
        if 'algo' in locals() and algo is not None: # Check if algo was successfully assigned
            algo.stop()
        ray.shutdown()
        logger.info("Ray shut down. Training process finished.")

if __name__ == "__main__":
    main()