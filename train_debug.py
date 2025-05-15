import numpy as np
import pandas as pd
import logging
import datetime
import uuid
import os
import ray
from ray.rllib.algorithms.dqn import DQNConfig # Using DQN as per existing agent
from ray.tune.logger import pretty_print

# Import our debug wrapper
from debug_replay_buffer import debug_sampling_process

from reinforcestrategycreator.data_fetcher import fetch_historical_data
from reinforcestrategycreator.technical_analyzer import calculate_indicators
# Import TradingEnv and the registration function
from reinforcestrategycreator.trading_environment import TradingEnv, register_rllib_env
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
END_DATE = "2023-12-31"
# RLlib training is often defined by iterations or total timesteps, not just episodes
NUM_TRAINING_ITERATIONS = 1  # REDUCED TO 1 FOR DEBUGGING
INITIAL_TRAINING_EPISODES_FOR_DATA_EST = 5 # Used for estimating data length if needed
GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS = 1 # Number of extra train() calls for draining
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 30 # Timeout for the draining phase
MAX_EPISODES_ALLOWED = 11 # Hard cap on total episodes (10 target + 1 margin) - use to catch overruns
 
# Environment parameters (to be passed in env_config)
ENV_INITIAL_BALANCE = 10000.0
ENV_TRANSACTION_FEE_PERCENT = 0.001
ENV_WINDOW_SIZE = 5 # Default from original TradingEnv
ENV_SHARPE_WINDOW_SIZE = 100
ENV_DRAWDOWN_PENALTY = 0.005
ENV_TRADING_PENALTY = 0.002
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

# Parallelism - Reduce for debugging
NUM_ROLLOUT_WORKERS = 0  # Use 0 for debugging to avoid multi-process complexity

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

def main():
    # Clear any lingering system-wide shutdown flag from previous runs
    TradingEnv.clear_system_wide_graceful_shutdown()
    TradingEnv.reset_run_episode_counter() # Reset global episode counter for the run
    logger.info(f"Starting RLlib debug training run for {TICKER} from {START_DATE} to {END_DATE}")
    
    # Monkey patch removed as direct modification to RLlib source is used.
    logger.info("Direct modification to RLlib source is used for debugging EpisodeReplayBuffer.")
    
    ray.init(ignore_reinit_error=True, log_to_driver=True) # Log to driver for easier debugging initially
    logger.info(f"Ray initialized. Dashboard URL: {ray.dashboard}")

    # --- 1. Data Pipeline ---
    logger.info("Fetching historical data...")
    try:
        data_df = fetch_historical_data(TICKER, START_DATE, END_DATE)
        if data_df.empty:
            logger.error("Failed to fetch data or data is empty.")
            ray.shutdown()
            return
        logger.info(f"Data fetched successfully: {data_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        ray.shutdown()
        return

    data_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data_df.columns]
    logger.info(f"Renamed columns to lowercase: {list(data_df.columns)}")

    logger.info("Adding technical indicators...")
    try:
        data_with_indicators = calculate_indicators(data_df.copy())
        initial_rows = len(data_with_indicators)
        data_with_indicators.dropna(inplace=True)
        rows_dropped = initial_rows - len(data_with_indicators)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation.")
        if data_with_indicators.empty:
            logger.error("Data is empty after dropping NaNs from indicators.")
            ray.shutdown()
            return
        logger.info("Technical indicators added successfully.")
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}", exc_info=True)
        ray.shutdown()
        return

    # --- Register Custom Environment ---
    register_rllib_env() # Call the registration function

    # --- RLlib Algorithm Configuration ---
    env_config_params = {
        "df": data_with_indicators,
        "initial_balance": ENV_INITIAL_BALANCE,
        "transaction_fee_percent": ENV_TRANSACTION_FEE_PERCENT,
        "window_size": ENV_WINDOW_SIZE,
        "sharpe_window_size": ENV_SHARPE_WINDOW_SIZE,
        "use_sharpe_ratio": ENV_USE_SHARPE_RATIO,
        "trading_frequency_penalty": ENV_TRADING_PENALTY,
        "drawdown_penalty": ENV_DRAWDOWN_PENALTY,
        "risk_fraction": ENV_RISK_FRACTION,
        "stop_loss_pct": ENV_STOP_LOSS_PCT,
        "normalization_window_size": ENV_NORMALIZATION_WINDOW_SIZE,
        "max_total_episodes_for_run": 10, # Will be set dynamically later, placeholder for now
        # Add any other parameters TradingEnv expects
    }
    
    # --- Database Logging Setup (run_id needs to be defined before config) ---
    run_id = f"RLlibDBG-{TICKER}-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated Debug Run ID: {run_id}")
    
    # Create the callback config using the run_id
    callback_config = {"run_id": run_id}
    logger.info(f"Created callback_config with run_id: {run_id}")

    config = (
        DQNConfig()
        # New API stack is enabled by default, so removing the explicit disabling.
        .environment(env="TradingEnv-v0", env_config=env_config_params)
        .framework("torch") # Switch to PyTorch
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length=5, # Further reduced for more granular control
            exploration_config={} # Use empty dict for default exploration with new API
        )
        .training(
            lr=AGENT_LEARNING_RATE,
            gamma=AGENT_GAMMA,
            train_batch_size=AGENT_TRAIN_BATCH_SIZE,
            n_step=1,  # IMPORTANT: Set to 1 for debugging, as this is what causes the issue
            # Using EpisodeReplayBuffer with the new API stack for proper episode handling
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 50000, # Using AGENT_BUFFER_SIZE would be 10000, but let's keep 50000
                "worker_side_prioritization": False,
            },
            target_network_update_freq=AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS,
            training_intensity=1.0  # Add this line
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            }
        )
        .resources(
            num_gpus=0  # Don't use GPU for debugging
        )
        .callbacks(lambda: DatabaseLoggingCallbacks(callback_config))
        .reporting(min_time_s_per_iteration=0)  # Set to 0 for more granular control
        # Using the new RLlib API stack (default) to avoid ABCMeta TypeError
        # Removed disabling of API stack and validation
    )
    config.log_level = "ERROR"
    
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

    # --- Create TrainingRun DB Record ---
    # This must happen BEFORE algo.build_algo() or algo.train()
    # so that callbacks can find the parent run_id.
    run_params = {
        "ticker": TICKER, "start_date": START_DATE, "end_date": END_DATE,
        "rllib_num_iterations": NUM_TRAINING_ITERATIONS,
        "rllib_num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "agent_learning_rate": AGENT_LEARNING_RATE, "agent_gamma": AGENT_GAMMA,
        "agent_buffer_size": AGENT_BUFFER_SIZE, "agent_train_batch_size": AGENT_TRAIN_BATCH_SIZE,
        "agent_target_network_update_freq": AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS,
        "agent_initial_epsilon": AGENT_INITIAL_EPSILON, "agent_final_epsilon": AGENT_FINAL_EPSILON,
        "agent_epsilon_timesteps": AGENT_EPSILON_TIMESTEPS,
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
                run_id=run_id, # This is the run_id generated earlier
                start_time=datetime.datetime.now(datetime.UTC),
                parameters=run_params,
                status='starting_rllib_debug' # Indicate it's a debug run
            )
            db.add(training_run_record)
            db.commit() # Commit immediately to make it available for foreign key constraints
            db.refresh(training_run_record)
            training_run_db_id = training_run_record.run_id
            logger.info(f"TrainingRun record created in DB with run_id: {training_run_db_id}")
        except Exception as e:
            logger.error(f"Failed to create TrainingRun DB record: {e}", exc_info=True)
            ray.shutdown()
            return

    # --- Build and Train Algorithm ---
    try:
        # Ensure callback_config is part of the main config if not already set by .callbacks()
        # This might be redundant if .callbacks(lambda: DatabaseLoggingCallbacks(callback_config)) works as expected
        # but let's ensure it's explicitly there.
        if not hasattr(config, "callbacks_config") or not config.callbacks_config:
             config.callbacks(lambda: DatabaseLoggingCallbacks(callback_config))
        
        algo = config.build_algo()
        logger.info("RLlib DQN Algorithm built successfully.")

        # Training loop - stop after 10 episodes
        episodes_completed_count = 0
        max_episodes_to_run = 10 # This is the target
        env_config_params["max_total_episodes_for_run"] = max_episodes_to_run # Update env_config
        max_iterations = 100 # Safety break for the loop
        logger.info(f"Starting training loop, will stop after {max_episodes_to_run} episodes or {max_iterations} iterations (hard cap: {MAX_EPISODES_ALLOWED}).")
        
        # Track if we've reached the episode limit to prevent additional episodes during graceful shutdown
        episode_limit_reached = False
        _training_should_stop_flag = [False] # Flag to signal stopping from the wrapper
        
        # Modify the config to limit training time and batch size to give more control
        
        # Add a custom stop condition function that will be checked by RLlib during training
        logger.info("Setting up custom episode limit stop condition")
        
        # Install custom evaluator to strictly enforce episode limits
        # This provides a direct hook into RLlib's training loop
        orig_training_step = algo.training_step
        
        def limited_train_one_step(*args, **kwargs):
            """Wrapper around RLlib's training_step that enforces episode limits using TradingEnv counter."""
            current_env_episode_count = TradingEnv.get_current_run_episodes_completed()

            # algo and i are available in the outer scope of main()
            # max_episodes_to_run and MAX_EPISODES_ALLOWED are also in outer scope

            logger.info(
                f"LIMITED_TRAIN_WRAPPER: Env episodes: {current_env_episode_count}, "
                f"Target: {max_episodes_to_run}, Hard Cap: {MAX_EPISODES_ALLOWED}"
            )

            # The counter increments at the start of an episode in TradingEnv.reset().
            # So, if max_episodes_to_run is 10, we want to stop *before* starting the 11th episode.
            # This means if current_env_episode_count is already 11 (meaning 10 have conceptually finished
            # and the 11th is about to start or has just started), we stop.
            # Or, more simply, if current_env_episode_count > max_episodes_to_run.
            # Let's use current_env_episode_count > max_episodes_to_run to be consistent with how the env terminates.
            # The MAX_EPISODES_ALLOWED is a hard stop.
            if current_env_episode_count > max_episodes_to_run or \
               current_env_episode_count > MAX_EPISODES_ALLOWED: # MAX_EPISODES_ALLOWED is 11
                logger.warning(
                    f"LIMITED_TRAIN_WRAPPER: Stopping. Env episodes: {current_env_episode_count} "
                    f"(Target: {max_episodes_to_run}, Hard Cap: {MAX_EPISODES_ALLOWED}). Setting stop flag."
                )
                _training_should_stop_flag[0] = True
                # Raise StopIteration to forcefully break out of algo.train()'s internal loop
                raise StopIteration("Episode limit reached by custom wrapper.")
            
            # Otherwise proceed with original training step
            # The original training_step (for new API stack) should also return None
            return orig_training_step(*args, **kwargs) # This should return None if orig_training_step adheres
        
        # Install our wrapper
        algo.training_step = limited_train_one_step
        
        # Keep track of episodes from previous iterations
        previous_episodes_completed = 0
        
        for i in range(max_iterations):
            # Pre-check to avoid even starting an iteration
            if previous_episodes_completed >= max_episodes_to_run:
                logger.info(f"Already reached {previous_episodes_completed} episodes before iteration {i+1}. Stopping.")
                episode_limit_reached = True
                break
            
            # Calculate remaining episodes target
            remaining_episodes = max_episodes_to_run - previous_episodes_completed
            # Use extremely small rollout fragment length when we're close to target
            if remaining_episodes <= 2:
                suggested_rollout_length = 1  # Collect single steps for maximum control near limit
            else:
                suggested_rollout_length = min(5, max(1, remaining_episodes * 2))
            
            # Dynamically adjust the rollout fragment length across all runners
            logger.info(f"Setting rollout_fragment_length to {suggested_rollout_length} (remaining episodes: {remaining_episodes})")
            try:
                # With new API stack, try two potential methods to adjust rollout_fragment_length
                try:
                    # First try the method that works with new API
                    algo.workers.foreach_env_runner(
                        lambda runner: setattr(runner, "rollout_fragment_length", suggested_rollout_length)
                        if hasattr(runner, "rollout_fragment_length") else None
                    )
                except AttributeError:
                    # Fallback for different API structure
                    logger.info("Trying alternate method to set rollout_fragment_length")
                    try:
                        if hasattr(algo, "workers") and hasattr(algo.workers, "foreach_worker"):
                            algo.workers.foreach_worker(
                                lambda w: setattr(w.sample_collector, "rollout_fragment_length", suggested_rollout_length)
                                if hasattr(w, "sample_collector") and hasattr(w.sample_collector, "rollout_fragment_length") else None
                            )
                    except Exception as e2:
                        logger.warning(f"Could not adjust rollout_fragment_length with alternate method: {e2}")
            except Exception as e:
                logger.warning(f"Could not adjust rollout_fragment_length: {e}")
            
            # Execute a small training step
            try:
                result = algo.train()

                # This flag check is a fallback if StopIteration wasn't raised but flag was set by wrapper.
                # This might happen if algo.train() completed its iteration *just as* the flag was set,
                # but before StopIteration was raised in a subsequent internal call to the wrapper.
                if _training_should_stop_flag[0]:
                    logger.info("Training stop flag detected by main loop (after algo.train completed). Setting episode_limit_reached and breaking.")
                    episode_limit_reached = True
                    # If result exists, use its episode count, otherwise assume max_episodes_to_run for safety
                    episodes_completed_count = result.get("episodes_total", max_episodes_to_run) if result else max_episodes_to_run
                    previous_episodes_completed = episodes_completed_count
                    break
            
            except StopIteration as e:
                logger.info(f"StopIteration caught: {e}. Breaking training loop.")
                episode_limit_reached = True
                # 'result' is not assigned in this path.
                # We need to ensure 'episodes_completed_count' and 'previous_episodes_completed'
                # reflect that the target was met for subsequent logic (like drain skipping).
                # The wrapper already logged the env_episode_count.
                # Let's set episodes_completed_count to max_episodes_to_run to ensure drain is skipped.
                episodes_completed_count = max_episodes_to_run
                previous_episodes_completed = episodes_completed_count
                result = {} # Ensure result is a dict for downstream checks, even if empty
                break
            
            logger.info(f"Training iteration {i+1} complete.")
            
            # Get updated episode counts
            episodes_this_iter = result.get("episodes_this_iteration", 0)
            episodes_completed_count = result.get("episodes_total", 0)
            
            logger.info(f"Episodes this iteration: {episodes_this_iter}, Total episodes completed: {episodes_completed_count}")
            
            # Store the current count for the next iteration
            previous_episodes_completed = episodes_completed_count
            
            # Check if we've reached or exceeded our target
            if episodes_completed_count >= max_episodes_to_run:
                logger.info(f"Reached {episodes_completed_count} episodes (target {max_episodes_to_run}). Stopping training loop.")
                episode_limit_reached = True
                break
            if i == max_iterations - 1:
                logger.warning(f"Reached max_iterations ({max_iterations}) before completing {max_episodes_to_run} episodes. Stopping.")

        # --- Graceful Shutdown Logic ---
        logger.info("Initiating graceful shutdown...")
        # Access workers property to get the EnvRunnerGroup (or equivalent)
        # This object structure can be tricky with num_workers=0 vs >0
        
        if NUM_ROLLOUT_WORKERS == 0:
            logger.info("Signaling driver's environment to initiate graceful shutdown (NUM_ROLLOUT_WORKERS == 0).")
            try:
                driver_env_runner = algo.env_runner # This is a SingleAgentEnvRunner
                actual_env = None
                
                if hasattr(driver_env_runner, 'env'):
                    wrapped_env = driver_env_runner.env
                    logger.info(f"Retrieved wrapped_env from driver_env_runner.env, type: {type(wrapped_env)}")
                    
                    potential_vector_env = wrapped_env
                    if hasattr(wrapped_env, 'unwrapped'):
                        potential_vector_env = wrapped_env.unwrapped
                        logger.info(f"Unwrapped to potential_vector_env, type: {type(potential_vector_env)}")
                    
                    # Check if it's a vectorized environment that supports 'call'
                    if hasattr(potential_vector_env, 'call') and callable(potential_vector_env.call):
                        logger.info(f"Attempting to call 'signal_graceful_shutdown' on sub-environments of vector_env (type: {type(potential_vector_env)}).")
                        try:
                            # The 'call' method executes a named method on each sub-environment.
                            # We assume 'signal_graceful_shutdown' exists on the sub-environments (our TradingEnv).
                            potential_vector_env.call('signal_graceful_shutdown')
                            logger.info("Signal 'signal_graceful_shutdown' called on sub-environments via vector_env.call().")
                        except Exception as e_vec_call:
                            logger.error(f"Error calling 'signal_graceful_shutdown' via vector_env.call(): {e_vec_call}", exc_info=True)
                    elif hasattr(potential_vector_env, 'signal_graceful_shutdown'): # Fallback for non-vectorized but unwrapped
                         logger.info(f"Potential vector env (type: {type(potential_vector_env)}) is not a vector env with 'call', but trying direct call.")
                         try:
                            potential_vector_env.signal_graceful_shutdown()
                            logger.info(f"Signal sent directly to environment (type: {type(potential_vector_env)}).")
                         except AttributeError:
                            logger.error(f"Environment (type: {type(potential_vector_env)}) does NOT have 'signal_graceful_shutdown' method.", exc_info=True)
                         except Exception as e_direct_call:
                            logger.error(f"Error calling signal_graceful_shutdown directly on env (type: {type(potential_vector_env)}): {e_direct_call}", exc_info=True)
                    else:
                        logger.warning(f"Environment (type: {type(potential_vector_env)}) does not support 'call' and does not have 'signal_graceful_shutdown' directly.")
                else:
                    logger.warning("driver_env_runner does not have an 'env' attribute.")
            except Exception as e_local_signal:
                logger.error(f"Error during local environment signaling setup: {e_local_signal}", exc_info=True)
        else: # NUM_ROLLOUT_WORKERS > 0
            worker_manager = algo.workers # This should be the EnvRunnerGroup
            logger.info(f"Signaling {worker_manager.num_remote_workers()} remote workers to initiate graceful shutdown in their environments.")
            try:
                # Use a lambda that checks if 'signal_graceful_shutdown' exists
                def safe_signal(env):
                    if hasattr(env, 'signal_graceful_shutdown'):
                        env.signal_graceful_shutdown()
                    else:
                        # This log will appear in worker logs, not driver, if it happens.
                        # Consider a way to aggregate such warnings if needed.
                        print(f"Warning: Env {type(env)} in remote worker does not have signal_graceful_shutdown.")

                worker_manager.foreach_env(safe_signal)
                logger.info("Signal attempt sent to all remote environments via foreach_env.")
            except Exception as e_remote_signal:
                logger.error(f"Error signaling remote workers via foreach_env: {e_remote_signal}", exc_info=True)

        # Only perform drain iterations if we haven't hit the episode limit
        # This prevents generating additional episodes during graceful shutdown
        if not episode_limit_reached:
            logger.info(f"Performing {GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS} drain iterations...")
            drain_start_time = datetime.datetime.now()
            
            # Set an extremely small rollout fragment length for draining iterations
            # This minimizes the chance of additional episodes being generated
            try:
                logger.info("Setting rollout_fragment_length to 1 for drain iterations")
                # With new API stack, try two potential methods to adjust rollout_fragment_length
                try:
                    # First try the method that works with new API
                    algo.workers.foreach_env_runner(
                        lambda runner: setattr(runner, "rollout_fragment_length", 1)
                        if hasattr(runner, "rollout_fragment_length") else None
                    )
                except AttributeError:
                    # Fallback for different API structure
                    logger.info("Trying alternate method to set rollout_fragment_length for drain")
                    try:
                        if hasattr(algo, "workers") and hasattr(algo.workers, "foreach_worker"):
                            algo.workers.foreach_worker(
                                lambda w: setattr(w.sample_collector, "rollout_fragment_length", 1)
                                if hasattr(w, "sample_collector") and hasattr(w.sample_collector, "rollout_fragment_length") else None
                            )
                    except Exception as e2:
                        logger.warning(f"Could not adjust rollout_fragment_length for drain with alternate method: {e2}")
            except Exception as e:
                logger.warning(f"Could not adjust rollout_fragment_length for draining: {e}")
            
            # Install an episode counter to detect if any new episodes are started
            # Use the previous_episodes_completed which is updated from train() result
            orig_episode_counter = previous_episodes_completed
            for i in range(GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS): # Should be 1 iteration
                logger.info(f"Drain iteration {i+1}/{GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS}...")
                # Perform a very small training step to allow episodes to complete and be logged
                drain_result = algo.train()
                # Log how many episodes were completed during drain iteration
                drain_episodes = drain_result.get("episodes_this_iteration", 0)
                total_episodes = drain_result.get("episodes_total", 0)
                logger.info(f"Drain iteration generated {drain_episodes} episodes. Total episodes now: {total_episodes}")
                
                # Check if we're generating new episodes during drain
                if total_episodes > orig_episode_counter:
                    logger.warning(f"Drain iteration generated {total_episodes - orig_episode_counter} new episodes! This is unexpected.")
                    # If we exceeded our target during drain, stop immediately
                    if total_episodes > max_episodes_to_run:
                        logger.warning(f"Episode limit exceeded during drain. Current: {total_episodes}, Target: {max_episodes_to_run}")
                        break
                
                if (datetime.datetime.now() - drain_start_time).total_seconds() > GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                    logger.warning(f"Graceful shutdown drain timeout ({GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS}s) reached during drain train() call.")
                    break
            logger.info("Graceful shutdown drain phase complete.")
        else:
            logger.info("Skipping drain iterations as episode limit was already reached.")
  
    except Exception as e:
        logger.error(f"Error during RLlib training or graceful shutdown: {e}", exc_info=True)
    finally:
        # Clear the system-wide flag again to ensure clean state for any subsequent processes
        TradingEnv.clear_system_wide_graceful_shutdown()

        # Ensure algo is defined before calling stop()
        if 'algo' in locals() and algo is not None:
            logger.info("Stopping RLlib algorithm...")
            algo.stop()
            logger.info("RLlib algorithm stopped.")
        else:
            logger.warning("'algo' object not found or is None, skipping algo.stop().")

        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("Ray shut down. Debug training process finished.")
        logger.info("Check replay_buffer_debug.log for detailed debug information about the replay buffer.")

if __name__ == "__main__":
    main()