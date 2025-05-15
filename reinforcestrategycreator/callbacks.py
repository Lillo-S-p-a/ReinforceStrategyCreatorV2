import datetime
import inspect # Keep for potential future use when inspecting episode object
import os # Add this import
import numpy as np
import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
# from ray.rllib.env.episode import Episode # Reverted this import
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.evaluation.rollout_worker import RolloutWorker # Corrected import path
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch # Import SampleBatch
from ray.rllib.env.single_agent_episode import SingleAgentEpisode # Import SingleAgentEpisode
from typing import Dict, Optional, Any # Keep Any for now, or use Union[Episode, SingleAgentEpisode] if needed

# Forward-declare RLModule and EnvRunner if full import causes issues or for cleaner typing
# These are typically available in newer RLlib versions.
# from ray.rllib.core.rl_module.rl_module import RLModule
# from ray.rllib.env.env_runner import EnvRunner

from sqlalchemy import func
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import (
    Episode as DbEpisode,
    Trade as DbTrade,
    TrainingRun,
    Step as DbStep, # Added
    TradingOperation as DbTradingOperation, # Added
    OperationType) # Added
from reinforcestrategycreator.trading_environment import TradingEnv

# Set up a specific logger for this module
logger = logging.getLogger('callbacks') # Use a specific name
logger.setLevel(logging.INFO) # Set default level for this logger

# Create a file handler for this logger
log_file_path = 'callbacks_debug.log'
try:
    # Remove old log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
except OSError as e:
    print(f"Error removing old log file {log_file_path}: {e}")

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers: # Avoid adding multiple handlers if script is reloaded
    logger.addHandler(file_handler)

logger.info("Callbacks logger initialized and configured to write to callbacks_debug.log")

class DatabaseLoggingCallbacks(DefaultCallbacks):
    """
    Callback class for logging training data to the database.
    
    This class handles logging of episodes, steps, and trading operations to the database.
    It also includes functionality to gracefully finalize incomplete episodes when a training
    run is terminated.
    """
    def __init__(self, legacy_callbacks_dict: Dict = None): # Re-added legacy_callbacks_dict
        super().__init__()
        self.run_id = None # Initialize self.run_id
        
        # Debug info about what we're receiving
        logger.info(f"DatabaseLoggingCallbacks init with: {legacy_callbacks_dict}")
        
        # Try multiple ways to access the run_id from different locations
        if legacy_callbacks_dict:
            if "run_id" in legacy_callbacks_dict:
                self.run_id = legacy_callbacks_dict["run_id"]
                logger.info(f"Found run_id in legacy_callbacks_dict: {self.run_id}")
            elif "callbacks_config" in legacy_callbacks_dict and legacy_callbacks_dict["callbacks_config"]:
                if "run_id" in legacy_callbacks_dict["callbacks_config"]:
                    self.run_id = legacy_callbacks_dict["callbacks_config"]["run_id"]
                    logger.info(f"Found run_id in callbacks_config: {self.run_id}")
        
        # Note: Even if run_id is not found during initialization, it may be set later
        # via the set_run_id method or retrieved from algorithm config
        if not self.run_id:
            logger.warning("WARNING: run_id not found in callback initialization! Will attempt to get it during execution.")
            
        logger.info(f"DatabaseLoggingCallbacks initialized with run_id: {self.run_id}")
        
    def set_run_id(self, run_id: str) -> None:
        """Allow setting the run_id after initialization"""
        self.run_id = run_id
        logger.info(f"Run ID set to: {self.run_id}")

    def get_run_id_from_algorithm(self, worker) -> str:
        """Try to get run_id from algorithm's config if available"""
        # This method is reinstated as 'worker' will be available with the old signature.
        try:
            # Try getting from callbacks_config on the worker's algorithm_config
            if hasattr(worker, "config") and worker.config and "callbacks_config" in worker.config:
                 if worker.config["callbacks_config"] and "run_id" in worker.config["callbacks_config"]:
                    logger.info(f"Found run_id in worker.config['callbacks_config']")
                    return worker.config["callbacks_config"]["run_id"]

            # Fallback: Try getting from worker.env.callbacks_config if worker is an EnvRunner and env has it
            # This is more speculative.
            if hasattr(worker, "env") and hasattr(worker.env, "config") and worker.env.config:
                if "callbacks_config" in worker.env.config and worker.env.config["callbacks_config"]:
                    if "run_id" in worker.env.config["callbacks_config"]:
                        logger.info(f"Found run_id in worker.env.config['callbacks_config']")
                        return worker.env.config["callbacks_config"]["run_id"]
            
            # Try algorithm instance directly if worker is an Algorithm object (less likely for episode callbacks)
            if hasattr(worker, "get_policy") and hasattr(worker.get_policy(), "config"):
                config = worker.get_policy().config
                if config and "callbacks_config" in config and config["callbacks_config"]:
                    if "run_id" in config["callbacks_config"]:
                        logger.info(f"Found run_id in worker.get_policy().config['callbacks_config']")
                        return config["callbacks_config"]["run_id"]
            
            logger.warning("Could not find run_id through worker object examination.")
            return None
        except Exception as e:
            logger.warning(f"Error retrieving run_id from algorithm via worker: {e}")
            return None

    def on_episode_start(
        self,
        *,
        base_env: Optional[BaseEnv] = None, # Make it optional to bypass TypeError
        episode: Any,  # Reverted to Any
        env_index: Optional[int] = None,
        env_runner: Optional["EnvRunner"] = None, # New API stack specific
        rl_module: Optional["RLModule"] = None,   # New API stack specific
        # For compatibility / older RLlib versions that might still pass these:
        worker: Optional["RolloutWorker"] = None,
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs,
    ) -> None:
        """Called at the beginning of each episode.
        Handles both new (EnvRunner) and old (RolloutWorker) API stacks.
        """
        logger.info(f"on_episode_start: received base_env parameter. Type: {type(base_env)}, Is None: {base_env is None}")
        if "base_env" in kwargs: # Check if RLlib somehow passed it via **kwargs despite named param
             logger.warning("on_episode_start: 'base_env' key was also found directly in **kwargs. This is unexpected.")
             # If base_env param is None but it's in kwargs, we might want to use kwargs['base_env']
             if base_env is None and kwargs.get("base_env") is not None: # Use .get for safety
                 logger.info("on_episode_start: Using 'base_env' from **kwargs as parameter was None.")
                 base_env = kwargs.pop("base_env") # Use it and remove from kwargs to avoid issues

        try:
            # Use getattr for episode.id_ for robustness as episode_id is preferred
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.info(f"--- on_episode_start CALLED for RLlib episode_id: {rllib_episode_id_str} ---")
            
            # Log details about received parameters (base_env type logged above)
            logger.info(f"on_episode_start details: env_runner type: {type(env_runner)}, rl_module type: {type(rl_module)}, worker type: {type(worker)}")
            # Note: kwargs might have had 'base_env' popped if the above condition was met
            kwarg_keys = list(kwargs.keys())
            logger.info(f"on_episode_start additional kwargs received (after potential pop): {kwarg_keys}")

            # Determine the correct runner object (EnvRunner for new, RolloutWorker for old)
            # The `env_runner` argument is preferred for the new stack.
            # `worker` is the argument for the old stack.
            current_runner_obj = env_runner if env_runner is not None else worker

            if current_runner_obj:
                logger.info(f"Using runner object of type: {type(current_runner_obj)} for run_id retrieval.")
            else:
                logger.warning("No env_runner or worker object explicitly passed to on_episode_start.")

            # Ensure self.run_id is set.
            if not self.run_id:
                logger.warning("self.run_id not set during __init__. Attempting to retrieve from runner object or legacy dict...")
                if current_runner_obj: # Try getting from the resolved runner object
                    # Check if callbacks_config is directly on the runner_obj (e.g., if it's an Algorithm instance)
                    if hasattr(current_runner_obj, "callbacks_config") and current_runner_obj.callbacks_config and "run_id" in current_runner_obj.callbacks_config:
                        self.run_id = current_runner_obj.callbacks_config["run_id"]
                        logger.info(f"Found run_id in current_runner_obj.callbacks_config: {self.run_id}")
                    # Check if it's on runner_obj.config.callbacks_config (e.g., EnvRunner.config or RolloutWorker.config)
                    elif hasattr(current_runner_obj, "config") and hasattr(current_runner_obj.config, "callbacks_config"):
                        cb_cfg = current_runner_obj.config.callbacks_config
                        if cb_cfg and "run_id" in cb_cfg:
                            self.run_id = cb_cfg["run_id"]
                            logger.info(f"Found run_id in current_runner_obj.config.callbacks_config: {self.run_id}")
                
                # Fallback to legacy_callbacks_dict if still no run_id
                if not self.run_id and hasattr(self, '_legacy_callbacks_dict') and self._legacy_callbacks_dict:
                    if "run_id" in self._legacy_callbacks_dict:
                        self.run_id = self._legacy_callbacks_dict["run_id"]
                        logger.info(f"Found run_id in _legacy_callbacks_dict (fallback in on_episode_start): {self.run_id}")
                    elif "callbacks_config" in self._legacy_callbacks_dict and \
                         isinstance(self._legacy_callbacks_dict["callbacks_config"], dict) and \
                         "run_id" in self._legacy_callbacks_dict["callbacks_config"]:
                        self.run_id = self._legacy_callbacks_dict["callbacks_config"]["run_id"]
                        logger.info(f"Found run_id in _legacy_callbacks_dict['callbacks_config'] (fallback in on_episode_start): {self.run_id}")

            if not self.run_id:
                logger.error("CRITICAL: self.run_id could not be determined in on_episode_start. Skipping DB logging.")
                return
            
            # Prepare kwargs for _log_episode_start_data, removing explicitly passed args
            call_kwargs = {k: v for k, v in kwargs.items() if k not in ['worker', 'base_env', 'policies', 'episode', 'env_index', 'env_runner', 'rl_module']}
            
            # The `policies` argument is not available in the new stack's on_episode_start.
            # It's also not used by _log_episode_start_data's new signature.
            # The `worker` argument to _log_episode_start_data will now be `env_runner` from this scope.
            self._log_episode_start_data(
                episode=episode,
                base_env=base_env,
                env_runner=current_runner_obj, # Pass the resolved runner object
                env_index=env_index,
                rl_module=rl_module, # Pass rl_module if available
                **call_kwargs
            )
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_episode_start: {e_outer}", exc_info=True)

    def _log_episode_start_data(
            self,
            *,
            episode: Any, # Reverted to Any
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env_index: Optional[int] = None,
            rl_module: Optional["RLModule"] = None,
            **kwargs,
        ) -> None:
            """Helper method to log episode start data."""
            try:
                rllib_episode_id = getattr(episode, 'id_', None)
                if not rllib_episode_id:
                    # Fallback for older RLlib versions or different episode objects
                    rllib_episode_id = getattr(episode, 'episode_id', f"unknown_rllib_ep_id_{datetime.datetime.now().timestamp()}")
                    logger.warning(f"Using fallback rllib_episode_id: {rllib_episode_id}")

                logger.info(f"--- _log_episode_start_data ENTERED for RLlib episode_id: {rllib_episode_id} ---")
                # Log the type of env_runner received for debugging
                logger.info(f"_log_episode_start_data received env_runner of type: {type(env_runner)}")
                logger.info(f"Run ID for this episode: {self.run_id}")


                if not self.run_id: # Attempt to get run_id from env_runner if not already set on self
                    if env_runner and hasattr(env_runner, "config") and hasattr(env_runner.config, "callbacks_config"):
                        cb_cfg = env_runner.config.callbacks_config
                        if cb_cfg and "run_id" in cb_cfg:
                            self.run_id = cb_cfg["run_id"]
                            logger.info(f"Retrieved run_id '{self.run_id}' from env_runner.config.callbacks_config in _log_episode_start_data")
                    
                if not self.run_id:
                    logger.error("CRITICAL: self.run_id not set in _log_episode_start_data. Cannot log episode start.")
                    return

                # Initial balance will be retrieved from episode.last_info_for() in _log_episode_end_data

                with get_db_session() as db:
                    # Create a new episode record
                    db_episode = DbEpisode(
                        rllib_episode_id=str(rllib_episode_id), # Ensure it's a string
                        run_id=self.run_id,
                        start_time=datetime.datetime.now(datetime.timezone.utc),
                        # initial_portfolio_value will be updated in _log_episode_end_data
                        status="started"
                    )
                    db.add(db_episode)
                    db.commit()
                    db.refresh(db_episode)
                    # Store the database episode ID in the RLlib episode's custom_data
                    # This is crucial for linking data in on_episode_end
                    episode.custom_data["db_episode_id"] = db_episode.episode_id # Corrected attribute
                    episode.custom_data["db_run_id"] = self.run_id # Also store run_id for safety
                    logger.info(f"Episode {db_episode.episode_id} (RLlib ID: {rllib_episode_id}) started and logged to DB. DB ID stored in custom_data.")

            except Exception as e:
                logger.error(f"Error in _log_episode_start_data for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e}", exc_info=True)
                # Do not re-raise, allow the main callback to continue if possible

    def _log_episode_end_data(
        self,
        *,
        episode: Any,
        worker: Optional[EnvRunnerGroup] = None, 
        base_env: Optional[BaseEnv] = None,    
        policies: Optional[Dict[str, Policy]] = None, 
        env_index: Optional[int] = None,       
        **kwargs,
    ) -> None:
        """Helper method to log episode end data. Can be called from on_episode_end or on_sample_end."""
        try:
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.info(f"--- _log_episode_end_data ENTERED for RLlib episode_id: {rllib_episode_id_str} ---")
            logger.info(f"Args for _log_episode_end_data: Worker type: {type(worker)}, BaseEnv type: {type(base_env)}, EnvIndex: {env_index}, Policies: {policies is not None}, Kwarg keys: {list(kwargs.keys())}")
            logger.info(f"Episode object type: {type(episode)}")

            # Ensure run_id and db_episode_id are available from custom_data
            run_id = episode.custom_data.get("db_run_id", self.run_id)
            db_episode_id = episode.custom_data.get("db_episode_id")

            if not run_id or not db_episode_id:
                logger.error(f"CRITICAL: run_id ({run_id}) or db_episode_id ({db_episode_id}) not found in episode.custom_data for RLlib episode_id {rllib_episode_id_str}. Aborting DB log.")
                return

            if episode.custom_data.get("_db_logged_end", False):
                logger.info(f"Episode {rllib_episode_id_str} (DB ID: {db_episode_id}) end already processed by this callback. Skipping.")
                return

            # Retrieve metrics from the final info dictionary
            last_info_dict = {}
            try:
                if hasattr(episode, 'last_info_for') and callable(getattr(episode, 'last_info_for')):
                    last_info_dict = episode.last_info_for()
            except Exception as e_info:
                logger.warning(f"Error retrieving last_info_for: {e_info}")

            # Attempt to get metrics from actual_env.cached_final_info_for_callback if last_info_dict is insufficient
            if not last_info_dict or not isinstance(last_info_dict, dict) or not last_info_dict.get('final_portfolio_value'):
                logger.warning(f"last_info_dict from episode.last_info_for() is missing or incomplete for RLlib episode {rllib_episode_id_str}. Attempting fallbacks.")
                
                # Fallback 1: Try worker.env.cached_final_info_for_callback (if worker is SingleAgentEnvRunner)
                # or base_env.cached_final_info_for_callback (if worker is RolloutWorker and base_env is the actual env)
                
                # The 'worker' parameter here can be SingleAgentEnvRunner (from on_episode_end)
                # or RolloutWorker (from on_sample_end).
                # The 'base_env' parameter here can be the actual TradingEnv (from on_episode_end if 'env' was in kwargs, or from on_sample_end)
                # or None.

                # Fallback 1: Try direct base_env (if it's a TradingEnv instance)
                # This should be the primary fallback now, as on_episode_end is modified to pass the actual TradingEnv as base_env.
                # Also covers on_sample_end where base_env is already the TradingEnv.
                if base_env and isinstance(base_env, TradingEnv):
                    logger.info(f"Fallback 1: Checking direct base_env (type: {type(base_env)}) for episode {rllib_episode_id_str}.")
                    if hasattr(base_env, 'cached_final_info_for_callback') and base_env.cached_final_info_for_callback:
                        cached_info = base_env.cached_final_info_for_callback
                        if isinstance(cached_info, dict) and cached_info.get('final_portfolio_value') is not None:
                            logger.info(f"Successfully retrieved cached_final_info_for_callback from direct base_env for episode {rllib_episode_id_str}.")
                            last_info_dict = cached_info
                        else:
                            logger.warning(f"cached_final_info_for_callback from direct base_env for episode {rllib_episode_id_str} was None, not a dict, or incomplete.")
                    else:
                        logger.warning(f"Direct base_env (TradingEnv) for episode {rllib_episode_id_str} does not have a valid cached_final_info_for_callback.")
                else:
                    logger.info(f"Fallback 1: Direct base_env is not a TradingEnv instance (type: {type(base_env)}) or is None for episode {rllib_episode_id_str}. Proceeding to next fallback.")

                # Fallback 2: Try via worker (SingleAgentEnvRunner) -> worker.env (SyncVectorEnv) -> worker.env.envs[0] (TradingEnv)
                # This is a secondary fallback, e.g. if base_env wasn't correctly passed or in on_sample_end context if base_env was not the TradingEnv.
                if (not last_info_dict or not isinstance(last_info_dict, dict) or not last_info_dict.get('final_portfolio_value')):
                    logger.info(f"Fallback 1 did not yield complete info. Attempting Fallback 2 (worker.env.envs[0]) for episode {rllib_episode_id_str}.")
                    if worker and hasattr(worker, 'env') and hasattr(worker.env, 'envs') and isinstance(worker.env.envs, list) and len(worker.env.envs) > 0:
                        actual_env_candidate = worker.env.envs[0]
                        logger.info(f"Fallback 2: Checking worker.env.envs[0] (type: {type(actual_env_candidate)}) for episode {rllib_episode_id_str}.")
                        if isinstance(actual_env_candidate, TradingEnv):
                            if hasattr(actual_env_candidate, 'cached_final_info_for_callback') and actual_env_candidate.cached_final_info_for_callback:
                                cached_info = actual_env_candidate.cached_final_info_for_callback
                                if isinstance(cached_info, dict) and cached_info.get('final_portfolio_value') is not None:
                                    logger.info(f"Successfully retrieved cached_final_info_for_callback from worker.env.envs[0] for episode {rllib_episode_id_str}.")
                                    last_info_dict = cached_info
                                else:
                                    logger.warning(f"cached_final_info_for_callback from worker.env.envs[0] for episode {rllib_episode_id_str} was None, not a dict, or incomplete.")
                            else:
                                logger.warning(f"worker.env.envs[0] (TradingEnv) for episode {rllib_episode_id_str} does not have a valid cached_final_info_for_callback.")
                        else:
                            logger.warning(f"worker.env.envs[0] is not a TradingEnv instance for episode {rllib_episode_id_str}. Type: {type(actual_env_candidate)}")
                    else:
                        logger.warning(f"Fallback 2: Worker or worker.env.envs structure not suitable for episode {rllib_episode_id_str}.")
            
            if not last_info_dict or not isinstance(last_info_dict, dict) or not last_info_dict.get('final_portfolio_value'): # Re-check after all fallbacks
                logger.error(f"CRITICAL: Could not retrieve valid last_info_dict (even after all fallbacks) for RLlib episode_id {rllib_episode_id_str}. Cannot log episode end metrics.")
                return

            initial_portfolio_value = last_info_dict.get('initial_portfolio_value', last_info_dict.get('initial_balance'))
            final_portfolio_value = last_info_dict.get('final_portfolio_value', last_info_dict.get('portfolio_value'))
            pnl_val = last_info_dict.get('pnl')
            sharpe_ratio_val = last_info_dict.get('sharpe_ratio')
            max_drawdown_val = last_info_dict.get('max_drawdown')
            win_rate_val = last_info_dict.get('win_rate')
            total_reward_val = last_info_dict.get('total_reward', getattr(episode, 'total_reward', None))
            total_steps_val = last_info_dict.get('total_steps', last_info_dict.get('episode_length', getattr(episode, 'length', None)))

            completed_trades = last_info_dict.get("completed_trades", [])
            if not isinstance(completed_trades, list):
                logger.warning(f"completed_trades in last_info_dict is not a list: {completed_trades}. Using empty list.")
                completed_trades = []

            # Ensure numeric values are not None before logging
            safe_initial_pf = float(initial_portfolio_value) if initial_portfolio_value is not None else 0.0
            safe_final_pf = float(final_portfolio_value) if final_portfolio_value is not None else 0.0
            safe_pnl = float(pnl_val) if pnl_val is not None else 0.0
            safe_sharpe = float(sharpe_ratio_val) if sharpe_ratio_val is not None else 0.0
            safe_mdd = float(max_drawdown_val) if max_drawdown_val is not None else 0.0
            safe_win_rate = float(win_rate_val) if win_rate_val is not None else 0.0
            safe_total_reward = float(total_reward_val) if total_reward_val is not None else 0.0
            safe_total_steps = int(total_steps_val) if total_steps_val is not None else 0

            with get_db_session() as db:
                db_episode = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()

                if db_episode:
                    db_episode.end_time = datetime.datetime.now(datetime.timezone.utc)
                    db_episode.status = "completed"
                    db_episode.final_portfolio_value = safe_final_pf
                    db_episode.pnl = safe_pnl
                    db_episode.sharpe_ratio = safe_sharpe
                    db_episode.max_drawdown = safe_mdd
                    db_episode.win_rate = safe_win_rate
                    db_episode.total_reward = safe_total_reward
                    db_episode.total_steps = safe_total_steps

                    # Update initial_portfolio_value if it wasn't set in _log_episode_start_data
                    if db_episode.initial_portfolio_value is None:
                        db_episode.initial_portfolio_value = safe_initial_pf
                        logger.info(f"Updated initial_portfolio_value for episode {db_episode_id} from end_info: {safe_initial_pf}")

                    # Log trades associated with this episode
                    for trade_data in completed_trades:
                        try:
                            # Filter trade_data to only include valid DbTrade fields
                            valid_trade_keys = [
                                "entry_time", "exit_time", "entry_price", "exit_price",
                                "quantity", "direction", "pnl", "costs"
                            ]
                            db_trade_kwargs = {
                                key: trade_data.get(key) for key in valid_trade_keys if key in trade_data
                            }
                            db_trade_kwargs["episode_id"] = db_episode.episode_id

                            db_trade = DbTrade(**db_trade_kwargs)
                            db.add(db_trade)
                        except Exception as e_trade:
                            logger.error(f"Error logging trade for episode {db_episode.episode_id}: {trade_data}. Error: {e_trade}", exc_info=True)

                    db.commit()
                    episode.custom_data["_db_logged_end"] = True # Mark as logged
                    logger.info(f"Episode {db_episode.episode_id} (RLlib ID: {rllib_episode_id_str}) end data logged to DB.")
                else:
                    logger.error(f"DB Episode record with ID {db_episode_id} not found for update. Cannot log episode end data.")
        except Exception as e:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in _log_episode_end_data for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e}", exc_info=True)
            # Do not re-raise

            
        

    def on_episode_end(
        self,
        *,
        episode: Any, # Make episode the primary required kwarg
        **kwargs,    # Capture all other arguments
    ) -> None:
        """Called at the end of each episode.
        Simplified signature to rely on kwargs for other parameters.
        """
        try: # Outer try-except for the entire method
            logger.info(f"--- on_episode_end CALLED for RLlib episode_id: {getattr(episode, 'id_', 'UNKNOWN_RLIB_EPISODE_ID')} ---")
            logger.info(f"on_episode_end kwargs received: {list(kwargs.keys())}")

            # Extract known optional params from kwargs if they exist, otherwise pass None
            # These will be passed explicitly.
            # In the new API stack, 'env_runner' is passed in kwargs, not 'worker'.
            # 'base_env' is also often passed directly as 'env' in kwargs by the EnvRunner.
            extracted_env_runner = kwargs.get("env_runner") # This is the SingleAgentEnvRunner
            extracted_policies = kwargs.get("policies") # May or may not be present
            extracted_env_index = kwargs.get("env_index") # Index of the env within the runner

            # Determine the actual TradingEnv instance to pass as base_env
            final_base_env_to_pass = None
            
            # Attempt to get the initial environment stack from common kwargs
            env_stack = None
            if extracted_env_runner and hasattr(extracted_env_runner, 'env'):
                env_stack = extracted_env_runner.env
                logger.info(f"In on_episode_end: Initial env stack from env_runner.env is {type(env_stack)}")
            elif "env" in kwargs:
                env_stack = kwargs["env"]
                logger.info(f"In on_episode_end: Initial env stack from kwargs['env'] is {type(env_stack)}")
            elif "base_env" in kwargs: # Less likely for SingleAgentEnvRunner but check
                env_stack = kwargs["base_env"]
                logger.info(f"In on_episode_end: Initial env stack from kwargs['base_env'] is {type(env_stack)}")

            if env_stack:
                # First, unwrap the outer layers until we find a vector env or the TradingEnv itself
                current_env = env_stack
                max_unwraps = 10  # Safety limit for unwrapping
                for i in range(max_unwraps):
                    if isinstance(current_env, TradingEnv):
                        final_base_env_to_pass = current_env
                        logger.info(f"In on_episode_end: Found TradingEnv directly (possibly after some unwraps). Type: {type(final_base_env_to_pass)}")
                        break
                    if hasattr(current_env, 'envs') and isinstance(current_env.envs, list) and len(current_env.envs) > 0:
                        # Likely a vector env (e.g., SyncVectorEnv)
                        logger.info(f"In on_episode_end: Found vector env: {type(current_env)}. Accessing envs[0].")
                        # Now, take the first actual environment and unwrap it
                        env_to_unwrap_further = current_env.envs[0]
                        for j in range(max_unwraps):
                            if isinstance(env_to_unwrap_further, TradingEnv):
                                final_base_env_to_pass = env_to_unwrap_further
                                logger.info(f"In on_episode_end: Successfully unwrapped envs[0] to TradingEnv. Type: {type(final_base_env_to_pass)}")
                                break
                            if hasattr(env_to_unwrap_further, 'env'):
                                logger.info(f"In on_episode_end (inner unwrap): Unwrapping {type(env_to_unwrap_further)}.")
                                env_to_unwrap_further = env_to_unwrap_further.env
                            else:
                                logger.warning(f"In on_episode_end (inner unwrap): Cannot unwrap {type(env_to_unwrap_further)} further.")
                                break
                        if final_base_env_to_pass: # Found from inner loop
                            break
                    
                    if hasattr(current_env, 'env'): # Common attribute for wrappers
                        logger.info(f"In on_episode_end (outer unwrap): Unwrapping {type(current_env)}.")
                        current_env = current_env.env
                    else:
                        logger.warning(f"In on_episode_end (outer unwrap): Cannot unwrap {type(current_env)} further.")
                        break # Stop if no more 'env' attribute
                else: # If loop finished without break
                    if not final_base_env_to_pass:
                        logger.warning(f"In on_episode_end: Max unwraps reached or could not find TradingEnv. Last checked type: {type(current_env)}")

            if final_base_env_to_pass is None:
                logger.error("CRITICAL in on_episode_end: Could not determine the actual TradingEnv instance to pass as base_env to _log_episode_end_data. Logging will likely fail.")
            else:
                logger.info(f"In on_episode_end: final_base_env_to_pass is type {type(final_base_env_to_pass)}")



            # Prepare a new kwargs dict for spreading, excluding the ones we pass explicitly.
            # 'env_runner' will be passed as 'worker' to _log_episode_end_data.
            remaining_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ['env_runner', 'env', 'base_env', 'policies', 'env_index', 'episode', 'worker']
            }
            logger.info(f"Extracted for _log_episode_end_data: env_runner(as worker)={type(extracted_env_runner)}, actual_TradingEnv(as base_env)={type(final_base_env_to_pass)}, policies={'set' if extracted_policies else 'None'}, env_index={extracted_env_index}")
            logger.info(f"Spreading remaining_kwargs to _log_episode_end_data: {list(remaining_kwargs.keys())}")

            self._log_episode_end_data(
                episode=episode,
                worker=extracted_env_runner, # Pass the SingleAgentEnvRunner as 'worker'
                base_env=final_base_env_to_pass,    # Pass the actual TradingEnv as 'base_env'
                policies=extracted_policies,
                env_index=extracted_env_index,
                **remaining_kwargs
            )
        except Exception as e_outer: # Catch-all for the entire method
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_episode_end: {e_outer}", exc_info=True)

    def on_sample_end(
        self,
        *,
        worker: Optional[RolloutWorker] = None, # Changed to keyword-only with default, RolloutWorker type
        samples: Optional["SampleBatch"] = None, # Changed to keyword-only with default
        **kwargs,
    ) -> None:
        """Processes completed episodes from a sample batch."""
        try:
            logger.info(f"--- on_sample_end CALLED --- (Worker type: {type(worker)}, Samples type: {type(samples)})")

            # Enhanced logging for samples object
            logger.info(f"on_sample_end: Received samples of type: {type(samples)}")
            if isinstance(samples, list):
                logger.info(f"on_sample_end: samples is a list with length {len(samples)}")
            if hasattr(samples, "count"): # SampleBatch has a 'count' attribute
                logger.info(f"on_sample_end: samples has a count attribute: {samples.count}")

            if not samples:
                logger.warning("on_sample_end called with no samples. Nothing to process.")
                return

            # Ensure run_id is available
            if not self.run_id and worker:
                self.run_id = self.get_run_id_from_algorithm(worker)
            if not self.run_id and "callbacks_config" in kwargs:
                 callbacks_cfg = kwargs["callbacks_config"]
                 if callbacks_cfg and "run_id" in callbacks_cfg:
                     self.run_id = callbacks_cfg["run_id"]
            
            if not self.run_id:
                logger.error("CRITICAL: self.run_id could not be determined in on_sample_end. Cannot process episodes from batch.")
                return

            # Iterate over episodes in the batch
            # The structure of `samples` can vary. For `SampleBatch.TYPE_EPISODES`, it's a list of Episode objects.
            # For `SampleBatch.TYPE_TRAJECTORIES`, we need to reconstruct or identify episodes.
            # RLlib's default behavior is to provide `SingleAgentEpisode` objects when `batch_mode="complete_episodes"`.
            
            # Check if samples is a SampleBatch and contains episode objects directly
            # This is common with `batch_mode="complete_episodes"`
            if isinstance(samples, SampleBatch) and samples.is_single_trajectory(): # A single trajectory might be a full episode
                # This case is less common for 'complete_episodes' but good to check
                # We need to check if this trajectory represents a completed episode
                if samples.terminateds[-1] or samples.truncateds[-1]:
                    # This is tricky because SampleBatch itself isn't an 'Episode' object in the same way.
                    # We might need to rely on episode IDs if they are present in the batch.
                    # For now, this path is less robust for direct episode metric logging.
                    # The `split_by_episode` method is more reliable.
                    logger.info("on_sample_end: Received a single trajectory SampleBatch. Attempting to process if it's a completed episode.")
                    # This requires more complex logic to map SampleBatch to an Episode-like structure for _log_episode_end_data
                    # For now, we'll rely on split_by_episode for more robust handling.
                    pass # Placeholder for potential future handling if needed

            # More robust: Use split_by_episode if available on the samples object
            # This method is designed to yield individual Episode objects from a batch.
            if hasattr(samples, "split_by_episode") and callable(samples.split_by_episode):
                num_episodes_in_batch = 0
                processed_episode_ids_in_batch = set()

                for episode_batch in samples.split_by_episode(batch_size=None): # Process one episode at a time
                    num_episodes_in_batch += 1
                    # `episode_batch` here should be an Episode-like object (e.g., SingleAgentEpisode)
                    # or a SampleBatch representing a single episode.
                    # We need to ensure we pass an Episode-like object to _log_episode_end_data.

                    # If episode_batch is a SampleBatch representing one episode, we might need to adapt it
                    # or rely on the fact that on_episode_end would have been called for it.
                    # However, if on_episode_end is not reliably called, this is a fallback.

                    # Let's assume episode_batch is an Episode-like object here.
                    # We need to get its unique ID.
                    current_rllib_episode_id = getattr(episode_batch, 'id_', getattr(episode_batch, 'episode_id', None))
                    if not current_rllib_episode_id:
                        logger.warning("on_sample_end: Could not get RLlib episode ID from episode_batch. Skipping.")
                        continue
                    
                    if current_rllib_episode_id in processed_episode_ids_in_batch:
                        logger.info(f"on_sample_end: RLlib episode {current_rllib_episode_id} already processed in this batch. Skipping.")
                        continue

                    # Check if the episode is actually terminated or truncated
                    # For SingleAgentEpisode, check `is_terminated` and `is_truncated`
                    is_done = False
                    if isinstance(episode_batch, SingleAgentEpisode):
                        is_done = episode_batch.is_terminated or episode_batch.is_truncated
                        logger.info(f"on_sample_end: Processing SingleAgentEpisode {current_rllib_episode_id}. is_terminated={episode_batch.is_terminated}, is_truncated={episode_batch.is_truncated}")
                    elif isinstance(episode_batch, SampleBatch): # If it's a SampleBatch for an episode
                        if episode_batch.count > 0: # Ensure not empty
                           is_done = episode_batch.terminateds[-1] or episode_batch.truncateds[-1]
                           logger.info(f"on_sample_end: Processing SampleBatch for episode {current_rllib_episode_id}. terminateds[-1]={episode_batch.terminateds[-1]}, truncateds[-1]={episode_batch.truncateds[-1]}")
                        else:
                            logger.warning(f"on_sample_end: SampleBatch for episode {current_rllib_episode_id} is empty. Skipping.")
                            continue
                    else:
                        logger.warning(f"on_sample_end: episode_batch is of unexpected type {type(episode_batch)}. Cannot determine if done. Skipping.")
                        continue # Cannot determine if done

                    if is_done:
                        logger.info(f"on_sample_end: Episode {current_rllib_episode_id} is marked as done. Calling _log_episode_end_data.")
                        # Pass relevant parts of the worker or kwargs if needed by _log_episode_end_data
                        # The `worker` here is the RolloutWorker. `base_env` might be on worker.env.
                        # `policies` might be on worker.policy_map.
                        
                        # Construct a minimal set of arguments for _log_episode_end_data
                        # It primarily needs the 'episode' object.
                        # Other arguments like base_env, policies, env_index might be harder to get reliably here
                        # if they are not part of the episode_batch object itself.
                        # _log_episode_end_data is designed to be robust to missing optional args.
                        
                        # Try to get base_env and policies from the worker if available
                        current_base_env = getattr(worker, 'env', None) if worker else None
                        current_policies = getattr(worker, 'policy_map', None) if worker else None
                        # env_index is tricky here, might not be directly available per episode from a batch.
                        # _log_episode_end_data should handle env_index=None.

                        self._log_episode_end_data(
                            episode=episode_batch, # This is the crucial part
                            worker=worker, # Pass the RolloutWorker
                            base_env=current_base_env,
                            policies=current_policies,
                            env_index=None, # env_index is harder to determine here
                            **kwargs # Pass original kwargs
                        )
                        processed_episode_ids_in_batch.add(current_rllib_episode_id)
                    else:
                        logger.info(f"on_sample_end: Episode {current_rllib_episode_id} from batch is not done. Not logging end.")
                
                if num_episodes_in_batch == 0:
                    logger.info("on_sample_end: samples.split_by_episode() yielded no episodes.")
                else:
                    logger.info(f"on_sample_end: Processed {len(processed_episode_ids_in_batch)} completed episodes from a batch of {num_episodes_in_batch} episode parts.")

            else: # Fallback if split_by_episode is not available (older RLlib or different SampleBatch type)
                logger.warning("on_sample_end: samples object does not have split_by_episode. Trying to iterate if it's a list of episodes (less common).")
                logger.info(f"on_sample_end: samples type: {type(samples)}")
                if isinstance(samples, list): # e.g. if samples was already a list of Episode objects
                    if len(samples) > 0:
                        logger.info(f"on_sample_end: samples[0] type: {type(samples[0])}")
                        if isinstance(samples[0], dict):
                            logger.info(f"on_sample_end: samples[0] keys: {list(samples[0].keys())}")
                    processed_episode_ids_in_batch = set()
                    for i, episode_obj in enumerate(samples):
                        if not hasattr(episode_obj, 'is_terminated') or not hasattr(episode_obj, 'is_truncated'): # Basic check
                            logger.warning(f"Item {i} in samples list is not an episode-like object with is_terminated/is_truncated. Type: {type(episode_obj)}. Skipping.")
                            continue
                        
                        current_rllib_episode_id = getattr(episode_obj, 'id_', getattr(episode_obj, 'episode_id', None))
                        if not current_rllib_episode_id:
                             logger.warning(f"on_sample_end (list iteration): Could not get RLlib episode ID from episode object at index {i}. Skipping.")
                             continue
                        
                        if current_rllib_episode_id in processed_episode_ids_in_batch:
                            logger.info(f"on_sample_end (list iteration): RLlib episode {current_rllib_episode_id} already processed in this list. Skipping.")
                            continue

                        is_done = episode_obj.is_terminated or episode_obj.is_truncated
                        if is_done:
                            logger.info(f"on_sample_end (list iteration): Episode {current_rllib_episode_id} is done. Calling _log_episode_end_data.")
                            current_base_env = getattr(worker, 'env', None) if worker else None
                            current_policies = getattr(worker, 'policy_map', None) if worker else None
                            self._log_episode_end_data(
                                episode=episode_obj,
                                worker=worker,
                                base_env=current_base_env,
                                policies=current_policies,
                                env_index=None, # env_index is hard to determine here
                                **kwargs
                            )
                            processed_episode_ids_in_batch.add(current_rllib_episode_id)
                        else:
                             logger.info(f"on_sample_end (list iteration): Episode {current_rllib_episode_id} from list is not done. Not logging end.")
                    logger.info(f"on_sample_end (list iteration): Processed {len(processed_episode_ids_in_batch)} completed episodes from list.")
                else:
                    logger.warning("on_sample_end: samples object is not a list and does not have split_by_episode. Cannot process episodes from this SampleBatch for _log_episode_end_data.")

        except Exception as e:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in on_sample_end: {e}", exc_info=True)


    def on_episode_step(
        self,
        *,
        episode: Any, # Make episode the primary required kwarg
        **kwargs,    # Capture all other arguments
    ) -> None:
        """Called after each step within an episode."""
        try:
            rllib_episode_id_str = getattr(episode, 'id_', getattr(episode, 'episode_id', 'UNKNOWN_RLIB_EPISODE_ID'))
            logger.debug(f"--- on_episode_step CALLED for RLlib episode_id: {rllib_episode_id_str}, Length: {len(episode) if hasattr(episode, '__len__') else 'N/A'} ---")

            if not self.run_id:
                worker = kwargs.get("worker") # Try to get worker from kwargs
                if worker: self.run_id = self.get_run_id_from_algorithm(worker)
                if not self.run_id and "callbacks_config" in kwargs and kwargs["callbacks_config"] and "run_id" in kwargs["callbacks_config"]:
                     self.run_id = kwargs["callbacks_config"]["run_id"]

                if not self.run_id:
                    logger.error(f"CRITICAL: self.run_id is NOT SET in on_episode_step for RLlib episode {rllib_episode_id_str}. Cannot log step.")
                    return

            db_episode_id = episode.custom_data.get("db_episode_id")
            if not db_episode_id:
                # Try to find it if not set (e.g., if on_episode_start failed or was missed)
                # This is a fallback and might be slow if called every step.
                logger.warning(f"db_episode_id not in custom_data for RLlib episode {rllib_episode_id_str} during on_episode_step. Attempting DB lookup.")
                try:
                    with get_db_session() as db_find_ep:
                        db_ep_obj = db_find_ep.query(DbEpisode.episode_id).filter(
                            DbEpisode.run_id == self.run_id,
                            DbEpisode.rllib_episode_id == str(rllib_episode_id_str)
                        ).first()
                        if db_ep_obj:
                            db_episode_id = db_ep_obj.episode_id
                            episode.custom_data["db_episode_id"] = db_episode_id # Cache it
                            logger.info(f"Found and cached db_episode_id {db_episode_id} for RLlib_ep {rllib_episode_id_str} in on_episode_step.")
                        else:
                            logger.error(f"Could not find matching DB episode for RLlib_ep {rllib_episode_id_str} and run {self.run_id} in on_episode_step. Cannot log step.")
                            return
                except Exception as e_find:
                    logger.error(f"Error looking up db_episode_id in on_episode_step: {e_find}", exc_info=True)
                    return
            
            # Get the last observation, action, reward, and info
            # For SingleAgentEpisode, these are typically accessed via methods
            last_observation = None
            last_action = None
            last_reward = None
            last_info_for_step = {} # Store info dict for this step

            if isinstance(episode, SingleAgentEpisode):
                if episode.get_observations(): last_observation = episode.get_observations()[-1]
                if episode.get_actions(): last_action = episode.get_actions()[-1]
                if episode.get_rewards(): last_reward = episode.get_rewards()[-1]
                if episode.get_infos(): 
                    temp_info = episode.get_infos()[-1]
                    if isinstance(temp_info, dict): last_info_for_step = temp_info
            else: # Older Episode API might have these as direct attributes or via agent_id methods
                # This part is more complex for older multi-agent; focusing on single agent for now
                # last_observation = episode.last_observation_for() # Needs agent_id for multi-agent
                # last_action = episode.last_action_for()
                # last_reward = episode.last_reward_for()
                # last_info_for_step = episode.last_info_for()
                logger.debug("Using direct attribute access for obs/act/reward/info for non-SingleAgentEpisode (may be limited).")
                if hasattr(episode, '_agent_to_last_obs'): last_observation = episode._agent_to_last_obs.get(Policy.DEFAULT_POLICY_ID) # Example
                if hasattr(episode, '_agent_to_last_action'): last_action = episode._agent_to_last_action.get(Policy.DEFAULT_POLICY_ID)
                if hasattr(episode, '_agent_to_last_reward'): last_reward = episode._agent_to_last_reward.get(Policy.DEFAULT_POLICY_ID)
                if hasattr(episode, '_agent_to_last_info'): 
                    temp_info = episode._agent_to_last_info.get(Policy.DEFAULT_POLICY_ID, {})
                    if isinstance(temp_info, dict): last_info_for_step = temp_info


            current_step_number = len(episode) # RLlib episode length is 1-based for current step
            
            # Extract trading operation details if present in info
            operation_type_str = last_info_for_step.get("operation_type_for_log") # Corrected key
            operation_price = last_info_for_step.get("execution_price_this_step")    # Corrected key
            operation_quantity = last_info_for_step.get("shares_transacted_this_step") # Corrected key
            # operation_cost, current_balance_after_op, current_position_after_op are not part of DbTradingOperation model
            
            # Log step data
            with get_db_session() as db:
                # Extract values from last_info_for_step, providing defaults or None if not found
                step_portfolio_value = last_info_for_step.get('portfolio_value')
                step_asset_price = last_info_for_step.get('current_price') # Mapped from 'current_price' in env
                step_action = last_info_for_step.get('action_taken')      # Mapped from 'action_taken' in env
                step_position = last_info_for_step.get('current_position') # Mapped from 'current_position' in env

                # Ensure numeric types are float or None
                step_portfolio_value = float(step_portfolio_value) if step_portfolio_value is not None else None
                step_asset_price = float(step_asset_price) if step_asset_price is not None else None
                
                # Action and Position are expected to be strings or convertible to strings.
                # If they are integers (e.g. from action_space), convert them.
                if isinstance(step_action, (int, np.integer)): # np.integer for numpy int types
                    # Map discrete action to string if necessary, or store as string
                    # Example mapping: 0 -> "Flat", 1 -> "Long", 2 -> "Short"
                    # This mapping should align with how actions are interpreted/defined.
                    # For now, just converting to string.
                    step_action = str(step_action)
                
                if isinstance(step_position, (int, np.integer)):
                    # Example mapping: 0 -> "Flat", 1 -> "Long", -1 -> "Short"
                    # This mapping should align with how positions are defined.
                    # For now, just converting to string.
                    step_position = str(step_position)

                db_step = DbStep(
                    episode_id=db_episode_id,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                    reward=float(last_reward) if last_reward is not None else None,
                    portfolio_value=step_portfolio_value,
                    asset_price=step_asset_price,
                    action=step_action,
                    position=step_position
                    # obs, action can be large; consider how/if to store them (e.g., hash, summary, or omit)
                    # observation_data=str(last_observation)[:255] if last_observation is not None else None, # Example: truncate
                    # action_data=str(last_action)[:255] if last_action is not None else None # Example: truncate
                )
                db.add(db_step)
                db.flush() # Ensure db_step.step_id is populated
                
                # Log trading operation if details are present
                if operation_type_str and operation_price is not None and operation_quantity is not None:
                    # Ensure operation_type_str is actually a string before .upper()
                    if not isinstance(operation_type_str, str):
                        # If it's an Enum member already, get its name
                        if isinstance(operation_type_str, OperationType):
                            operation_type_str = operation_type_str.name
                        else:
                            logger.warning(f"operation_type_str is not a string or OperationType enum: {type(operation_type_str)}. Value: {operation_type_str}. Skipping operation log.")
                            operation_type_str = None # Prevent further processing

                    if operation_type_str: # Proceed if we have a valid string
                        try:
                            op_type_enum = OperationType[operation_type_str.upper()] # Convert string to Enum
                            db_trading_op = DbTradingOperation(
                                episode_id=db_episode_id,
                                step_id=db_step.step_id, # Correctly use the DbStep's ID
                                timestamp=datetime.datetime.now(datetime.timezone.utc), # Could also use env's current time if available
                                operation_type=op_type_enum,
                                price=float(operation_price), # Ensure float
                                size=float(operation_quantity) # Use 'size' and ensure float
                                # Removed cost, balance_after_operation, position_after_operation
                            )
                            db.add(db_trading_op)
                            logger.debug(f"Logged trading operation: {op_type_enum} at RLlib step {current_step_number}, db_step_id {db_step.step_id}")
                        except KeyError:
                            logger.warning(f"Invalid operation_type string '{operation_type_str}' at RLlib step {current_step_number}. Cannot log operation.")
                        except Exception as e_op_log:
                            logger.error(f"Error logging trading operation at RLlib step {current_step_number}: {e_op_log}", exc_info=True)
                
                db.commit() # Commit step and any operation
                logger.debug(f"Step {current_step_number} for episode {db_episode_id} logged. Reward: {last_reward}")

        except Exception as e:
            logger.error(f"Error in on_episode_step for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e}", exc_info=True)
            
    def on_algorithm_shutdown(self, *, algorithm, **kwargs) -> None:
        """
        Called when the algorithm is shut down.
        
        This method ensures that all in-progress episodes are properly finalized
        when the algorithm is shut down, either normally or due to an exception.
        
        Args:
            algorithm: The training algorithm instance.
            **kwargs: Additional keyword arguments.
        """
        try:
            logger.info("on_algorithm_shutdown called. Finalizing any incomplete episodes...")
            
            # Ensure run_id is set
            if not self.run_id and hasattr(algorithm, "config") and algorithm.config:
                if "callbacks_config" in algorithm.config and algorithm.config["callbacks_config"]:
                    if "run_id" in algorithm.config["callbacks_config"]:
                        self.run_id = algorithm.config["callbacks_config"]["run_id"]
                        logger.info(f"Set run_id to {self.run_id} from algorithm.config in on_algorithm_shutdown")
            
            if not self.run_id:
                logger.error("run_id not set in on_algorithm_shutdown. Cannot finalize incomplete episodes.")
                return
                
            # Finalize all incomplete episodes
            self.finalize_incomplete_episodes()
            
        except Exception as e:
            logger.critical(f"Error in on_algorithm_shutdown: {e}", exc_info=True)
            
    def on_train_result(self, *, algorithm, result, **kwargs) -> None:
        """
        Called at the end of each training iteration.
        
        This method checks if the training run is about to be terminated and calls
        finalize_incomplete_episodes if necessary.
        
        Args:
            algorithm: The training algorithm instance.
            result: The training result dictionary.
            **kwargs: Additional keyword arguments.
        """
        try:
            logger.info(f"on_train_result called with result keys: {list(result.keys())}")
            
            # Ensure run_id is set
            if not self.run_id and hasattr(algorithm, "config") and algorithm.config:
                if "callbacks_config" in algorithm.config and algorithm.config["callbacks_config"]:
                    if "run_id" in algorithm.config["callbacks_config"]:
                        self.run_id = algorithm.config["callbacks_config"]["run_id"]
                        logger.info(f"Set run_id to {self.run_id} from algorithm.config in on_train_result")
            
            if not self.run_id:
                logger.error("run_id not set in on_train_result. Cannot check for training completion.")
                return
                
            # Check if this is the final iteration
            current_iteration = result.get("training_iteration", 0)
            iterations_since_restore = result.get("iterations_since_restore", 0)
            
            # Try to get the total number of iterations from the algorithm config
            max_iterations = None
            if hasattr(algorithm, "config") and algorithm.config:
                # Different RLlib algorithms might store this differently
                # Try common patterns
                if hasattr(algorithm, "_num_iterations") and algorithm._num_iterations is not None:
                    max_iterations = algorithm._num_iterations
                elif hasattr(algorithm.config, "num_iterations") and algorithm.config.num_iterations is not None:
                    max_iterations = algorithm.config.num_iterations
                elif hasattr(algorithm.config, "training") and hasattr(algorithm.config.training, "num_iterations"):
                    max_iterations = algorithm.config.training.num_iterations
            
            # If we couldn't find max_iterations in the config, try to get it from the result
            if max_iterations is None and "num_training_iterations" in result:
                max_iterations = result["num_training_iterations"]
                
            # Log what we found
            logger.info(f"Current iteration: {current_iteration}, Iterations since restore: {iterations_since_restore}, Max iterations: {max_iterations}")
            
            # Check if this is the final iteration
            is_final_iteration = False
            if max_iterations is not None:
                is_final_iteration = current_iteration >= max_iterations or iterations_since_restore >= max_iterations
                
            # Also check the 'done' flag in the result
            if "done" in result and result["done"]:
                is_final_iteration = True
                
            # For manual training loops (like in train.py), check if this is the last iteration
            # by comparing current_iteration with NUM_TRAINING_ITERATIONS (if available)
            if hasattr(algorithm, "config") and algorithm.config:
                if "callbacks_config" in algorithm.config and algorithm.config["callbacks_config"]:
                    if "num_training_iterations" in algorithm.config["callbacks_config"]:
                        num_training_iterations = algorithm.config["callbacks_config"]["num_training_iterations"]
                        if current_iteration >= num_training_iterations:
                            is_final_iteration = True
                            logger.info(f"Final iteration detected based on num_training_iterations={num_training_iterations}")
            
            if is_final_iteration:
                logger.info(f"Final iteration detected (iteration {current_iteration}). Finalizing incomplete episodes...")
                self.finalize_incomplete_episodes()
            else:
                logger.info(f"Not the final iteration (iteration {current_iteration}). Continuing training...")
                
        except Exception as e:
            logger.critical(f"Error in on_train_result: {e}", exc_info=True)
            
    def finalize_incomplete_episodes(self) -> None:
        """
        Finalize all incomplete episodes for the current run_id.
        
        This method is called when the training run is terminated to ensure that all
        in-progress episodes are properly finalized in the database. It updates the
        status of all episodes with a status of "started" to "completed" and fills in
        default values for the missing metrics.
        """
        if not self.run_id:
            logger.error("Cannot finalize incomplete episodes: run_id is not set.")
            return
            
        try:
            with get_db_session() as db:
                # Query for all episodes with status "started" for the current run_id
                incomplete_episodes = db.query(DbEpisode).filter(
                    DbEpisode.run_id == self.run_id,
                    DbEpisode.status == "started"
                ).all()
                
                if not incomplete_episodes:
                    logger.info(f"No incomplete episodes found for run_id {self.run_id}.")
                    return
                    
                logger.info(f"Found {len(incomplete_episodes)} incomplete episodes for run_id {self.run_id}. Finalizing...")
                
                # Update each incomplete episode
                for db_episode in incomplete_episodes:
                    # Set end time to current time
                    db_episode.end_time = datetime.datetime.now(datetime.timezone.utc)
                    
                    # Update status to "completed"
                    db_episode.status = "completed"
                    
                    # If initial_portfolio_value is not set, use a default value
                    if db_episode.initial_portfolio_value is None:
                        db_episode.initial_portfolio_value = 10000.0  # Default initial balance
                        
                    # Calculate final_portfolio_value based on the last step's portfolio_value
                    # or use initial_portfolio_value if no steps are found
                    last_step = db.query(DbStep).filter(
                        DbStep.episode_id == db_episode.episode_id
                    ).order_by(DbStep.timestamp.desc()).first()
                    
                    if last_step and last_step.portfolio_value is not None:
                        db_episode.final_portfolio_value = last_step.portfolio_value
                    else:
                        # If no steps with portfolio_value, use initial_portfolio_value
                        db_episode.final_portfolio_value = db_episode.initial_portfolio_value
                        
                    # Calculate PnL
                    db_episode.pnl = db_episode.final_portfolio_value - db_episode.initial_portfolio_value
                    
                    # Set default values for other metrics
                    db_episode.sharpe_ratio = 0.0
                    db_episode.max_drawdown = 0.0
                    db_episode.win_rate = 0.0
                    
                    # Count total steps
                    total_steps = db.query(DbStep).filter(
                        DbStep.episode_id == db_episode.episode_id
                    ).count()
                    db_episode.total_steps = total_steps
                    
                    # Calculate total reward (sum of rewards from all steps)
                    total_reward = db.query(func.sum(DbStep.reward)).filter(
                        DbStep.episode_id == db_episode.episode_id,
                        DbStep.reward.isnot(None)
                    ).scalar() or 0.0
                    db_episode.total_reward = total_reward
                    
                    logger.info(f"Finalized episode {db_episode.episode_id}: final_pf={db_episode.final_portfolio_value}, pnl={db_episode.pnl}, total_steps={db_episode.total_steps}, total_reward={db_episode.total_reward}")
                
                # Commit all changes
                db.commit()
                logger.info(f"Successfully finalized {len(incomplete_episodes)} incomplete episodes for run_id {self.run_id}.")
                
        except Exception as e:
            logger.critical(f"Error finalizing incomplete episodes for run_id {self.run_id}: {e}", exc_info=True)