import datetime
import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
# Episode class is not directly importable in new RLlib API stack for callbacks.
# The 'episode' object passed to callbacks is an internal type.
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.policy import Policy
from typing import Dict, Optional, Any # Added Any for Episode type hint

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode as DbEpisode, Trade as DbTrade, TrainingRun

logger = logging.getLogger(__name__)

class DatabaseLoggingCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict = None): # Re-added legacy_callbacks_dict
        super().__init__()
        self.run_id = None # Initialize self.run_id
        if legacy_callbacks_dict and "run_id" in legacy_callbacks_dict:
            self.run_id = legacy_callbacks_dict["run_id"]
        logger.info(f"DatabaseLoggingCallbacks initialized. Run ID: {self.run_id}")

    def on_episode_start(
        self,
        *,
        episode: Any,
        env_index: Optional[int] = None,
        worker: Optional[EnvRunnerGroup] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs,
    ) -> None:
        # Now use self.run_id which should be set in __init__
        if not self.run_id:
            logger.warning("self.run_id not set in DatabaseLoggingCallbacks for on_episode_start. Skipping DB logging.")
            return

        # base_env and env_index are crucial for getting the actual environment instance
        if base_env is None or env_index is None:
            logger.warning("base_env or env_index not available in on_episode_start. Cannot get sub_environment.")
            # Try to get env from episode if possible (might be Ray 2.3+ style)
            # This is a fallback, might not always work or be the right env instance
            env = episode.env if hasattr(episode, 'env') else None
            if not env:
                 logger.error("Could not retrieve environment instance in on_episode_start.")
                 return
        else:
            env = base_env.get_sub_environments()[env_index]
        initial_portfolio_value = getattr(env, 'initial_balance', None) # TradingEnv specific

        try:
            with get_db_session() as db:
                # Check if TrainingRun exists using self.run_id
                training_run = db.query(TrainingRun).filter(TrainingRun.run_id == self.run_id).first()
                if not training_run:
                    logger.error(f"TrainingRun with run_id {self.run_id} not found. Cannot log episode.")
                    return

                db_episode = DbEpisode(
                    run_id=self.run_id,
                    start_time=datetime.datetime.now(datetime.UTC),
                    initial_portfolio_value=initial_portfolio_value,
                    # Other fields will be updated in on_episode_end
                )
                db.add(db_episode)
                db.commit()
                db.refresh(db_episode)
                episode.custom_data["db_episode_id"] = db_episode.episode_id
                logger.debug(f"Logged episode start for RLlib episode {episode.id_}, DB episode ID: {db_episode.episode_id}, Run ID: {self.run_id}") # Use self.run_id
        except Exception as e:
            logger.error(f"Error in on_episode_start DB logging: {e}", exc_info=True)

    def on_episode_end(
        self,
        *, # Re-added * to enforce keyword-only arguments
        episode: Any,
        env_index: Optional[int] = None,
        # Make these optional
        worker: Optional[EnvRunnerGroup] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[str, Policy]] = None,
        **kwargs,
    ) -> None:
        db_episode_id = episode.custom_data.get("db_episode_id")
        # run_id is now expected to be set in self.run_id during __init__
        if not db_episode_id or not self.run_id:
            logger.warning(f"db_episode_id or self.run_id not found for RLlib episode {episode.id_}. Skipping DB logging for episode end.")
            return

        env = base_env.get_sub_environments()[env_index]
        
        final_portfolio_value = getattr(env, 'portfolio_value', None)
        initial_portfolio_value = getattr(env, 'initial_balance', None)
        pnl = None
        if final_portfolio_value is not None and initial_portfolio_value is not None:
            pnl = final_portfolio_value - initial_portfolio_value

        # Attempt to get custom metrics from the episode if TradingEnv populates them
        sharpe_ratio = episode.custom_metrics.get("sharpe_ratio", None)
        max_drawdown = episode.custom_metrics.get("max_drawdown", None)
        win_rate = episode.custom_metrics.get("win_rate", None)
        
        try:
            with get_db_session() as db:
                db_episode = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                if db_episode:
                    db_episode.end_time = datetime.datetime.now(datetime.UTC)
                    db_episode.final_portfolio_value = final_portfolio_value
                    db_episode.pnl = pnl
                    db_episode.total_reward = episode.total_reward
                    db_episode.total_steps = episode.length
                    db_episode.sharpe_ratio = sharpe_ratio
                    db_episode.max_drawdown = max_drawdown
                    db_episode.win_rate = win_rate
                    
                    # Log completed trades
                    completed_trades = getattr(env, '_completed_trades', [])
                    if completed_trades:
                        logger.debug(f"Logging {len(completed_trades)} trades for DB episode ID: {db_episode_id}")
                    for trade_info in completed_trades:
                        db_trade = DbTrade(
                            episode_id=db_episode_id,
                            entry_time=trade_info.get("entry_time"),
                            exit_time=trade_info.get("exit_time"),
                            entry_price=trade_info.get("entry_price"),
                            exit_price=trade_info.get("exit_price"),
                            quantity=trade_info.get("quantity"),
                            direction=trade_info.get("direction"),
                            pnl=trade_info.get("pnl"),
                            costs=trade_info.get("costs", 0.0) # Assuming costs might not always be present
                        )
                        db.add(db_trade)
                    
                    if hasattr(env, '_clear_episode_trades'): # Method to clear trades after logging
                        env._clear_episode_trades()
                    elif hasattr(env, '_completed_trades'): # Or just clear the list
                         env._completed_trades = []


                    db.commit()
                    logger.debug(f"Logged episode end for RLlib episode {episode.id_}, DB episode ID: {db_episode_id}") # Changed to episode.id_
                else:
                    logger.warning(f"Could not find DB episode with ID {db_episode_id} to update.")
        except Exception as e:
            logger.error(f"Error in on_episode_end DB logging: {e}", exc_info=True)

    # on_episode_step could be used for very granular logging if needed,
    # but often on_episode_end is sufficient for aggregated metrics and trades.
    # def on_episode_step(self, *, worker: "WorkerSet", base_env: BaseEnv, episode: Episode, env_index: Optional[int] = None, **kwargs) -> None:
    #     # Example: log step-specific data if necessary
    #     # db_episode_id = episode.user_data.get("db_episode_id")
    #     # if db_episode_id:
    #     #     # Log step data
    #     pass