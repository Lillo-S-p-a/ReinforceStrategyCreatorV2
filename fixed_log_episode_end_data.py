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
                
            if not last_info_dict or not isinstance(last_info_dict, dict):
                logger.error(f"CRITICAL: Could not retrieve valid last_info_dict for RLlib episode_id {rllib_episode_id_str}. Cannot log episode end metrics.")
                return

            # Extract metrics from last_info_dict
            initial_portfolio_value = last_info_dict.get('initial_portfolio_value', last_info_dict.get('initial_balance'))
            final_portfolio_value = last_info_dict.get('final_portfolio_value', last_info_dict.get('portfolio_value'))
            pnl_val = last_info_dict.get('pnl')
            sharpe_ratio_val = last_info_dict.get('sharpe_ratio')
            max_drawdown_val = last_info_dict.get('max_drawdown')
            win_rate_val = last_info_dict.get('win_rate')
            total_reward_val = last_info_dict.get('total_reward', getattr(episode, 'total_reward', None))
            total_steps_val = last_info_dict.get('total_steps', last_info_dict.get('episode_length', getattr(episode, 'length', None)))

            # Extract completed trades
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

            # Update the database record
            with get_db_session() as db:
                db_episode = db.query(DbEpisode).filter(DbEpisode.episode_id == db_episode_id).first()
                
                if db_episode:
                    # Update episode metrics
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
                            db_trade = DbTrade(
                                episode_id=db_episode.episode_id,
                                entry_step=trade_data.get('entry_step'),
                                exit_step=trade_data.get('exit_step'),
                                entry_time=trade_data.get('entry_time'),
                                exit_time=trade_data.get('exit_time'),
                                entry_price=trade_data.get('entry_price'),
                                exit_price=trade_data.get('exit_price'),
                                quantity=trade_data.get('quantity'),
                                direction=trade_data.get('direction'),
                                pnl=trade_data.get('pnl'),
                                costs=trade_data.get('costs')
                            )
                            db.add(db_trade)
                        except Exception as e_trade:
                            logger.error(f"Error logging trade for episode {db_episode.episode_id}: {trade_data}. Error: {e_trade}", exc_info=True)
                    
                    # Commit changes to database
                    db.commit()
                    episode.custom_data["_db_logged_end"] = True  # Mark as logged
                    logger.info(f"Episode {db_episode.episode_id} (RLlib ID: {rllib_episode_id_str}) end data logged to DB.")
                else:
                    logger.error(f"DB Episode record with ID {db_episode_id} not found for update. Cannot log episode end data.")
                    
        except Exception as e_outer:
            logger.critical(f"CRITICAL UNCAUGHT EXCEPTION in _log_episode_end_data for RLlib episode {getattr(episode, 'id_', 'unknown_rllib_id')}: {e_outer}", exc_info=True)
            # Do not re-raise