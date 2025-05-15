#!/usr/bin/env python3
"""
Script to fix incomplete episodes in the database.

This script updates episodes with IDs from 253 to 315 that are in "started" status
to "completed" status and fills in default values for the missing metrics.
"""

import datetime
import logging
from sqlalchemy import func

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode as DbEpisode, Step as DbStep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_incomplete_episodes(start_id=253, end_id=315):
    """
    Fix incomplete episodes with IDs in the specified range.
    
    Args:
        start_id (int): The starting episode ID.
        end_id (int): The ending episode ID.
    """
    try:
        with get_db_session() as db:
            # Query for all episodes with IDs in the specified range and status "started"
            incomplete_episodes = db.query(DbEpisode).filter(
                DbEpisode.episode_id >= start_id,
                DbEpisode.episode_id <= end_id,
                DbEpisode.status == "started"
            ).all()
            
            if not incomplete_episodes:
                logger.info(f"No incomplete episodes found with IDs from {start_id} to {end_id}.")
                return
                
            logger.info(f"Found {len(incomplete_episodes)} incomplete episodes with IDs from {start_id} to {end_id}. Fixing...")
            
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
                
                logger.info(f"Fixed episode {db_episode.episode_id}: final_pf={db_episode.final_portfolio_value}, pnl={db_episode.pnl}, total_steps={db_episode.total_steps}, total_reward={db_episode.total_reward}")
            
            # Commit all changes
            db.commit()
            logger.info(f"Successfully fixed {len(incomplete_episodes)} incomplete episodes with IDs from {start_id} to {end_id}.")
            
    except Exception as e:
        logger.critical(f"Error fixing incomplete episodes: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting fix_incomplete_episodes script...")
    fix_incomplete_episodes()
    logger.info("Script completed.")