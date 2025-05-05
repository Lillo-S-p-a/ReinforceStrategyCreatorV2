#!/usr/bin/env python3
# verify_operations_fix.py
import sys
import os
import datetime
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to allow imports from reinforcestrategycreator
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import TradingOperation, Episode, Step, OperationType

def check_episode_exists(run_id, episode_id):
    """
    Check if the specified episode exists and belongs to the specified run.
    
    Args:
        run_id (str): The run ID to check
        episode_id (int): The episode ID to check
    
    Returns:
        bool: True if the episode exists, False otherwise
    """
    try:
        with get_db_session() as db:
            episode = db.query(Episode).filter(
                Episode.episode_id == episode_id
            ).first()
            
            if not episode:
                print(f"No episode found with ID {episode_id}")
                return False
            
            if episode.run_id != run_id:
                print(f"Episode {episode_id} exists but belongs to run '{episode.run_id}', not '{run_id}'")
                return False
            
            print(f"Episode {episode_id} exists and belongs to run '{run_id}'")
            return True
    except Exception as e:
        print(f"Error querying database for episode: {e}")
        return False

def check_trading_operations(run_id, episode_id):
    """
    Check if any TradingOperation records exist for the specified run_id and episode_id.
    
    Args:
        run_id (str): The run ID to check
        episode_id (int): The episode ID to check
    
    Returns:
        bool: True if records exist, False otherwise
    """
    try:
        with get_db_session() as db:
            # Check if any trading operations exist for this episode
            count = db.query(TradingOperation).filter(
                TradingOperation.episode_id == episode_id
            ).count()
            
            exists = count > 0
            print(f"Trading operations for episode_id={episode_id}: {'EXIST' if exists else 'DO NOT EXIST'}")
            print(f"Found {count} trading operation records")
            
            return exists
    except Exception as e:
        print(f"Error querying database for trading operations: {e}")
        return False

def fix_missing_operations(run_id, episode_id):
    """
    Fix missing TradingOperation records by generating them from Step records.
    
    Args:
        run_id (str): The run ID to fix
        episode_id (int): The episode ID to fix
    
    Returns:
        bool: True if fix was successful, False otherwise
    """
    try:
        with get_db_session() as db:
            # Get all steps for this episode
            steps = db.query(Step).filter(
                Step.episode_id == episode_id
            ).order_by(Step.timestamp).all()
            
            if not steps:
                print(f"No steps found for episode_id={episode_id}")
                return False
            
            print(f"Found {len(steps)} steps for episode_id={episode_id}")
            
            # Track current position to detect changes
            current_position = "flat"  # Start with flat position
            operations_created = 0
            
            for i, step in enumerate(steps):
                # Skip the first step as it's the initial state
                if i == 0:
                    continue
                
                # Get the action and position from the step
                action = step.action
                position = step.position
                
                # Determine if a trading operation occurred based on position change
                if position != current_position:
                    # Determine operation type based on position transition
                    operation_type = None
                    
                    if current_position == "flat" and position == "long":
                        operation_type = OperationType.ENTRY_LONG
                        size = 1.0  # Placeholder size
                    elif current_position == "flat" and position == "short":
                        operation_type = OperationType.ENTRY_SHORT
                        size = 1.0  # Placeholder size
                    elif current_position == "long" and position == "flat":
                        operation_type = OperationType.EXIT_LONG
                        size = 1.0  # Placeholder size
                    elif current_position == "short" and position == "flat":
                        operation_type = OperationType.EXIT_SHORT
                        size = 1.0  # Placeholder size
                    elif current_position == "long" and position == "short":
                        # This is a two-step operation: exit long, then enter short
                        # First, create exit long operation
                        exit_long_op = TradingOperation(
                            step_id=step.step_id,
                            episode_id=episode_id,
                            timestamp=step.timestamp,
                            operation_type=OperationType.EXIT_LONG,
                            size=1.0,  # Placeholder size
                            price=100.0  # Placeholder price
                        )
                        db.add(exit_long_op)
                        operations_created += 1
                        
                        # Then set operation_type for enter short
                        operation_type = OperationType.ENTRY_SHORT
                        size = 1.0  # Placeholder size
                    elif current_position == "short" and position == "long":
                        # This is a two-step operation: exit short, then enter long
                        # First, create exit short operation
                        exit_short_op = TradingOperation(
                            step_id=step.step_id,
                            episode_id=episode_id,
                            timestamp=step.timestamp,
                            operation_type=OperationType.EXIT_SHORT,
                            size=1.0,  # Placeholder size
                            price=100.0  # Placeholder price
                        )
                        db.add(exit_short_op)
                        operations_created += 1
                        
                        # Then set operation_type for enter long
                        operation_type = OperationType.ENTRY_LONG
                        size = 1.0  # Placeholder size
                    
                    # Create the trading operation if an operation type was determined
                    if operation_type:
                        trading_op = TradingOperation(
                            step_id=step.step_id,
                            episode_id=episode_id,
                            timestamp=step.timestamp,
                            operation_type=operation_type,
                            size=size,
                            price=100.0  # Placeholder price
                        )
                        db.add(trading_op)
                        operations_created += 1
                
                # Update current position for next iteration
                current_position = position
            
            # Commit the changes
            db.commit()
            print(f"Created {operations_created} trading operation records")
            
            return operations_created > 0
    except Exception as e:
        print(f"Error fixing missing trading operations: {e}")
        return False

if __name__ == "__main__":
    run_id = "RUN-SPY-20250505113632-48ccd3c0"
    episode_id = 56
    
    print(f"Checking for episode with run_id='{run_id}' and episode_id={episode_id}")
    episode_exists = check_episode_exists(run_id, episode_id)
    
    if episode_exists:
        print(f"\nChecking for trading operations with episode_id={episode_id}")
        operations_exist = check_trading_operations(run_id, episode_id)
        
        if not operations_exist:
            print(f"\nAttempting to fix missing trading operations for episode_id={episode_id}")
            fix_success = fix_missing_operations(run_id, episode_id)
            
            if fix_success:
                print(f"\nSuccessfully created trading operations for episode_id={episode_id}")
                print(f"Verifying trading operations now exist:")
                check_trading_operations(run_id, episode_id)
            else:
                print(f"\nFailed to create trading operations for episode_id={episode_id}")
        else:
            print(f"\nTrading operations already exist for episode_id={episode_id}, no fix needed")
    else:
        print(f"Cannot fix trading operations because episode does not exist or doesn't match run_id")