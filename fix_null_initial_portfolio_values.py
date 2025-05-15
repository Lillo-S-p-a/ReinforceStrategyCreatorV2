#!/usr/bin/env python3
"""
Script to fix NULL initial_portfolio_value in episodes table.

This script connects to the database and updates all episodes with NULL
initial_portfolio_value to the default value of 10000.0 for a specific run_id.
"""
import sys
import os
import argparse
from sqlalchemy import update

# Add project root to path to allow imports from reinforcestrategycreator
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode

def fix_null_initial_portfolio_values(run_id, initial_value=10000.0):
    """
    Fix NULL initial_portfolio_value in episodes table for a specific run_id.
    
    Args:
        run_id (str): The run_id to fix episodes for
        initial_value (float): The value to set for NULL initial_portfolio_value
    
    Returns:
        int: Number of episodes updated
    """
    print(f"Fixing NULL initial_portfolio_value for run_id: {run_id}")
    
    try:
        with get_db_session() as db:
            # Count episodes with NULL initial_portfolio_value
            null_count = db.query(Episode).filter(
                Episode.run_id == run_id,
                Episode.initial_portfolio_value.is_(None)
            ).count()
            
            print(f"Found {null_count} episodes with NULL initial_portfolio_value")
            
            if null_count == 0:
                print("No episodes to fix. Exiting.")
                return 0
            
            # Update episodes with NULL initial_portfolio_value
            stmt = update(Episode).where(
                Episode.run_id == run_id,
                Episode.initial_portfolio_value.is_(None)
            ).values(initial_portfolio_value=initial_value)
            
            result = db.execute(stmt)
            db.commit()
            
            print(f"Updated {result.rowcount} episodes with initial_portfolio_value = {initial_value}")
            return result.rowcount
            
    except Exception as e:
        print(f"Error fixing NULL initial_portfolio_value: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Fix NULL initial_portfolio_value in episodes table for a specific run_id"
    )
    parser.add_argument(
        "run_id",
        help="The run_id to fix episodes for"
    )
    parser.add_argument(
        "--initial-value",
        type=float,
        default=10000.0,
        help="The value to set for NULL initial_portfolio_value (default: 10000.0)"
    )
    args = parser.parse_args()
    
    fix_null_initial_portfolio_values(args.run_id, args.initial_value)

if __name__ == "__main__":
    main()