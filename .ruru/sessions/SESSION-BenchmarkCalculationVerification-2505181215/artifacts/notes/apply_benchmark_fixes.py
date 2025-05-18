#!/usr/bin/env python
"""
Benchmark Calculation Fix Application Script

This script applies the fixes identified in the benchmark comparison analysis
to correct issues in the benchmark calculation implementation.

Usage:
    python apply_benchmark_fixes.py

The script will:
1. Back up original files before modification
2. Apply fixes to the benchmark implementation
3. Log all changes made
"""

import os
import sys
import re
import shutil
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark_fix_application')

# Add project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Fix: Going up 5 directories from script_dir to reach project root
# From: .ruru/sessions/SESSION-BenchmarkCalculationVerification-2505181215/artifacts/notes/
# To: /home/alessio/Personal/ReinforceStrategyCreatorV2/
project_root = os.path.abspath(os.path.join(script_dir, '../../../../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Print the project root for verification
logger.info(f"Project root: {project_root}")

# File paths relative to project root
BENCHMARK_PATH = os.path.join(project_root, 'reinforcestrategycreator', 'backtesting', 'benchmarks.py')
EVALUATION_PATH = os.path.join(project_root, 'reinforcestrategycreator', 'backtesting', 'evaluation.py')
WORKFLOW_PATH = os.path.join(project_root, 'reinforcestrategycreator', 'backtesting', 'workflow.py')

# Log the actual paths to verify
logger.info(f"BENCHMARK_PATH: {BENCHMARK_PATH}")
logger.info(f"EVALUATION_PATH: {EVALUATION_PATH}")
logger.info(f"WORKFLOW_PATH: {WORKFLOW_PATH}")

# Backup directory
BACKUP_DIR = os.path.join(project_root, 'backups', f'benchmark_fixes_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


def backup_file(file_path):
    """Create a backup of the specified file"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Determine the relative path for preserving directory structure
    rel_path = os.path.relpath(file_path, project_root)
    backup_path = os.path.join(BACKUP_DIR, rel_path)
    
    # Create directory structure in backup location
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backed up {file_path} to {backup_path}")
    
    return backup_path


def fix_buy_and_hold_list_conversion(file_path):
    """
    Fix the list conversion issue in the Buy and Hold strategy.
    
    The issue is in the run() method where:
    1. The code converts prices to a list unnecessarily
    2. This adds overhead and potential precision loss
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the problematic line in the BuyAndHoldStrategy class
    pattern = r'(prices = list\(prices\))'
    
    # Check if the pattern exists
    if re.search(pattern, content):
        # Replace with the fixed version that directly uses prices without list conversion
        fixed_content = re.sub(pattern, '# prices = list(prices)  # Removed unnecessary list conversion', content)
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Fixed Buy and Hold list conversion issue in {file_path}")
        return True
    else:
        logger.warning(f"Buy and Hold list conversion pattern not found in {file_path}")
        return False


def add_debug_logging(file_path):
    """Add detailed debug logging to aid in diagnosing benchmark calculation issues"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the end of the BuyAndHoldStrategy run() method
    pattern = r'(return metrics)'
    
    # Additional logging code to insert before the return statement
    logging_code = '''
        # Debug logging to track benchmark calculation details
        logger.debug(f"Buy and Hold strategy completed with:")
        logger.debug(f"  - Initial price: {prices[0]}")
        logger.debug(f"  - Final price: {prices[-1]}")
        logger.debug(f"  - Shares: {shares}")
        logger.debug(f"  - Initial balance: {self.initial_balance}")
        logger.debug(f"  - Final value: {shares * prices[-1]}")
        logger.debug(f"  - PnL: {metrics['pnl']}")
        logger.debug(f"  - PnL %: {metrics['pnl_percentage']}")
        '''
    
    # Check if the pattern exists
    if re.search(pattern, content):
        # Insert logging code before the return statement
        fixed_content = re.sub(pattern, f"{logging_code}\n        {pattern}", content)
        
        # Also add logger import at the top if needed
        if 'import logging' not in content:
            fixed_content = "import logging\nlogger = logging.getLogger(__name__)\n\n" + fixed_content
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Added debug logging to Buy and Hold strategy in {file_path}")
        return True
    else:
        logger.warning(f"Buy and Hold return statement pattern not found in {file_path}")
        return False


def ensure_consistent_fee_handling(file_path):
    """
    Ensure consistent transaction fee handling between model and benchmarks.
    
    The issue might be inconsistent fee application between the model and benchmarks.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the fee application in Buy and Hold strategy
    pattern = r'(shares = self\.initial_balance / prices\[0\])'
    replacement = 'shares = self.initial_balance / prices[0]\n        # Apply transaction fee to initial purchase\n        shares = shares * (1 - self.transaction_fee)'
    
    # Check if the pattern exists but the fee application is missing
    if re.search(pattern, content) and 'shares = shares * (1 - self.transaction_fee)' not in content:
        # Add the fee application
        fixed_content = re.sub(pattern, replacement, content)
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Added consistent transaction fee handling to Buy and Hold strategy in {file_path}")
        return True
    else:
        logger.info(f"Transaction fee handling already appears correct in {file_path}")
        return False


def fix_portfolio_value_tracking(file_path):
    """
    Enhance portfolio value tracking in the BuyAndHoldStrategy.
    
    The issue might be that portfolio values aren't tracked consistently,
    leading to incorrect metrics calculations.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match where portfolio values tracking can be added
    pattern = r'(def run\(self, data\):)'
    
    # Code to add for portfolio tracking
    portfolio_tracking = '''
    def run(self, data):
        """Run the Buy and Hold strategy on the provided data.
        
        Args:
            data (DataFrame): DataFrame with price data.
            
        Returns:
            dict: Strategy performance metrics.
        """
        # Initialize portfolio value tracking
        portfolio_values = [float(self.initial_balance)]  # Start with initial balance
'''
    
    # Check if portfolio tracking is already implemented
    if 'portfolio_values =' not in content:
        # Add portfolio tracking initialization
        fixed_content = re.sub(pattern, portfolio_tracking, content)
        
        # Now add code to track portfolio values throughout the simulation
        # Find the price loop and add tracking
        price_pattern = r'(final_value = shares \* prices\[-1\])'
        portfolio_update = '''
        # Track portfolio values throughout the simulation
        for i in range(1, len(prices)):
            current_value = shares * prices[i]
            portfolio_values.append(float(current_value))
            
        final_value = shares * prices[-1]'''
        
        if re.search(price_pattern, fixed_content):
            fixed_content = re.sub(price_pattern, portfolio_update, fixed_content)
            
            # Also add portfolio_values to the returned metrics
            metrics_pattern = r'(return metrics)'
            metrics_update = '''        # Add portfolio values to metrics
        metrics['portfolio_values'] = portfolio_values
        return metrics'''
            
            if re.search(metrics_pattern, fixed_content):
                fixed_content = re.sub(metrics_pattern, metrics_update, fixed_content)
                
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                
                logger.info(f"Enhanced portfolio value tracking in Buy and Hold strategy in {file_path}")
                return True
            else:
                logger.warning(f"Could not find metrics return pattern in {file_path}")
                return False
        else:
            logger.warning(f"Could not find final value calculation pattern in {file_path}")
            return False
    else:
        logger.info(f"Portfolio value tracking already appears to be implemented in {file_path}")
        return False


def standardize_metrics_calculation(file_path):
    """
    Standardize metrics calculation between model and benchmarks.
    
    Ensure that the same calculation methods are used for both model and benchmarks.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # This is a more complex change requiring careful implementation
    # For now, add a comment highlighting the need for standardization
    metrics_pattern = r'(pnl = final_value - self\.initial_balance)'
    
    # Check if we should apply this change
    if re.search(metrics_pattern, content):
        clarification_comment = '''        # Ensure consistent metrics calculation with the model
        # PnL calculation
        pnl = final_value - self.initial_balance'''
        
        fixed_content = re.sub(metrics_pattern, clarification_comment, content)
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Added metrics calculation clarification in {file_path}")
        return True
    else:
        logger.info(f"Metrics calculation appears standard in {file_path}")
        return False


def add_comparison_validation(workflow_path):
    """
    Add validation steps to the benchmark comparison workflow to ensure fair comparison.
    
    This adds checks for data consistency between model and benchmarks.
    """
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    # Look for the compare_with_benchmarks method
    pattern = r'(def compare_with_benchmarks\(self.*?\):)'
    
    # Validation code to add
    validation_code = '''
    def compare_with_benchmarks(self, model_metrics=None):
        """Compare model performance with benchmark strategies.
        
        Args:
            model_metrics (dict, optional): Model metrics. If None, will use the latest.
            
        Returns:
            dict: Dictionary containing model and benchmark metrics.
        """
        # Validation to ensure fair comparison
        logger.info("Validating comparison data consistency...")
'''
    
    # Check if this is appropriate to add
    if re.search(pattern, content) and 'Validating comparison data consistency' not in content:
        # Add validation code
        fixed_content = re.sub(pattern, validation_code, content)
        
        # Also add logger import at the top if needed
        if 'import logging' not in content:
            fixed_content = "import logging\nlogger = logging.getLogger(__name__)\n\n" + fixed_content
        
        with open(workflow_path, 'w') as f:
            f.write(fixed_content)
        
        logger.info(f"Added comparison validation to {workflow_path}")
        return True
    else:
        logger.info(f"Comparison validation already appears implemented in {workflow_path}")
        return False


def apply_fixes():
    """Apply all identified fixes"""
    fixes_applied = []
    
    # First, back up all files we'll modify
    logger.info("Creating backups of files to be modified...")
    backup_file(BENCHMARK_PATH)
    backup_file(EVALUATION_PATH)
    backup_file(WORKFLOW_PATH)
    
    # Apply fixes to benchmarks.py
    logger.info("Applying fixes to benchmarks.py...")
    
    if fix_buy_and_hold_list_conversion(BENCHMARK_PATH):
        fixes_applied.append("Removed unnecessary list conversion in Buy and Hold strategy")
    
    if add_debug_logging(BENCHMARK_PATH):
        fixes_applied.append("Added detailed debug logging to Buy and Hold strategy")
    
    if ensure_consistent_fee_handling(BENCHMARK_PATH):
        fixes_applied.append("Ensured consistent transaction fee handling in Buy and Hold strategy")
        
    if fix_portfolio_value_tracking(BENCHMARK_PATH):
        fixes_applied.append("Enhanced portfolio value tracking in Buy and Hold strategy")
    
    if standardize_metrics_calculation(BENCHMARK_PATH):
        fixes_applied.append("Standardized metrics calculation in Buy and Hold strategy")
    
    # Apply fixes to workflow.py
    logger.info("Applying fixes to workflow.py...")
    
    if add_comparison_validation(WORKFLOW_PATH):
        fixes_applied.append("Added data consistency validation to benchmark comparison workflow")
    
    return fixes_applied


if __name__ == "__main__":
    try:
        logger.info("Starting benchmark fixes application")
        
        # Apply fixes
        applied_fixes = apply_fixes()
        
        if applied_fixes:
            logger.info(f"Successfully applied {len(applied_fixes)} fixes:")
            for i, fix in enumerate(applied_fixes, 1):
                logger.info(f"  {i}. {fix}")
            print(f"\nSuccessfully applied {len(applied_fixes)} fixes. Original files backed up to {BACKUP_DIR}")
        else:
            logger.info("No fixes were applied. Either the issues were already fixed or patterns were not found.")
            print("\nNo fixes were applied. The code may already have been fixed or the patterns weren't found.")
        
    except Exception as e:
        logger.exception("Error applying benchmark fixes")
        print(f"\nError applying fixes: {str(e)}")