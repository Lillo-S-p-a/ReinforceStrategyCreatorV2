#!/usr/bin/env python3
"""
Script to demonstrate the refactored backtesting module with model improvements.
This script shows how to use the BacktestingWorkflow class and explores
potential improvements to the reinforcement learning trading model.
"""

import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Import the refactored BacktestingWorkflow class
from reinforcestrategycreator.backtesting import BacktestingWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the improved backtesting workflow."""
    logger.info("Starting improved backtesting workflow")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"improved_backtest_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define improved configuration with optimized hyperparameters
    improved_config = {
        # Basic configuration
        "initial_balance": 10000,
        "transaction_fee": 0.001,
        
        # Improved RL hyperparameters
        "learning_rate": 0.0005,       # Lower learning rate for more stable learning
        "gamma": 0.98,                 # Slightly adjusted discount factor
        "epsilon": 1.0,                # Starting exploration rate
        "epsilon_decay": 0.998,        # Slower decay for better exploration
        "epsilon_min": 0.05,           # Higher minimum exploration
        "batch_size": 64,              # Larger batch size for better gradient estimates
        "memory_size": 10000,          # Larger replay buffer
        "update_target_frequency": 10, # More frequent target network updates
        
        # Training parameters
        "episodes": 150,               # More episodes for cross-validation
        "final_episodes": 300,         # More episodes for final model
        
        # Enhanced reward function parameters
        "use_risk_adjusted_reward": True,
        "sharpe_weight": 0.5,
        "drawdown_penalty": 0.3,
        
        # Feature engineering
        "use_technical_indicators": True,
        "use_market_indicators": True,
        "normalize_features": True,
        
        # SMA benchmark parameters
        "sma_short_window": 15,
        "sma_long_window": 40,
    }
    
    # Create backtesting workflow with improved configuration
    workflow = BacktestingWorkflow(
        config=improved_config,
        results_dir=results_dir,
        asset="SPY",
        start_date="2018-01-01",       # Extended training period
        end_date="2023-01-01",
        cv_folds=5,
        test_ratio=0.2,
        random_seed=42
    )
    
    # Run the complete workflow
    logger.info("Fetching and preparing data")
    workflow.fetch_data()
    
    logger.info("Performing cross-validation to find optimal hyperparameters")
    cv_results = workflow.perform_cross_validation()
    
    logger.info("Selecting best model configuration")
    best_model_info = workflow.select_best_model()
    logger.info(f"Best model has Sharpe ratio: {best_model_info['metrics']['sharpe_ratio']:.4f}")
    
    logger.info("Training final model with best parameters")
    workflow.train_final_model()
    
    logger.info("Evaluating model on test data")
    test_metrics = workflow.evaluate_final_model()
    
    logger.info("Generating comprehensive report")
    report_path = workflow.generate_report(format="html")
    
    logger.info("Exporting model for production use")
    model_path = workflow.export_for_trading()
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("BACKTESTING RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Final PnL: ${test_metrics['pnl']:.2f}")
    logger.info(f"PnL Percentage: {test_metrics['pnl_percentage']:.2f}%")
    logger.info(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    logger.info(f"Max Drawdown: {test_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {test_metrics['win_rate']*100:.2f}%")
    logger.info(f"Total Trades: {test_metrics['trades']}")
    logger.info("="*50)
    logger.info(f"Full report saved to: {report_path}")
    logger.info(f"Model exported to: {model_path}")
    logger.info("="*50)
    
    # Open the report in the default browser
    logger.info("Opening report in browser")
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except Exception as e:
        logger.error(f"Failed to open report: {e}")
    
    logger.info("Workflow complete!")
    
    return report_path, model_path

if __name__ == "__main__":
    main()