#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality.

This script tests the hyperparameter optimization functionality
in the backtesting workflow.
"""

import os
import logging
import ray
import time
import json
from datetime import datetime

from reinforcestrategycreator.backtesting import BacktestingWorkflow
from reinforcestrategycreator.backtesting.hyperparameter_optimization import HyperparameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the hyperparameter optimization test."""
    start_time = time.time()
    logger.info("Starting hyperparameter optimization test")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"hpo_test_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define test configuration with hyperparameter search space
    test_config = {
        # Basic configuration
        "initial_balance": 100000,
        "commission_pct": 0.03,
        "slippage_bps": 3,
        
        # Hyperparameter search space
        "hyperparameters": {
            "learning_rate": [0.001, 0.0005, 0.0001],
            "batch_size": [32, 64, 128],
            "layers": [
                [64, 32],
                [128, 64],
                [256, 128],
                [128, 64, 32]
            ],
            "gamma": [0.95, 0.97, 0.99],
            "epsilon_decay": [0.99, 0.995, 0.998],
            "epsilon_min": [0.01, 0.05, 0.1]
        },
        
        # Training parameters
        "episodes": 50,  # Reduced for faster testing
        "final_episodes": 100,
        
        # Cross-validation metric weights
        "cross_validation": {
            "metric_weights": {
                "sharpe_ratio": 0.4,
                "pnl": 0.3,
                "win_rate": 0.2,
                "max_drawdown": 0.1
            }
        }
    }
    
    # Initialize Ray for distributed processing
    if not ray.is_initialized():
        num_cpus = os.cpu_count()
        cpus_to_use = max(num_cpus - 1, 1) if num_cpus else 2
        
        logger.info(f"Initializing Ray with {cpus_to_use} CPUs")
        ray.init(
            num_cpus=cpus_to_use,
            ignore_reinit_error=True,
            log_to_driver=True,
            include_dashboard=False
        )
    
    # Create backtesting workflow with HPO enabled
    workflow = BacktestingWorkflow(
        config=test_config,
        results_dir=results_dir,
        asset="SPY",
        start_date="2020-01-01",
        end_date="2022-12-31",
        cv_folds=5,  # Reduced for faster testing
        test_ratio=0.2,
        random_seed=42,
        use_hpo=True,
        hpo_num_samples=10,  # Number of hyperparameter configurations to try
        hpo_max_concurrent_trials=4  # Maximum number of concurrent trials
    )
    
    # Fetch data
    logger.info("Fetching and preparing data")
    workflow.fetch_data()
    
    # Run hyperparameter optimization
    logger.info("Running hyperparameter optimization")
    best_hpo_params = workflow.perform_hyperparameter_optimization()
    
    # Log best hyperparameters
    logger.info(f"Best hyperparameters: {json.dumps(best_hpo_params, indent=2)}")
    
    # Run cross-validation with best hyperparameters
    logger.info("Running cross-validation with best hyperparameters")
    cv_results = workflow.perform_cross_validation()
    
    # Select best model
    logger.info("Selecting best model")
    best_model_info = workflow.select_best_model()
    
    # Log best model metrics
    metrics = best_model_info['metrics']
    source = best_model_info.get('source', 'cv')
    logger.info(f"Selected best model from {'HPO' if source == 'hpo' else 'cross-validation'} with:")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"PnL: ${metrics['pnl']:.2f}")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    # Train final model
    logger.info("Training final model with best hyperparameters")
    workflow.train_final_model()
    
    # Evaluate final model
    logger.info("Evaluating final model")
    test_metrics = workflow.evaluate_final_model()
    
    # Generate report
    logger.info("Generating report")
    report_path = workflow.generate_report(format="html")
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("HYPERPARAMETER OPTIMIZATION TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Final PnL: ${test_metrics['pnl']:.2f}")
    logger.info(f"PnL Percentage: {test_metrics['pnl_percentage']:.2f}%")
    logger.info(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.4f}")
    logger.info(f"Max Drawdown: {test_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {test_metrics['win_rate']*100:.2f}%")
    logger.info("="*50)
    logger.info(f"Full report saved to: {report_path}")
    logger.info("="*50)
    
    # Shutdown Ray
    if ray.is_initialized():
        logger.info("Shutting down Ray")
        ray.shutdown()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Test complete! Total execution time: {elapsed_time:.2f} seconds")
    
    return report_path

if __name__ == "__main__":
    main()