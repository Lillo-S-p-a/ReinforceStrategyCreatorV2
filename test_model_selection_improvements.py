#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to demonstrate improvements in model selection and training.
Compares the original approach (Sharpe-only) with our enhanced multi-metric selection
and advanced training techniques (transfer learning and ensemble models).
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import datetime
import logging
from copy import deepcopy

from reinforcestrategycreator.backtesting.workflow import BacktestingWorkflow
from reinforcestrategycreator.backtesting.cross_validation import CrossValidator
from reinforcestrategycreator.backtesting.model import ModelTrainer

# Setup logging
def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"model_selection_test_{timestamp}.log"
logger = setup_logger(__name__, log_file)

class ModelSelectionTester:
    """Class to test and compare different model selection and training approaches."""
    
    def __init__(self, config_path, data_path):
        """Initialize with configuration and data paths."""
        self.config_path = config_path
        self.data_path = data_path
        self.results_dir = Path(f"test_results_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Check if config exists, if not create a default one for testing
        if not os.path.exists(config_path):
            logger.info(f"Config file not found at {config_path}, creating a default configuration")
            self._create_default_config(config_path)
        
        # Create base workflow
        try:
            self.base_workflow = BacktestingWorkflow(
                config_path=config_path,
                data_path=data_path
            )
            
            logger.info(f"Initialized model selection tester with config: {config_path}")
            logger.info(f"Results will be saved to: {self.results_dir}")
        except Exception as e:
            logger.error(f"Error initializing workflow: {str(e)}")
            raise
    
    def _create_default_config(self, config_path: str):
        """Create a default configuration file for testing."""
        import json
        
        default_config = {
            "model": {
                "type": "dqn",
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "batch_size": 32,
                "memory_size": 10000,
                "layers": [64, 32]
            },
            "training": {
                "episodes": 100,
                "steps_per_episode": 1000,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "hyperparameters": {
                "learning_rate": [0.001, 0.0001],
                "batch_size": [32, 64],
                "layers": [[64, 32], [128, 64]]
            },
            "cross_validation": {
                "folds": 5,
                "metric_weights": {
                    "sharpe_ratio": 0.4,
                    "pnl": 0.3,
                    "win_rate": 0.2,
                    "max_drawdown": 0.1
                }
            },
            "data": {
                "features": ["price", "volume", "ma_20", "ma_50", "rsi"],
                "target": "returns"
            },
            "random_seed": 42
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
            
        logger.info(f"Created default configuration at {config_path}")
        
    def run_original_approach(self):
        """Run the original model selection approach (Sharpe ratio only)."""
        logger.info("===== Running Original Approach (Sharpe ratio only) =====")
        
        try:
            # Create a copy of the workflow for original approach
            workflow = deepcopy(self.base_workflow)
            
            # Ensure the cross-validator is initialized
            if not hasattr(workflow, 'cross_validator'):
                workflow.cross_validator = CrossValidator(
                    train_data=workflow.train_data,
                    config=workflow.config,
                    cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                    models_dir=workflow.models_dir,
                    random_seed=workflow.random_seed
                )
            
            # Modify the cross-validator to use only Sharpe ratio (original behavior)
            workflow.cross_validator.use_multi_metric = False
            logger.info("Configured cross-validator to use only Sharpe ratio (original approach)")
            
            # Check if we have training data
            if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                logger.info("Fetching training data...")
                workflow.fetch_data()
            
            # Run cross-validation
            logger.info("Running cross-validation with original Sharpe-only approach")
            try:
                workflow.run_cross_validation()
            except AttributeError:
                # If run_cross_validation doesn't exist, try perform_cross_validation instead
                logger.info("Using alternative cross-validation method")
                workflow.perform_cross_validation()
            
            # Select best model (original method)
            logger.info("Selecting best model using original approach")
            best_model_info = workflow.select_best_model()
            
            # Train final model without advanced techniques
            logger.info("Training final model without advanced techniques")
            workflow.train_final_model(use_transfer_learning=False, use_ensemble=False)
            
            # Run final backtest
            logger.info("Running final backtest")
            final_results = workflow.run_final_backtest()
            logger.info(f"Original approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['total_pnl']:.2f}")
            
            # Store results
            self.original_results = {
                'best_model_info': best_model_info,
                'final_backtest': final_results
            }
            
            # Save results
            with open(self.results_dir / "original_approach_results.json", 'w') as f:
                json.dump({
                    'best_model_info': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in best_model_info.items() if k != 'model'
                    },
                    'final_metrics': final_results
                }, f, indent=4)
                
            logger.info(f"Original approach selected model from fold {best_model_info.get('fold', -1)}")
            metrics = best_model_info['metrics']
            logger.info(f"Metrics - Sharpe: {metrics['sharpe_ratio']:.4f}, "
                       f"PnL: ${metrics['pnl']:.2f}, "
                       f"Win Rate: {metrics['win_rate']*100:.2f}%, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            
            return self.original_results
            
        except Exception as e:
            logger.error(f"Error in original approach: {str(e)}")
            logger.exception("Exception details:")
            self.original_results = {
                'error': str(e)
            }
            return self.original_results
        
        # Save results
        with open(self.results_dir / "original_approach_results.json", 'w') as f:
            json.dump({
                'best_model_info': {
                    k: v if not isinstance(v, np.ndarray) else v.tolist() 
                    for k, v in best_model_info.items() if k != 'model'
                },
                'final_metrics': final_results
            }, f, indent=4)
            
        logger.info(f"Original approach selected model from fold {best_model_info.get('fold', -1)}")
        metrics = best_model_info['metrics']
        logger.info(f"Metrics - Sharpe: {metrics['sharpe_ratio']:.4f}, "
                   f"PnL: ${metrics['pnl']:.2f}, "
                   f"Win Rate: {metrics['win_rate']*100:.2f}%, "
                   f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        return self.original_results
    
    def run_enhanced_approach(self):
        """Run the enhanced model selection approach with advanced training."""
        logger.info("===== Running Enhanced Approach (Multi-metric + Advanced Training) =====")
        
        try:
            # Create a copy of the workflow for enhanced approach
            workflow = deepcopy(self.base_workflow)
            
            # Ensure the cross-validator is initialized
            if not hasattr(workflow, 'cross_validator'):
                workflow.cross_validator = CrossValidator(
                    train_data=workflow.train_data,
                    config=workflow.config,
                    cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                    models_dir=workflow.models_dir,
                    random_seed=workflow.random_seed
                )
            
            # Ensure multi-metric selection is enabled
            workflow.cross_validator.use_multi_metric = True
            logger.info("Configured cross-validator to use multi-metric selection (enhanced approach)")
            
            # Check if we have training data
            if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                logger.info("Fetching training data...")
                workflow.fetch_data()
            
            # Run cross-validation
            logger.info("Running cross-validation with enhanced multi-metric approach")
            try:
                workflow.run_cross_validation()
            except AttributeError:
                # If run_cross_validation doesn't exist, try perform_cross_validation instead
                logger.info("Using alternative cross-validation method")
                workflow.perform_cross_validation()
            
            # Generate comprehensive CV report
            logger.info("Generating comprehensive cross-validation report")
            cv_report = workflow.cross_validator.generate_cv_report()
            cv_report_path = self.results_dir / "enhanced_cv_report.csv"
            cv_report.to_csv(cv_report_path)
            logger.info(f"Saved detailed cross-validation report to {cv_report_path}")
            
            # Select best model using multi-metric approach
            logger.info("Selecting best model using enhanced multi-metric approach")
            best_model_info = workflow.select_best_model()
            
            # Train final model with advanced techniques
            logger.info("Training final model with transfer learning and ensemble techniques")
            workflow.train_final_model(use_transfer_learning=True, use_ensemble=True)
            
            # Run final backtest
            logger.info("Running final backtest with enhanced model")
            final_results = workflow.run_final_backtest()
            logger.info(f"Enhanced approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['total_pnl']:.2f}")
            
            # Store results
            self.enhanced_results = {
                'best_model_info': best_model_info,
                'final_backtest': final_results,
                'cv_report': cv_report
            }
            
            # Save results
            with open(self.results_dir / "enhanced_approach_results.json", 'w') as f:
                json.dump({
                    'best_model_info': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in best_model_info.items() if k != 'model'
                    },
                    'final_metrics': final_results
                }, f, indent=4)
                
            logger.info(f"Enhanced approach selected model from fold {best_model_info.get('fold', -1)}")
            metrics = best_model_info['metrics']
            logger.info(f"Metrics - Sharpe: {metrics['sharpe_ratio']:.4f}, "
                       f"PnL: ${metrics['pnl']:.2f}, "
                       f"Win Rate: {metrics['win_rate']*100:.2f}%, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            
            return self.enhanced_results
            
        except Exception as e:
            logger.error(f"Error in enhanced approach: {str(e)}")
            logger.exception("Exception details:")
            self.enhanced_results = {
                'error': str(e)
            }
            return self.enhanced_results
    
    def run_ablation_study(self):
        """Run ablation study to test the impact of individual enhancements."""
        logger.info("===== Running Ablation Study =====")
        
        configurations = [
            {
                'name': 'multi_metric_only',
                'description': 'Multi-metric selection only',
                'use_multi_metric': True,
                'use_transfer_learning': False,
                'use_ensemble': False
            },
            {
                'name': 'transfer_learning_only',
                'description': 'Transfer learning only',
                'use_multi_metric': False,
                'use_transfer_learning': True,
                'use_ensemble': False
            },
            {
                'name': 'ensemble_only',
                'description': 'Ensemble model only',
                'use_multi_metric': False,
                'use_transfer_learning': False,
                'use_ensemble': True
            }
        ]
        
        ablation_results = {}
        
        for config in configurations:
            logger.info(f"Testing configuration: {config['description']}")
            
            try:
                # Create a copy of the workflow
                workflow = deepcopy(self.base_workflow)
                
                # Ensure the cross-validator is initialized
                if not hasattr(workflow, 'cross_validator'):
                    workflow.cross_validator = CrossValidator(
                        train_data=workflow.train_data,
                        config=workflow.config,
                        cv_folds=workflow.config.get("cross_validation", {}).get("folds", 5),
                        models_dir=workflow.models_dir,
                        random_seed=workflow.random_seed
                    )
                
                # Configure settings
                workflow.cross_validator.use_multi_metric = config['use_multi_metric']
                logger.info(f"Configured cross-validator with multi-metric={config['use_multi_metric']}")
                
                # Check if we have training data
                if not hasattr(workflow, 'train_data') or workflow.train_data is None:
                    logger.info("Fetching training data...")
                    workflow.fetch_data()
                
                # Run cross-validation
                logger.info(f"Running cross-validation with {config['description']} configuration")
                try:
                    workflow.run_cross_validation()
                except AttributeError:
                    # If run_cross_validation doesn't exist, try perform_cross_validation instead
                    logger.info("Using alternative cross-validation method")
                    workflow.perform_cross_validation()
                
                # Select best model
                logger.info("Selecting best model")
                best_model_info = workflow.select_best_model()
                
                # Train final model with specific configuration
                logger.info(f"Training final model with transfer_learning={config['use_transfer_learning']}, ensemble={config['use_ensemble']}")
                workflow.train_final_model(
                    use_transfer_learning=config['use_transfer_learning'],
                    use_ensemble=config['use_ensemble']
                )
                
                # Run final backtest
                logger.info("Running final backtest")
                final_results = workflow.run_final_backtest()
                logger.info(f"{config['description']} final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['total_pnl']:.2f}")
                
                # Store results
                ablation_results[config['name']] = {
                    'best_model_info': best_model_info,
                    'final_backtest': final_results
                }
            except Exception as e:
                logger.error(f"Error in {config['description']} configuration: {str(e)}")
                logger.exception("Exception details:")
                ablation_results[config['name']] = {
                    'error': str(e)
                }
                
                # Create minimal error report
                with open(self.results_dir / f"ablation_{config['name']}_error.json", 'w') as f:
                    json.dump({
                        'configuration': config,
                        'error': str(e)
                    }, f, indent=4)
                
                # Continue with the next configuration
                continue
            
            # Save results
            with open(self.results_dir / f"ablation_{config['name']}_results.json", 'w') as f:
                json.dump({
                    'configuration': config,
                    'best_model_info': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist() 
                        for k, v in best_model_info.items() if k != 'model'
                    },
                    'final_metrics': final_results
                }, f, indent=4)
                
            logger.info(f"Configuration '{config['name']}' selected model from fold {best_model_info.get('fold', -1)}")
            metrics = best_model_info['metrics']
            logger.info(f"Metrics - Sharpe: {metrics['sharpe_ratio']:.4f}, "
                       f"PnL: ${metrics['pnl']:.2f}, "
                       f"Win Rate: {metrics['win_rate']*100:.2f}%, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        self.ablation_results = ablation_results
        return ablation_results
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report of all tested approaches."""
        logger.info("Generating comparison report")
        
        # Ensure we have results to compare
        if not hasattr(self, 'original_results') or not hasattr(self, 'enhanced_results'):
            logger.error("Missing results for comparison. Run both approaches first.")
            return None
        
        # Create DataFrame for metrics comparison
        approaches = ['Original', 'Enhanced']
        metrics_data = []
        
        # Add original approach metrics
        orig_metrics = self.original_results['best_model_info']['metrics']
        metrics_data.append([
            orig_metrics['sharpe_ratio'],
            orig_metrics['pnl'],
            orig_metrics['win_rate'] * 100,
            orig_metrics['max_drawdown'] * 100,
            self.original_results['final_backtest']['sharpe_ratio'],
            self.original_results['final_backtest']['total_pnl'],
            self.original_results['final_backtest']['win_rate'] * 100,
            self.original_results['final_backtest']['max_drawdown'] * 100
        ])
        
        # Add enhanced approach metrics
        enh_metrics = self.enhanced_results['best_model_info']['metrics']
        metrics_data.append([
            enh_metrics['sharpe_ratio'],
            enh_metrics['pnl'],
            enh_metrics['win_rate'] * 100,
            enh_metrics['max_drawdown'] * 100,
            self.enhanced_results['final_backtest']['sharpe_ratio'],
            self.enhanced_results['final_backtest']['total_pnl'],
            self.enhanced_results['final_backtest']['win_rate'] * 100,
            self.enhanced_results['final_backtest']['max_drawdown'] * 100
        ])
        
        # Add ablation study metrics if available
        if hasattr(self, 'ablation_results'):
            for config_name, results in self.ablation_results.items():
                approaches.append(config_name.replace('_', ' ').title())
                metrics = results['best_model_info']['metrics']
                metrics_data.append([
                    metrics['sharpe_ratio'],
                    metrics['pnl'],
                    metrics['win_rate'] * 100,
                    metrics['max_drawdown'] * 100,
                    results['final_backtest']['sharpe_ratio'],
                    results['final_backtest']['total_pnl'],
                    results['final_backtest']['win_rate'] * 100,
                    results['final_backtest']['max_drawdown'] * 100
                ])
        
        # Create DataFrame
        columns = [
            'CV Sharpe', 'CV PnL ($)', 'CV Win Rate (%)', 'CV Max Drawdown (%)',
            'Final Sharpe', 'Final PnL ($)', 'Final Win Rate (%)', 'Final Max Drawdown (%)'
        ]
        df_comparison = pd.DataFrame(metrics_data, index=approaches, columns=columns)
        
        # Calculate improvement percentages
        improvement_row = pd.Series({
            col: ((df_comparison.loc['Enhanced', col] - df_comparison.loc['Original', col]) / 
                  abs(df_comparison.loc['Original', col])) * 100
            for col in columns
        })
        
        # For max drawdown, a negative percentage means improvement (less drawdown)
        for col in ['CV Max Drawdown (%)', 'Final Max Drawdown (%)']:
            improvement_row[col] = -improvement_row[col]
            
        # Add improvement row
        df_comparison.loc['% Improvement'] = improvement_row
        
        # Save to CSV
        report_path = self.results_dir / "approach_comparison.csv"
        df_comparison.to_csv(report_path)
        logger.info(f"Saved comparison report to {report_path}")
        
        # Generate visualizations
        self._generate_visualizations(df_comparison)
        
        return df_comparison
    
    def _generate_visualizations(self, df_comparison):
        """Generate visualizations comparing the different approaches."""
        logger.info("Generating visualization charts")
        
        # Bar chart comparing key metrics
        plt.figure(figsize=(14, 8))
        
        # Select key metrics for visualization
        key_metrics = ['Final Sharpe', 'Final PnL ($)', 'Final Win Rate (%)']
        
        # Select approaches (exclude % Improvement row)
        approaches = df_comparison.index[:-1]
        
        # Create grouped bar chart
        bar_width = 0.25
        x = np.arange(len(approaches))
        
        for i, metric in enumerate(key_metrics):
            plt.bar(x + i*bar_width, df_comparison.loc[approaches, metric], 
                    width=bar_width, label=metric)
        
        plt.xlabel('Approach')
        plt.ylabel('Value')
        plt.title('Comparison of Key Metrics Across Approaches')
        plt.xticks(x + bar_width, approaches)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        chart_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(chart_path)
        logger.info(f"Saved metrics comparison chart to {chart_path}")
        
        # Create improvement chart
        plt.figure(figsize=(10, 6))
        
        # Get improvement percentages (excluding max drawdown which is inverted)
        improvements = df_comparison.loc['% Improvement', 
                                         [c for c in df_comparison.columns if 'Max Drawdown' not in c]]
        
        # Create horizontal bar chart
        bars = plt.barh(improvements.index, improvements.values)
        
        # Color bars based on positive/negative values
        for i, bar in enumerate(bars):
            if improvements.values[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.xlabel('Improvement (%)')
        plt.title('Enhanced Approach % Improvement Over Original')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values to the end of each bar
        for i, v in enumerate(improvements.values):
            plt.text(v + 1, i, f"{v:.1f}%", va='center')
        
        # Save figure
        plt.tight_layout()
        chart_path = self.results_dir / "improvement_chart.png"
        plt.savefig(chart_path)
        logger.info(f"Saved improvement chart to {chart_path}")
        
        # If we have CV report, visualize fold performance
        if 'cv_report' in self.enhanced_results:
            self._visualize_cv_performance(self.enhanced_results['cv_report'])
    
    def _visualize_cv_performance(self, cv_report):
        """Create visualizations of cross-validation fold performance."""
        # Filter for key metrics
        metrics_to_plot = ['sharpe_ratio', 'pnl', 'win_rate', 'max_drawdown']
        
        plt.figure(figsize=(16, 12))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            
            # Extract data for this metric across folds
            metric_data = cv_report.pivot(index='fold', columns='model_config', values=metric)
            
            # Plot as heatmap
            im = plt.imshow(metric_data.values, cmap='viridis', aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, label=metric)
            
            # Add labels
            plt.title(f'Cross-Validation Performance: {metric}')
            plt.xlabel('Model Configuration')
            plt.ylabel('Fold')
            
            # Set ticks
            plt.xticks(range(len(metric_data.columns)), 
                      [f"Config {i}" for i in range(len(metric_data.columns))], 
                      rotation=45)
            plt.yticks(range(len(metric_data.index)), metric_data.index)
            
        plt.tight_layout()
        chart_path = self.results_dir / "cv_performance_heatmap.png"
        plt.savefig(chart_path)
        logger.info(f"Saved cross-validation performance heatmap to {chart_path}")
        
        # Create parallel coordinates plot for multi-dimensional analysis
        plt.figure(figsize=(12, 6))
        
        # Prepare data - normalize each metric to [0,1] range for comparison
        normalized_data = pd.DataFrame()
        
        for metric in metrics_to_plot:
            if metric == 'max_drawdown':
                # Invert max_drawdown so higher is better (consistent with other metrics)
                normalized_data[metric] = 1 - (cv_report[metric] - cv_report[metric].min()) / \
                                         (cv_report[metric].max() - cv_report[metric].min())
            else:
                normalized_data[metric] = (cv_report[metric] - cv_report[metric].min()) / \
                                         (cv_report[metric].max() - cv_report[metric].min())
        
        # Add fold and config columns
        normalized_data['fold'] = cv_report['fold']
        normalized_data['config'] = cv_report['model_config']
        
        # Plot parallel coordinates
        pd.plotting.parallel_coordinates(normalized_data, 'config', colormap='viridis')
        
        plt.title('Multi-Metric Performance by Model Configuration')
        plt.grid(True)
        plt.tight_layout()
        
        chart_path = self.results_dir / "multi_metric_parallel_plot.png"
        plt.savefig(chart_path)
        logger.info(f"Saved multi-metric parallel coordinates plot to {chart_path}")
        
    def run_complete_test(self):
        """Run the complete test suite and generate comprehensive report."""
        logger.info("Starting complete model selection improvement test")
        success = True
        
        try:
            # Run original approach
            logger.info("Step 1: Running original approach")
            original_results = self.run_original_approach()
            if 'error' in original_results:
                logger.warning("Original approach encountered errors but will continue with test")
            
            # Run enhanced approach
            logger.info("Step 2: Running enhanced approach")
            enhanced_results = self.run_enhanced_approach()
            if 'error' in enhanced_results:
                logger.warning("Enhanced approach encountered errors but will continue with test")
            
            # Run ablation study
            logger.info("Step 3: Running ablation study")
            ablation_results = self.run_ablation_study()
            
            # Generate comparison report
            logger.info("Step 4: Generating comparison report")
            comparison_report = self.generate_comparison_report()
            
            if comparison_report is not None:
                logger.info("Test complete! Results summary:")
                logger.info("\n" + str(comparison_report))
            else:
                logger.warning("Could not generate comparison report due to missing results")
                success = False
                
            logger.info(f"All test results saved to: {self.results_dir}")
            return comparison_report
            
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}")
            logger.exception("Exception details:")
            success = False
            return None
        finally:
            if success:
                logger.info("All test steps completed. Some steps might have encountered non-fatal errors.")
            else:
                logger.warning("Test completed with some failures. Check logs for details.")

def main():
    """Main function to run the test."""
    try:
        # Default config and data paths
        config_path = "config/backtesting_config.json"
        data_path = "data/processed_market_data.csv"
        
        # Create tester
        logger.info(f"Initializing tester with config: {config_path} and data: {data_path}")
        tester = ModelSelectionTester(config_path, data_path)
        
        # Run the complete test
        logger.info("Starting complete test...")
        report = tester.run_complete_test()
        
        if report is not None:
            print("\nTest completed successfully.")
            print(f"Results saved to: {tester.results_dir}")
            print("\nComparison Report:")
            print(report)
        else:
            print("\nTest completed with errors. Check the logs for details.")
            print(f"Partial results may be available in: {tester.results_dir}")
        
        return report
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        logger.exception("Exception details:")
        print(f"Critical error: {str(e)}")
        print("Check the logs for details.")
        return None

if __name__ == "__main__":
    main()