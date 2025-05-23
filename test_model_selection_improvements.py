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
            # Load configuration from file
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Initialize the workflow with proper parameters
            self.base_workflow = BacktestingWorkflow(
                config=config,
                results_dir=str(self.results_dir),
                asset="SPY",
                start_date="2020-01-01",
                end_date="2023-01-01",
                cv_folds=5,
                test_ratio=0.2
            )
            
            logger.info(f"Initialized model selection tester with config: {config_path}")
            logger.info(f"Results will be saved to: {self.results_dir}")
        except Exception as e:
            logger.error(f"Error initializing workflow: {str(e)}")
            # Create a minimal error report
            error_report_path = self.results_dir / "initialization_error.json"
            with open(error_report_path, 'w') as f:
                json.dump({
                    'error': str(e),
                    'config_path': config_path,
                    'data_path': data_path,
                    'timestamp': timestamp
                }, f, indent=4)
            logger.info(f"Error report saved to {error_report_path}")
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
                try:
                    workflow.fetch_data()
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}")
                    # Create sample data directly in the workflow
                    logger.info("Attempting to create sample data directly")
                    from reinforcestrategycreator.backtesting.data import DataManager
                    workflow.data_manager = DataManager(
                        asset="SPY",
                        start_date="2020-01-01",
                        end_date="2023-01-01",
                        test_ratio=0.2
                    )
                    # Create minimal sample data
                    sample_data = self._create_minimal_sample_data()
                    workflow.data = sample_data
                    workflow.train_data = sample_data
                    workflow.test_data = sample_data.iloc[-50:].copy()
            
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
            final_results = workflow.evaluate_final_model()
            logger.info(f"Original approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")
            
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
            try:
                cv_report = workflow.cross_validator.generate_cv_report()
                # Check if cv_report is a DataFrame or string
                if isinstance(cv_report, pd.DataFrame):
                    cv_report_path = self.results_dir / "enhanced_cv_report.csv"
                    cv_report.to_csv(cv_report_path)
                    logger.info(f"Saved detailed cross-validation report to {cv_report_path}")
                else:
                    # If it's a string, save it as text
                    cv_report_path = self.results_dir / "enhanced_cv_report.txt"
                    with open(cv_report_path, 'w') as f:
                        f.write(str(cv_report))
                    logger.info(f"Saved CV report as text to {cv_report_path}")
            except Exception as e:
                logger.error(f"Error generating CV report: {str(e)}")
                cv_report = str(e)
            
            # Select best model using multi-metric approach
            logger.info("Selecting best model using enhanced multi-metric approach")
            best_model_info = workflow.select_best_model()
            
            # Train final model with advanced techniques
            logger.info("Training final model with transfer learning and ensemble techniques")
            workflow.train_final_model(use_transfer_learning=True, use_ensemble=True)
            
            # Run final backtest
            logger.info("Running final backtest with enhanced model")
            final_results = workflow.evaluate_final_model()
            logger.info(f"Enhanced approach final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")
            
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
                final_results = workflow.evaluate_final_model()
                logger.info(f"{config['description']} final backtest results: Sharpe={final_results['sharpe_ratio']:.4f}, PnL=${final_results['pnl']:.2f}")
                
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
        
        # Helper function to safely extract metrics
        def extract_metrics(result_dict, is_final=False):
            try:
                if 'error' in result_dict:
                    logger.warning(f"Error found in results: {result_dict['error']}")
                    return [0.0, 0.0, 0.0, 0.0]  # Default values for error case
                
                if is_final:
                    metrics = result_dict['final_backtest']
                else:
                    metrics = result_dict['best_model_info']['metrics']
                
                return [
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('pnl', 0.0),
                    metrics.get('win_rate', 0.0) * 100,
                    metrics.get('max_drawdown', 0.0) * 100
                ]
            except KeyError as e:
                logger.error(f"KeyError extracting metrics: {str(e)}")
                return [0.0, 0.0, 0.0, 0.0]  # Default values for error case
        
        # Add original approach metrics
        try:
            cv_metrics = extract_metrics(self.original_results, is_final=False)
            final_metrics = extract_metrics(self.original_results, is_final=True)
            metrics_data.append(cv_metrics + final_metrics)
        except Exception as e:
            logger.error(f"Error processing original approach metrics: {str(e)}")
            metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add enhanced approach metrics
        try:
            cv_metrics = extract_metrics(self.enhanced_results, is_final=False)
            final_metrics = extract_metrics(self.enhanced_results, is_final=True)
            metrics_data.append(cv_metrics + final_metrics)
        except Exception as e:
            logger.error(f"Error processing enhanced approach metrics: {str(e)}")
            metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add ablation study metrics if available
        if hasattr(self, 'ablation_results'):
            for config_name, results in self.ablation_results.items():
                approaches.append(config_name.replace('_', ' ').title())
                try:
                    if 'error' in results:
                        logger.warning(f"Error in ablation study {config_name}: {results['error']}")
                        metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        continue
                        
                    cv_metrics = extract_metrics(results, is_final=False)
                    final_metrics = extract_metrics(results, is_final=True)
                    metrics_data.append(cv_metrics + final_metrics)
                except Exception as e:
                    logger.error(f"Error processing ablation metrics for {config_name}: {str(e)}")
                    metrics_data.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Create DataFrame
        columns = [
            'CV Sharpe', 'CV PnL ($)', 'CV Win Rate (%)', 'CV Max Drawdown (%)',
            'Final Sharpe', 'Final PnL ($)', 'Final Win Rate (%)', 'Final Max Drawdown (%)'
        ]
        df_comparison = pd.DataFrame(metrics_data, index=approaches, columns=columns)
        
        # Calculate improvement percentages
        improvement_row = pd.Series()
        try:
            for col in columns:
                original_val = df_comparison.loc['Original', col]
                enhanced_val = df_comparison.loc['Enhanced', col]
                
                # Avoid division by zero
                if original_val != 0:
                    improvement = ((enhanced_val - original_val) / abs(original_val)) * 100
                else:
                    # If original is zero, use absolute improvement or 100% if enhanced is positive
                    improvement = 100.0 if enhanced_val > 0 else 0.0
                    
                improvement_row[col] = improvement
        except Exception as e:
            logger.error(f"Error calculating improvement percentages: {str(e)}")
            # Create default improvement row
            improvement_row = pd.Series({col: 0.0 for col in columns})
        
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
            try:
                self._visualize_cv_performance(self.enhanced_results['cv_report'])
            except Exception as e:
                logger.error(f"Error visualizing CV performance: {str(e)}")
                # Create a simple error visualization
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"CV Visualization Error:\n{str(e)}",
                        ha='center', va='center', fontsize=12, color='red')
                plt.axis('off')
                
                chart_path = self.results_dir / "cv_visualization_error.png"
                plt.savefig(chart_path)
                logger.info(f"Saved error information to {chart_path}")
    
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
        
    def _create_minimal_sample_data(self):
        """Create a minimal sample DataFrame for testing when data fetching fails."""
        logger.info("Creating minimal sample data for testing")
        
        # Create date range for sample data (smaller dataset for faster testing)
        dates = pd.date_range(start='2020-01-01', end='2020-03-01', freq='D')
        
        # Generate base price data
        base_price = 100
        price_volatility = 2
        
        # Create sample data with required columns
        opens = np.random.normal(base_price, price_volatility, len(dates))
        closes = np.random.normal(base_price, price_volatility, len(dates))
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(1, 0.5, len(dates)))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(1, 0.5, len(dates)))
        
        data = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'price': closes,
            'ma_20': np.random.normal(base_price, price_volatility/2, len(dates)),
            'ma_50': np.random.normal(base_price, price_volatility/2, len(dates)),
            'rsi': np.random.uniform(30, 70, len(dates)),
            'returns': np.random.normal(0, 0.01, len(dates))
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data, index=dates)
        logger.info(f"Created minimal sample data with {len(df)} rows and columns: {', '.join(df.columns)}")
        return df

    def run_complete_test(self):
        """Run the complete test suite and generate comprehensive report."""
        logger.info("Starting complete model selection improvement test")
        success = True
        results_summary = {}
        
        try:
            # Run original approach
            logger.info("Step 1: Running original approach")
            try:
                original_results = self.run_original_approach()
                if 'error' in original_results:
                    logger.warning(f"Original approach encountered errors: {original_results['error']}")
                    results_summary['original'] = {'status': 'error', 'error': original_results['error']}
                else:
                    results_summary['original'] = {'status': 'success'}
            except Exception as e:
                logger.error(f"Exception in original approach: {str(e)}")
                logger.exception("Exception details:")
                results_summary['original'] = {'status': 'exception', 'error': str(e)}
            
            # Run enhanced approach
            logger.info("Step 2: Running enhanced approach")
            try:
                enhanced_results = self.run_enhanced_approach()
                if 'error' in enhanced_results:
                    logger.warning(f"Enhanced approach encountered errors: {enhanced_results['error']}")
                    results_summary['enhanced'] = {'status': 'error', 'error': enhanced_results['error']}
                else:
                    results_summary['enhanced'] = {'status': 'success'}
            except Exception as e:
                logger.error(f"Exception in enhanced approach: {str(e)}")
                logger.exception("Exception details:")
                results_summary['enhanced'] = {'status': 'exception', 'error': str(e)}
            
            # Run ablation study
            logger.info("Step 3: Running ablation study")
            try:
                ablation_results = self.run_ablation_study()
                results_summary['ablation'] = {'status': 'completed'}
            except Exception as e:
                logger.error(f"Exception in ablation study: {str(e)}")
                logger.exception("Exception details:")
                results_summary['ablation'] = {'status': 'exception', 'error': str(e)}
            
            # Generate comparison report
            logger.info("Step 4: Generating comparison report")
            comparison_report = None
            try:
                comparison_report = self.generate_comparison_report()
                
                if comparison_report is not None:
                    logger.info("Test complete! Results summary:")
                    logger.info("\n" + str(comparison_report))
                    results_summary['comparison'] = {'status': 'success'}
                else:
                    logger.warning("Could not generate comparison report due to missing results")
                    success = False
                    results_summary['comparison'] = {'status': 'failed', 'reason': 'missing results'}
            except Exception as e:
                logger.error(f"Exception generating comparison report: {str(e)}")
                logger.exception("Exception details:")
                success = False
                results_summary['comparison'] = {'status': 'exception', 'error': str(e)}
            
            # Save results summary
            with open(self.results_dir / "test_summary.json", 'w') as f:
                json.dump(results_summary, f, indent=4)
                
            logger.info(f"All test results saved to: {self.results_dir}")
            return comparison_report
            
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}")
            logger.exception("Exception details:")
            success = False
            
            # Save error report
            with open(self.results_dir / "test_execution_error.json", 'w') as f:
                json.dump({
                    'error': str(e),
                    'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                }, f, indent=4)
                
            return None
        finally:
            if success:
                logger.info("All test steps completed. Some steps might have encountered non-fatal errors.")
            else:
                logger.warning("Test completed with some failures. Check logs for details.")
def create_sample_data(data_path):
    """Create a sample market data file for testing if it doesn't exist."""
    try:
        if os.path.exists(data_path):
            logger.info(f"Using existing data file at {data_path}")
            return True
            
        logger.info(f"Sample data file not found at {data_path}, creating test data")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create date range for sample data
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        
        # Generate base price data
        base_price = 100
        price_volatility = 2
        
        # Create sample data with required columns
        # Ensure high > open/close and low < open/close for each day
        opens = np.random.normal(base_price, price_volatility, len(dates))
        closes = np.random.normal(base_price, price_volatility, len(dates))
        
        # Generate highs and lows that are consistent with open/close
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(1, 0.5, len(dates)))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(1, 0.5, len(dates)))
        
        data = {
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'price': closes,  # Add price column (same as close for simplicity)
            'ma_20': np.random.normal(base_price, price_volatility/2, len(dates)),
            'ma_50': np.random.normal(base_price, price_volatility/2, len(dates)),
            'rsi': np.random.uniform(30, 70, len(dates))
        }
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Ensure all required columns are present and properly formatted
        required_columns = ['high', 'low', 'close', 'price', 'volume', 'ma_20', 'ma_50', 'rsi']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Adding missing column: {col}")
                df[col] = df['close'] if col in ['price', 'high', 'low'] else np.random.normal(100, 1, len(df))
        
        # Add returns column
        df['returns'] = df['close'].pct_change().fillna(0)
        
        df.to_csv(data_path)
        
        logger.info(f"Created sample data file at {data_path} with {len(df)} rows and columns: {', '.join(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        return False

def main():
    """Main function to run the test."""
    try:
        # Default config and data paths
        config_path = "config/backtesting_config.json"
        data_path = "data/processed_market_data.csv"
        
        # Create sample data if needed
        if not create_sample_data(data_path):
            logger.error("Failed to create or verify sample data, cannot proceed with test")
            return None
            
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