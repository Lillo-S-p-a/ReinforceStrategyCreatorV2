"""
Main workflow orchestration module for backtesting.

This module provides the main BacktestingWorkflow class that orchestrates
the entire backtesting process using the modular components.
"""

import os
import logging
import random
import datetime
import numpy as np
import pandas as pd
import webbrowser
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from reinforcestrategycreator.rl_agent import StrategyAgent as RLAgent
from reinforcestrategycreator.backtesting.data import DataManager
from reinforcestrategycreator.backtesting.cross_validation import CrossValidator
from reinforcestrategycreator.backtesting.model import ModelTrainer
from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator, BenchmarkEvaluator
from reinforcestrategycreator.backtesting.visualization import Visualizer
from reinforcestrategycreator.backtesting.reporting import ReportGenerator
from reinforcestrategycreator.backtesting.export import ModelExporter

# Configure logging
logger = logging.getLogger(__name__)


class BacktestingWorkflow:
    """
    A comprehensive backtesting workflow for reinforcement learning trading strategies.
    
    This class implements the entire workflow from data preparation through 
    cross-validation, hyperparameter optimization, model selection, final evaluation,
    benchmark comparison, and results visualization.
    
    This is the main orchestration class that uses the modular components
    to implement the complete backtesting workflow.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 results_dir: Optional[str] = None,
                 asset: str = "SPY",
                 start_date: str = "2020-01-01",
                 end_date: str = "2023-01-01",
                 cv_folds: int = 5,
                 test_ratio: float = 0.2,
                 random_seed: int = 42) -> None:
        """
        Initialize the backtesting workflow with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            results_dir: Directory to save results (default: timestamped directory)
            asset: Asset symbol to backtest
            start_date: Start date for data in YYYY-MM-DD format
            end_date: End date for data in YYYY-MM-DD format
            cv_folds: Number of cross-validation folds
            test_ratio: Ratio of data to use for final testing
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.cv_folds = cv_folds
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create results directory with timestamp
        if results_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = os.path.join("results", f"backtest_{asset}_{timestamp}")
        else:
            self.results_dir = results_dir
            
        # Create subdirectories
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.models_dir = os.path.join(self.results_dir, "models")
        self.reports_dir = os.path.join(self.results_dir, "reports")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Initialize component modules
        self.data_manager = DataManager(
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            test_ratio=test_ratio
        )
        
        # Initialize metrics calculator with sharpe_window_size from config
        sharpe_window_size = config.get("sharpe_window_size", None)
        self.metrics_calculator = MetricsCalculator(sharpe_window_size=sharpe_window_size)
        logger.info(f"Initialized MetricsCalculator with sharpe_window_size={sharpe_window_size}")
        
        self.visualizer = Visualizer(
            plots_dir=self.plots_dir
        )
        
        self.report_generator = ReportGenerator(
            reports_dir=self.reports_dir
        )
        
        self.model_exporter = ModelExporter(
            export_dir="production_models"
        )
        
        # Initialize containers for results
        self.data = None
        self.train_data = None
        self.test_data = None
        self.cv_results = []
        self.best_params = None
        self.best_model = None
        self.test_metrics = None
        self.benchmark_metrics = None
        
        logger.info(f"Initialized backtesting workflow for {asset} from {start_date} to {end_date}")
        logger.info(f"Results will be saved to {self.results_dir}")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch and prepare data with technical indicators.
        
        Returns:
            DataFrame containing price data and technical indicators
        """
        # Use the DataManager to fetch and prepare data
        data = self.data_manager.fetch_data()
        
        # Store data references
        self.data = self.data_manager.get_data()
        self.train_data, self.test_data = self.data_manager.get_train_test_data()
        
        return data
    
    def perform_cross_validation(self) -> List[Dict[str, Any]]:
        """
        Execute time-series cross-validation.
        
        Returns:
            List of dictionaries containing results for each fold
        """
        logger.info(f"Performing {self.cv_folds}-fold time-series cross-validation")
        
        if self.train_data is None:
            logger.warning("No training data available. Fetching data first.")
            self.fetch_data()
        
        # Initialize cross-validator
        cross_validator = CrossValidator(
            train_data=self.train_data,
            config=self.config,
            cv_folds=self.cv_folds,
            models_dir=self.models_dir,
            random_seed=self.random_seed
        )
        
        # Perform cross-validation
        cv_results = cross_validator.perform_cross_validation()
        
        # Store CV results
        self.cv_results = cv_results
        
        # Create CV comparison plot
        self.visualizer.create_cv_comparison_plot(cv_results)
        
        # Save CV summary
        self._save_cv_summary()
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def _save_cv_summary(self) -> None:
        """
        Save cross-validation summary data.
        """
        if not self.cv_results:
            logger.warning("No CV results to save")
            return
            
        try:
            # Extract summary data
            summary = {
                "folds": len(self.cv_results),
                "results": self.cv_results
            }
            
            # Calculate average metrics
            metrics_keys = ["pnl", "sharpe_ratio", "max_drawdown", "win_rate"]
            avg_metrics = {key: 0.0 for key in metrics_keys}
            count = 0
            
            for result in self.cv_results:
                if "error" not in result:
                    for key in metrics_keys:
                        avg_metrics[key] += result["val_metrics"][key]
                    count += 1
            
            if count > 0:
                for key in metrics_keys:
                    avg_metrics[key] /= count
                    
            summary["average_metrics"] = avg_metrics
            
            # Save to file
            import json
            with open(os.path.join(self.results_dir, "cv_summary.json"), "w") as f:
                json.dump(summary, f, indent=4)
                
            logger.info(f"CV summary saved to {self.results_dir}/cv_summary.json")
            
        except Exception as e:
            logger.error(f"Error saving CV summary: {e}", exc_info=True)
    
    def select_best_model(self) -> Dict[str, Any]:
        """
        Select best model using enhanced multi-metric selection criteria.
        
        Returns:
            Dictionary containing best parameters and model path
        """
        logger.info("Selecting best model using enhanced multi-metric selection criteria")
        
        if not self.cv_results:
            logger.warning("No CV results available. Running cross-validation first.")
            self.perform_cross_validation()
            
        if not self.cv_results:
            raise ValueError("Failed to generate CV results")
            
        try:
            # Initialize cross-validator with existing results
            cross_validator = CrossValidator(
                train_data=self.train_data,
                config=self.config,
                cv_folds=self.cv_folds,
                models_dir=self.models_dir,
                random_seed=self.random_seed
            )
            # Set the CV results
            cross_validator.cv_results = self.cv_results
            
            # Generate and log the CV report
            cv_report = cross_validator.generate_cv_report()
            logger.info(f"Cross-validation Performance Report:\n{cv_report}")
            
            # Use the enhanced multi-metric selection
            best_model_info = cross_validator.select_best_model()
            
            # Store best parameters
            self.best_params = best_model_info["params"]
            
            # Log detailed selection metrics
            metrics = best_model_info["metrics"]
            fold_num = best_model_info.get("fold", -1)
            logger.info(f"Selected best model from fold {fold_num} with metrics:")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"PnL: ${metrics['pnl']:.2f}")
            logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            
            return best_model_info
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}", exc_info=True)
            raise
    
    def train_final_model(self,
                         use_transfer_learning: bool = True,
                         use_ensemble: bool = False) -> RLAgent:  # Using RLAgent alias
        """
        Train final model on complete dataset with enhanced techniques.
        
        Args:
            use_transfer_learning: Whether to initialize from best CV model weights
            use_ensemble: Whether to create an ensemble from top-performing models
            
        Returns:
            Trained RL agent
        """
        logger.info(f"Training final model with {'transfer learning' if use_transfer_learning else 'scratch initialization'}" +
                  f"{' and ensemble creation' if use_ensemble else ''}")
        
        if self.best_params is None:
            logger.warning("No best parameters available. Selecting best model first.")
            self.select_best_model()
            
        if self.train_data is None:
            logger.warning("No training data available. Fetching data first.")
            self.fetch_data()
            
        # Initialize model trainer
        model_trainer = ModelTrainer(
            config=self.config,
            models_dir=self.models_dir,
            random_seed=self.random_seed
        )
        
        # Provide CV results for ensemble creation if enabled
        if use_ensemble:
            # Pass the CV results to the model trainer's config for ensemble creation
            model_trainer.config["cv_results"] = self.cv_results
            logger.info(f"Providing {len(self.cv_results)} CV results for potential ensemble creation")
        
        # Train final model with enhanced options
        agent = model_trainer.train_final_model(
            train_data=self.train_data,
            best_params=self.best_params,
            use_transfer_learning=use_transfer_learning,
            use_ensemble=use_ensemble
        )
        
        # Store best model
        self.best_model = agent
        
        return agent
    
    def evaluate_final_model(self) -> Dict[str, float]:
        """
        Evaluate on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating final model on test data")
        
        if self.best_model is None:
            logger.warning("No final model available. Training final model first.")
            self.train_final_model()
            
        if self.test_data is None:
            logger.warning("No test data available. Fetching data first.")
            self.fetch_data()
            
        # Initialize model trainer for evaluation
        model_trainer = ModelTrainer(
            config=self.config,
            models_dir=self.models_dir,
            random_seed=self.random_seed
        )
        
        # Evaluate model
        metrics = model_trainer.evaluate_model(
            model=self.best_model,
            test_data=self.test_data
        )
        
        # Store test metrics
        self.test_metrics = metrics
        
        # Compare with benchmarks
        self._compare_with_benchmarks(self.test_data)
        
        # Save test results
        self._save_test_results()
        
        # Create visualizations
        self._visualize_test_results()
        
        logger.info(f"Final model evaluation completed with PnL: ${metrics['pnl']:.2f}")
        
        return metrics
    
    def _compare_with_benchmarks(self, test_data: pd.DataFrame) -> dict:
        """
        Compare model performance with benchmark strategies.
        
        Args:
            test_data: DataFrame containing the test data
            
        Returns:
            dict: Comparison metrics against benchmarks
        """
        logger.info("Comparing model performance with benchmarks")
        
        # Initialize benchmark evaluator
        benchmark_evaluator = BenchmarkEvaluator(
            config=self.config,
            metrics_calculator=self.metrics_calculator
        )
        
        # Compare with benchmarks
        benchmark_results = benchmark_evaluator.compare_with_benchmarks(
            test_data=test_data,
            model_metrics=self.test_metrics
        )
        
        # Store benchmark metrics
        self.benchmark_metrics = benchmark_results["benchmarks"]
        
        return benchmark_results["relative_performance"]
    
    def _save_test_results(self) -> None:
        """Save test results and metrics to files."""
        logger.info("Saving test results")
        
        # Save test metrics
        import json
        with open(os.path.join(self.results_dir, "test_metrics.json"), "w") as f:
            json.dump(self.test_metrics, f, indent=4)
            
        # Save benchmark comparison
        if hasattr(self, "benchmark_metrics"):
            with open(os.path.join(self.results_dir, "benchmark_metrics.json"), "w") as f:
                json.dump(self.benchmark_metrics, f, indent=4)
                
        logger.info(f"Test results saved to {self.results_dir}")
    
    def _visualize_test_results(self) -> None:
        """Create visualizations of test results."""
        logger.info("Creating test result visualizations")
        
        try:
            # Create performance comparison chart
            self.visualizer.create_benchmark_comparison_chart(
                test_metrics=self.test_metrics,
                benchmark_metrics=self.benchmark_metrics
            )
            
            # Create metrics summary chart
            self.visualizer.create_metrics_summary_chart(
                test_metrics=self.test_metrics
            )
            
        except Exception as e:
            logger.error(f"Error creating test result visualizations: {e}", exc_info=True)
    
    def generate_report(self, format: str = "html") -> str:
        """
        Create a comprehensive backtesting report.
        
        Args:
            format: Report format ('html', 'markdown', or 'pdf')
            
        Returns:
            str: Path to the generated report file
        """
        logger.info(f"Generating {format} backtesting report")
        
        if self.test_metrics is None:
            logger.warning("No test metrics available. Evaluating final model first.")
            self.evaluate_final_model()
        
        # Prepare report data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            "asset": self.asset,
            "period": f"{self.start_date} to {self.end_date}",
            "timestamp": timestamp,
            "test_metrics": self.test_metrics,
            "benchmark_metrics": self.benchmark_metrics if hasattr(self, "benchmark_metrics") else {},
            "cv_results": self.cv_results,
            "best_params": self.best_params,
            "plots": {
                "benchmark_comparison": os.path.join(self.plots_dir, "benchmark_comparison.png"),
                "metrics_summary": os.path.join(self.plots_dir, "metrics_summary.png"),
                "cv_comparison": os.path.join(self.plots_dir, "cv_comparison.png") if os.path.exists(os.path.join(self.plots_dir, "cv_comparison.png")) else None
            }
        }
        
        # Generate report
        report_path = self.report_generator.generate_report(
            data=report_data,
            format=format
        )
        
        return report_path
    
    def export_for_trading(self, export_dir: Optional[str] = None) -> str:
        """
        Export the final model for paper/live trading.
        
        Args:
            export_dir: Directory to export the model (default: 'production_models')
            
        Returns:
            str: Path to the exported model file
        """
        logger.info("Exporting model for trading")
        
        if self.best_model is None:
            logger.warning("No final model available. Training final model first.")
            self.train_final_model()
        
        # Set export directory if provided
        if export_dir is not None:
            self.model_exporter.export_dir = export_dir
            os.makedirs(export_dir, exist_ok=True)
        
        # Export model
        model_path = self.model_exporter.export_model(
            model=self.best_model,
            asset=self.asset,
            start_date=self.start_date,
            end_date=self.end_date,
            params=self.best_params,
            test_metrics=self.test_metrics,
            benchmark_metrics=self.benchmark_metrics
        )
        
        return model_path
    
    def run_workflow(self) -> Dict[str, Any]:
        """
        Run the complete backtesting workflow.
        
        This method orchestrates the entire process from data preparation
        through cross-validation, model selection, final evaluation,
        benchmark comparison, report generation, and model export.
        
        Returns:
            Dict[str, Any]: Summary of workflow results
        """
        logger.info("Starting complete backtesting workflow")
        start_time = datetime.datetime.now()
        
        try:
            # Step 1: Fetch and prepare data
            logger.info("Step 1/7: Fetching and preparing data")
            self.fetch_data()
            
            # Step 2: Perform cross-validation
            logger.info("Step 2/7: Performing cross-validation")
            cv_results = self.perform_cross_validation()
            
            # Step 3: Select best model
            logger.info("Step 3/7: Selecting best model")
            best_model_info = self.select_best_model()
            
            # Step 4: Train final model
            logger.info("Step 4/7: Training final model")
            self.train_final_model()
            
            # Step 5: Evaluate final model
            logger.info("Step 5/7: Evaluating final model")
            test_metrics = self.evaluate_final_model()
            
            # Step 6: Generate report
            logger.info("Step 6/7: Generating report")
            report_path = self.generate_report(format="html")
            
            # Step 7: Export model for trading
            logger.info("Step 7/7: Exporting model for trading")
            model_path = self.export_for_trading()
            
            # Calculate execution time
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Prepare workflow summary
            summary = {
                "asset": self.asset,
                "period": f"{self.start_date} to {self.end_date}",
                "execution_time_seconds": execution_time,
                "cv_folds": len(cv_results),
                "test_metrics": test_metrics,
                "best_params": self.best_params,
                "report_path": report_path,
                "model_path": model_path,
                "results_dir": self.results_dir
            }
            
            logger.info(f"Backtesting workflow completed in {execution_time:.2f} seconds")
            logger.info(f"Results saved to {self.results_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in backtesting workflow: {e}", exc_info=True)
            raise