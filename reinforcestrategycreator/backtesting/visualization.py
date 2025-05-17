"""
Visualization module for backtesting.

This module provides functionality for creating visualizations of
backtesting results, including performance metrics, benchmark comparisons,
and cross-validation results.
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates visualizations for backtesting results.
    
    This class provides methods for creating various visualizations,
    including performance metrics, benchmark comparisons, and
    cross-validation results.
    """
    
    def __init__(self, plots_dir: str = "plots") -> None:
        """
        Initialize the visualizer.
        
        Args:
            plots_dir: Directory to save plots
        """
        self.plots_dir = plots_dir
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def visualize_fold_results(self, fold_results: Dict[str, Any], fold: int) -> Optional[str]:
        """
        Create visualizations for each fold.
        
        Args:
            fold_results: Results dictionary for the fold
            fold: Fold number
            
        Returns:
            Path to the saved plot or None if visualization failed
        """
        if "error" in fold_results:
            logger.warning(f"Cannot visualize fold {fold+1} due to error")
            return None
            
        try:
            # Create metrics visualization
            metrics = fold_results["val_metrics"]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics.keys(), metrics.values())
            plt.title(f"Validation Metrics - Fold {fold+1}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f"fold_{fold}_metrics.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Visualization created for fold {fold+1}")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error visualizing fold {fold+1} results: {e}", exc_info=True)
            return None
    
    def create_cv_comparison_plot(self, cv_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Create comparison plot across folds.
        
        Args:
            cv_results: List of cross-validation results
            
        Returns:
            Path to the saved plot or None if visualization failed
        """
        if not cv_results:
            logger.warning("No CV results available for comparison plot")
            return None
            
        try:
            # Extract metrics from each fold
            folds = []
            pnls = []
            sharpe_ratios = []
            
            for result in cv_results:
                if "error" not in result:
                    folds.append(f"Fold {result['fold']+1}")
                    pnls.append(result["val_metrics"]["pnl"])
                    sharpe_ratios.append(result["val_metrics"]["sharpe_ratio"])
            
            if not folds:
                logger.warning("No valid fold results for comparison plot")
                return None
                
            # Create figure with two subplots
            plt.figure(figsize=(12, 10))
            
            # PnL comparison
            plt.subplot(2, 1, 1)
            plt.bar(folds, pnls)
            plt.title("PnL Across CV Folds")
            plt.ylabel("PnL ($)")
            plt.grid(True, axis='y')
            
            # Sharpe ratio comparison
            plt.subplot(2, 1, 2)
            plt.bar(folds, sharpe_ratios)
            plt.title("Sharpe Ratio Across CV Folds")
            plt.ylabel("Sharpe Ratio")
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, "cv_comparison.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info("CV comparison plot created")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating CV comparison plot: {e}", exc_info=True)
            return None
    
    def create_benchmark_comparison_chart(self, 
                                         test_metrics: Dict[str, float], 
                                         benchmark_metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """
        Create a chart comparing model performance with benchmarks.
        
        Args:
            test_metrics: Dictionary of model performance metrics
            benchmark_metrics: Dictionary of benchmark metrics
            
        Returns:
            Path to the saved plot or None if visualization failed
        """
        if not benchmark_metrics:
            logger.warning("No benchmark metrics available for comparison chart")
            return None
            
        try:
            # Extract PnL values
            strategies = ["Model"] + list(benchmark_metrics.keys())
            pnl_values = [test_metrics["pnl"]] + [m["pnl"] for m in benchmark_metrics.values()]
            sharpe_values = [test_metrics["sharpe_ratio"]] + [m["sharpe_ratio"] for m in benchmark_metrics.values()]
            
            # Create figure with two subplots
            plt.figure(figsize=(12, 10))
            
            # PnL comparison
            plt.subplot(2, 1, 1)
            bars = plt.bar(strategies, pnl_values, color=['blue'] + ['gray'] * len(benchmark_metrics))
            plt.title("PnL Comparison with Benchmark Strategies")
            plt.ylabel("PnL ($)")
            plt.grid(True, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'${height:.2f}', ha='center', va='bottom')
            
            # Sharpe ratio comparison
            plt.subplot(2, 1, 2)
            bars = plt.bar(strategies, sharpe_values, color=['blue'] + ['gray'] * len(benchmark_metrics))
            plt.title("Sharpe Ratio Comparison with Benchmark Strategies")
            plt.ylabel("Sharpe Ratio")
            plt.grid(True, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, "benchmark_comparison.png")
            plt.savefig(plot_path)
            
            logger.info(f"Benchmark comparison chart saved to {self.plots_dir}")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating benchmark comparison chart: {e}", exc_info=True)
            return None
    
    def create_metrics_summary_chart(self, test_metrics: Dict[str, float]) -> Optional[str]:
        """
        Create a chart summarizing key performance metrics.
        
        Args:
            test_metrics: Dictionary of model performance metrics
            
        Returns:
            Path to the saved plot or None if visualization failed
        """
        try:
            # Extract metrics
            metrics = {
                "Sharpe Ratio": test_metrics["sharpe_ratio"],
                "Max Drawdown (%)": test_metrics["max_drawdown"] * 100,  # Convert to percentage
                "Win Rate (%)": test_metrics["win_rate"] * 100  # Convert to percentage
            }
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 6))
            y_pos = range(len(metrics))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Define colors based on metric values
            colors = ['green' if name == 'Sharpe Ratio' or name == 'Win Rate (%)' else 'red' for name in metric_names]
            
            bars = plt.barh(y_pos, metric_values, color=colors)
            plt.yticks(y_pos, metric_names)
            plt.xlabel('Value')
            plt.title('Key Performance Metrics (Test Period)')
            plt.grid(True, axis='x')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, "metrics_summary.png")
            plt.savefig(plot_path)
            
            logger.info(f"Metrics summary chart saved to {self.plots_dir}")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating metrics summary chart: {e}", exc_info=True)
            return None
    
    def create_equity_curve(self, portfolio_values: List[float], dates: List[str]) -> Optional[str]:
        """
        Create an equity curve visualization.
        
        Args:
            portfolio_values: List of portfolio values over time
            dates: List of corresponding dates
            
        Returns:
            Path to the saved plot or None if visualization failed
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, portfolio_values)
            plt.title("Portfolio Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, "equity_curve.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Equity curve saved to {self.plots_dir}")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}", exc_info=True)
            return None