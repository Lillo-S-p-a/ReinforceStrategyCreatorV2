"""Performance visualization module for trading strategies.

This module provides the PerformanceVisualizer class for creating various
performance charts and graphs for model evaluation results.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Visualizer for trading strategy performance metrics.
    
    Provides methods to create various performance visualizations including:
    - Cumulative returns / P&L curves
    - Drawdown plots
    - Metrics comparison charts
    - Learning curves
    - Risk-return scatter plots
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance visualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config or {}
        
        # Figure size defaults
        self.default_figsize = self.config.get("default_figsize", (12, 8))
        self.small_figsize = self.config.get("small_figsize", (10, 6))
        
        # Style settings
        self.style = self.config.get("style", "seaborn-v0_8-darkgrid")
        self.color_palette = self.config.get("color_palette", "husl")
        
        # DPI for saving figures
        self.save_dpi = self.config.get("save_dpi", 300)
        
        # Apply style settings
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
    
    def plot_cumulative_returns(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        benchmark_values: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        dates: Optional[Union[List, pd.DatetimeIndex]] = None,
        title: str = "Cumulative Returns",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Plot cumulative returns / P&L curve.
        
        Args:
            portfolio_values: Portfolio values over time
            benchmark_values: Optional benchmark values for comparison
            dates: Optional dates for x-axis
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values)
        
        # Calculate cumulative returns
        initial_value = portfolio_values[0]
        cumulative_returns = ((portfolio_values / initial_value) - 1) * 100
        
        # Create x-axis values
        if dates is not None:
            x_values = pd.to_datetime(dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            x_values = range(len(portfolio_values))
        
        # Plot portfolio returns
        ax.plot(x_values, cumulative_returns, label='Strategy', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_values is not None:
            benchmark_values = np.array(benchmark_values)
            benchmark_initial = benchmark_values[0]
            benchmark_returns = ((benchmark_values / benchmark_initial) - 1) * 100
            ax.plot(x_values, benchmark_returns, label='Benchmark', 
                   linewidth=2, color='orange', linestyle='--')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if dates
        if dates is not None:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved cumulative returns plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        dates: Optional[Union[List, pd.DatetimeIndex]] = None,
        title: str = "Portfolio Drawdown",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Plot drawdown chart.
        
        Args:
            portfolio_values: Portfolio values over time
            dates: Optional dates for x-axis
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Convert to numpy array
        portfolio_values = np.array(portfolio_values)
        
        # Calculate drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = ((portfolio_values - cumulative_max) / cumulative_max) * 100
        
        # Create x-axis values
        if dates is not None:
            x_values = pd.to_datetime(dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        else:
            x_values = range(len(portfolio_values))
        
        # Plot drawdown
        ax.fill_between(x_values, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        ax.plot(x_values, drawdown, color='darkred', linewidth=1.5)
        
        # Add max drawdown line
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = drawdown[max_dd_idx]
        ax.axhline(y=max_dd_value, color='darkred', linestyle='--', alpha=0.7,
                  label=f'Max Drawdown: {max_dd_value:.2f}%')
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)  # Drawdown is always negative or zero
        
        # Rotate x-axis labels if dates
        if dates is not None:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved drawdown plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        title: str = "Metrics Comparison",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Plot comparison of metrics across different models or strategies.
        
        Args:
            metrics_dict: Dictionary with strategy names as keys and metrics dicts as values
            metrics_to_plot: List of metric names to plot (plots all if None)
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data
        if not metrics_dict:
            logger.warning("No metrics provided for comparison")
            return None
        
        # Get all available metrics
        all_metrics = set()
        for metrics in metrics_dict.values():
            all_metrics.update(metrics.keys())
        
        # Filter metrics to plot
        if metrics_to_plot:
            metrics_to_use = [m for m in metrics_to_plot if m in all_metrics]
        else:
            # Default important metrics
            default_metrics = ['sharpe_ratio', 'max_drawdown', 'pnl_percentage', 
                             'win_rate', 'profit_factor', 'calmar_ratio']
            metrics_to_use = [m for m in default_metrics if m in all_metrics]
        
        if not metrics_to_use:
            logger.warning("No valid metrics found to plot")
            return None
        
        # Create subplots
        n_metrics = len(metrics_to_use)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_use):
            ax = axes[idx]
            
            # Prepare data for this metric
            strategies = []
            values = []
            
            for strategy, metrics in metrics_dict.items():
                if metric in metrics:
                    strategies.append(strategy)
                    values.append(metrics[metric])
            
            # Create bar plot
            bars = ax.bar(strategies, values)
            
            # Color bars based on positive/negative values
            for bar, value in zip(bars, values):
                if value < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
            
            # Formatting
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if many strategies
            if len(strategies) > 3:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved metrics comparison plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_learning_curves(
        self,
        training_history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        title: str = "Learning Curves",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Plot learning curves from training history.
        
        Args:
            training_history: Dictionary with metric names as keys and lists of values
            metrics: Specific metrics to plot (plots all if None)
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        if not training_history:
            logger.warning("No training history provided")
            return None
        
        # Filter metrics to plot
        if metrics:
            metrics_to_plot = [m for m in metrics if m in training_history]
        else:
            metrics_to_plot = list(training_history.keys())
        
        if not metrics_to_plot:
            logger.warning("No valid metrics found in training history")
            return None
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = training_history[metric]
            
            # Plot with smoothing
            ax.plot(values, alpha=0.3, color='blue', label='Raw')
            
            # Add smoothed line if enough data points
            if len(values) > 10:
                window_size = min(len(values) // 10, 50)
                smoothed = pd.Series(values).rolling(window=window_size, center=True).mean()
                ax.plot(smoothed, color='darkblue', linewidth=2, label='Smoothed')
            
            # Formatting
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_xlabel('Episode/Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved learning curves plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_risk_return_scatter(
        self,
        results_list: List[Dict[str, Any]],
        x_metric: str = "volatility",
        y_metric: str = "pnl_percentage",
        size_metric: Optional[str] = "sharpe_ratio",
        labels: Optional[List[str]] = None,
        title: str = "Risk-Return Profile",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Plot risk-return scatter plot for multiple strategies.
        
        Args:
            results_list: List of evaluation results dictionaries
            x_metric: Metric to use for x-axis (default: volatility)
            y_metric: Metric to use for y-axis (default: pnl_percentage)
            size_metric: Metric to use for bubble size (optional)
            labels: Labels for each result (uses model IDs if None)
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Extract data
        x_values = []
        y_values = []
        size_values = []
        plot_labels = []
        
        for idx, result in enumerate(results_list):
            metrics = result.get('metrics', {})
            
            # Get metric values
            x_val = metrics.get(x_metric, 0)
            y_val = metrics.get(y_metric, 0)
            
            x_values.append(x_val)
            y_values.append(y_val)
            
            # Get size value if specified
            if size_metric:
                size_val = abs(metrics.get(size_metric, 1))  # Use abs to ensure positive
                size_values.append(size_val * 100)  # Scale for visibility
            
            # Get label
            if labels and idx < len(labels):
                plot_labels.append(labels[idx])
            else:
                model_id = result.get('model', {}).get('id', f'Model {idx+1}')
                plot_labels.append(model_id)
        
        # Create scatter plot
        if size_metric and size_values:
            scatter = ax.scatter(x_values, y_values, s=size_values, alpha=0.6, 
                               c=range(len(x_values)), cmap='viridis')
            # Add size legend
            handles, labels_legend = scatter.legend_elements(prop="sizes", alpha=0.6, 
                                                            num=4, fmt="{x:.0f}")
            size_legend = ax.legend(handles, labels_legend, loc="upper left", 
                                  title=size_metric.replace('_', ' ').title())
            ax.add_artist(size_legend)
        else:
            ax.scatter(x_values, y_values, s=100, alpha=0.6, 
                      c=range(len(x_values)), cmap='viridis')
        
        # Add labels for each point
        for i, label in enumerate(plot_labels):
            ax.annotate(label, (x_values[i], y_values[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Formatting
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved risk-return scatter plot to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_performance_dashboard(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Figure:
        """Create a comprehensive performance dashboard with multiple plots.
        
        Args:
            evaluation_results: Complete evaluation results dictionary
            save_path: Path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Extract data
        metrics = evaluation_results.get('metrics', {})
        model_info = evaluation_results.get('model', {})
        
        # Title
        fig.suptitle(f"Performance Dashboard - {model_info.get('id', 'Model')}", 
                    fontsize=16, fontweight='bold')
        
        # 1. Key Metrics Summary (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metrics_summary(ax1, metrics)
        
        # 2. Model Info (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_model_info(ax2, evaluation_results)
        
        # 3. Cumulative Returns (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        # Note: This would need actual portfolio values from evaluation
        self._plot_placeholder(ax3, "Cumulative Returns")
        
        # 4. Drawdown (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_placeholder(ax4, "Drawdown")
        
        # 5. Benchmark Comparison (bottom)
        if 'benchmarks' in evaluation_results:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_benchmark_comparison(ax5, evaluation_results)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            logger.info(f"Saved performance dashboard to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_metrics_summary(self, ax: Axes, metrics: Dict[str, float]) -> None:
        """Plot key metrics summary as a table."""
        # Select key metrics
        key_metrics = ['pnl_percentage', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'profit_factor', 'trades_count']
        
        # Prepare data
        metric_data = []
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if metric in ['win_rate']:
                        formatted_value = f"{value:.1%}"
                    elif metric in ['pnl_percentage']:
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                metric_data.append([metric.replace('_', ' ').title(), formatted_value])
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metric_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(metric_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Key Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    def _plot_model_info(self, ax: Axes, results: Dict[str, Any]) -> None:
        """Plot model information."""
        ax.axis('tight')
        ax.axis('off')
        
        model_info = results.get('model', {})
        eval_info = {
            'Evaluation ID': results.get('evaluation_id', 'N/A'),
            'Model Type': model_info.get('type', 'N/A'),
            'Version': model_info.get('version', 'N/A'),
            'Timestamp': results.get('timestamp', 'N/A')[:19]  # Trim to date/time
        }
        
        info_text = '\n'.join([f'{k}: {v}' for k, v in eval_info.items()])
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        ax.set_title('Evaluation Info', fontsize=12, fontweight='bold')
    
    def _plot_benchmark_comparison(self, ax: Axes, results: Dict[str, Any]) -> None:
        """Plot benchmark comparison."""
        benchmarks = results.get('benchmarks', {})
        model_metrics = results.get('metrics', {})
        
        if not benchmarks:
            self._plot_placeholder(ax, "No Benchmark Data")
            return
        
        # Prepare data
        strategies = ['Model'] + list(benchmarks.keys())
        metrics_to_compare = ['pnl_percentage', 'sharpe_ratio', 'max_drawdown']
        
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_compare):
            values = [model_metrics.get(metric, 0)]
            for strategy in benchmarks:
                values.append(benchmarks[strategy].get(metric, 0))
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Value')
        ax.set_title('Benchmark Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_placeholder(self, ax: Axes, title: str) -> None:
        """Plot placeholder for missing data."""
        ax.text(0.5, 0.5, f'{title}\n(Data not available)', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')