"""Visualization utilities for cross-validation results."""

import logging
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from .cross_validator import CVResults


class CVVisualizer:
    """Visualization tools for cross-validation results."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: tuple = (10, 6)):
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.logger = logging.getLogger("CVVisualizer")
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_fold_metrics(
        self,
        cv_results: CVResults,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot metrics across folds as a bar chart.
        
        Args:
            cv_results: Cross-validation results
            metrics: Metrics to plot (None for all)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get summary DataFrame
        df = cv_results.get_summary_df()
        
        # Select metrics to plot
        if metrics is None:
            # Get all metric columns (excluding fold and training_time)
            metrics = [col for col in df.columns 
                      if col not in ['fold', 'training_time']]
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create bar plot
            df.plot(x='fold', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric)
            ax.set_xlabel('Fold')
            ax.set_ylabel('Value')
            
            # Add mean line
            mean_val = df[metric].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.4f}')
            ax.legend()
        
        # Remove empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved fold metrics plot to {save_path}")
        
        return fig
    
    def plot_metric_distribution(
        self,
        cv_results: CVResults,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot distribution of metrics across folds as box plots.
        
        Args:
            cv_results: Cross-validation results
            metrics: Metrics to plot (None for all)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get summary DataFrame
        df = cv_results.get_summary_df()
        
        # Select metrics to plot
        if metrics is None:
            metrics = [col for col in df.columns 
                      if col not in ['fold', 'training_time']]
        
        # Prepare data for box plot
        plot_data = []
        for metric in metrics:
            for value in df[metric]:
                plot_data.append({
                    'Metric': metric,
                    'Value': value
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create box plot
        sns.boxplot(data=plot_df, x='Metric', y='Value', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Metric Distribution Across Folds')
        
        # Add points for individual folds
        sns.stripplot(data=plot_df, x='Metric', y='Value', 
                     color='black', alpha=0.5, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved metric distribution plot to {save_path}")
        
        return fig
    
    def plot_train_val_comparison(
        self,
        cv_results: CVResults,
        metric: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot train vs validation metric comparison.
        
        Args:
            cv_results: Cross-validation results
            metric: Metric to compare
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get summary DataFrame
        df = cv_results.get_summary_df()
        
        # Check if both train and val versions exist
        train_metric = f"train_{metric}" if not metric.startswith("train_") else metric
        val_metric = f"val_{metric}" if not metric.startswith("val_") else metric
        
        if train_metric not in df.columns or val_metric not in df.columns:
            raise ValueError(f"Both {train_metric} and {val_metric} must be present")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot data
        x = df['fold']
        width = 0.35
        
        ax.bar(x - width/2, df[train_metric], width, label='Train', alpha=0.8)
        ax.bar(x + width/2, df[val_metric], width, label='Validation', alpha=0.8)
        
        # Add mean lines
        train_mean = df[train_metric].mean()
        val_mean = df[val_metric].mean()
        
        ax.axhline(y=train_mean, color='blue', linestyle='--', alpha=0.5,
                  label=f'Train Mean: {train_mean:.4f}')
        ax.axhline(y=val_mean, color='orange', linestyle='--', alpha=0.5,
                  label=f'Val Mean: {val_mean:.4f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'Train vs Validation: {metric}')
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in x])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved train/val comparison plot to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        results_dict: Dict[str, CVResults],
        metric: str,
        use_validation: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot comparison of multiple models.
        
        Args:
            results_dict: Dictionary mapping model names to CVResults
            metric: Metric to compare
            use_validation: Whether to use validation metrics (vs training)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        comparison_data = []
        
        prefix = "val_" if use_validation else "train_"
        full_metric = f"{prefix}{metric}" if not metric.startswith(prefix) else metric
        
        for model_name, cv_results in results_dict.items():
            if full_metric in cv_results.aggregated_metrics:
                stats = cv_results.aggregated_metrics[full_metric]
                comparison_data.append({
                    'Model': model_name,
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Min': stats['min'],
                    'Max': stats['max']
                })
        
        if not comparison_data:
            raise ValueError(f"No data found for metric {full_metric}")
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Mean')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create bar plot with error bars
        x = np.arange(len(df))
        ax.bar(x, df['Mean'], yerr=df['Std'], capsize=5, alpha=0.8)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(df['Mean'], df['Std'])):
            ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                   ha='center', va='bottom')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.set_ylabel(full_metric)
        ax.set_title(f'Model Comparison: {full_metric}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved model comparison plot to {save_path}")
        
        return fig
    
    def plot_learning_curves(
        self,
        cv_results: CVResults,
        metric: str = "loss",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plot learning curves for each fold.
        
        Args:
            cv_results: Cross-validation results
            metric: Metric to plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        # Collect histories from all folds
        for fold_result in cv_results.fold_results:
            history = fold_result.additional_info.get('history', {})
            fold_idx = fold_result.fold_idx
            
            # Plot training metric
            if metric in history:
                axes[0].plot(history[metric], label=f'Fold {fold_idx + 1}', alpha=0.7)
            
            # Plot validation metric
            val_metric = f"val_{metric}"
            if val_metric in history:
                axes[1].plot(history[val_metric], label=f'Fold {fold_idx + 1}', alpha=0.7)
        
        # Configure subplots
        axes[0].set_title(f'Training {metric}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel(metric)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title(f'Validation {metric}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved learning curves plot to {save_path}")
        
        return fig
    
    def create_cv_report(
        self,
        cv_results: CVResults,
        output_dir: Union[str, Path],
        include_plots: List[str] = None
    ) -> Path:
        """Create a comprehensive CV report with plots.
        
        Args:
            cv_results: Cross-validation results
            output_dir: Directory to save the report
            include_plots: List of plots to include 
                         ['fold_metrics', 'distribution', 'train_val', 'learning']
            
        Returns:
            Path to the report directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if include_plots is None:
            include_plots = ['fold_metrics', 'distribution', 'train_val', 'learning']
        
        # Save plots
        if 'fold_metrics' in include_plots:
            self.plot_fold_metrics(
                cv_results, 
                save_path=output_dir / "fold_metrics.png"
            )
        
        if 'distribution' in include_plots:
            self.plot_metric_distribution(
                cv_results,
                save_path=output_dir / "metric_distribution.png"
            )
        
        if 'train_val' in include_plots:
            # Plot train/val comparison for the scoring metric
            scoring_metric = cv_results.config.get('scoring_metric', 'loss')
            try:
                self.plot_train_val_comparison(
                    cv_results,
                    scoring_metric,
                    save_path=output_dir / f"train_val_{scoring_metric}.png"
                )
            except ValueError:
                self.logger.warning(f"Could not create train/val plot for {scoring_metric}")
        
        if 'learning' in include_plots:
            try:
                self.plot_learning_curves(
                    cv_results,
                    save_path=output_dir / "learning_curves.png"
                )
            except Exception as e:
                self.logger.warning(f"Could not create learning curves: {e}")
        
        # Save summary statistics
        summary_df = cv_results.get_summary_df()
        summary_df.to_csv(output_dir / "fold_summary.csv", index=False)
        
        # Create text report
        report_path = output_dir / "cv_report.txt"
        with open(report_path, 'w') as f:
            f.write("CROSS-VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"  Number of folds: {cv_results.config.get('n_folds', 'N/A')}\n")
            f.write(f"  CV method: {cv_results.config.get('cv_config', {}).get('method', 'N/A')}\n")
            f.write(f"  Scoring metric: {cv_results.config.get('scoring_metric', 'N/A')}\n")
            f.write(f"  Scoring mode: {cv_results.config.get('scoring_mode', 'N/A')}\n\n")
            
            # Aggregated metrics
            f.write("Aggregated Metrics:\n")
            for metric, stats in cv_results.aggregated_metrics.items():
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.6f}\n")
                f.write(f"    Std:  {stats['std']:.6f}\n")
                f.write(f"    Min:  {stats['min']:.6f}\n")
                f.write(f"    Max:  {stats['max']:.6f}\n")
            
            # Best fold
            f.write(f"\nBest fold: {cv_results.best_fold_idx + 1}\n")
            best_fold = cv_results.fold_results[cv_results.best_fold_idx]
            f.write("Best fold metrics:\n")
            for metric, value in best_fold.val_metrics.items():
                f.write(f"  val_{metric}: {value:.6f}\n")
            
            # Timing
            f.write(f"\nTotal time: {cv_results.total_time:.2f} seconds\n")
            avg_time = np.mean([fr.training_time for fr in cv_results.fold_results])
            f.write(f"Average fold time: {avg_time:.2f} seconds\n")
        
        self.logger.info(f"Created CV report in {output_dir}")
        return output_dir