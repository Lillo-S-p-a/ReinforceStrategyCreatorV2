"""Visualization utilities for Hyperparameter Optimization results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Install for visualization support.")


class HPOVisualizer:
    """Visualizer for HPO results analysis."""
    
    def __init__(self, style: str = "seaborn"):
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib and Seaborn are required for visualization")
        
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_optimization_history(
        self,
        results: Dict[str, Any],
        metric: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot optimization history showing metric improvement over trials.
        
        Args:
            results: HPO results dictionary
            metric: Metric name (uses default from results if not provided)
            save_path: Path to save the plot
        """
        if "all_trials" not in results:
            raise ValueError("No trials found in results")
        
        # Extract metric name
        if metric is None:
            metric = results.get("metric", "loss")
        
        # Prepare data
        trials_data = []
        for i, trial in enumerate(results["all_trials"]):
            if trial.get("metric") is not None:
                trials_data.append({
                    "trial": i + 1,
                    "metric": trial["metric"],
                    "status": trial.get("status", "COMPLETED")
                })
        
        if not trials_data:
            print("No successful trials to plot")
            return
        
        df = pd.DataFrame(trials_data)
        
        # Calculate cumulative best
        mode = results.get("mode", "min")
        if mode == "min":
            df["cumulative_best"] = df["metric"].cummin()
        else:
            df["cumulative_best"] = df["metric"].cummax()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all trials
        ax.scatter(df["trial"], df["metric"], alpha=0.6, label="Trial results")
        
        # Plot cumulative best
        ax.plot(df["trial"], df["cumulative_best"], 
                color="red", linewidth=2, label="Best so far")
        
        # Highlight best trial
        best_idx = df["metric"].idxmin() if mode == "min" else df["metric"].idxmax()
        ax.scatter(df.loc[best_idx, "trial"], df.loc[best_idx, "metric"],
                  color="green", s=200, marker="*", label="Best trial")
        
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(f"{metric}")
        ax.set_title(f"HPO Progress: {metric} over trials")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_parameter_importance(
        self,
        analysis: Dict[str, Any],
        top_k: int = 10,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot parameter importance from analysis results.
        
        Args:
            analysis: Analysis dictionary from HPOptimizer.analyze_results()
            top_k: Number of top parameters to show
            save_path: Path to save the plot
        """
        if "parameter_importance" not in analysis:
            raise ValueError("No parameter importance data in analysis")
        
        # Get top k parameters
        importance_data = list(analysis["parameter_importance"].items())[:top_k]
        if not importance_data:
            print("No parameter importance data to plot")
            return
        
        params, importances = zip(*importance_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel("Importance Score")
        ax.set_title("Parameter Importance (Correlation with Metric)")
        ax.grid(True, axis="x", alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_parallel_coordinates(
        self,
        results: Dict[str, Any],
        params_to_plot: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot parallel coordinates for top trials.
        
        Args:
            results: HPO results dictionary
            params_to_plot: List of parameters to include (auto-selects if None)
            top_k: Number of top trials to plot
            save_path: Path to save the plot
        """
        if "all_trials" not in results:
            raise ValueError("No trials found in results")
        
        # Prepare data
        trials_data = []
        for trial in results["all_trials"]:
            if trial.get("metric") is not None:
                trial_flat = {"metric": trial["metric"]}
                trial_flat.update(trial.get("params", {}))
                trials_data.append(trial_flat)
        
        if not trials_data:
            print("No successful trials to plot")
            return
        
        df = pd.DataFrame(trials_data)
        
        # Sort by metric and get top k
        mode = results.get("mode", "min")
        df = df.sort_values("metric", ascending=(mode == "min")).head(top_k)
        
        # Select parameters to plot
        if params_to_plot is None:
            # Auto-select numeric parameters
            params_to_plot = [col for col in df.columns 
                            if col != "metric" and df[col].dtype in [np.float64, np.int64]]
        
        if not params_to_plot:
            print("No numeric parameters to plot")
            return
        
        # Normalize data for plotting
        plot_data = df[params_to_plot + ["metric"]].copy()
        for col in plot_data.columns:
            col_min = plot_data[col].min()
            col_max = plot_data[col].max()
            if col_max > col_min:
                plot_data[col] = (plot_data[col] - col_min) / (col_max - col_min)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines
        for idx, row in plot_data.iterrows():
            values = row.values
            positions = range(len(values))
            
            # Color by metric value
            color_val = row["metric"]
            ax.plot(positions, values, alpha=0.5, linewidth=2)
        
        # Set labels
        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels(plot_data.columns, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title(f"Parallel Coordinates: Top {len(df)} Trials")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_parameter_distributions(
        self,
        results: Dict[str, Any],
        params_to_plot: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot parameter value distributions across all trials.
        
        Args:
            results: HPO results dictionary
            params_to_plot: List of parameters to plot (auto-selects if None)
            save_path: Path to save the plot
        """
        if "all_trials" not in results:
            raise ValueError("No trials found in results")
        
        # Prepare data
        params_data = {}
        for trial in results["all_trials"]:
            if trial.get("params"):
                for param, value in trial["params"].items():
                    if param not in params_data:
                        params_data[param] = []
                    params_data[param].append(value)
        
        if not params_data:
            print("No parameter data to plot")
            return
        
        # Select parameters to plot
        if params_to_plot is None:
            params_to_plot = list(params_data.keys())
        
        # Filter to only numeric parameters
        numeric_params = []
        for param in params_to_plot:
            if param in params_data:
                values = params_data[param]
                if values and isinstance(values[0], (int, float)):
                    numeric_params.append(param)
        
        if not numeric_params:
            print("No numeric parameters to plot")
            return
        
        # Create subplots
        n_params = len(numeric_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot distributions
        for i, param in enumerate(numeric_params):
            ax = axes[i]
            values = params_data[param]
            
            # Plot histogram
            ax.hist(values, bins=20, alpha=0.7, edgecolor="black")
            ax.set_xlabel(param)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {param}")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Parameter Value Distributions", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_metric_vs_parameter(
        self,
        results: Dict[str, Any],
        parameter: str,
        metric: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Plot metric values against a specific parameter.
        
        Args:
            results: HPO results dictionary
            parameter: Parameter name to plot
            metric: Metric name (uses default from results if not provided)
            save_path: Path to save the plot
        """
        if "all_trials" not in results:
            raise ValueError("No trials found in results")
        
        # Extract metric name
        if metric is None:
            metric = results.get("metric", "loss")
        
        # Prepare data
        param_values = []
        metric_values = []
        
        for trial in results["all_trials"]:
            if trial.get("metric") is not None and trial.get("params", {}).get(parameter) is not None:
                param_values.append(trial["params"][parameter])
                metric_values.append(trial["metric"])
        
        if not param_values:
            print(f"No data found for parameter '{parameter}'")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(param_values, metric_values, alpha=0.6)
        
        # Add trend line if numeric
        if isinstance(param_values[0], (int, float)):
            # Fit polynomial
            z = np.polyfit(param_values, metric_values, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(param_values), max(param_values), 100)
            ax.plot(x_smooth, p(x_smooth), "r-", alpha=0.8, linewidth=2, label="Trend")
        
        ax.set_xlabel(parameter)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {parameter}")
        ax.grid(True, alpha=0.3)
        
        if isinstance(param_values[0], (int, float)):
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def create_summary_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> None:
        """Create a comprehensive summary report with plots.
        
        Args:
            results: HPO results dictionary
            analysis: Analysis dictionary from HPOptimizer.analyze_results()
            output_dir: Directory to save the report
            include_plots: Whether to generate and include plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results and analysis as JSON
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate plots if requested
        if include_plots and MATPLOTLIB_AVAILABLE:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Optimization history
            self.plot_optimization_history(
                results, 
                save_path=plots_dir / "optimization_history.png"
            )
            
            # Parameter importance
            if "parameter_importance" in analysis:
                self.plot_parameter_importance(
                    analysis,
                    save_path=plots_dir / "parameter_importance.png"
                )
            
            # Parameter distributions
            self.plot_parameter_distributions(
                results,
                save_path=plots_dir / "parameter_distributions.png"
            )
            
            # Parallel coordinates
            self.plot_parallel_coordinates(
                results,
                save_path=plots_dir / "parallel_coordinates.png"
            )
        
        # Create summary text report
        report_lines = [
            "# Hyperparameter Optimization Summary Report",
            f"\nRun Name: {results.get('run_name', 'Unknown')}",
            f"Timestamp: {results.get('timestamp', 'Unknown')}",
            f"\n## Configuration",
            f"- Number of trials: {results.get('num_trials', 'Unknown')}",
            f"- Search algorithm: {results.get('search_algorithm', 'Unknown')}",
            f"- Scheduler: {results.get('scheduler', 'Unknown')}",
            f"- Metric: {results.get('metric', 'Unknown')} ({results.get('mode', 'Unknown')})",
            f"\n## Results",
            f"- Best score: {results.get('best_score', 'Unknown')}",
            f"- Best parameters:",
        ]
        
        if results.get("best_params"):
            for param, value in results["best_params"].items():
                report_lines.append(f"  - {param}: {value}")
        
        if analysis:
            report_lines.extend([
                f"\n## Analysis",
                f"- Total trials: {analysis.get('total_trials', 'Unknown')}",
                f"- Successful trials: {analysis.get('successful_trials', 'Unknown')}",
                f"\n### Top 5 Trials:",
            ])
            
            for trial in analysis.get("top_k_trials", [])[:5]:
                report_lines.append(
                    f"- Rank {trial['rank']}: {trial['metric']:.4f} - {trial['params']}"
                )
            
            if "parameter_importance" in analysis:
                report_lines.append("\n### Parameter Importance:")
                for param, importance in list(analysis["parameter_importance"].items())[:10]:
                    report_lines.append(f"- {param}: {importance:.3f}")
        
        # Save report
        with open(output_dir / "summary_report.md", "w") as f:
            f.write("\n".join(report_lines))
        
        print(f"Summary report saved to: {output_dir}")