"""Example script demonstrating visualization and reporting functionality.

This script shows how to:
1. Generate performance visualizations
2. Create evaluation reports in various formats
3. Compare multiple model evaluations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.visualization.performance_visualizer import PerformanceVisualizer
from src.visualization.report_generator import ReportGenerator


def generate_sample_evaluation_results():
    """Generate sample evaluation results for demonstration."""
    # Simulate portfolio values over time
    np.random.seed(42)
    days = 252  # One trading year
    initial_value = 10000
    
    # Generate returns with some trend and volatility
    daily_returns = np.random.normal(0.0005, 0.02, days)
    portfolio_values = [initial_value]
    
    for ret in daily_returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)
    
    # Generate dates
    dates = pd.date_range(start='2024-01-01', periods=days + 1, freq='D')
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Create evaluation results
    results = {
        'evaluation_id': f'eval_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'evaluation_name': 'Demo Model Evaluation',
        'timestamp': datetime.now().isoformat(),
        'model': {
            'id': 'demo_ppo_model',
            'version': '1.0.0',
            'type': 'PPO',
            'hyperparameters': {
                'learning_rate': 0.0003,
                'batch_size': 64,
                'n_steps': 2048,
                'gamma': 0.99
            }
        },
        'data': {
            'source_id': 'demo_market_data',
            'version': '2024.1',
            'shape': [days, 10],
            'columns': ['open', 'high', 'low', 'close', 'volume', 
                       'sma_20', 'sma_50', 'rsi', 'macd', 'signal']
        },
        'metrics': {
            'pnl': final_value - initial_value,
            'pnl_percentage': total_return * 100,
            'total_return': total_return,
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252),
            'max_drawdown': calculate_max_drawdown(portfolio_values),
            'win_rate': 0.55,
            'profit_factor': 1.8,
            'trades_count': 150,
            'average_win': 85.50,
            'average_loss': 47.50,
            'volatility': np.std(daily_returns) * np.sqrt(252)
        },
        'benchmarks': {
            'buy_and_hold': {
                'pnl': 500.00,
                'pnl_percentage': 5.0,
                'sharpe_ratio': 0.8,
                'max_drawdown': 0.25,
                'win_rate': 0.0,
                'trades_count': 1
            },
            'sma_crossover': {
                'pnl': 650.00,
                'pnl_percentage': 6.5,
                'sharpe_ratio': 0.95,
                'max_drawdown': 0.18,
                'win_rate': 0.45,
                'trades_count': 80
            }
        },
        'relative_performance': {
            'buy_and_hold': {
                'absolute_difference': (final_value - initial_value) - 500.00,
                'percentage_difference': ((total_return * 100) - 5.0) / 5.0 * 100,
                'sharpe_ratio_difference': 0.4
            },
            'sma_crossover': {
                'absolute_difference': (final_value - initial_value) - 650.00,
                'percentage_difference': ((total_return * 100) - 6.5) / 6.5 * 100,
                'sharpe_ratio_difference': 0.25
            }
        }
    }
    
    return results, portfolio_values, dates


def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values."""
    values = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(values)
    drawdowns = (cumulative_max - values) / cumulative_max
    return float(np.max(drawdowns))


def main():
    """Main function to demonstrate visualization and reporting."""
    print("=== Visualization & Reporting Example ===\n")
    
    # Create output directory
    output_dir = Path("output/visualization_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    visualizer = PerformanceVisualizer(config={
        'default_figsize': (12, 8),
        'save_dpi': 150
    })
    
    report_generator = ReportGenerator(config={
        'pdf_options': {
            'page-size': 'A4',
            'orientation': 'Portrait'
        }
    })
    
    # Generate sample data
    print("1. Generating sample evaluation results...")
    results, portfolio_values, dates = generate_sample_evaluation_results()
    print(f"   - Generated {len(portfolio_values)} days of portfolio data")
    print(f"   - Final PnL: ${results['metrics']['pnl']:.2f}")
    print(f"   - Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
    
    # Create visualizations
    print("\n2. Creating visualizations...")
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Cumulative returns plot
    print("   - Generating cumulative returns plot...")
    fig = visualizer.plot_cumulative_returns(
        portfolio_values,
        dates=dates,
        title="Model Performance - Cumulative Returns",
        save_path=viz_dir / "cumulative_returns.png",
        show=False
    )
    
    # Drawdown plot
    print("   - Generating drawdown plot...")
    fig = visualizer.plot_drawdown(
        portfolio_values,
        dates=dates,
        title="Model Performance - Drawdown Analysis",
        save_path=viz_dir / "drawdown.png",
        show=False
    )
    
    # Metrics comparison
    print("   - Generating metrics comparison...")
    all_metrics = {"Model": results['metrics']}
    all_metrics.update(results['benchmarks'])
    
    fig = visualizer.plot_metrics_comparison(
        all_metrics,
        metrics_to_plot=['pnl_percentage', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
        title="Model vs Benchmarks - Key Metrics",
        save_path=viz_dir / "metrics_comparison.png",
        show=False
    )
    
    # Performance dashboard
    print("   - Generating performance dashboard...")
    fig = visualizer.create_performance_dashboard(
        results,
        save_path=viz_dir / "performance_dashboard.png",
        show=False
    )
    
    # Risk-return scatter (with multiple models)
    print("   - Generating risk-return scatter plot...")
    # Create additional model results for comparison
    results_list = [results]
    for i in range(2):
        modified_results = results.copy()
        modified_results['model']['id'] = f'model_variant_{i+1}'
        modified_results['metrics']['volatility'] = results['metrics']['volatility'] * (1 + 0.1 * (i+1))
        modified_results['metrics']['pnl_percentage'] = results['metrics']['pnl_percentage'] * (1 - 0.1 * (i+1))
        results_list.append(modified_results)
    
    fig = visualizer.plot_risk_return_scatter(
        results_list,
        title="Risk-Return Profile Comparison",
        save_path=viz_dir / "risk_return_scatter.png",
        show=False
    )
    
    # Generate reports
    print("\n3. Generating evaluation reports...")
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Prepare visualization paths for reports
    visualization_paths = {
        'cumulative_returns': str(viz_dir / "cumulative_returns.png"),
        'drawdown': str(viz_dir / "drawdown.png"),
        'metrics_comparison': str(viz_dir / "metrics_comparison.png"),
        'performance_dashboard': str(viz_dir / "performance_dashboard.png")
    }
    
    # Markdown report
    print("   - Generating Markdown report...")
    markdown_report = report_generator.generate_report(
        results,
        format_type='markdown',
        output_path=reports_dir / "evaluation_report.md",
        include_visualizations=True,
        visualization_paths=visualization_paths
    )
    
    # HTML report
    print("   - Generating HTML report...")
    html_report = report_generator.generate_report(
        results,
        format_type='html',
        output_path=reports_dir / "evaluation_report.html",
        include_visualizations=True,
        visualization_paths=visualization_paths
    )
    
    # PDF report (if wkhtmltopdf is installed)
    print("   - Attempting to generate PDF report...")
    try:
        pdf_report = report_generator.generate_report(
            results,
            format_type='pdf',
            output_path=reports_dir / "evaluation_report.pdf",
            include_visualizations=True,
            visualization_paths=visualization_paths
        )
        print("     ✓ PDF report generated successfully")
    except Exception as e:
        print(f"     ✗ PDF generation failed: {e}")
        print("     (Install wkhtmltopdf to enable PDF generation)")
    
    # Summary report for multiple evaluations
    print("\n4. Generating summary report for multiple models...")
    summary_report = report_generator.create_summary_report(
        results_list,
        output_path=reports_dir / "models_summary.md",
        format_type='markdown'
    )
    
    # Learning curves example (if training history available)
    print("\n5. Generating learning curves...")
    # Simulate training history
    episodes = 1000
    training_history = {
        'episode_reward': np.cumsum(np.random.normal(0.1, 0.5, episodes)),
        'loss': np.exp(-np.linspace(0, 3, episodes)) + np.random.normal(0, 0.01, episodes),
        'learning_rate': np.linspace(0.001, 0.0001, episodes)
    }
    
    fig = visualizer.plot_learning_curves(
        training_history,
        metrics=['episode_reward', 'loss'],
        title="Training Progress",
        save_path=viz_dir / "learning_curves.png",
        show=False
    )
    
    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  Visualizations:")
    for viz_file in viz_dir.glob("*.png"):
        print(f"    - {viz_file.name}")
    print("  Reports:")
    for report_file in reports_dir.glob("*"):
        print(f"    - {report_file.name}")
    
    # Display sample of the markdown report
    print("\n=== Sample of Markdown Report ===")
    print(markdown_report[:500] + "...\n")


if __name__ == "__main__":
    main()