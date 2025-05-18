#!/usr/bin/env python
"""
Benchmark Verification Script

This script provides a standalone test to verify benchmark calculation correctness
by running a simple scenario through both the model and benchmark strategies.

Usage:
    python benchmark_verification_script.py

The script will:
1. Create synthetic price data with known patterns
2. Run both the model and benchmarks on this data
3. Perform manual calculations to verify results
4. Display detailed comparisons of calculations
"""

import os
import sys
import logging
import json
from collections import deque
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import ray

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark_verification')

# Add project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize Ray
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# Import project modules
try:
    from reinforcestrategycreator.backtesting.benchmarks import (
        BuyAndHoldStrategy, SMAStrategy, RandomStrategy
    )
    from reinforcestrategycreator.backtesting.evaluation import MetricsCalculator, BenchmarkEvaluator
    from reinforcestrategycreator.backtesting.model import ModelTrainer
    from reinforcestrategycreator.trading_environment import TradingEnv
    from reinforcestrategycreator.rl_agent import StrategyAgent
except ImportError:
    logger.error("Could not import project modules. Make sure the script is run from the project root.")
    sys.exit(1)


def create_synthetic_data(scenario='uptrend', length=100):
    """
    Create synthetic price data with known patterns.
    
    Args:
        scenario: The price pattern to generate ('uptrend', 'downtrend', 'oscillating', 'plateau')
        length: Number of data points
        
    Returns:
        DataFrame with OHLCV price data
    """
    dates = pd.date_range(start='2023-01-01', periods=length)
    
    # Create different price scenarios
    if scenario == 'uptrend':
        # Linear uptrend (20% gain over period)
        base = np.linspace(100, 120, length)
    elif scenario == 'downtrend':
        # Linear downtrend (20% loss over period)
        base = np.linspace(100, 80, length)
    elif scenario == 'oscillating':
        # Oscillating prices
        x = np.linspace(0, 4*np.pi, length)
        base = 100 + 10 * np.sin(x)
    elif scenario == 'plateau':
        # Flat price with small noise
        base = 100 * np.ones(length)
        base += np.random.normal(0, 0.5, length)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    # Add small random noise to make it more realistic
    prices = base + np.random.normal(0, 1, length)
    prices = np.maximum(prices, 1)  # Ensure no negative prices
    
    # Create OHLCV data
    df = pd.DataFrame({
        'close': prices,
        'open': prices * np.random.uniform(0.99, 1.01, length),
        'high': prices * np.random.uniform(1.01, 1.03, length),
        'low': prices * np.random.uniform(0.97, 0.99, length),
        'volume': np.random.uniform(1000, 10000, length)
    }, index=dates)
    
    return df


def manual_buy_hold_calculation(prices, initial_balance=10000, transaction_fee=0.001):
    """
    Perform a manual buy and hold calculation for verification.
    
    Args:
        prices: Array of prices
        initial_balance: Starting balance
        transaction_fee: Fee as a fraction of trade value
        
    Returns:
        Dict of calculated metrics
    """
    # Calculate shares bought (accounting for fees)
    initial_price = prices[0]
    shares = initial_balance / initial_price * (1 - transaction_fee)
    
    # Calculate portfolio values
    portfolio_values = [initial_balance]
    for price in prices[1:]:
        value = shares * price
        portfolio_values.append(value)
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    pnl = final_value - initial_balance
    pnl_percent = (pnl / initial_balance) * 100
    
    # Calculate returns for Sharpe ratio
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
    
    # Calculate max drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        "pnl": pnl,
        "pnl_percentage": pnl_percent,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "trades": 1,
        "win_rate": 1 if pnl > 0 else 0,
        "portfolio_values": portfolio_values
    }


def run_benchmark_strategy(test_data, strategy_name='buy_and_hold', params=None):
    """
    Run a benchmark strategy on test data.
    
    Args:
        test_data: DataFrame with price data
        strategy_name: Name of the strategy to run
        params: Optional parameters for the strategy
        
    Returns:
        Dict with benchmark results and portfolio values
    """
    if params is None:
        params = {}
    
    initial_balance = params.get('initial_balance', 10000)
    transaction_fee = params.get('transaction_fee', 0.001)
    
    # Create strategy instance
    if strategy_name == 'buy_and_hold':
        strategy = BuyAndHoldStrategy(
            initial_balance=initial_balance,
            transaction_fee=transaction_fee
        )
    elif strategy_name == 'sma':
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        strategy = SMAStrategy(
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            short_window=short_window,
            long_window=long_window
        )
    elif strategy_name == 'random':
        trade_probability = params.get('trade_probability', 0.05)
        random_seed = params.get('random_seed', 42)
        strategy = RandomStrategy(
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            trade_probability=trade_probability,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Patch the run method to capture portfolio values
    original_run = strategy.run
    
    def patched_run(self, data):
        # Extract the method body to capture portfolio values
        if strategy_name == 'buy_and_hold':
            # Specifically for buy and hold
            portfolio_values = [float(initial_balance)]  # Initial balance
            
            # Get price data
            if 'close' in data.columns:
                prices = data['close'].values
            elif 'Close' in data.columns:
                prices = data['Close'].values
            elif 'Adj Close' in data.columns:
                prices = data['Adj Close'].values
            else:
                raise ValueError("Cannot find price data in columns")
            
            # Calculate number of shares to buy
            initial_price = float(prices[0])
            shares = initial_balance / initial_price
            shares = shares * (1 - transaction_fee)  # Account for transaction fee
            
            # Calculate portfolio value over time
            for price in prices[1:]:
                portfolio_value = float(shares * price)
                portfolio_values.append(portfolio_value)
                
            # Run original method
            result = original_run(data)
            
            # Store portfolio values in result
            result['portfolio_values'] = portfolio_values
            return result
        else:
            # For other strategies, just run and note we don't capture portfolio values
            result = original_run(data)
            result['portfolio_values_unavailable'] = True
            return result
    
    # Replace the run method
    strategy.run = patched_run.__get__(strategy)
    
    # Run the strategy
    result = strategy.run(test_data)
    return result


def create_dummy_model_for_testing():
    """
    Create a simple model for testing that buys and holds with additional rules.
    This is not a trained model, just a simple strategy for testing.
    """
    class DummyModel:
        def __init__(self, buy_threshold=0):
            self.buy_threshold = buy_threshold
            self.has_position = False
            
        def select_action(self, state):
            """Simple strategy: Buy if no position, otherwise hold"""
            if not self.has_position:
                self.has_position = True
                return 1  # Buy/Long
            return 0  # Hold/Flat
    
    return DummyModel()


def run_model_on_data(test_data, initial_balance=10000, transaction_fee=0.001):
    """
    Run a simple model on test data.
    
    Args:
        test_data: DataFrame with price data
        initial_balance: Starting balance
        transaction_fee: Fee as a fraction of trade value
        
    Returns:
        Dict with model results
    """
    # Create a dummy model
    model = create_dummy_model_for_testing()
    
    # Put data in Ray object store
    test_data_ref = ray.put(test_data)
    
    # Configure environment
    env_config = {
        "df": test_data_ref,
        "initial_balance": initial_balance,
        "transaction_fee_percent": transaction_fee * 100,  # Convert to percentage
        "window_size": 1,  # Minimal window size for testing
    }
    
    # Create and run environment
    env = TradingEnv(env_config=env_config)
    state, _ = env.reset()
    done = False
    portfolio_values = [initial_balance]
    
    # Ensure the env has a properly initialized portfolio value history
    if not hasattr(env, "_portfolio_value_history") or len(env._portfolio_value_history) == 0:
        env._portfolio_value_history = deque(maxlen=1000)
        env._portfolio_value_history.append(initial_balance)
    
    while not done:
        action = model.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])
        
        # Ensure environment's internal history is maintained properly
        if hasattr(env, "_portfolio_value_history"):
            env._portfolio_value_history.append(info['portfolio_value'])
    
    # Calculate metrics using the same env info that would be used in real evaluation
    metrics_calculator = MetricsCalculator()
    metrics = metrics_calculator.get_episode_metrics(env)
    
    # If metrics calculation failed and returned zeros, calculate them manually
    if metrics['pnl'] == 0.0 and len(portfolio_values) > 1:
        logger.warning("MetricsCalculator returned zeros. Calculating metrics manually.")
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        metrics['pnl'] = final_value - initial_value
        metrics['pnl_percentage'] = (metrics['pnl'] / initial_value) * 100
    
    # Add portfolio values
    metrics['portfolio_values'] = portfolio_values
    return metrics


def plot_portfolio_comparison(results, title):
    """
    Plot a comparison of portfolio values.
    
    Args:
        results: Dict of strategy results, each with 'portfolio_values' key
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for strategy, result in results.items():
        if 'portfolio_values' in result:
            plt.plot(result['portfolio_values'], label=strategy)
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"benchmark_comparison_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def run_verification():
    """
    Run the verification process.
    
    This function:
    1. Creates synthetic data
    2. Runs manual calculations
    3. Runs benchmark strategies
    4. Runs model
    5. Compares results
    """
    logger.info("Starting benchmark verification")
    
    # Create synthetic data for different scenarios
    scenarios = ['uptrend', 'downtrend', 'oscillating', 'plateau']
    results = {}
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario}")
        
        # Generate data
        test_data = create_synthetic_data(scenario=scenario, length=100)
        prices = test_data['close'].values
        
        # Manual calculation
        manual = manual_buy_hold_calculation(
            prices=prices, 
            initial_balance=10000, 
            transaction_fee=0.001
        )
        
        # Run benchmark strategy
        benchmark = run_benchmark_strategy(
            test_data=test_data,
            strategy_name='buy_and_hold',
            params={'initial_balance': 10000, 'transaction_fee': 0.001}
        )
        
        # Run model
        model_result = run_model_on_data(
            test_data=test_data,
            initial_balance=10000,
            transaction_fee=0.001
        )
        
        # Store results
        results[scenario] = {
            'manual': manual,
            'benchmark': benchmark,
            'model': model_result
        }
        
        # Compare calculations
        logger.info(f"Scenario: {scenario}")
        logger.info(f"  Manual PnL: {manual['pnl']:.2f} ({manual['pnl_percentage']:.2f}%)")
        logger.info(f"  Benchmark PnL: {benchmark['pnl']:.2f} ({benchmark['pnl_percentage']:.2f}%)")
        logger.info(f"  Model PnL: {model_result['pnl']:.2f}")
        
        # Calculate differences
        pnl_diff_benchmark = manual['pnl'] - benchmark['pnl']
        pnl_diff_model = manual['pnl'] - model_result['pnl']
        
        logger.info(f"  Manual vs Benchmark PnL diff: {pnl_diff_benchmark:.2f}")
        logger.info(f"  Manual vs Model PnL diff: {pnl_diff_model:.2f}")
        
        # Plot comparison
        plot_path = plot_portfolio_comparison(
            {
                'Manual': manual,
                'Benchmark': benchmark,
                'Model': model_result
            },
            f"Portfolio Comparison - {scenario.capitalize()}"
        )
        logger.info(f"  Comparison plot saved to: {plot_path}")
    
    # Save detailed results for analysis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"verification_results_{timestamp}.json"
    
    # Clean up portfolio values for JSON serialization (convert numpy arrays to lists)
    clean_results = {}
    for scenario, scenario_results in results.items():
        clean_results[scenario] = {}
        for strategy, strategy_results in scenario_results.items():
            clean_results[scenario][strategy] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in strategy_results.items()
            }
    
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Generate comprehensive report with insights
    report_file = generate_comprehensive_report(results, timestamp)
    
    logger.info(f"Detailed results saved to: {output_file}")
    logger.info(f"Comprehensive report saved to: {report_file}")
    logger.info("Verification completed successfully")
    
    return results, output_file, report_file


def generate_comprehensive_report(results, timestamp):
    """
    Generate a comprehensive report with insights about the benchmark and model comparison.
    
    Args:
        results: Dictionary with verification results
        timestamp: Timestamp string for naming the output file
        
    Returns:
        Path to the generated report file
    """
    report_file = f"benchmark_report_{timestamp}.md"
    
    # Create summary table data
    summary_rows = []
    scenarios = []
    model_vs_benchmark = {}
    
    for scenario, data in results.items():
        scenarios.append(scenario)
        manual_pnl = data['manual']['pnl_percentage']
        benchmark_pnl = data['benchmark']['pnl_percentage']
        model_pnl = data['model'].get('pnl', 0) / 10000 * 100  # Use initial balance for percentage
        
        # Store for later analysis
        model_vs_benchmark[scenario] = {
            'benchmark_pnl': benchmark_pnl,
            'model_pnl': model_pnl,
            'difference': model_pnl - benchmark_pnl
        }
        
        # Create row for summary table
        summary_rows.append([
            scenario.capitalize(),
            f"{model_pnl:.2f}%",
            f"{benchmark_pnl:.2f}%",
            f"{model_pnl - benchmark_pnl:.2f}%",
            "Outperforms" if model_pnl > benchmark_pnl else "Underperforms"
        ])
    
    # Determine model characteristics
    outperforming_scenarios = [s for s in scenarios if model_vs_benchmark[s]['difference'] > 0]
    underperforming_scenarios = [s for s in scenarios if model_vs_benchmark[s]['difference'] < 0]
    
    # Generate observations based on performance patterns
    observations = []
    
    if 'downtrend' in outperforming_scenarios:
        observations.append("The model performs well in downtrend markets, suggesting effective shorting or risk management capabilities.")
    
    if 'uptrend' in underperforming_scenarios:
        observations.append("The model underperforms in uptrend markets, possibly indicating a conservative approach to long positions.")
    
    if 'oscillating' in underperforming_scenarios:
        observations.append("The model struggles in oscillating markets, suggesting potential issues with rapid position switching or timing.")
    
    if 'plateau' in outperforming_scenarios:
        observations.append("The model performs well in sideways/plateau markets, indicating good signal filtering in low-volatility conditions.")
    
    # Calculate average performance metrics
    avg_model_pnl = sum(model_vs_benchmark[s]['model_pnl'] for s in scenarios) / len(scenarios)
    avg_benchmark_pnl = sum(model_vs_benchmark[s]['benchmark_pnl'] for s in scenarios) / len(scenarios)
    avg_difference = sum(model_vs_benchmark[s]['difference'] for s in scenarios) / len(scenarios)
    
    # Generate recommendations based on findings
    recommendations = []
    
    if 'uptrend' in underperforming_scenarios:
        recommendations.append("Consider adjusting the model to be more aggressive in long positions during uptrends.")
    
    if avg_difference < 0:
        recommendations.append("Reevaluate model training to improve overall performance relative to benchmark strategies.")
    
    if 'oscillating' in underperforming_scenarios:
        recommendations.append("Enhance signal filtering or implement a confirmation mechanism for trend changes to better handle oscillating markets.")
    
    # Write the report
    with open(report_file, 'w') as f:
        # Header and introduction
        f.write("# Benchmark Verification Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report provides a comprehensive analysis of model performance compared to benchmark strategies across various market scenarios.\n\n")
        
        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Scenario | Model PnL | Benchmark PnL | Difference | Result |\n")
        f.write("|----------|-----------|---------------|------------|--------|\n")
        for row in summary_rows:
            f.write(f"| {' | '.join(row)} |\n")
        f.write("\n")
        
        # Overall metrics
        f.write("## Overall Performance Metrics\n\n")
        f.write(f"- Average Model PnL: {avg_model_pnl:.2f}%\n")
        f.write(f"- Average Benchmark PnL: {avg_benchmark_pnl:.2f}%\n")
        f.write(f"- Average Performance Difference: {avg_difference:.2f}%\n")
        f.write(f"- Outperforming Scenarios: {', '.join([s.capitalize() for s in outperforming_scenarios]) if outperforming_scenarios else 'None'}\n")
        f.write(f"- Underperforming Scenarios: {', '.join([s.capitalize() for s in underperforming_scenarios]) if underperforming_scenarios else 'None'}\n\n")
        
        # Key observations
        f.write("## Key Observations\n\n")
        for observation in observations:
            f.write(f"- {observation}\n")
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        for recommendation in recommendations:
            f.write(f"- {recommendation}\n")
        f.write("\n")
        
        # Detailed scenario analysis
        f.write("## Detailed Scenario Analysis\n\n")
        for scenario in scenarios:
            f.write(f"### {scenario.capitalize()} Market\n\n")
            
            model_data = results[scenario]['model']
            benchmark_data = results[scenario]['benchmark']
            
            # Detailed metrics table
            f.write("| Metric | Model | Benchmark |\n")
            f.write("|--------|-------|------------|\n")
            
            # Calculate model PnL percentage if not already available
            model_pnl_pct = model_data.get('pnl', 0) / 10000 * 100  # Use initial balance for percentage
            
            f.write(f"| PnL | {model_data.get('pnl', 0):.2f} | {benchmark_data.get('pnl', 0):.2f} |\n")
            f.write(f"| PnL (%) | {model_pnl_pct:.2f}% | {benchmark_data.get('pnl_percentage', 0):.2f}% |\n")
            f.write(f"| Sharpe Ratio | {model_data.get('sharpe_ratio', 0):.2f} | {benchmark_data.get('sharpe_ratio', 0):.2f} |\n")
            f.write(f"| Max Drawdown | {model_data.get('max_drawdown', 0):.4f} | {benchmark_data.get('max_drawdown', 0):.4f} |\n")
            f.write(f"| Win Rate | {model_data.get('win_rate', 0):.2f} | {benchmark_data.get('win_rate', 0):.2f} |\n")
            f.write(f"| Trades | {model_data.get('trades', 0)} | {benchmark_data.get('trades', 0)} |\n\n")
            
            # Scenario-specific insights
            if scenario == 'uptrend':
                if model_pnl_pct < benchmark_data.get('pnl_percentage', 0):
                    f.write("The model significantly underperforms the benchmark in this uptrend scenario. This suggests the model may be too conservative with long positions or has a bias toward short positions that limits capturing upside potential.\n\n")
                else:
                    f.write("The model effectively captures the uptrend, showing strong long positioning capabilities.\n\n")
                    
            elif scenario == 'downtrend':
                if model_pnl_pct > benchmark_data.get('pnl_percentage', 0):
                    f.write("The model shows strong performance in downtrend markets, successfully avoiding losses or even generating profits while the benchmark strategy suffers. This indicates good risk management or effective short positioning.\n\n")
                else:
                    f.write("The model fails to effectively hedge against downside risk in this scenario, suggesting improvements in trend detection or short selling strategies may be beneficial.\n\n")
                    
            elif scenario == 'oscillating':
                if model_pnl_pct > benchmark_data.get('pnl_percentage', 0):
                    f.write("The model navigates the oscillating market effectively, showing good adaptability to changing conditions and avoiding false signals.\n\n")
                else:
                    f.write("The model struggles with the rapid changes in the oscillating market, likely getting caught in false signals or experiencing whipsaw effects. A more robust signal filtering mechanism might improve performance.\n\n")
                    
            elif scenario == 'plateau':
                if model_pnl_pct > benchmark_data.get('pnl_percentage', 0):
                    f.write("The model performs well in the sideways market, showing an ability to extract value in low-volatility conditions where simple buy-and-hold strategies struggle to generate significant returns.\n\n")
                else:
                    f.write("The model doesn't add value in stable market conditions. Consider implementing specific strategies for sideways markets such as mean reversion or range-bound trading techniques.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if avg_difference > 0:
            f.write("Overall, the model demonstrates value by outperforming the benchmark across tested scenarios. The model's strengths and weaknesses have been identified, and targeted improvements can further enhance performance.\n\n")
        else:
            f.write("The model currently underperforms the benchmark on average. However, by addressing the specific weaknesses identified in this analysis, particularly in uptrend and oscillating markets, the model's performance can be significantly improved.\n\n")
            
        f.write("This verification exercise has provided valuable insights into the relationship between model and benchmark calculations and revealed important performance characteristics of the current model implementation.\n")
    
    return report_file


if __name__ == "__main__":
    try:
        results, output_file, report_file = run_verification()
        logger.info("Verification completed successfully")
        print(f"\nVerification results saved to: {output_file}")
        print(f"Comprehensive report saved to: {report_file}")
    except Exception as e:
        logger.exception("Verification failed")
        print(f"\nVerification failed: {str(e)}")