import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import argparse
import logging
from datetime import datetime
from tensorflow import keras # Ensure Keras is available
import numpy as np # For calculations if needed
import pandas as pd # For Series if needed by metrics functions

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_agent(model_path: str):
    logger.info(f"Loading agent from: {model_path}")
    try:
        from reinforcestrategycreator.rl_agent import StrategyAgent

        loaded_keras_model = keras.models.load_model(model_path)
        logger.info(f"Keras model loaded successfully from {model_path}")

        if not loaded_keras_model.inputs:
            logger.error("Loaded Keras model has no inputs defined.")
            return None
        state_size = loaded_keras_model.input_shape[-1]

        if not loaded_keras_model.outputs:
            logger.error("Loaded Keras model has no outputs defined.")
            return None
        action_size = loaded_keras_model.output_shape[-1]
        
        logger.info(f"Inferred state_size: {state_size}, action_size: {action_size} from loaded model.")

        agent_instance = StrategyAgent(state_size=state_size, action_size=action_size)
        agent_instance.model = loaded_keras_model
        agent_instance.target_model = keras.models.clone_model(loaded_keras_model)
        agent_instance.target_model.set_weights(loaded_keras_model.get_weights())
        agent_instance.epsilon = agent_instance.epsilon_min 
        
        logger.info(f"StrategyAgent instance created and configured with loaded Keras model.")
        return agent_instance
        
    except ImportError as e_import:
        logger.error(f"Could not import StrategyAgent or TensorFlow/Keras. Specific error: {e_import}. Ensure reinforcestrategycreator and tensorflow are installed and in PYTHONPATH.")
        return None
    except Exception as e:
        logger.error(f"Error loading agent from {model_path}: {e}", exc_info=True)
        return None


def backtest_agent(agent, start_date_str: str, end_date_str: str):
    logger.info(f"Backtesting agent from {start_date_str} to {end_date_str}")
    try:
        from reinforcestrategycreator.trading_environment import TradingEnv 
        from reinforcestrategycreator.data_fetcher import fetch_historical_data
        from reinforcestrategycreator.technical_analyzer import calculate_indicators # Added import

        raw_historical_data_df = fetch_historical_data("SPY", start_date_str, end_date_str)

        if raw_historical_data_df is None or raw_historical_data_df.empty:
            logger.error(f"Failed to fetch raw historical data for agent backtest (SPY, {start_date_str}-{end_date_str}).")
            return [], []
        
        logger.debug(f"Raw historical_data_df columns: {raw_historical_data_df.columns.tolist()}")
        logger.debug(f"Number of columns in raw_historical_data_df: {len(raw_historical_data_df.columns)}")

        # Add technical indicators
        historical_data_with_indicators_df = calculate_indicators(raw_historical_data_df)
        logger.debug(f"Columns after adding indicators: {historical_data_with_indicators_df.columns.tolist()}")
        logger.debug(f"Number of columns after adding indicators: {len(historical_data_with_indicators_df.columns)}")
# Lowercase columns for consistency with TradingEnv expectations
        logger.info("Lowercasing column names for consistency...")
        if isinstance(historical_data_with_indicators_df.columns, pd.MultiIndex):
             # Handle MultiIndex specifically if present after indicators
             try:
                 historical_data_with_indicators_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in historical_data_with_indicators_df.columns]
             except IndexError:
                 logger.warning("Could not access first element of MultiIndex tuple for all columns during lowercasing. Attempting simple lowercase.")
                 historical_data_with_indicators_df.columns = [str(col).lower() for col in historical_data_with_indicators_df.columns]
        else:
             # Handle regular index
             historical_data_with_indicators_df.columns = [col.lower() for col in historical_data_with_indicators_df.columns]
        logger.info(f"Renamed columns: {list(historical_data_with_indicators_df.columns)}")


        env = TradingEnv( 
            df=historical_data_with_indicators_df, # Use data with indicators
            initial_balance=100000,
        )
        logger.info(f"TradingEnv initialized for agent backtest with data from {env.df.index.min()} to {env.df.index.max()}")

        obs, info = env.reset()
        done = False
        portfolio_values = [env.initial_balance]
        trades = []
        step_count = 0

        while not done:
            step_count += 1
            action = agent.get_action(obs) 
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_val = env.portfolio_value 
            portfolio_values.append(current_val)

            if "trade" in info and info["trade"]:
                trades.append(info["trade"])
            
            if step_count % 100 == 0:
                 logger.debug(f"Backtest step {step_count}, Portfolio: {current_val:.2f}")

        logger.info(f"Agent backtest completed. Total steps: {step_count}. Final portfolio value: {portfolio_values[-1]:.2f}")
        return portfolio_values, trades

    except ImportError as e:
        logger.error(f"Could not import necessary modules for backtesting: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Error during agent backtesting: {e}", exc_info=True)
        return [], []

def _calculate_total_return_pct(portfolio_values: list) -> float:
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    if initial_value == 0:
        return 0.0
    return ((final_value - initial_value) / initial_value) * 100.0

def calculate_metrics(portfolio_values, trades, risk_free_rate: float):
    logger.info("Calculating agent performance metrics...")
    if not portfolio_values:
        logger.warning("Portfolio values list is empty. Cannot calculate metrics for agent.")
        return 0.0, 0.0, 0.0
    try:
        from reinforcestrategycreator.metrics_calculator import calculate_sharpe_ratio, calculate_max_drawdown
        
        if len(portfolio_values) < 2:
            daily_returns = pd.Series(dtype=float) 
        else:
            pv_series = pd.Series(portfolio_values)
            daily_returns = pv_series.pct_change().dropna()

        total_return_pct = _calculate_total_return_pct(portfolio_values)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
        max_drawdown_pct = calculate_max_drawdown(portfolio_values) * 100.0 

        logger.info(f"Agent Metrics: Total Return={total_return_pct:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown_pct:.2f}%")
        return total_return_pct, sharpe_ratio, max_drawdown_pct
    except ImportError as e: 
        logger.error(f"Could not import functions from reinforcestrategycreator.metrics_calculator. Error: {e}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        logger.error(f"Error calculating agent metrics: {e}", exc_info=True)
        return 0.0, 0.0, 0.0


def fetch_benchmark_data(symbol: str, start_date_str: str, end_date_str: str):
    logger.info(f"Fetching benchmark data for {symbol} from {start_date_str} to {end_date_str}")
    try:
        from reinforcestrategycreator.data_fetcher import fetch_historical_data 
        benchmark_df = fetch_historical_data(symbol, start_date_str, end_date_str)
        if benchmark_df is None or benchmark_df.empty:
            logger.error(f"Failed to fetch benchmark data for {symbol} ({start_date_str}-{end_date_str}).")
            return []
        
        close_col_present = False
        if isinstance(benchmark_df.columns, pd.MultiIndex):
            if ('Close', symbol) in benchmark_df.columns:
                close_col_present = True
        elif "Close" in benchmark_df.columns:
            close_col_present = True

        if not close_col_present:
            logger.error(f"'Close' column not found for {symbol}. Columns: {benchmark_df.columns}")
            return []
        
        logger.info(f"Successfully fetched benchmark data for {symbol} with {len(benchmark_df)} data points.")
        logger.debug(f"Type of benchmark_df: {type(benchmark_df)}")
        logger.debug(f"benchmark_df columns: {benchmark_df.columns}")
        
        if isinstance(benchmark_df.columns, pd.MultiIndex):
            close_series = benchmark_df[('Close', symbol)]
        else:
            close_series = benchmark_df["Close"]
            
        logger.debug(f"Type of close_series: {type(close_series)}")
        logger.debug(f"First 5 values of close_series: \n{close_series.head()}")
        return close_series.tolist()
    except ImportError as e: 
        logger.error(f"Could not import fetch_historical_data. Ensure reinforcestrategycreator.data_fetcher defines it. Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching benchmark data for {symbol}: {e}", exc_info=True)
        return []


def calculate_benchmark_metrics(benchmark_prices, risk_free_rate: float):
    logger.info("Calculating benchmark performance metrics...")
    if not benchmark_prices: 
        logger.warning("Benchmark prices list is empty. Cannot calculate metrics for benchmark.")
        return 0.0, 0.0, 0.0
    try:
        from reinforcestrategycreator.metrics_calculator import calculate_sharpe_ratio, calculate_max_drawdown
        
        if len(benchmark_prices) < 2:
            benchmark_returns = pd.Series(dtype=float)
        else:
            price_series = pd.Series(benchmark_prices)
            benchmark_returns = price_series.pct_change().dropna()

        total_return_pct = _calculate_total_return_pct(benchmark_prices)
        sharpe_ratio = calculate_sharpe_ratio(benchmark_returns, risk_free_rate=risk_free_rate)
        max_drawdown_pct = calculate_max_drawdown(benchmark_prices) * 100.0 

        logger.info(f"Benchmark Metrics: Total Return={total_return_pct:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown_pct:.2f}%")
        return total_return_pct, sharpe_ratio, max_drawdown_pct
    except ImportError as e: 
        logger.error(f"Could not import functions from reinforcestrategycreator.metrics_calculator. Error: {e}")
        return 0.0, 0.0, 0.0
    except Exception as e:
        logger.error(f"Error calculating benchmark metrics: {e}", exc_info=True)
        return 0.0, 0.0, 0.0


def display_results(
    start_date: str,
    end_date: str,
    agent_metrics: tuple,
    benchmark_metrics: tuple,
    output_file: str = None,
):
    agent_return, agent_sharpe, agent_drawdown = agent_metrics
    bench_return, bench_sharpe, bench_drawdown = benchmark_metrics
    results_str = f"""
Evaluation Period: {start_date} to {end_date}

RL Agent Performance:
  Total Return: {agent_return:.2f}%
  Sharpe Ratio: {agent_sharpe:.2f}
  Max Drawdown: {agent_drawdown:.2f}%

SPY Benchmark Performance:
  Total Return: {bench_return:.2f}%
  Sharpe Ratio: {bench_sharpe:.2f}
  Max Drawdown: {bench_drawdown:.2f}%
"""
    logger.info(results_str)
    if output_file:
        logger.info(f"Saving results to {output_file}")
        results_data = {
            "evaluation_period": {"start_date": start_date, "end_date": end_date},
            "rl_agent_performance": {
                "total_return_pct": agent_return,
                "sharpe_ratio": agent_sharpe,
                "max_drawdown_pct": agent_drawdown,
            },
            "spy_benchmark_performance": {
                "total_return_pct": bench_return,
                "sharpe_ratio": bench_sharpe,
                "max_drawdown_pct": bench_drawdown,
            },
        }
        try:
            if output_file.endswith(".json"):
                import json
                with open(output_file, 'w') as f:
                    json.dump(results_data, f, indent=4)
                logger.info(f"Results successfully saved to JSON: {output_file}")
            else:
                logger.warning(f"Unsupported output file format: {output_file}. Please use .json.")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL Trading Agent against SPY Benchmark"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the saved agent model checkpoint."
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01", help="Backtest start date (YYYY-MM-DD). Default: 2020-01-01"
    )
    parser.add_argument(
        "--end-date", type=str, default="2023-12-31", help="Backtest end date (YYYY-MM-DD). Default: 2023-12-31"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.0, help="Annualized risk-free rate for Sharpe Ratio. Default: 0.0"
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Optional path to save results (e.g., results.json or results.csv)."
    )
    args = parser.parse_args()
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD for start-date and end-date.")
        return

    logger.info("Starting evaluation script...")
    logger.info(f"Parameters: {args}")

    agent = load_agent(args.model_path)
    if agent is None:
        logger.error("Failed to load agent. Exiting.")
        return

    if hasattr(agent, 'select_action') and not hasattr(agent, 'get_action'):
        agent.get_action = agent.select_action
        logger.info("Aliased agent.get_action to agent.select_action for compatibility.")
    elif not hasattr(agent, 'get_action'):
        logger.error("Loaded agent object does not have a get_action or select_action method.")
        return

    agent_portfolio_values, agent_trades = backtest_agent(
        agent, args.start_date, args.end_date
    )
    agent_metrics = calculate_metrics(
        agent_portfolio_values, agent_trades, args.risk_free_rate
    )
    benchmark_prices = fetch_benchmark_data("SPY", args.start_date, args.end_date)
    benchmark_metrics = calculate_benchmark_metrics(
        benchmark_prices, args.risk_free_rate
    )
    display_results(
        args.start_date, args.end_date, agent_metrics, benchmark_metrics, args.output_file
    )
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()