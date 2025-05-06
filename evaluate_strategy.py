import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_agent(model_path: str):
    """
    Loads the RL agent from the specified path.
    Placeholder for actual agent loading logic.
    """
    logger.info(f"Loading agent from: {model_path}")
    try:
        from reinforcestrategycreator.rl_agent import RLAgent
        # Assuming RLAgent has a static load method
        # This might need adjustment based on actual RLAgent implementation
        # For example, if it expects a directory or specific file names within a path
        agent = RLAgent.load_model(model_path) # Or RLAgent.load(model_path)
        logger.info(f"Agent loaded successfully from {model_path}")
        return agent
    except ImportError:
        logger.error("Could not import RLAgent. Ensure reinforcestrategycreator is installed and in PYTHONPATH.")
        return None
    except Exception as e:
        logger.error(f"Error loading agent from {model_path}: {e}")
        return None


def backtest_agent(agent, start_date_str: str, end_date_str: str):
    """
    Backtests the loaded RL agent over the specified period.
    Placeholder for actual backtesting logic.
    """
    logger.info(f"Backtesting agent from {start_date_str} to {end_date_str}")
    try:
        from reinforcestrategycreator.trading_environment import TradingEnvironment
        from reinforcestrategycreator.data_fetcher import DataFetcher
        from reinforcestrategycreator.config import LOG_LEVEL # Assuming a config file for common params

        # It's generally better to fetch data once if the agent and benchmark use the same underlying asset.
        # However, the task implies separate fetching steps. For now, let's assume SPY data for the agent's environment.
        # This might need adjustment if the agent is trained on a different asset or feature set.
        data_fetcher = DataFetcher()
        # The environment likely expects a DataFrame with specific columns (Open, High, Low, Close, Volume, etc.)
        # and a DatetimeIndex.
        # The symbol used here should ideally be configurable or derived from the agent's training.
        # For now, using 'SPY' as a placeholder, assuming the agent trades SPY.
        historical_data_df = data_fetcher.fetch_data_for_env("SPY", start_date_str, end_date_str)

        if historical_data_df is None or historical_data_df.empty:
            logger.error(f"Failed to fetch historical data for agent backtest (SPY, {start_date_str}-{end_date_str}).")
            return [], []

        # Ensure 'balance' is a valid attribute or use 'initial_balance'
        env = TradingEnvironment(
            data_df=historical_data_df,
            initial_balance=100000,  # Standard initial balance
            # Default values from TradingEnvironment, adjust if needed for evaluation
            # window_size=50, # Example, ensure it matches agent's expected observation space
            # fee_bps=10, # Example transaction fee in basis points
            # ticker="SPY", # Example
            # features=None, # Example, ensure it matches agent's expected observation space
            # episode_max_steps=None, # For evaluation, run through all data
            # log_level=LOG_LEVEL
        )
        logger.info(f"TradingEnvironment initialized for agent backtest with data from {env.data_df.index.min()} to {env.data_df.index.max()}")

        obs, info = env.reset()
        done = False
        portfolio_values = [env.initial_balance] # Start with initial balance
        trades = []
        step_count = 0

        while not done:
            step_count += 1
            # Assuming agent has a 'predict' or 'get_action' method that takes an observation
            # and returns an action compatible with the environment's action space.
            # The 'deterministic=True' flag is common for evaluation.
            action = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_val = env.current_portfolio_value() # Ensure this method exists and is correct
            portfolio_values.append(current_val)

            if "trade" in info and info["trade"]: # Assuming 'trade' key holds trade details
                trades.append(info["trade"])
            
            if step_count % 100 == 0: # Log progress periodically
                 logger.debug(f"Backtest step {step_count}, Portfolio: {current_val:.2f}")


        logger.info(f"Agent backtest completed. Total steps: {step_count}. Final portfolio value: {portfolio_values[-1]:.2f}")
        return portfolio_values, trades

    except ImportError as e:
        logger.error(f"Could not import necessary modules for backtesting: {e}")
        return [], []
    except Exception as e:
        logger.error(f"Error during agent backtesting: {e}", exc_info=True)
        return [], []


def calculate_metrics(portfolio_values, trades, risk_free_rate: float):
    """
    Calculates performance metrics from backtest results.
    Placeholder for actual metrics calculation.
    """
    logger.info("Calculating agent performance metrics...")
    if not portfolio_values:
        logger.warning("Portfolio values list is empty. Cannot calculate metrics for agent.")
        return 0.0, 0.0, 0.0
    try:
        from reinforcestrategycreator.metrics_calculator import MetricsCalculator

        # Assuming daily portfolio values. Adjust frequency if different for Sharpe.
        metrics_calculator = MetricsCalculator(risk_free_rate=risk_free_rate, trading_days_per_year=252)

        total_return_pct = metrics_calculator.calculate_total_return_pct(portfolio_values)
        sharpe_ratio = metrics_calculator.calculate_sharpe_ratio_from_portfolio_values(portfolio_values)
        max_drawdown_pct = metrics_calculator.calculate_max_drawdown_pct(portfolio_values)

        logger.info(f"Agent Metrics: Total Return={total_return_pct:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown_pct:.2f}%")
        return total_return_pct, sharpe_ratio, max_drawdown_pct
    except ImportError:
        logger.error("Could not import MetricsCalculator. Ensure reinforcestrategycreator is installed.")
        return 0.0, 0.0, 0.0
    except Exception as e:
        logger.error(f"Error calculating agent metrics: {e}", exc_info=True)
        return 0.0, 0.0, 0.0


def fetch_benchmark_data(symbol: str, start_date_str: str, end_date_str: str):
    """
    Fetches historical data for the benchmark symbol.
    Placeholder for actual benchmark data fetching.
    """
    logger.info(f"Fetching benchmark data for {symbol} from {start_date_str} to {end_date_str}")
    try:
        from reinforcestrategycreator.data_fetcher import DataFetcher
        data_fetcher = DataFetcher()
        # fetch_data usually returns a DataFrame. We need 'Close' prices for benchmark calculation.
        benchmark_df = data_fetcher.fetch_data(symbol, start_date_str, end_date_str, timeframe="1D")
        if benchmark_df is None or benchmark_df.empty:
            logger.error(f"Failed to fetch benchmark data for {symbol} ({start_date_str}-{end_date_str}).")
            return []
        
        # Ensure 'Close' column exists
        if "Close" not in benchmark_df.columns:
            logger.error(f"'Close' column not found in benchmark data for {symbol}.")
            return []
            
        logger.info(f"Successfully fetched benchmark data for {symbol} with {len(benchmark_df)} data points.")
        return benchmark_df["Close"].tolist()
    except ImportError:
        logger.error("Could not import DataFetcher. Ensure reinforcestrategycreator is installed.")
        return []
    except Exception as e:
        logger.error(f"Error fetching benchmark data for {symbol}: {e}", exc_info=True)
        return []


def calculate_benchmark_metrics(benchmark_prices, risk_free_rate: float):
    """
    Calculates performance metrics for the benchmark.
    Placeholder for actual benchmark metrics calculation.
    """
    logger.info("Calculating benchmark performance metrics...")
    if not benchmark_prices:
        logger.warning("Benchmark prices list is empty. Cannot calculate metrics for benchmark.")
        return 0.0, 0.0, 0.0
    try:
        from reinforcestrategycreator.metrics_calculator import MetricsCalculator

        # For a buy-and-hold benchmark, the list of closing prices can be treated as portfolio values.
        metrics_calculator = MetricsCalculator(risk_free_rate=risk_free_rate, trading_days_per_year=252)

        total_return_pct = metrics_calculator.calculate_total_return_pct(benchmark_prices)
        sharpe_ratio = metrics_calculator.calculate_sharpe_ratio_from_portfolio_values(benchmark_prices) # or a dedicated method if available
        max_drawdown_pct = metrics_calculator.calculate_max_drawdown_pct(benchmark_prices)

        logger.info(f"Benchmark Metrics: Total Return={total_return_pct:.2f}%, Sharpe Ratio={sharpe_ratio:.2f}, Max Drawdown={max_drawdown_pct:.2f}%")
        return total_return_pct, sharpe_ratio, max_drawdown_pct
    except ImportError:
        logger.error("Could not import MetricsCalculator. Ensure reinforcestrategycreator is installed.")
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
    """
    Displays or saves the evaluation results.
    """
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
            # elif output_file.endswith(".csv"): # Example for CSV
                # import pandas as pd
                # df_results = pd.DataFrame.from_dict({
                #     "Metric": ["Total Return %", "Sharpe Ratio", "Max Drawdown %"],
                #     "RL Agent": [agent_return, agent_sharpe, agent_drawdown],
                #     "SPY Benchmark": [bench_return, bench_sharpe, bench_drawdown]
                # })
                # df_results.to_csv(output_file, index=False)
                # logger.info(f"Results successfully saved to CSV: {output_file}")
            else:
                logger.warning(f"Unsupported output file format: {output_file}. Please use .json.")
                # Fallback to printing if format is not supported for saving
                # Or, save as JSON with a warning if extension is unknown
                # For now, only JSON is fully implemented for saving.
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL Trading Agent against SPY Benchmark"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved agent model checkpoint.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Backtest start date (YYYY-MM-DD). Default: 2020-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-12-31",
        help="Backtest end date (YYYY-MM-DD). Default: 2023-12-31",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate for Sharpe Ratio. Default: 0.0",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save results (e.g., results.json or results.csv).",
    )

    args = parser.parse_args()

    try:
        # Validate dates
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error(
            "Invalid date format. Please use YYYY-MM-DD for start-date and end-date."
        )
        return

    logger.info("Starting evaluation script...")
    logger.info(f"Parameters: {args}")

    # 1. Load Agent
    agent = load_agent(args.model_path)
    if agent is None:
        logger.error("Failed to load agent. Exiting.")
        return

    # 2. Backtest Agent
    agent_portfolio_values, agent_trades = backtest_agent(
        agent, args.start_date, args.end_date
    )

    # 3. Calculate Agent Metrics
    agent_metrics = calculate_metrics(
        agent_portfolio_values, agent_trades, args.risk_free_rate
    )

    # 4. Fetch SPY Benchmark Data
    benchmark_prices = fetch_benchmark_data("SPY", args.start_date, args.end_date)

    # 5. Calculate SPY Benchmark Metrics
    benchmark_metrics = calculate_benchmark_metrics(
        benchmark_prices, args.risk_free_rate
    )

    # 6. Output Results
    display_results(
        args.start_date, args.end_date, agent_metrics, benchmark_metrics, args.output_file
    )

    logger.info("Evaluation script finished.")


if __name__ == "__main__":
    main()