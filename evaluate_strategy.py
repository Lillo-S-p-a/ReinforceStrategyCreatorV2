import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import argparse
import logging
from datetime import datetime
from tensorflow import keras # Ensure Keras is available
import numpy as np # For calculations if needed
import pandas as pd # For Series if needed by metrics functions
import json # Added for config loading

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config variable
CONFIG = {}

def load_config_eval(config_path):
    """Loads configuration from a JSON file for evaluation."""
    global CONFIG
    try:
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {config_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}")
        raise

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


def backtest_agent(agent, ticker: str, start_date_str: str, end_date_str: str, transaction_cost_pct: float, initial_balance: float, env_config: dict):
    logger.info(f"Backtesting agent for {ticker} from {start_date_str} to {end_date_str} with transaction cost {transaction_cost_pct*100:.3f}%")
    try:
        from reinforcestrategycreator.trading_environment import TradingEnv
        from reinforcestrategycreator.data_fetcher import fetch_historical_data
        from reinforcestrategycreator.technical_analyzer import calculate_indicators

        raw_historical_data_df = fetch_historical_data(ticker, start_date_str, end_date_str)

        if raw_historical_data_df is None or raw_historical_data_df.empty:
            logger.error(f"Failed to fetch raw historical data for agent backtest ({ticker}, {start_date_str}-{end_date_str}).")
            return [], [], pd.Series(dtype='float64'), [] # Return empty daily returns and dates
        
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


        # Prepare TradingEnv parameters from env_config (subset of CONFIG['env_params'])
        env_init_params = {
            'df': historical_data_with_indicators_df,
            'initial_balance': initial_balance,
            'transaction_fee_percent': transaction_cost_pct, # Crucial: use from config
            'sharpe_window_size': env_config.get('sharpe_window_size', 100),
            'drawdown_penalty': env_config.get('drawdown_penalty', 0.05),
            'trading_frequency_penalty': env_config.get('trading_frequency_penalty', 0.0001),
            'risk_fraction': env_config.get('risk_fraction', 0.1),
            'stop_loss_pct': env_config.get('stop_loss_pct', 5.0),
            'take_profit_pct': env_config.get('take_profit_pct', None), # Add if in config
            'use_sharpe_ratio': env_config.get('use_sharpe_ratio_reward', True) # Add if in config
        }

        env = TradingEnv(**env_init_params)
        logger.info(f"TradingEnv initialized for agent backtest with data from {env.df.index.min()} to {env.df.index.max()}")

        obs, info = env.reset()
        done = False
        portfolio_values = [env.initial_balance]
        # Store daily portfolio values for metrics calculation
        daily_portfolio_values_for_metrics = pd.Series(index=historical_data_with_indicators_df.index, dtype=float)
        daily_portfolio_values_for_metrics.iloc[0] = env.initial_balance # Start with initial balance
        
        trades = [] # Will store completed trades from env
        step_count = 0
        
        # For daily returns and dates, ensure we use the environment's dataframe index
        # This will be populated as the environment steps through the data
        agent_daily_returns = pd.Series(index=env.df.index, dtype=float)
        
        current_date_index = 0 # To track position in env.df.index

        while not done:
            step_count += 1
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_val = env.portfolio_value
            portfolio_values.append(current_val)

            # Store daily portfolio value at the current step's date
            if current_date_index < len(env.df.index):
                current_step_date = env.df.index[current_date_index]
                daily_portfolio_values_for_metrics[current_step_date] = current_val
            
            # Calculate daily return based on the previous day's portfolio value
            if current_date_index > 0:
                prev_val = daily_portfolio_values_for_metrics.iloc[current_date_index -1]
                if prev_val != 0: # Avoid division by zero
                     agent_daily_returns.iloc[current_date_index] = (current_val - prev_val) / prev_val
                else:
                     agent_daily_returns.iloc[current_date_index] = 0.0
            else: # First day, return is 0
                agent_daily_returns.iloc[current_date_index] = 0.0

            current_date_index +=1

            # No need to check for "trade" in info, env.get_completed_trades() will be used
            
            if step_count % 100 == 0:
                 logger.debug(f"Backtest step {step_count}, Portfolio: {current_val:.2f}")
        
        trades = env.get_completed_trades() # Get all completed trades at the end
        logger.info(f"Agent backtest completed. Total steps: {step_count}. Final portfolio value: {portfolio_values[-1]:.2f}. Total trades: {len(trades)}")
        
        # Fill any remaining NaNs in daily_portfolio_values_for_metrics (e.g., if backtest ends early)
        daily_portfolio_values_for_metrics = daily_portfolio_values_for_metrics.ffill().bfill()
        
        # Ensure agent_daily_returns has the same length as the backtest period
        # If the loop finished early, fill remaining returns with 0
        if current_date_index < len(agent_daily_returns):
            agent_daily_returns.iloc[current_date_index:] = 0.0
            
        return portfolio_values, trades, agent_daily_returns.dropna(), daily_portfolio_values_for_metrics.index.to_list()


    except ImportError as e:
        logger.error(f"Could not import necessary modules for backtesting: {e}")
        return [], [], pd.Series(dtype='float64'), []
    except Exception as e:
        logger.error(f"Error during agent backtesting: {e}", exc_info=True)
        return [], [], pd.Series(dtype='float64'), []

def _calculate_total_return_pct(portfolio_values: list) -> float:
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    if initial_value == 0:
        return 0.0
    return ((final_value - initial_value) / initial_value) * 100.0

# Renamed and updated to calculate all agent metrics
def calculate_agent_metrics_full(
    portfolio_values: list,
    agent_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series, # Added for Alpha/Beta
    agent_trades: list, # Added for trade metrics
    annual_risk_free_rate: float,
    start_date_str: str, # Added for annualized trades
    end_date_str: str,   # Added for annualized trades
    annualization_factor: int = 252
):
    logger.info("Calculating full agent performance metrics...")
    if not portfolio_values:
        logger.warning("Agent portfolio values list is empty. Cannot calculate metrics.")
        return {} # Return empty dict
    
    metrics_results = {}
    try:
        from reinforcestrategycreator.metrics_calculator import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_sortino_ratio,
            calculate_annualized_volatility,
            calculate_beta,
            calculate_alpha
        )
        
        metrics_results["total_return_pct"] = _calculate_total_return_pct(portfolio_values)
        
        # Calculate annualized agent return for Alpha
        # Using mean daily return * annualization_factor for simplicity and consistency with volatility
        mean_agent_daily_return = agent_daily_returns.mean() if not agent_daily_returns.empty else 0.0
        metrics_results["annualized_agent_return_for_alpha"] = mean_agent_daily_return * annualization_factor
        
        metrics_results["sharpe_ratio"] = calculate_sharpe_ratio(
            agent_daily_returns, annual_risk_free_rate=annual_risk_free_rate, annualization_factor=annualization_factor
        )
        metrics_results["max_drawdown_pct"] = calculate_max_drawdown(portfolio_values) * 100.0
        metrics_results["sortino_ratio"] = calculate_sortino_ratio(
            agent_daily_returns, annual_risk_free_rate=annual_risk_free_rate, annualization_factor=annualization_factor
        )
        metrics_results["annualized_volatility_pct"] = calculate_annualized_volatility(
            agent_daily_returns, annualization_factor=annualization_factor
        ) * 100.0 # As percentage

        # Beta and Alpha require benchmark returns
        metrics_results["beta"] = calculate_beta(agent_daily_returns, benchmark_daily_returns)
        
        # For Alpha, we need annualized benchmark return. This will be passed or calculated similarly.
        # Assuming benchmark_metrics will provide its own annualized return for alpha.
        # Here, we'll calculate alpha using the agent's annualized return and expect benchmark's to be passed if needed.
        # Or, it's better if calculate_alpha takes both annualized returns directly.
        # Let's assume calculate_benchmark_metrics_full will provide 'annualized_benchmark_return_for_alpha'
        # This will be calculated after benchmark metrics are available.

        # Trade Metrics
        metrics_results["total_trades"] = len(agent_trades)
        try:
            test_period_start = datetime.strptime(start_date_str, "%Y-%m-%d")
            test_period_end = datetime.strptime(end_date_str, "%Y-%m-%d")
            duration_days = (test_period_end - test_period_start).days
            duration_years = duration_days / 365.25 if duration_days > 0 else 0
            if duration_years > 0:
                metrics_results["average_annual_trades"] = metrics_results["total_trades"] / duration_years
            else:
                metrics_results["average_annual_trades"] = 0 if metrics_results["total_trades"] == 0 else float('inf') # Or handle as appropriate
        except ValueError:
            logger.error("Invalid date format for calculating average annual trades.")
            metrics_results["average_annual_trades"] = 0

        logger.info(f"Agent Metrics Calculated: { {k: (f'{v:.2f}' if isinstance(v, float) else v) for k, v in metrics_results.items()} }")
        return metrics_results
    except ImportError as e:
        logger.error(f"Could not import functions from reinforcestrategycreator.metrics_calculator. Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error calculating full agent metrics: {e}", exc_info=True)
        return {}


def fetch_benchmark_data(symbol: str, start_date_str: str, end_date_str: str) -> tuple[pd.Series, pd.Series, list]:
    logger.info(f"Fetching benchmark data for {symbol} from {start_date_str} to {end_date_str}")
    try:
        from reinforcestrategycreator.data_fetcher import fetch_historical_data
        benchmark_df = fetch_historical_data(symbol, start_date_str, end_date_str)
        if benchmark_df is None or benchmark_df.empty:
            logger.error(f"Failed to fetch benchmark data for {symbol} ({start_date_str}-{end_date_str}).")
            return pd.Series(dtype='float64'), pd.Series(dtype='float64'), [] # Return empty series and list

        # Ensure columns are lowercase for consistency
        if isinstance(benchmark_df.columns, pd.MultiIndex):
            benchmark_df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in benchmark_df.columns]
        else:
            benchmark_df.columns = [col.lower() for col in benchmark_df.columns]

        if "close" not in benchmark_df.columns: # Check for lowercase 'close'
            logger.error(f"'close' column not found for {symbol}. Columns: {benchmark_df.columns}")
            return pd.Series(dtype='float64'), pd.Series(dtype='float64'), []

        logger.info(f"Successfully fetched benchmark data for {symbol} with {len(benchmark_df)} data points.")
        
        close_prices = benchmark_df["close"]
        daily_returns = close_prices.pct_change().dropna()
        dates = benchmark_df.index.to_list() # Get dates from the benchmark data index

        return close_prices, daily_returns, dates # Return prices, returns, and dates
    except ImportError as e:
        logger.error(f"Could not import fetch_historical_data. Error: {e}")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), []
    except Exception as e:
        logger.error(f"Error fetching benchmark data for {symbol}: {e}", exc_info=True)
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), []


def calculate_benchmark_metrics_full(
    benchmark_prices: pd.Series,
    benchmark_daily_returns: pd.Series,
    annual_risk_free_rate: float,
    annualization_factor: int = 252
):
    logger.info("Calculating full benchmark performance metrics...")
    if benchmark_prices.empty:
        logger.warning("Benchmark prices series is empty. Cannot calculate metrics for benchmark.")
        return {}
    
    metrics_results = {}
    try:
        from reinforcestrategycreator.metrics_calculator import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_sortino_ratio,
            calculate_annualized_volatility
        )
        
        metrics_results["total_return_pct"] = _calculate_total_return_pct(benchmark_prices.tolist())
        
        # Calculate annualized benchmark return for Alpha
        mean_benchmark_daily_return = benchmark_daily_returns.mean() if not benchmark_daily_returns.empty else 0.0
        metrics_results["annualized_benchmark_return_for_alpha"] = mean_benchmark_daily_return * annualization_factor

        metrics_results["sharpe_ratio"] = calculate_sharpe_ratio(
            benchmark_daily_returns, annual_risk_free_rate=annual_risk_free_rate, annualization_factor=annualization_factor
        )
        metrics_results["max_drawdown_pct"] = calculate_max_drawdown(benchmark_prices.tolist()) * 100.0
        metrics_results["sortino_ratio"] = calculate_sortino_ratio(
            benchmark_daily_returns, annual_risk_free_rate=annual_risk_free_rate, annualization_factor=annualization_factor
        )
        metrics_results["annualized_volatility_pct"] = calculate_annualized_volatility(
            benchmark_daily_returns, annualization_factor=annualization_factor
        ) * 100.0 # As percentage

        logger.info(f"Benchmark Metrics Calculated: { {k: (f'{v:.2f}' if isinstance(v, float) else v) for k, v in metrics_results.items()} }")
        return metrics_results
    except ImportError as e:
        logger.error(f"Could not import functions from reinforcestrategycreator.metrics_calculator. Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error calculating full benchmark metrics: {e}", exc_info=True)
        return {}


def display_results(
    experiment_id: str,
    start_date: str,
    end_date: str,
    agent_metrics: dict,
    benchmark_metrics: dict,
    benchmark_ticker: str,
    eval_params_config: dict, # For objective thresholds
    output_file_base: str = "evaluation_results",
):
    results_str_list = [
        f"\n--- Evaluation Results for Experiment ID: {experiment_id} ---",
        f"Evaluation Period: {start_date} to {end_date}",
        f"Benchmark Ticker: {benchmark_ticker.upper()}",
        "\nRL Agent Performance:",
        f"  Total Cumulative Return: {agent_metrics.get('total_return_pct', 0.0):.2f}%",
        f"  Annualized Sharpe Ratio: {agent_metrics.get('sharpe_ratio', 0.0):.2f}",
        f"  Annualized Sortino Ratio: {agent_metrics.get('sortino_ratio', 0.0):.2f}",
        f"  Max Drawdown: {agent_metrics.get('max_drawdown_pct', 0.0):.2f}%",
        f"  Annualized Volatility: {agent_metrics.get('annualized_volatility_pct', 0.0):.2f}%",
        f"  Beta: {agent_metrics.get('beta', 0.0):.2f}",
        f"  Alpha: {agent_metrics.get('alpha', 0.0):.4f}", # Alpha is often small
        f"  Total Number of Trades: {agent_metrics.get('total_trades', 0)}",
        f"  Average Annual Trades: {agent_metrics.get('average_annual_trades', 0.0):.2f}",
        f"\n{benchmark_ticker.upper()} Benchmark Performance:",
        f"  Total Cumulative Return: {benchmark_metrics.get('total_return_pct', 0.0):.2f}%",
        f"  Annualized Sharpe Ratio: {benchmark_metrics.get('sharpe_ratio', 0.0):.2f}",
        f"  Annualized Sortino Ratio: {benchmark_metrics.get('sortino_ratio', 0.0):.2f}",
        f"  Max Drawdown: {benchmark_metrics.get('max_drawdown_pct', 0.0):.2f}%",
        f"  Annualized Volatility: {benchmark_metrics.get('annualized_volatility_pct', 0.0):.2f}%",
    ]

    # Objective Checks
    required_sharpe = eval_params_config.get('required_sharpe_ratio', 0.5)
    required_annual_trades = eval_params_config.get('required_annual_trades', 0)

    sharpe_objective_met = agent_metrics.get('sharpe_ratio', -float('inf')) > required_sharpe
    return_objective_met = agent_metrics.get('total_return_pct', -float('inf')) > benchmark_metrics.get('total_return_pct', -float('inf'))
    trades_objective_met = agent_metrics.get('average_annual_trades', -1) >= required_annual_trades
    
    all_objectives_met = sharpe_objective_met and return_objective_met and trades_objective_met

    results_str_list.extend([
        "\nObjective Checks:",
        f"  Agent Sharpe Ratio > {required_sharpe}: {'MET' if sharpe_objective_met else 'NOT MET'} (Agent: {agent_metrics.get('sharpe_ratio', 0.0):.2f})",
        f"  Agent Return > Benchmark Return: {'MET' if return_objective_met else 'NOT MET'} (Agent: {agent_metrics.get('total_return_pct', 0.0):.2f}%, Benchmark: {benchmark_metrics.get('total_return_pct', 0.0):.2f}%)",
        f"  Agent Avg Annual Trades >= {required_annual_trades}: {'MET' if trades_objective_met else 'NOT MET'} (Agent: {agent_metrics.get('average_annual_trades', 0.0):.2f})",
        f"  Overall Objectives Met: {'YES' if all_objectives_met else 'NO'}"
    ])
    
    final_results_str = "\n".join(results_str_list)
    logger.info(final_results_str)
    
    output_filename = f"{output_file_base}_{experiment_id}.json"
    logger.info(f"Saving results to {output_filename}")
    
    results_data = {
        "experiment_id": experiment_id,
        "evaluation_period": {"start_date": start_date, "end_date": end_date},
        "benchmark_ticker": benchmark_ticker,
        "rl_agent_performance": agent_metrics,
        "benchmark_performance": benchmark_metrics,
        "objectives": {
            "required_sharpe_ratio": required_sharpe,
            "required_annual_trades": required_annual_trades,
            "sharpe_objective_met": sharpe_objective_met,
            "return_objective_met": return_objective_met,
            "trades_objective_met": trades_objective_met,
            "all_objectives_met": all_objectives_met
        }
        # "agent_trades_summary": agent_trades, # Could add summary of trades if needed
    }
    try:
        with open(output_filename, 'w') as f:
            json.dump(results_data, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o) # Handle numpy floats
        logger.info(f"Results successfully saved to JSON: {output_filename}")
    except Exception as e:
        logger.error(f"Error saving results to {output_filename}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL Trading Agent against a Benchmark"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the JSON configuration file."
    )
    # Model path will now come from config
    # parser.add_argument(
    #     "--model-path", type=str, required=True, help="Path to the saved agent model checkpoint."
    # )
    # Dates and risk-free rate will also come from config
    # parser.add_argument(
    #     "--start-date", type=str, default="2020-01-01", help="Backtest start date (YYYY-MM-DD). Default: 2020-01-01"
    # )
    # parser.add_argument(
    #     "--end-date", type=str, default="2023-12-31", help="Backtest end date (YYYY-MM-DD). Default: 2023-12-31"
    # )
    # parser.add_argument(
    #     "--risk-free-rate", type=float, default=0.0, help="Annualized risk-free rate for Sharpe Ratio. Default: 0.0"
    # )
    # Output file name will be derived from experiment_id in config
    # parser.add_argument(
    #     "--output-file", type=str, default=None, help="Optional path to save results (e.g., results.json or results.csv)."
    # )
    args = parser.parse_args()

    try:
        load_config_eval(args.config)
    except Exception:
        return # Exit if config loading fails

    # Extract parameters from CONFIG
    eval_params_config = CONFIG.get('evaluation_params', {}) # Keep this for objective thresholds
    data_params_config = CONFIG.get('data_params', {})
    env_params_config = CONFIG.get('env_params', {}) # For TradingEnv initialization
    experiment_id = CONFIG.get('experiment_id', 'DEFAULT_EXP')
    
    model_path = eval_params_config.get('model_path')
    if not model_path:
        logger.error("`model_path` not found in evaluation_params in the config file.")
        return

    start_date_str = data_params_config.get('test_start_date', "2024-01-01")
    end_date_str = data_params_config.get('test_end_date', "2024-03-31")
    annual_risk_free_rate = eval_params_config.get('risk_free_rate', 0.0) # Assumed annualized
    benchmark_ticker = eval_params_config.get('benchmark_ticker', "SPY")
    agent_ticker = data_params_config.get('ticker', "SPY")
    transaction_cost_pct = env_params_config.get('transaction_cost_pct', 0.001)
    initial_balance = env_params_config.get('initial_balance', 100000)
    annualization_factor = eval_params_config.get('annualization_factor', 252) # Get from config or default

    try:
        datetime.strptime(start_date_str, "%Y-%m-%d")
        datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format in config. Please use YYYY-MM-DD for test_start_date and test_end_date.")
        return

    logger.info(f"Starting evaluation script with config: {args.config}")
    logger.info(f"Experiment ID: {experiment_id}, Model: {model_path}")
    logger.info(f"Agent Ticker: {agent_ticker}, Benchmark Ticker: {benchmark_ticker}")
    logger.info(f"Test Period: {start_date_str} to {end_date_str}, Annual Risk-Free Rate: {annual_risk_free_rate}, Annualization Factor: {annualization_factor}")

    agent = load_agent(model_path)
    if agent is None:
        logger.error("Failed to load agent. Exiting.")
        return

    if hasattr(agent, 'select_action') and not hasattr(agent, 'get_action'):
        agent.get_action = agent.select_action
        logger.info("Aliased agent.get_action to agent.select_action for compatibility.")
    elif not hasattr(agent, 'get_action'):
        logger.error("Loaded agent object does not have a get_action or select_action method.")
        return

    agent_portfolio_values, agent_trades, agent_daily_returns, _ = backtest_agent( # agent_dates not directly used here
        agent, agent_ticker, start_date_str, end_date_str, transaction_cost_pct, initial_balance, env_params_config
    )
    
    benchmark_prices, benchmark_daily_returns, _ = fetch_benchmark_data( # benchmark_dates not directly used here
        benchmark_ticker, start_date_str, end_date_str
    )

    # Calculate full benchmark metrics first to get annualized benchmark return for Alpha
    benchmark_metrics_dict = calculate_benchmark_metrics_full(
        benchmark_prices, benchmark_daily_returns, annual_risk_free_rate, annualization_factor
    )
    
    # Now calculate full agent metrics, including Alpha
    agent_metrics_dict = calculate_agent_metrics_full(
        agent_portfolio_values, agent_daily_returns, benchmark_daily_returns, # Pass benchmark_daily_returns
        agent_trades, annual_risk_free_rate, start_date_str, end_date_str, annualization_factor
    )
    
    # Calculate Alpha for the agent (if not already done, or to ensure it's using the final benchmark annualized return)
    # This requires the annualized returns from both agent and benchmark, and beta.
    if "beta" in agent_metrics_dict and \
       "annualized_agent_return_for_alpha" in agent_metrics_dict and \
       "annualized_benchmark_return_for_alpha" in benchmark_metrics_dict:
        
        from reinforcestrategycreator.metrics_calculator import calculate_alpha # Import directly if not done above
        agent_metrics_dict["alpha"] = calculate_alpha(
            annualized_agent_return=agent_metrics_dict["annualized_agent_return_for_alpha"],
            annualized_benchmark_return=benchmark_metrics_dict["annualized_benchmark_return_for_alpha"],
            annual_risk_free_rate=annual_risk_free_rate,
            beta=agent_metrics_dict["beta"]
        )
    else:
        agent_metrics_dict["alpha"] = 0.0 # Default if components are missing
        logger.warning("Could not calculate Alpha due to missing components (beta, annualized returns).")

    display_results(
        experiment_id, start_date_str, end_date_str,
        agent_metrics_dict, benchmark_metrics_dict,
        benchmark_ticker, eval_params_config # Pass eval_params_config for objective thresholds
    )
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()