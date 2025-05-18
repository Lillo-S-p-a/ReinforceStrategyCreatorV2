# Benchmark Calculation Fix Plan

## Overview
This document outlines specific steps to investigate and fix the discrepancies between the RL model and benchmark strategy performance metrics. Based on our code analysis, we've identified several potential issues and created a structured plan to address them.

## Immediate Code Fixes

### 1. Fix Buy and Hold Strategy Conversion Issue

In `benchmarks.py`, the `BuyAndHoldStrategy.run()` method has an unnecessary list conversion that may be affecting calculations:

```python
# Convert portfolio values to numpy array to ensure consistency
portfolio_values = np.array(portfolio_values, dtype=float).tolist()
```

**Fix:**
```python
# Fix: Keep values as numpy array for consistent calculation
portfolio_values = np.array(portfolio_values, dtype=float)
```

### 2. Add Logging for Data Consistency Checks

Add logging statements in both model evaluation and benchmark execution to ensure they're working with the same data:

In `evaluation.py`, `BenchmarkEvaluator.compare_with_benchmarks()`:

```python
# Add before line 175
logger.info(f"Data used for benchmarks - shape: {test_data.shape}, range: {test_data.index[0]} to {test_data.index[-1]}")
logger.info(f"First few closing prices: {test_data['close'].head(3).values} ... Last few: {test_data['close'].tail(3).values}")
```

In `model.py`, `ModelTrainer.evaluate_model()`:

```python
# Add before line 163
logger.info(f"Data used for model evaluation - shape: {test_data.shape}, range: {test_data.index[0]} to {test_data.index[-1]}")
logger.info(f"First few closing prices: {test_data['close'].head(3).values} ... Last few: {test_data['close'].tail(3).values}")
```

### 3. Fix Metrics Calculation Consistency

Ensure both systems use exactly the same metrics calculation approach. Update `benchmarks.py` to use the same MetricsCalculator as the model:

**Add to `workflow.py` in `_compare_with_benchmarks()`:**

```python
# Update line 372-375 to pass the same metrics_calculator to all benchmark strategies
metrics_calculator = self.metrics_calculator
# Initialize benchmark evaluator with the shared metrics calculator
benchmark_evaluator = BenchmarkEvaluator(
    config=self.config,
    metrics_calculator=metrics_calculator
)

# Pass metrics_calculator to benchmark strategies too
for strategy_name, strategy in benchmark_evaluator.strategies.items():
    strategy.set_metrics_calculator(metrics_calculator)
```

**Add to `benchmarks.py` in `BenchmarkStrategy` class:**

```python
def set_metrics_calculator(self, metrics_calculator):
    """Set an external metrics calculator to ensure consistent calculations"""
    self._external_metrics_calculator = metrics_calculator

def calculate_metrics(self, 
                     portfolio_values: List[float],
                     trades: int,
                     profitable_trades: int) -> Dict[str, float]:
    """
    Calculate performance metrics.
    
    If an external metrics calculator is set, use that for consistency.
    Otherwise, fall back to internal implementation.
    """
    if hasattr(self, '_external_metrics_calculator'):
        # Create a dummy environment object with portfolio history
        class DummyEnv:
            def __init__(self, portfolio_values, trades, profitable_trades):
                self._portfolio_value_history = portfolio_values
                self._completed_trades = [{'pnl': 1}] * profitable_trades + [{'pnl': -1}] * (trades - profitable_trades)
                
        dummy_env = DummyEnv(portfolio_values, trades, profitable_trades)
        return self._external_metrics_calculator.get_episode_metrics(dummy_env)
    else:
        # Original implementation follows...
```

### 4. Fix Transaction Fee Consistency

Ensure the transaction fee is identical in both systems:

In `workflow.py`, update `_compare_with_benchmarks()`:

```python
# Update line 372-375
benchmark_evaluator = BenchmarkEvaluator(
    config={
        **self.config,
        # Explicitly set transaction fee to match the model
        "transaction_fee": self.config.get("transaction_fee_percent", self.config.get("transaction_fee", 0.001))
    },
    metrics_calculator=self.metrics_calculator
)
```

## Investigation Steps

### 1. Detailed Trade Comparison

Create a script to run both the model and benchmarks on the same small data subset and log each trade:

```python
def compare_trading_decisions(test_data_sample):
    """Compare trading decisions between model and benchmarks"""
    # Initialize model and benchmark
    model = load_trained_model()  # Load your trained model
    benchmark = BuyAndHoldStrategy(initial_balance=10000, transaction_fee=0.001)
    
    # Create a small sample of data (e.g., 100 data points)
    test_sample = test_data_sample.iloc[:100]
    
    # Track model trades
    model_trades = []
    env = TradingEnvironment(config={"df": ray.put(test_sample), "initial_balance": 10000})
    state = env.reset()
    done = False
    
    while not done:
        action = model.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        
        # Log trade details
        if 'operation_type_for_log' in info:
            model_trades.append({
                'step': info['step'],
                'price': info['current_price'],
                'action': info['action_taken'],
                'operation': info['operation_type_for_log'],
                'shares': info['shares_transacted_this_step'],
                'balance': info['balance'],
                'portfolio_value': info['portfolio_value']
            })
    
    # Run benchmark on same data
    benchmark_result = benchmark.run(test_sample)
    
    # Output comparison
    print("Model trades:", model_trades)
    print("Benchmark final metrics:", benchmark_result)
    
    return model_trades, benchmark_result
```

### 2. Portfolio Value Tracking

Add extensive logging of portfolio values at each step:

In `trading_environment.py`, update `step()`:

```python
# Add after line 377
if hasattr(self, '_debug_portfolio_history'):
    self._debug_portfolio_history.append({
        'step': self.current_step,
        'price': self.current_price,
        'balance': self.balance,
        'shares': self.shares_held,
        'portfolio_value': self.portfolio_value,
        'action': action
    })
```

In `benchmarks.py`, update each strategy's `run()` method:

```python
# Add near the beginning of the function
debug_portfolio_history = []

# Add inside the price loop where portfolio values are calculated
debug_portfolio_history.append({
    'step': i,
    'price': price,
    'portfolio_value': portfolio_value,
    # Add other relevant values based on the strategy
})

# Add before return
logger.debug(f"Portfolio value progression: {debug_portfolio_history[:5]} ... {debug_portfolio_history[-5:]}")
```

### 3. Hand Calculation Validation

Create a test script that runs a simple preset scenario through both systems:

```python
def validate_calculations():
    """Validate calculations with a simple predefined scenario"""
    # Create synthetic data with a simple trend
    dates = pd.date_range(start='2023-01-01', periods=20)
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
              110, 109, 108, 107, 106, 105, 104, 103, 102, 101]
    
    test_data = pd.DataFrame({
        'close': prices,
        'open': [p-0.5 for p in prices],
        'high': [p+0.5 for p in prices],
        'low': [p-1 for p in prices],
        'volume': [1000000] * 20
    }, index=dates)
    
    # Run model
    model_result = run_model_on_data(test_data)
    
    # Run benchmarks
    buy_hold = BuyAndHoldStrategy(initial_balance=10000, transaction_fee=0.001)
    bh_result = buy_hold.run(test_data)
    
    # Manual calculation (buy and hold)
    initial_price = prices[0]
    final_price = prices[-1]
    shares = 10000 / initial_price * (1 - 0.001)  # Account for transaction fee
    manual_final_value = shares * final_price
    manual_pnl = manual_final_value - 10000
    manual_pnl_pct = (manual_pnl / 10000) * 100
    
    print(f"Manual calculation - Final value: {manual_final_value:.2f}, PnL: {manual_pnl:.2f} ({manual_pnl_pct:.2f}%)")
    print(f"Buy & Hold result: {bh_result}")
    print(f"Model result: {model_result}")
    
    return {
        "manual": {"final_value": manual_final_value, "pnl": manual_pnl, "pnl_pct": manual_pnl_pct},
        "buy_hold": bh_result,
        "model": model_result
    }
```

## Proposed Code Modifications

### 1. Refactor `BenchmarkStrategy.calculate_metrics()`

Rewrite to ensure perfect parity with the model's metrics calculation:

```python
def calculate_metrics(self, 
                     portfolio_values: List[float],
                     trades: int,
                     profitable_trades: int) -> Dict[str, float]:
    """
    Calculate performance metrics consistently with the model's approach.
    """
    # Ensure portfolio_values is a numpy array
    portfolio_values = np.array(portfolio_values, dtype=float)
    
    # Calculate returns exactly as in MetricsCalculator
    if len(portfolio_values) <= 1:
        return {
            "pnl": 0.0,
            "pnl_percentage": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades": 0
        }
        
    # Calculate returns (identical to MetricsCalculator)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics (identical to MetricsCalculator)
    initial_value = float(portfolio_values[0])
    final_value = float(portfolio_values[-1])
    pnl = final_value - initial_value
    pnl_percentage = (pnl / initial_value) * 100
    
    # Calculate Sharpe ratio exactly as in MetricsCalculator
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0
        
    # Calculate max drawdown exactly as in MetricsCalculator  
    peak = portfolio_values[0]
    max_drawdown = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate win rate
    win_rate = profitable_trades / trades if trades > 0 else 0.0
    
    return {
        "pnl": pnl,
        "pnl_percentage": pnl_percentage,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": trades
    }
```

### 2. Add Verification Method to BacktestingWorkflow

```python
def verify_data_consistency(self):
    """Verify data consistency between model and benchmarks"""
    if self.test_data is None:
        logger.warning("No test data available. Fetching data first.")
        self.fetch_data()
    
    # Check column names and data shapes
    logger.info(f"Test data shape: {self.test_data.shape}")
    logger.info(f"Test data columns: {self.test_data.columns}")
    
    # Log price statistics
    if 'close' in self.test_data.columns:
        close_col = 'close'
    elif 'Close' in self.test_data.columns:
        close_col = 'Close'
    else:
        close_col = None
        logger.warning("Could not identify close price column")
    
    if close_col:
        prices = self.test_data[close_col]
        logger.info(f"Price range: {prices.min()} to {prices.max()}")
        logger.info(f"Price change: {(prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100:.2f}%")
    
    # Test identical metrics calculation
    sample_values = [10000, 10100, 10200, 10150, 10300]
    
    # Create dummy environment
    class DummyEnv:
        def __init__(self):
            self._portfolio_value_history = sample_values
            self._completed_trades = [{'pnl': 100}, {'pnl': 50}]
    
    dummy_env = DummyEnv()
    model_metrics = self.metrics_calculator.get_episode_metrics(dummy_env)
    
    # Calculate using benchmark method
    benchmark_strategy = self.strategies['buy_and_hold']
    benchmark_metrics = benchmark_strategy.calculate_metrics(
        portfolio_values=sample_values,
        trades=2,
        profitable_trades=2
    )
    
    # Compare results
    logger.info(f"Model metrics calculation: {model_metrics}")
    logger.info(f"Benchmark metrics calculation: {benchmark_metrics}")
    
    return {
        "model_metrics": model_metrics,
        "benchmark_metrics": benchmark_metrics,
        "match": all(abs(model_metrics[k] - benchmark_metrics[k]) < 1e-6 
                    for k in model_metrics if k in benchmark_metrics)
    }
```

## Implementation Plan

1. First implement the logging changes to gather more data about the discrepancy
2. Fix the list conversion issue in the Buy and Hold strategy
3. Implement the metrics calculation consistency fix
4. Run the verification methods to compare trading decisions
5. If issues persist, implement the more extensive refactoring of the metrics calculation

This systematic approach should isolate and resolve the issues causing the performance discrepancy between the RL model and benchmark strategies.