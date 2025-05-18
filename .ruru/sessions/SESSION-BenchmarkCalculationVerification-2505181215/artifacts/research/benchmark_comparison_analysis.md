# Benchmark Calculation Analysis

## Overview
This analysis compares the calculation methodologies between the RL model and benchmark strategies to identify potential causes for the significant performance disparity. The model reported a 40.79% PnL with 6.18 Sharpe ratio while benchmarks showed negative performance (down to -18.26% PnL).

## Code Review Findings

After examining the key components of the backtesting system, I've identified several potential areas where discrepancies might arise:

### 1. Data Handling Differences

#### RL Model:
- Uses the `TradingEnvironment` (trading_environment.py) which processes data through a normalized observation window.
- Data is processed with rolling z-score normalization (lines 925-997 in trading_environment.py).
- Uses Ray object store for data references, which might affect data consistency.

#### Benchmarks:
- Direct access to raw price data through DataFrame lookups (e.g., lines 156-172 in benchmarks.py).
- No equivalent normalization process applied to the data.
- Different column name detection logic that may not match the model's data access pattern.

**Potential Issue:** The model could be trading on normalized data while benchmarks use raw data, creating an "apples to oranges" comparison.

### 2. Trading Mechanics Differences

#### RL Model:
- Uses sophisticated position sizing with fractional shares (lines 582-596 in trading_environment.py).
- Implements stop-loss and take-profit mechanisms (lines 330-367).
- Has built-in risk management through drawdown penalties (lines 895-902).
- Records and handles execution details meticulously (lines 468-508).

#### Benchmarks:
- Simpler trading logic with basic position sizing (e.g., Buy & Hold: lines 179-180).
- SMA strategy uses signal generation that's more rigid (lines 287-298).
- No equivalent stop-loss/take-profit mechanisms.
- Less detailed trade tracking.

**Potential Issue:** The model's superior risk management capabilities might explain better performance, but differences in trading mechanics implementation could also introduce unfair advantages.

### 3. Performance Calculation Methods

#### RL Model:
- Metrics calculated by `MetricsCalculator.get_episode_metrics()` in evaluation.py (lines 31-112).
- Portfolio value history tracked continuously during training/testing.
- Sharpe ratio calculation uses episode returns with proper annualization (lines 435-445 in trading_environment.py).

#### Benchmarks:
- Metrics calculated by `BenchmarkStrategy.calculate_metrics()` (lines 51-107 in benchmarks.py).
- Portfolio values calculated retrospectively from price history.
- Similar Sharpe ratio calculation approach but potentially different data points.

**Potential Issue:** Despite similar calculation formulas, differences in how portfolio values are tracked and when metrics are calculated could lead to significant divergence.

### 4. Implementation Details

#### Critical Issue Found:
- In `BuyAndHoldStrategy.run()` (lines 137-209), the portfolio values are initially calculated correctly, but then converted back to a Python list:
  ```python
  # Convert portfolio values to numpy array to ensure consistency
  portfolio_values = np.array(portfolio_values, dtype=float).tolist()
  ```
  This unnecessary conversion to list might be introducing unexpected behavior.

- For SMA and Random strategies, there's no equivalent explicit conversion before metric calculation.

#### Environment Differences:
- The RL model uses a gymnasium environment that simulates trading with precise step-by-step balance updates.
- Benchmarks use a simplified simulation without the same sequential environment constraints.

**Potential Issue:** Different computational frameworks may lead to precision differences or calculation artifacts.

### 5. Anomalies in Test Results

The reported "trades: 0" in the test metrics alongside the 40.79% PnL is suspicious. If the model made no trades but still showed profit, this suggests a fundamental calculation error.

## Recommendations for Investigation

1. **Data Consistency Check**: Verify that both the model and benchmarks are using identical price data, particularly checking normalization effects.

2. **Trading Logic Verification**: Implement logging to capture each trading decision in both paradigms and compare side-by-side.

3. **Benchmark Code Fix**: Remove the unnecessary list conversion in `BuyAndHoldStrategy.run()` to maintain consistent array types.

4. **Portfolio Value Tracking**: Add detailed logging of portfolio values at each timestep for both the model and benchmarks.

5. **Transaction Fee Parity**: Confirm that identical transaction fees are applied in both systems.

6. **Initial Balance Verification**: Check that both start with the exact same initial balance.

7. **Temporal Alignment**: Ensure that both systems are evaluating performance over identical time periods.

8. **Hand Calculation Validation**: Perform manual calculations on a small subset of data to verify both approaches.

This analysis suggests that while the implementation approaches between model and benchmark evaluations are similar, subtle differences in data handling, trading mechanics, and calculation methods could be compounding to create the observed performance gap.