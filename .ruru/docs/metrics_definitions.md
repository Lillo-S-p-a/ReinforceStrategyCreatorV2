# Performance Metrics Definitions

This document defines the key performance metrics for the ReinforceStrategyCreator project, including their calculation logic, data sources, granularity, and units.

---

## 1. Financial Performance Metrics

### 1.1. Profit and Loss (PnL)

*   **Definition:** The net financial gain or loss over a specified period (e.g., a backtest, a trading session, or cumulatively).
*   **Calculation Logic:**
    *   `PnL = Ending Portfolio Value - Starting Portfolio Value - Total Transaction Costs`
    *   Alternatively, sum of profits/losses from individual closed trades: `PnL = Σ(Trade Exit Value - Trade Entry Value - Trade Transaction Costs)`
*   **Data Sources:**
    *   Portfolio value history (recorded at least at the start and end of the period).
    *   Trade execution logs (entry/exit prices, quantities, timestamps).
    *   Transaction cost configuration/logs.
*   **Granularity:** Per backtest, per trading session, daily, weekly, monthly, cumulative.
*   **Units:** Currency (e.g., $, €).

### 1.2. Risk-Adjusted Return (Sharpe Ratio)

*   **Definition:** Measures the excess return (or risk premium) per unit of deviation (risk). Higher values indicate better risk-adjusted performance. (Note: This is also used as the reward signal in the environment as per FR3.7).
*   **Calculation Logic:**
    *   `Sharpe Ratio = (Rp - Rf) / σp`
    *   Where:
        *   `Rp`: Mean portfolio return over the period (e.g., mean daily/weekly return).
        *   `Rf`: Risk-free rate over the period (e.g., corresponding T-bill rate). **Assumption:** If a dynamic source isn't available, a constant (e.g., 0% or a configurable value) might be used. This assumption MUST be documented.
        *   `σp`: Standard deviation of the portfolio's excess return (`Rp - Rf`) over the period.
*   **Data Sources:**
    *   Portfolio value history (sampled at regular intervals, e.g., daily, to calculate returns).
    *   Risk-free rate data source or configured constant.
*   **Granularity:** Calculated over a period (e.g., backtest duration, annually). Requires a series of returns (e.g., daily, weekly).
*   **Units:** Dimensionless.

### 1.3. Maximum Drawdown (MDD)

*   **Definition:** The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. Indicates downside risk over a specified time period.
*   **Calculation Logic:**
    *   Calculate portfolio value `P(t)` at various points `t` in the period.
    *   Calculate Drawdown `D(t) = 1 - P(t) / Peak(t)`, where `Peak(t)` is the maximum portfolio value up to time `t`.
    *   `MDD = max(D(t))` over the entire period.
*   **Data Sources:** Portfolio value history (sampled frequently enough to capture peaks and troughs, e.g., daily or per trade).
*   **Granularity:** Calculated over a period (e.g., backtest duration).
*   **Units:** Percentage (%).

---

## 2. RL Agent & Simulation Metrics

### 2.1. Total Reward (Cumulative Reward)

*   **Definition:** The sum of all reward signals received by the agent during a specific episode or training run. Reflects the agent's primary optimization objective within the simulation.
*   **Calculation Logic:** `Total Reward = Σ(Reward_t)` for all steps `t` in an episode/run.
*   **Data Sources:** RL environment reward logs (`training_log.csv` likely contains episode totals).
*   **Granularity:** Per episode, per training run.
*   **Units:** Dimensionless (depends on the specific reward function, which is Sharpe Ratio based here).

### 2.2. Steps Per Episode

*   **Definition:** The number of simulation steps taken within a single episode.
*   **Calculation Logic:** Count of steps from the start to the termination of an episode.
*   **Data Sources:** RL environment step logs (`training_log.csv` likely contains this).
*   **Granularity:** Per episode, average over multiple episodes.
*   **Units:** Count.

---

## 3. Trading Activity & Efficiency Metrics

### 3.1. Win Rate

*   **Definition:** The percentage of closed trades that resulted in a profit.
*   **Calculation Logic:** `Win Rate = (Number of Profitable Trades / Total Number of Closed Trades) * 100%`
    *   A trade is profitable if `(Trade Exit Value - Trade Entry Value - Trade Transaction Costs) > 0`.
*   **Data Sources:** Trade execution logs (entry/exit prices, quantities, costs).
*   **Granularity:** Calculated over a period (e.g., backtest duration, per N episodes).
*   **Units:** Percentage (%).

### 3.2. Trade Frequency

*   **Definition:** The number of trades executed within a specific period.
*   **Calculation Logic:** `Trade Frequency = Total Number of Closed Trades / Number of Periods` (e.g., trades per day, trades per week, trades per episode).
*   **Data Sources:** Trade execution logs (timestamps).
*   **Granularity:** Per period (day, week, month), per episode, average over backtest.
*   **Units:** Count per period (e.g., trades/day).

### 3.3. Success Rate (Episode Success Rate)

*   **Definition:** The percentage of episodes that meet a defined success criterion. **Proposed Criterion:** Episode ends with a positive PnL. (This definition should be confirmed or refined based on specific project goals).
*   **Calculation Logic:** `Success Rate = (Number of Episodes with PnL > 0 / Total Number of Episodes) * 100%`
*   **Data Sources:** Episode PnL results (derived from portfolio value history within each episode). `training_log.csv` might contain relevant data.
*   **Granularity:** Calculated over a set of episodes (e.g., during a training run or evaluation).
*   **Units:** Percentage (%).

---

## 4. System Resource Metrics

### 4.1. Training Time

*   **Definition:** The wall-clock time taken to complete a training run (e.g., N episodes).
*   **Calculation Logic:** `End Timestamp - Start Timestamp` of the training script execution.
*   **Data Sources:** Training script logs or system monitoring.
*   **Granularity:** Per training run.
*   **Units:** Seconds, minutes, hours.

### 4.2. CPU/GPU Utilization (Optional)

*   **Definition:** Average or peak percentage of CPU or GPU resources used during training or backtesting.
*   **Calculation Logic:** Measured using system monitoring tools (e.g., `nvidia-smi` for GPU, `top`/`htop` or Python libraries like `psutil` for CPU). Requires integration with monitoring.
*   **Data Sources:** System monitoring tools/logs.
*   **Granularity:** Average/Peak over training run, average/peak per episode (if logged).
*   **Units:** Percentage (%).

### 4.3. Memory Usage (Optional)

*   **Definition:** Average or peak RAM or GPU memory consumed during training or backtesting.
*   **Calculation Logic:** Measured using system monitoring tools.
*   **Data Sources:** System monitoring tools/logs.
*   **Granularity:** Average/Peak over training run, average/peak per episode (if logged).
*   **Units:** Megabytes (MB), Gigabytes (GB).

---