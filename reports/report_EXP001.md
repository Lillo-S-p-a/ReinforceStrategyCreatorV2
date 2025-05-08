# Experiment Report: EXP001

**Date:** 2025-05-08
**Experiment ID:** EXP001
**Branch:** `feature/rl-optimization-workflow`
**Configuration File:** [`configs/config_EXP001.json`](configs/config_EXP001.json)
**Model File:** [`models/model_EXP001_ep447.keras`](models/model_EXP001_ep447.keras)
**Evaluation Results File:** [`evaluation_results_EXP001.json`](evaluation_results_EXP001.json)

## Description

Baseline experiment using the refactored workflow. Trained the agent using parameters similar to Iteration 2, but with a defined train/test split and updated evaluation metrics. Note: Training ran for 10 episodes due to a likely override in `train.py`, instead of the configured 500 episodes.

*   **Training Period:** 2018-01-01 to 2021-12-31
*   **Testing Period:** 2022-01-01 to 2023-12-31
*   **Key Training Params:** 10 Episodes (actual), Gamma=0.99, LR=0.001, Epsilon Decay=0.9999, Target Update=100
*   **Key Env Params:** Window=5, Drawdown Penalty=0.05, Trading Penalty=0.005, Stop Loss=5.0%, Tx Cost=0.05%

## Hypothesis

Training for a limited number of episodes (10) with moderate penalties is unlikely to yield a profitable strategy but establishes a baseline for the new workflow and metrics.

## Evaluation Results (Test Set: 2022-01-01 to 2023-12-31)

| Metric                     | Agent     | Benchmark (SPY) | Objective Met |
| :------------------------- | :-------- | :-------------- | :------------ |
| Total Return (%)           | -4.95     | 2.65            | **NO**        |
| Annualized Sharpe Ratio    | -1.65     | 0.16            | **NO** (>0.5) |
| Annualized Sortino Ratio   | -1.37     | 0.17            | N/A           |
| Max Drawdown (%)           | 6.06      | 24.50           | N/A           |
| Annualized Volatility (%)  | 1.53      | 19.53           | N/A           |
| Beta                       | -0.00     | 1.00            | N/A           |
| Alpha                      | -0.025    | 0.00            | N/A           |
| Total Trades               | 132       | N/A             | N/A           |
| Avg Annual Trades          | 66.14     | N/A             | **NO** (>=100)|
| **Overall Objectives Met** |           |                 | **NO**        |

*(Transaction Cost: 0.05% applied during agent backtest)*

## Analysis

As expected, the agent trained for only 10 episodes performed poorly on the out-of-sample test data, significantly underperforming the SPY benchmark. Key objectives were not met. The trade frequency was also below the target.

## Next Steps (EXP002 Hypothesis)

Increase the number of training episodes significantly (e.g., 500) while keeping the moderate trading penalty (0.005) to see if longer training leads to better policy convergence and performance. Address the issue where the configured number of episodes was overridden in `train.py`.