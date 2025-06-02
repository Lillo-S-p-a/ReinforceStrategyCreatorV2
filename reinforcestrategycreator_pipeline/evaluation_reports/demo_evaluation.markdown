# Demo Model Evaluation

**Evaluation ID:** eval_model_demo_model_874b9f4a_20250530_201558  
**Generated:** 2025-05-30T20:15:58.641789

## Executive Summary

This report presents the evaluation results for model **model_demo_model_874b9f4a** (version v_20250530_201558).

### Key Performance Metrics

| Metric | Value |
|--------|-------|

| Pnl | 424.24 |

| Pnl Percentage | 4.24% |

| Sharpe Ratio | 1.5505 |

| Sortino Ratio | 0.4033 |

| Max Drawdown | 0.2799 |

| Calmar Ratio | 0.1522 |

| Volatility | 0.3065 |

| Trades Count | 65 |


## Model Information

- **Model ID:** model_demo_model_874b9f4a
- **Version:** v_20250530_201558
- **Type:** mock_model
- **Hyperparameters:**


## Data Information

- **Source:** sample_prices
- **Version:** None
- **Shape:** [252, 5]
- **Columns:** 5

## Performance Analysis

### Detailed Metrics

The model achieved the following performance metrics:


- **Pnl:** 424.24

- **Pnl Percentage:** 4.24%

- **Sharpe Ratio:** 1.5505

- **Sortino Ratio:** 0.4033

- **Max Drawdown:** 0.2799

- **Calmar Ratio:** 0.1522

- **Volatility:** 0.3065

- **Trades Count:** 65



## Benchmark Comparison

### Benchmark Performance

| Strategy | PnL % | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|----------|-------|--------------|--------------|----------|--------|

| Buy And Hold | 4.14% | 0.2202 | 0.2799 | 100.0% | 1 |

| Simple Moving Average | -3.37% | -0.1449 | 0.2227 | 33.3% | 3 |

| Random | -1.51% | 0.0051 | 0.2967 | 50.0% | 8 |


### Relative Performance


**vs Buy And Hold:**
- PnL Difference: 10.42 (2.52%)
- Sharpe Ratio Difference: 1.3303


**vs Simple Moving Average:**
- PnL Difference: 761.33 (-225.85%)
- Sharpe Ratio Difference: 1.6954


**vs Random:**
- PnL Difference: 574.91 (-381.58%)
- Sharpe Ratio Difference: 1.5455





## Visualizations


### Cumulative Returns
![cumulative_returns](/tmp/eval_model_demo_model_874b9f4a_20250530_201558/visualizations/cumulative_returns.png)


### Drawdown
![drawdown](/tmp/eval_model_demo_model_874b9f4a_20250530_201558/visualizations/drawdown.png)


### Metrics Comparison
![metrics_comparison](/tmp/eval_model_demo_model_874b9f4a_20250530_201558/visualizations/metrics_comparison.png)


### Performance Dashboard
![performance_dashboard](/tmp/eval_model_demo_model_874b9f4a_20250530_201558/visualizations/performance_dashboard.png)




## Conclusions

Based on the evaluation results:

1. The model shows positive returns with a PnL of 4.24%.
2. Risk-adjusted performance (Sharpe ratio) is 1.5505.
3. Maximum drawdown observed was 0.2799.

4. Compared to benchmarks, the model outperforms in terms of returns.


---
*Report generated automatically by the Model Evaluation Pipeline*