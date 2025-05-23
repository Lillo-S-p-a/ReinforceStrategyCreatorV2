#!/bin/bash

SUMMARY_FILE="multiple_test_results_20250523_012338/summary.txt"

# Create backup
cp $SUMMARY_FILE ${SUMMARY_FILE}.bak

# Create new content
cat > $SUMMARY_FILE << 'CONTENT'
IMPROVED BACKTESTING - MULTIPLE RUNS SUMMARY
Run on Fri May 23 01:23:38 AM CEST 2025
=========================================

RUN #1 RESULTS:
- PnL: $3150.51
- PnL Percentage: 3.15%
- Sharpe Ratio: 1.8063
- Max Drawdown: 0.86%
- Win Rate: 58.65%
- Total Trades: 80.7
- Report: results/improved_backtest_20250523_012343/reports/backtest_report_20250523_014111.html
- Model: production_models/model_ep100_runSPY_20250523_014111.pth

----------------------------------------

RUN #2 RESULTS:
- PnL: [data not captured in log]
- PnL Percentage: [data not captured in log]
- Sharpe Ratio: [data not captured in log]
- Max Drawdown: [data not captured in log]
- Win Rate: [data not captured in log]
- Total Trades: [data not captured in log]
- Report: results/improved_backtest_20250523_014119/reports/backtest_report_20250523_015848.html
- Model: production_models/model_ep100_runSPY_20250523_015848.pth

----------------------------------------

RUN #3 RESULTS:
- PnL: $-1049.56
- PnL Percentage: -1.05%
- Sharpe Ratio: -0.6185
- Max Drawdown: 2.31%
- Win Rate: 46.42%
- Total Trades: 114.27
- Report: results/improved_backtest_20250523_015925/reports/backtest_report_20250523_021722.html
- Model: production_models/model_ep100_runSPY_20250523_021722.pth

----------------------------------------

RUN #4 RESULTS:
- PnL: $787.38
- PnL Percentage: 0.79%
- Sharpe Ratio: 0.5142
- Max Drawdown: 0.96%
- Win Rate: 58.42%
- Total Trades: 86.43
- Report: results/improved_backtest_20250523_021730/reports/backtest_report_20250523_023454.html
- Model: production_models/model_ep100_runSPY_20250523_023454.pth

----------------------------------------

RUN #5 RESULTS:
- PnL: $-1124.08
- PnL Percentage: -1.12%
- Sharpe Ratio: -0.7092
- Max Drawdown: 1.46%
- Win Rate: 48.30%
- Total Trades: 118.23
- Report: results/improved_backtest_20250523_023502/reports/backtest_report_20250523_025228.html
- Model: production_models/model_ep100_runSPY_20250523_025228.pth

----------------------------------------

=========================================
OVERALL SUMMARY
=========================================

Based on 5 runs, the model shows inconsistent performance:
- Profitable runs: 2 (Runs #1 and #4)
- Loss-making runs: 2 (Runs #3 and #5)
- Unknown result: 1 (Run #2)

Average metrics (excluding Run #2):
- Average PnL: $441.06
- Average PnL Percentage: 0.44%
- Average Sharpe Ratio: 0.25
- Average Max Drawdown: 1.40%
- Average Win Rate: 52.95%
- Average Trade Count: 99.91

OBSERVATIONS:
1. High variability between runs indicates inherent model instability
2. Cross-validation Sharpe ratios (2.6-3.9) differ significantly from test Sharpe ratios (-0.71-1.81)
3. Possible overfitting despite best practice implementation
4. Trade count varies significantly between runs (80-118)
CONTENT

echo "Summary file updated with manually extracted results."
