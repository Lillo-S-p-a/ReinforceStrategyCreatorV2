#!/bin/bash

# Script to run the improved backtesting multiple times and collect results
echo "Running improved backtesting 5 times..."
echo "=========================================="

# Create a results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="multiple_test_results_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Create a summary file
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "IMPROVED BACKTESTING - MULTIPLE RUNS SUMMARY" > $SUMMARY_FILE
echo "Run on $(date)" >> $SUMMARY_FILE
echo "=========================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Run the backtesting 5 times
for i in {1..5}; do
    echo ""
    echo "Starting Run #$i of 5..."
    echo "----------------------------------------"
    
    # Create a log file for this run
    LOG_FILE="${RESULTS_DIR}/run_${i}.log"
    
    # Run the backtesting and capture the output
    python run_improved_backtesting.py | tee $LOG_FILE
    
    # Extract key metrics and save to summary
    echo "RUN #$i RESULTS:" >> $SUMMARY_FILE
    
    # Extract PnL
    PNL=$(grep "Final PnL:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- PnL: $PNL" >> $SUMMARY_FILE
    
    # Extract PnL Percentage
    PNL_PCT=$(grep "PnL Percentage:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- PnL Percentage: $PNL_PCT" >> $SUMMARY_FILE
    
    # Extract Sharpe Ratio
    SHARPE=$(grep "Sharpe Ratio:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- Sharpe Ratio: $SHARPE" >> $SUMMARY_FILE
    
    # Extract Max Drawdown
    DRAWDOWN=$(grep "Max Drawdown:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- Max Drawdown: $DRAWDOWN" >> $SUMMARY_FILE
    
    # Extract Win Rate
    WIN_RATE=$(grep "Win Rate:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- Win Rate: $WIN_RATE" >> $SUMMARY_FILE
    
    # Extract Total Trades
    TRADES=$(grep "Total Trades:" $LOG_FILE | tail -1 | awk '{print $3}')
    echo "- Total Trades: $TRADES" >> $SUMMARY_FILE
    
    # Extract Report Path
    REPORT_PATH=$(grep "Full report saved to:" $LOG_FILE | tail -1 | awk '{print $5}')
    echo "- Report: $REPORT_PATH" >> $SUMMARY_FILE
    
    # Extract Model Path
    MODEL_PATH=$(grep "Model exported to:" $LOG_FILE | tail -1 | awk '{print $4}')
    echo "- Model: $MODEL_PATH" >> $SUMMARY_FILE
    
    echo "" >> $SUMMARY_FILE
    echo "----------------------------------------" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    
    echo "Run #$i completed."
    echo ""
done

# Add final summary and stats
echo "=========================================" >> $SUMMARY_FILE
echo "OVERALL SUMMARY" >> $SUMMARY_FILE
echo "=========================================" >> $SUMMARY_FILE

# Print location of summary file
echo ""
echo "All runs completed!"
echo "Summary file created at: $SUMMARY_FILE"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Show the summary file
echo "Summary content:"
cat $SUMMARY_FILE