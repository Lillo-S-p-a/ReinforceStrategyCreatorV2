#!/bin/bash

# Script to run the model selection improvements test

# Create necessary directories
mkdir -p config
mkdir -p data
mkdir -p logs

# Check if we have test data, if not create a placeholder
if [ ! -f "data/processed_market_data.csv" ]; then
    echo "Creating sample test data..."
    echo "timestamp,open,high,low,close,volume,price,ma_20,ma_50,rsi,returns" > data/processed_market_data.csv
    # Add some sample data rows
    for i in {1..100}; do
        price=$(echo "scale=2; 100 + $RANDOM / 1000" | bc)
        echo "2025-01-$i,${price},${price},${price},${price},1000,${price},${price},${price},50,0.001" >> data/processed_market_data.csv
    done
    echo "Sample data created at data/processed_market_data.csv"
fi

# Run the test script
echo "Running model selection improvement tests..."
python test_model_selection_improvements.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Test completed. Check results in test_results_* directory"
else
    echo "Test failed. Check logs for details"
    exit 1
fi