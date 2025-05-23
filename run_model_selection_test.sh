#!/bin/bash

# Script to run the model selection improvements test

# Create necessary directories
mkdir -p config
mkdir -p data
mkdir -p logs

# Check if we have test data, if not create a placeholder
if [ ! -f "data/processed_market_data.csv" ]; then
    echo "Creating sample test data..."
    
    # Create a more realistic sample data file with proper OHLC values
    echo "timestamp,open,high,low,close,volume,price,ma_20,ma_50,rsi,returns" > data/processed_market_data.csv
    
    # Generate 200 days of data with more realistic price movements
    base_price=100.00
    current_price=$base_price
    
    for i in {1..200}; do
        # Generate random price movement (-1% to +1%)
        change=$(echo "scale=4; $current_price * (0.5 - $RANDOM / 32767) / 50" | bc)
        
        # Calculate OHLC values
        open_price=$(echo "scale=2; $current_price" | bc)
        close_price=$(echo "scale=2; $current_price + $change" | bc)
        
        # Make high higher than both open and close
        high_price=$(echo "scale=2; if($open_price > $close_price) then $open_price + ($RANDOM % 100) / 100 else $close_price + ($RANDOM % 100) / 100" | bc)
        
        # Make low lower than both open and close
        low_price=$(echo "scale=2; if($open_price < $close_price) then $open_price - ($RANDOM % 100) / 100 else $close_price - ($RANDOM % 100) / 100" | bc)
        
        # Ensure low is not higher than high (can happen due to bc precision)
        if (( $(echo "$low_price > $high_price" | bc -l) )); then
            temp=$low_price
            low_price=$high_price
            high_price=$temp
        fi
        
        # Calculate volume (random between 1000-10000)
        volume=$(( 1000 + $RANDOM % 9000 ))
        
        # Calculate simple moving averages (just use close price for simplicity)
        ma_20=$close_price
        ma_50=$close_price
        
        # RSI (random between 30-70)
        rsi=$(( 30 + $RANDOM % 40 ))
        
        # Returns (percentage change)
        returns=$(echo "scale=4; $change / $current_price" | bc)
        
        # Format date properly
        date=$(date -d "2025-01-01 +$i days" +"%Y-%m-%d" 2>/dev/null || date -v+${i}d -j -f "%Y-%m-%d" "2025-01-01" +"%Y-%m-%d" 2>/dev/null || echo "2025-01-$i")
        
        # Add row to CSV
        echo "$date,$open_price,$high_price,$low_price,$close_price,$volume,$close_price,$ma_20,$ma_50,$rsi,$returns" >> data/processed_market_data.csv
        
        # Update current price for next iteration
        current_price=$close_price
    done
    
    echo "Sample data created at data/processed_market_data.csv with 200 rows of realistic OHLC data"
fi

# Run the test script
echo "Running model selection improvement tests..."
python test_model_selection_improvements.py

# Check if the test generated any results
latest_results=$(find . -maxdepth 1 -type d -name "test_results_*" | sort -r | head -1)
if [ -n "$latest_results" ]; then
    echo "Test completed. Results available in $latest_results"
    
    # Check if comparison report was generated
    if [ -f "$latest_results/approach_comparison.csv" ]; then
        echo "Comparison report generated successfully."
        echo "Summary of results:"
        cat "$latest_results/approach_comparison.csv"
    else
        echo "No comparison report found. Check logs for errors."
    fi
else
    echo "No results directory found. Test may have failed completely."
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "Test completed. Check results in test_results_* directory"
else
    echo "Test failed. Check logs for details"
    exit 1
fi