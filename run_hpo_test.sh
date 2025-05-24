#!/bin/bash
# Script to run hyperparameter optimization test

echo "Starting hyperparameter optimization test..."

# Make the script executable
chmod +x test_hyperparameter_optimization.py

# Run the test
python test_hyperparameter_optimization.py

echo "Test completed. Check the results directory for output."