#!/bin/bash
# Script to run model selection test with HPO enabled

echo "Starting model selection test with HPO enabled..."

# Make the script executable
chmod +x test_model_selection_improvements.py

# Run the test with HPO enabled
python test_model_selection_improvements.py --hpo

echo "Test completed. Check the results directory for output."