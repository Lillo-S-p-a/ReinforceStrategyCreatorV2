#!/bin/bash
set -e

echo "Starting debug run of the trading RL training with enhanced logging"

# Run the debug version of the training script
python train_debug.py

echo "Debug run completed. Check replay_buffer_debug.log for detailed logging information"