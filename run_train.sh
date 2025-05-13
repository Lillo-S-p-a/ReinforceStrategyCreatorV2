#!/bin/bash

# Set environment variables to suppress warnings
export PYTHONWARNINGS="ignore::DeprecationWarning"
export RAY_DISABLE_DEPRECATION_WARNINGS=1
export RAY_DEDUP_LOGS=0
export RLLIB_DISABLE_API_STACK_WARNING=1

# Silence all Ray logs except errors
export RAY_LOGGING_LEVEL=ERROR

# Run the training script using poetry
poetry run python3 train.py
