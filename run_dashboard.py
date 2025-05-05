"""
Wrapper script to run the modularized dashboard.
This script imports the main module from the dashboard package and runs it.
"""
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the dashboard package
from dashboard.main import main

# Run the dashboard
if __name__ == "__main__":
    main()