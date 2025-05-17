"""
Simple test script to verify the TechnicalAnalyzer compatibility wrapper.
"""

import pandas as pd
import numpy as np

# Import the TechnicalAnalyzer class and calculate_indicators function
from reinforcestrategycreator.technical_analyzer import TechnicalAnalyzer, calculate_indicators

def main():
    """
    Test the TechnicalAnalyzer compatibility wrapper directly.
    """
    print("Testing TechnicalAnalyzer compatibility wrapper...")
    
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(105, 115, 100),
        'Low': np.random.uniform(95, 105, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.randint(1000000, 2000000, 100)
    }, index=dates)
    
    # Test the function-based approach
    print("\nTesting function-based approach (calculate_indicators)...")
    try:
        result_func = calculate_indicators(test_data)
        print(f"Successfully processed data with {len(result_func)} rows")
        
        # Check if technical indicators were added
        indicator_columns = [col for col in result_func.columns if col not in test_data.columns]
        print(f"Added indicator columns: {indicator_columns}")
        print("Function-based approach test passed!")
    except Exception as e:
        print(f"Error during function-based test: {e}")
    
    # Test the class-based approach
    print("\nTesting class-based approach (TechnicalAnalyzer.add_all_indicators)...")
    try:
        analyzer = TechnicalAnalyzer()
        result_class = analyzer.add_all_indicators(test_data)
        print(f"Successfully processed data with {len(result_class)} rows")
        
        # Check if technical indicators were added
        indicator_columns = [col for col in result_class.columns if col not in test_data.columns]
        print(f"Added indicator columns: {indicator_columns}")
        print("Class-based approach test passed!")
    except Exception as e:
        print(f"Error during class-based test: {e}")
    
    # Verify that both approaches produce the same result
    print("\nVerifying that both approaches produce the same result...")
    try:
        pd.testing.assert_frame_equal(result_func, result_class)
        print("Both approaches produce identical results!")
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"Error: Results from the two approaches differ: {e}")
        return False

if __name__ == "__main__":
    main()