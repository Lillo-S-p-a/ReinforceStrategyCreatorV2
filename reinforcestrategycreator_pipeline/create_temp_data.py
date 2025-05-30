import numpy as np
import pandas as pd
from pathlib import Path

def create_sample_data(n_samples=1000, n_features=10):
    """Create sample training data."""
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some pattern
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.1
    
    # Convert to DataFrame for compatibility with data manager
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    return df

if __name__ == "__main__":
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_df = create_sample_data(n_samples=2000, n_features=10) # Using 2000 samples
    file_path = data_dir / "training_data.csv"
    sample_df.to_csv(file_path, index=False)
    print(f"Sample data saved to {file_path}")