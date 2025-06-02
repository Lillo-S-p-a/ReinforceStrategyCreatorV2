"""Example usage of the Data Manager component."""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.manager import ConfigManager
from src.artifact_store.local_adapter import LocalFileSystemStore
from src.data.manager import DataManager


def main():
    """Demonstrate Data Manager usage."""
    print("=== Data Manager Example ===\n")
    
    # Initialize components
    print("1. Initializing components...")
    
    # Create config manager
    config_manager = ConfigManager(
        config_dir="configs",
        environment="development"
    )
    config_manager.load_config()
    
    # Create artifact store
    artifact_store = LocalFileSystemStore(
        root_path="./artifacts"
    )
    
    # Create data manager
    data_manager = DataManager(
        config_manager=config_manager,
        artifact_store=artifact_store,
        cache_dir="./cache/data"
    )
    print("✓ Components initialized\n")
    
    # Example 1: CSV Data Source
    print("2. Working with CSV data source...")
    
    # Create sample CSV file
    sample_csv = Path("./data/sample_stock_data.csv")
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    sample_data = pd.DataFrame({
        "date": dates,
        "symbol": "AAPL",
        "open": [150.0 + i * 0.5 for i in range(len(dates))],
        "high": [151.0 + i * 0.5 for i in range(len(dates))],
        "low": [149.0 + i * 0.5 for i in range(len(dates))],
        "close": [150.5 + i * 0.5 for i in range(len(dates))],
        "volume": [1000000 + i * 10000 for i in range(len(dates))]
    })
    sample_data.to_csv(sample_csv, index=False)
    print(f"✓ Created sample CSV: {sample_csv}")
    
    # Register CSV source
    csv_source = data_manager.register_source(
        source_id="stock_prices_csv",
        source_type="csv",
        config={
            "file_path": str(sample_csv),
            "parse_dates": ["date"],
            "index_col": "date"
        }
    )
    print("✓ Registered CSV data source\n")
    
    # Load data from CSV
    print("3. Loading data from CSV source...")
    df_csv = data_manager.load_data("stock_prices_csv")
    print(f"✓ Loaded {len(df_csv)} rows")
    print(f"Columns: {list(df_csv.columns)}")
    print(f"Date range: {df_csv.index.min()} to {df_csv.index.max()}\n")
    
    # Example 2: API Data Source (Mock)
    print("4. Working with API data source...")
    
    # Register API source (using a mock endpoint)
    api_source = data_manager.register_source(
        source_id="market_data_api",
        source_type="api",
        config={
            "endpoint": "https://api.example.com/v1/market-data",
            "method": "GET",
            "response_format": "json",
            "params": {
                "symbols": "AAPL,GOOGL,MSFT",
                "interval": "1d"
            },
            "auth": {
                "type": "api_key",
                "key_name": "X-API-Key",
                "key_value": os.getenv("MARKET_DATA_API_KEY", "demo-key")
            }
        }
    )
    print("✓ Registered API data source")
    print("  Note: This is a mock endpoint for demonstration\n")
    
    # Example 3: Data Versioning
    print("5. Demonstrating data versioning...")
    
    # Save a version of the CSV data
    version1 = data_manager.save_version(
        source_id="stock_prices_csv",
        data=df_csv,
        description="Initial stock price data for AAPL",
        tags=["stock", "AAPL", "daily"]
    )
    print(f"✓ Saved version: {version1}")
    
    # Modify the data (simulate an update)
    df_modified = df_csv.copy()
    df_modified["close"] = df_modified["close"] * 1.02  # 2% increase
    
    # Save another version
    version2 = data_manager.save_version(
        source_id="stock_prices_csv",
        data=df_modified,
        version="v2_adjusted",
        description="Adjusted stock prices with 2% increase",
        tags=["stock", "AAPL", "daily", "adjusted"]
    )
    print(f"✓ Saved version: {version2}")
    
    # List versions
    versions = data_manager.list_versions("stock_prices_csv")
    print(f"✓ Available versions: {versions}\n")
    
    # Example 4: Caching
    print("6. Demonstrating caching...")
    
    # First load (will hit the source)
    start_time = datetime.now()
    df1 = data_manager.load_data("stock_prices_csv")
    load_time1 = (datetime.now() - start_time).total_seconds()
    print(f"✓ First load took: {load_time1:.4f} seconds")
    
    # Second load (should hit cache)
    start_time = datetime.now()
    df2 = data_manager.load_data("stock_prices_csv")
    load_time2 = (datetime.now() - start_time).total_seconds()
    print(f"✓ Second load took: {load_time2:.4f} seconds (from cache)")
    print(f"  Cache speedup: {load_time1/load_time2:.1f}x faster\n")
    
    # Example 5: Data Lineage
    print("7. Examining data lineage...")
    
    lineage = data_manager.get_lineage("stock_prices_csv")
    print(f"✓ Found {len(lineage)} lineage entries:")
    
    for i, entry in enumerate(lineage[-3:], 1):  # Show last 3 entries
        print(f"  {i}. {entry['operation']} at {entry['timestamp']}")
        if "rows" in entry.get("details", {}):
            print(f"     Rows: {entry['details']['rows']}")
    print()
    
    # Example 6: Source Metadata
    print("8. Getting source metadata...")
    
    metadata = data_manager.get_source_metadata("stock_prices_csv")
    if metadata:
        print(f"✓ Source ID: {metadata.source_id}")
        print(f"  Type: {metadata.source_type}")
        print(f"  Schema: {metadata.schema}")
        print(f"  Properties: {metadata.properties}\n")
    
    # Example 7: Multi-source Data Loading
    print("9. Demonstrating multi-source capabilities...")
    
    # Create another CSV with different data
    sample_csv2 = Path("./data/sample_fundamentals.csv")
    fundamentals_data = pd.DataFrame({
        "date": dates[::3],  # Every 3 days
        "symbol": "AAPL",
        "pe_ratio": [25.5, 26.0, 26.5, 27.0],
        "market_cap": [2.5e12, 2.52e12, 2.54e12, 2.56e12]
    })
    fundamentals_data.to_csv(sample_csv2, index=False)
    
    # Register second source
    data_manager.register_source(
        source_id="fundamentals_csv",
        source_type="csv",
        config={
            "file_path": str(sample_csv2),
            "parse_dates": ["date"]
        }
    )
    
    # Load from both sources
    prices = data_manager.load_data("stock_prices_csv")
    fundamentals = data_manager.load_data("fundamentals_csv")
    
    print(f"✓ Loaded data from multiple sources:")
    print(f"  Stock prices: {len(prices)} rows")
    print(f"  Fundamentals: {len(fundamentals)} rows\n")
    
    # Example 8: Cache Management
    print("10. Cache management...")
    
    # Clear cache for specific source
    cleared = data_manager.clear_cache("stock_prices_csv")
    print(f"✓ Cleared {cleared} cache entries for stock_prices_csv")
    
    # Clear all cache
    cleared_all = data_manager.clear_cache()
    print(f"✓ Cleared {cleared_all} total cache entries\n")
    
    print("=== Example Complete ===")
    print("\nKey features demonstrated:")
    print("- CSV and API data source registration")
    print("- Data loading with automatic caching")
    print("- Data versioning with artifact store")
    print("- Lineage tracking for audit trail")
    print("- Multi-source data management")
    print("- Cache management and optimization")


if __name__ == "__main__":
    main()