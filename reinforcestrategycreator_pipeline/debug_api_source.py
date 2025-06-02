"""Debug script to understand ApiDataSource behavior."""

from src.data.api_source import ApiDataSource

# Create a simple API source
config = {
    "endpoint": "https://api.example.com/data",
    "response_format": "json",
    "retry_count": 0
}

source = ApiDataSource("test_api", config)

# Check what params are set
print(f"Initial params: {source.params}")
print(f"Config: {source.config}")

# Check if there are any class-level attributes
print(f"Class dict: {ApiDataSource.__dict__}")

# Check instance attributes
print(f"Instance dict: {source.__dict__}")