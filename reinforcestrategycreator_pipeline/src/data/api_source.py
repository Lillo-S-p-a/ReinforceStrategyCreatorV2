"""API data source implementation."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import DataSource


class ApiDataSource(DataSource):
    """Data source for loading data from APIs."""
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """Initialize API data source.
        
        Args:
            source_id: Unique identifier for this data source
            config: Configuration dictionary containing:
                - endpoint: API endpoint URL
                - method: HTTP method (default: 'GET')
                - headers: Dictionary of HTTP headers
                - params: Dictionary of query parameters
                - auth: Authentication configuration
                - timeout: Request timeout in seconds (default: 30)
                - retry_count: Number of retries (default: 3)
                - response_format: Expected response format ('json', 'csv')
                - data_path: Path to data in JSON response (e.g., 'data.items')
        """
        super().__init__(source_id, config)
        self.endpoint = config.get("endpoint", "")
        self.method = config.get("method", "GET").upper()
        self.headers = config.get("headers", {})
        self.params = config.get("params", {})
        self.auth = config.get("auth", None)
        self.timeout = config.get("timeout", 30)
        self.retry_count = config.get("retry_count", 3)
        self.response_format = config.get("response_format", "json")
        self.data_path = config.get("data_path", None)
        
        # Set up session with retry strategy
        self.session = self._create_session()
        
        # Validate configuration on initialization
        self.validate_config()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy.
        
        Returns:
            Configured requests Session
        """
        session = requests.Session()
        
        # Only configure retry strategy if retry_count > 0
        if self.retry_count > 0:
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.retry_count,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        
        # Set default headers
        if self.headers:
            session.headers.update(self.headers)
        
        return session
    
    def validate_config(self) -> bool:
        """Validate the API data source configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.endpoint:
            raise ValueError("API data source requires 'endpoint' in config")
        
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("API endpoint must start with http:// or https://")
        
        if self.method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            raise ValueError(f"Invalid HTTP method: {self.method}")
        
        if self.response_format not in ["json", "csv"]:
            raise ValueError(f"Unsupported response format: {self.response_format}")
        
        return True
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from the API.
        
        Args:
            **kwargs: Additional parameters:
                - extra_params: Additional query parameters
                - extra_headers: Additional headers
                - Any other kwargs are treated as additional query parameters
                
        Returns:
            DataFrame containing the loaded data
        """
        # Prepare request parameters
        params = self.params.copy()
        
        # Handle extra_params if provided
        if "extra_params" in kwargs:
            params.update(kwargs.pop("extra_params"))
        
        # Handle extra_headers if provided
        headers = self.headers.copy()
        if "extra_headers" in kwargs:
            headers.update(kwargs.pop("extra_headers"))
        
        # Treat any remaining kwargs as additional query parameters
        params.update(kwargs)
        
        # Update lineage before loading
        self.update_lineage("load_data", {
            "endpoint": self.endpoint,
            "method": self.method,
            "params": params,
            "headers": {k: v for k, v in headers.items() if k.lower() != "authorization"}
        })
        
        try:
            # Make the API request
            response = self.session.request(
                method=self.method,
                url=self.endpoint,
                params=params if self.method == "GET" else None,
                json=params if self.method in ["POST", "PUT", "PATCH"] else None,
                headers=headers,
                timeout=self.timeout,
                auth=self._get_auth()
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response based on format
            if self.response_format == "json":
                df = self._parse_json_response(response)
            elif self.response_format == "csv":
                df = self._parse_csv_response(response)
            else:
                raise ValueError(f"Unsupported response format: {self.response_format}")
            
            # Update lineage with success info
            self.update_lineage("load_complete", {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "rows": len(df),
                "columns": list(df.columns)
            })
            
            return df
            
        except Exception as e:
            # Update lineage with error info
            self.update_lineage("load_error", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            raise
    
    def _get_auth(self) -> Optional[Any]:
        """Get authentication object based on config.
        
        Returns:
            Authentication object or None
        """
        if not self.auth:
            return None
        
        auth_type = self.auth.get("type", "").lower()
        
        if auth_type == "basic":
            return (self.auth.get("username"), self.auth.get("password"))
        elif auth_type == "bearer":
            token = self.auth.get("token")
            if token:
                self.session.headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "api_key":
            key_name = self.auth.get("key_name", "X-API-Key")
            key_value = self.auth.get("key_value")
            if key_value:
                self.session.headers[key_name] = key_value
        
        return None
    
    def _parse_json_response(self, response: requests.Response) -> pd.DataFrame:
        """Parse JSON response into DataFrame.
        
        Args:
            response: API response object
            
        Returns:
            DataFrame containing the parsed data
        """
        data = response.json()
        
        # Navigate to data path if specified
        if self.data_path:
            for key in self.data_path.split("."):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise ValueError(f"Data path '{self.data_path}' not found in response")
        
        # Convert to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If dict, try to find a list value
            for value in data.values():
                if isinstance(value, list):
                    return pd.DataFrame(value)
            # Otherwise, treat as single row
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Cannot convert {type(data)} to DataFrame")
    
    def _parse_csv_response(self, response: requests.Response) -> pd.DataFrame:
        """Parse CSV response into DataFrame.
        
        Args:
            response: API response object
            
        Returns:
            DataFrame containing the parsed data
        """
        from io import StringIO
        return pd.read_csv(StringIO(response.text))
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema by making a sample request.
        
        Returns:
            Dictionary mapping column names to data types
        """
        try:
            # Make a request with limited data if possible
            params = self.params.copy()
            params.update({
                "limit": 10,  # Common parameter for limiting results
                "page_size": 10,  # Alternative parameter
                "per_page": 10,  # Another alternative
            })
            
            df_sample = self.load_data(extra_params=params)
            
            schema = {}
            for col, dtype in df_sample.dtypes.items():
                schema[str(col)] = str(dtype)
            
            return schema
            
        except Exception:
            # Return empty schema if we can't get sample data
            return {}
    
    def test_connection(self) -> bool:
        """Test the API connection.
        
        Returns:
            True if connection is successful
        """
        try:
            response = self.session.request(
                method="HEAD" if self.method == "GET" else self.method,
                url=self.endpoint,
                timeout=10,
                auth=self._get_auth()
            )
            return response.status_code < 400
        except Exception:
            return False