"""Unit tests for API data source."""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import timedelta
import requests
import pandas as pd

from reinforcestrategycreator_pipeline.src.data.api_source import ApiDataSource


class TestApiDataSource:
    """Test ApiDataSource class."""
    
    @pytest.fixture
    def mock_response(self):
        """Create a mock response object."""
        response = Mock(spec=requests.Response)
        response.status_code = 200
        # Create a mock timedelta for elapsed
        mock_elapsed = Mock(spec=timedelta)
        mock_elapsed.total_seconds.return_value = 0.5
        response.elapsed = mock_elapsed
        response.raise_for_status = Mock()
        return response
    
    def test_api_source_initialization(self):
        """Test initializing API data source."""
        config = {
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"X-API-Key": "test-key"},
            "params": {"limit": 100},
            "timeout": 60,
            "retry_count": 5,
            "response_format": "json"
        }
        
        source = ApiDataSource("test_api", config)
        
        assert source.source_id == "test_api"
        assert source.endpoint == "https://api.example.com/data"
        assert source.method == "GET"
        assert source.headers == {"X-API-Key": "test-key"}
        assert source.params == {"limit": 100}
        assert source.timeout == 60
        assert source.retry_count == 5
        assert source.response_format == "json"
    
    def test_validate_config_success(self):
        """Test successful config validation."""
        config = {
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "response_format": "json"
        }
        source = ApiDataSource("test_api", config)
        
        assert source.validate_config() is True
    
    def test_validate_config_missing_endpoint(self):
        """Test validation with missing endpoint."""
        config = {}
        
        with pytest.raises(ValueError, match="requires 'endpoint'"):
            ApiDataSource("test_api", config)
    
    def test_validate_config_invalid_endpoint(self):
        """Test validation with invalid endpoint."""
        config = {"endpoint": "not-a-url"}
        
        with pytest.raises(ValueError, match="must start with http"):
            ApiDataSource("test_api", config)
    
    def test_validate_config_invalid_method(self):
        """Test validation with invalid HTTP method."""
        config = {
            "endpoint": "https://api.example.com",
            "method": "INVALID"
        }
        
        with pytest.raises(ValueError, match="Invalid HTTP method"):
            ApiDataSource("test_api", config)
    
    def test_validate_config_invalid_format(self):
        """Test validation with unsupported response format."""
        config = {
            "endpoint": "https://api.example.com",
            "response_format": "xml"
        }
        
        with pytest.raises(ValueError, match="Unsupported response format"):
            ApiDataSource("test_api", config)
    
    @patch('reinforcestrategycreator_pipeline.src.data.api_source.requests.Session')
    @patch.object(ApiDataSource, 'get_schema', return_value={})
    def test_load_data_json_list(self, mock_get_schema, mock_session_class, mock_response):
        """Test loading JSON data that returns a list."""
        # Setup mock session instance
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock response with list data
        mock_response.json.return_value = [
            {"id": 1, "name": "Alice", "value": 100},
            {"id": 2, "name": "Bob", "value": 200}
        ]
        mock_session.request.return_value = mock_response
        
        # Mock mount method (called by retry setup)
        mock_session.mount = Mock()
        
        config = {
            "endpoint": "https://api.example.com/data",
            "response_format": "json",
            "retry_count": 0  # Disable retries for testing
        }
        
        source = ApiDataSource("test_api", config)
        df = source.load_data()
        
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
        assert df["name"].tolist() == ["Alice", "Bob"]
        
        # Verify request was made correctly
        mock_session.request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/data",
            params={},
            json=None,
            headers={},
            timeout=30,
            auth=None
        )
    
    @patch('requests.Session')
    def test_load_data_json_dict_with_list(self, mock_session_class, mock_response):
        """Test loading JSON data that returns a dict containing a list."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock response with dict containing list
        mock_response.json.return_value = {
            "status": "success",
            "data": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "count": 2
        }
        mock_session.request.return_value = mock_response
        
        config = {
            "endpoint": "https://api.example.com/data",
            "response_format": "json"
        }
        source = ApiDataSource("test_api", config)
        
        df = source.load_data()
        
        # Should extract the list from the dict
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
    
    @patch('requests.Session')
    def test_load_data_json_with_data_path(self, mock_session_class, mock_response):
        """Test loading JSON data with specified data path."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock nested response
        mock_response.json.return_value = {
            "response": {
                "results": {
                    "items": [
                        {"id": 1, "value": 100},
                        {"id": 2, "value": 200}
                    ]
                }
            }
        }
        mock_session.request.return_value = mock_response
        
        config = {
            "endpoint": "https://api.example.com/data",
            "response_format": "json",
            "data_path": "response.results.items"
        }
        source = ApiDataSource("test_api", config)
        
        df = source.load_data()
        
        assert len(df) == 2
        assert df["value"].tolist() == [100, 200]
    
    @patch('requests.Session')
    def test_load_data_csv_format(self, mock_session_class, mock_response):
        """Test loading CSV formatted data."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock CSV response
        mock_response.text = "id,name,value\n1,Alice,100\n2,Bob,200\n"
        mock_session.request.return_value = mock_response
        
        config = {
            "endpoint": "https://api.example.com/data.csv",
            "response_format": "csv"
        }
        source = ApiDataSource("test_api", config)
        
        df = source.load_data()
        
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
        assert df["name"].tolist() == ["Alice", "Bob"]
    
    @patch('reinforcestrategycreator_pipeline.src.data.api_source.requests.Session')
    @patch.object(ApiDataSource, 'get_schema', return_value={})
    def test_load_data_with_params(self, mock_get_schema, mock_session_class, mock_response):
        """Test loading data with query parameters."""
        # Setup mock session instance
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock response
        mock_response.json.return_value = [{"id": 1, "status": "active"}]
        mock_session.request.return_value = mock_response
        
        # Mock mount method (called by retry setup)
        mock_session.mount = Mock()
        
        config = {
            "endpoint": "https://api.example.com/data",
            "params": {
                "filter": "active",
                "limit": 10
            },
            "retry_count": 0  # Disable retries for testing
        }
        
        source = ApiDataSource("test_api", config)
        # Load with additional params
        df = source.load_data(page=2, sort="desc")
        
        assert len(df) == 1
        
        # Verify merged params
        mock_session.request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/data",
            params={
                "filter": "active",
                "limit": 10,
                "page": 2,
                "sort": "desc"
            },
            json=None,
            headers={},
            timeout=30,
            auth=None
        )
    
    @patch('reinforcestrategycreator_pipeline.src.data.api_source.requests.Session')
    @patch.object(ApiDataSource, 'get_schema', return_value={})
    def test_load_data_post_method(self, mock_get_schema, mock_session_class, mock_response):
        """Test loading data with POST method."""
        # Setup mock session instance
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock response
        mock_response.json.return_value = {"success": True, "data": [{"id": 1}]}
        mock_session.request.return_value = mock_response
        
        # Mock mount method (called by retry setup)
        mock_session.mount = Mock()
        
        config = {
            "endpoint": "https://api.example.com/data",
            "method": "POST",
            "params": {
                "filter": "active",
                "limit": 10
            },
            "retry_count": 0  # Disable retries for testing
        }
        
        source = ApiDataSource("test_api", config)
        # Load with additional params
        df = source.load_data(page_size=10, per_page=10)
        
        assert len(df) == 1
        
        # For POST, params should be in json body
        mock_session.request.assert_called_once_with(
            method="POST",
            url="https://api.example.com/data",
            params=None,
            json={
                "filter": "active",
                "limit": 10,
                "page_size": 10,
                "per_page": 10
            },
            headers={},
            timeout=30,
            auth=None
        )
    
    def test_auth_basic(self):
        """Test basic authentication setup."""
        config = {
            "endpoint": "https://api.example.com/data",
            "auth": {
                "type": "basic",
                "username": "user",
                "password": "pass"
            }
        }
        source = ApiDataSource("test_api", config)
        
        auth = source._get_auth()
        assert auth == ("user", "pass")
    
    def test_auth_bearer(self):
        """Test bearer token authentication setup."""
        config = {
            "endpoint": "https://api.example.com/data",
            "auth": {
                "type": "bearer",
                "token": "test-token-123"
            }
        }
        source = ApiDataSource("test_api", config)
        
        # Bearer token should be set in headers
        source._get_auth()
        assert source.session.headers["Authorization"] == "Bearer test-token-123"
    
    def test_auth_api_key(self):
        """Test API key authentication setup."""
        config = {
            "endpoint": "https://api.example.com/data",
            "auth": {
                "type": "api_key",
                "key_name": "X-Custom-Key",
                "key_value": "secret-key"
            }
        }
        source = ApiDataSource("test_api", config)
        
        # API key should be set in headers
        source._get_auth()
        assert source.session.headers["X-Custom-Key"] == "secret-key"
    
    @patch('requests.Session')
    def test_error_handling(self, mock_session_class, mock_response):
        """Test error handling during load."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock error response
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_session.request.return_value = mock_response
        
        config = {"endpoint": "https://api.example.com/data"}
        source = ApiDataSource("test_api", config)
        
        with pytest.raises(requests.HTTPError):
            source.load_data()
        
        # Check error was tracked in lineage
        metadata = source.get_metadata()
        operations = metadata.lineage.get("operations", [])
        error_op = next((op for op in operations if op["operation"] == "load_error"), None)
        
        assert error_op is not None
        assert error_op["details"]["error_type"] == "HTTPError"
        assert "404 Not Found" in error_op["details"]["error_message"]
    
    @patch('requests.Session')
    def test_get_schema(self, mock_session_class, mock_response):
        """Test getting schema from API."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock response for schema request
        mock_response.json.return_value = [
            {"id": 1, "name": "Test", "value": 100.5, "active": True}
        ]
        mock_session.request.return_value = mock_response
        
        config = {"endpoint": "https://api.example.com/data"}
        source = ApiDataSource("test_api", config)
        
        schema = source.get_schema()
        
        assert "id" in schema
        assert "name" in schema
        assert "value" in schema
        assert "active" in schema
        assert schema["id"] == "int64"
        assert schema["name"] == "object"
        assert schema["value"] == "float64"
        assert schema["active"] == "bool"
    
    @patch('requests.Session')
    def test_test_connection(self, mock_session_class, mock_response):
        """Test connection testing."""
        # Setup mock
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock successful response
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response
        
        config = {"endpoint": "https://api.example.com/data"}
        source = ApiDataSource("test_api", config)
        
        assert source.test_connection() is True
        
        # Test failed connection
        mock_response.status_code = 500
        assert source.test_connection() is False
        
        # Test connection error
        mock_session.request.side_effect = requests.ConnectionError()
        assert source.test_connection() is False
    
    def test_lineage_tracking(self):
        """Test that lineage is tracked during operations."""
        with patch('requests.Session') as mock_session_class:
            # Setup mock
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 1.5
            mock_response.json.return_value = [{"id": 1}, {"id": 2}]
            mock_response.raise_for_status = Mock()
            mock_session.request.return_value = mock_response
            
            config = {"endpoint": "https://api.example.com/data"}
            source = ApiDataSource("test_api", config)
            
            # Load data
            df = source.load_data()
            
            # Check lineage
            metadata = source.get_metadata()
            operations = metadata.lineage.get("operations", [])
            
            assert len(operations) >= 2  # load_data and load_complete
            
            # Check load_data operation
            load_op = next(op for op in operations if op["operation"] == "load_data")
            assert load_op["details"]["endpoint"] == "https://api.example.com/data"
            assert load_op["details"]["method"] == "GET"
            
            # Check load_complete operation
            complete_op = next(op for op in operations if op["operation"] == "load_complete")
            assert complete_op["details"]["status_code"] == 200
            assert complete_op["details"]["response_time"] == 1.5
            assert complete_op["details"]["rows"] == 2