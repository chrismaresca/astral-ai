# -------------------------------------------------------------------------------- #
# Base Provider Client Tests
# -------------------------------------------------------------------------------- #
"""
Tests for the BaseProviderClient class in Astral AI.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import time
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Astral AI imports
from astral_ai.providers._base_client import BaseProviderClient, read_config
from astral_ai.constants._models import ModelProvider
from astral_ai.errors.exceptions import (
    ProviderAuthenticationError,
    UnknownAuthMethodError,
    AuthMethodFailureError
)
from astral_ai._auth import auth_method, AUTH_CONFIG_TYPE, AUTH_ENV_VARS

# -------------------------------------------------------------------------------- #
# Mock Client Implementation
# -------------------------------------------------------------------------------- #


class MockProviderClient(BaseProviderClient):
    """A mock implementation of BaseProviderClient for testing."""
    
    _model_provider = "mock_provider"
    
    @auth_method("api_key")
    def auth_via_api_key(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> Any:
        """Authenticate using API key."""
        print(f"\n--- Attempting API Key Authentication ---")
        api_key = config.get("api_key") or env.get("MOCK_API_KEY")
        print(f"Using API key: {api_key}")
        
        if not api_key:
            print("No API key found")
            raise ProviderAuthenticationError("No API key provided for mock authentication.")
            
        if api_key == "invalid_key":
            print("Invalid API key provided")
            raise ProviderAuthenticationError("Invalid API key provided.")
            
        print("API Key authentication successful")
        return MagicMock(name="mock_client_from_api_key")
    
    @auth_method("oauth")
    def auth_via_oauth(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> Any:
        """Authenticate using OAuth token."""
        print(f"\n--- Attempting OAuth Authentication ---")
        oauth_token = config.get("oauth_token") or env.get("MOCK_OAUTH_TOKEN")
        print(f"Using OAuth token: {oauth_token}")
        
        if not oauth_token:
            print("No OAuth token found")
            raise ProviderAuthenticationError("No OAuth token provided for mock authentication.")
            
        if oauth_token == "invalid_token":
            print("Invalid OAuth token provided")
            raise ProviderAuthenticationError("Invalid OAuth token provided.")
            
        print("OAuth authentication successful")
        return MagicMock(name="mock_client_from_oauth")
    
    def create_completion_chat(self, request):
        """Mock implementation of create_completion_chat."""
        print("\n--- Called create_completion_chat ---")
        print(f"Request: {request}")
        return {"mock_chat_response": True}
    
    def create_completion_structured(self, request):
        """Mock implementation of create_completion_structured."""
        print("\n--- Called create_completion_structured ---")
        print(f"Request: {request}")
        return {"mock_structured_response": True}

# -------------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------------- #


@pytest.fixture
def mock_config_file_content():
    """Mock content for the config file."""
    return """
    mock_provider:
      auth_method: "api_key"
      api_key: "test_api_key_from_file"
      cache_client: true
    other_provider:
      auth_method: "oauth"
      oauth_token: "test_oauth_token"
    """


@pytest.fixture
def mock_api_key_env():
    """Set up environment with API key."""
    with patch.dict(os.environ, {"MOCK_API_KEY": "test_api_key_from_env"}, clear=True):
        yield


@pytest.fixture
def mock_oauth_env():
    """Set up environment with OAuth token."""
    with patch.dict(os.environ, {"MOCK_OAUTH_TOKEN": "test_oauth_token_from_env"}, clear=True):
        yield


@pytest.fixture
def mock_invalid_api_key_env():
    """Set up environment with invalid API key."""
    with patch.dict(os.environ, {"MOCK_API_KEY": "invalid_key"}, clear=True):
        yield


@pytest.fixture
def mock_config_api_key():
    """Config dictionary with API key."""
    return {
        "mock_provider": {
            "auth_method": "api_key",
            "api_key": "test_api_key_from_config",
            "cache_client": True
        }
    }


@pytest.fixture
def mock_config_oauth():
    """Config dictionary with OAuth token."""
    return {
        "mock_provider": {
            "auth_method": "oauth",
            "oauth_token": "test_oauth_token_from_config",
            "cache_client": True
        }
    }


@pytest.fixture
def mock_unknown_auth_method_config():
    """Config dictionary with unknown auth method."""
    return {
        "mock_provider": {
            "auth_method": "unknown_method",
            "cache_client": True
        }
    }

# -------------------------------------------------------------------------------- #
# Tests - Configuration Loading
# -------------------------------------------------------------------------------- #


def test_read_config_file_exists(mock_config_file_content):
    """Test reading config from file when file exists."""
    print("\n=== Starting test_read_config_file_exists ===")
    
    config_path = Path("astral.yaml")
    
    with patch("pathlib.Path.open", mock_open(read_data=mock_config_file_content)) as mock_file:
        with patch("pathlib.Path.exists", return_value=True):
            print("\n--- Reading configuration file ---")
            config = read_config(config_path)
            
            print(f"Loaded config: {config}")
            assert mock_file.call_count > 0, "open should be called at least once"
            
            assert config is not None, "Config should not be None"
            assert "mock_provider" in config, "mock_provider should be in config"
            assert config["mock_provider"]["auth_method"] == "api_key", "Auth method should be api_key"
            assert config["mock_provider"]["api_key"] == "test_api_key_from_file", "API key incorrect"
            
    print("✓ Configuration file read successfully")
    print("=== test_read_config_file_exists completed successfully ===")


def test_read_config_file_not_exists():
    """Test reading config when file doesn't exist."""
    print("\n=== Starting test_read_config_file_not_exists ===")
    
    config_path = Path("nonexistent.yaml")
    
    with patch("pathlib.Path.exists", return_value=False):
        print("\n--- Attempting to read non-existent configuration file ---")
        config = read_config(config_path)
        
        print(f"Result: {config}")
        assert config is None, "Config should be None when file doesn't exist"
        
    print("✓ Non-existent configuration file handled correctly")
    print("=== test_read_config_file_not_exists completed successfully ===")


def test_get_provider_config():
    """Test extracting provider-specific config."""
    print("\n=== Starting test_get_provider_config ===")
    
    full_config = {
        "mock_provider": {"api_key": "test_key", "option1": "value1"},
        "other_provider": {"oauth_token": "test_token", "option2": "value2"}
    }
    
    print(f"\n--- Full config: {full_config}")
    
    client = MockProviderClient(full_config)
    
    print(f"\n--- Provider-specific config: {client._config}")
    
    assert client._config == full_config["mock_provider"], "Provider config should match mock_provider section"
    assert "api_key" in client._config, "api_key should be in provider config"
    assert "option1" in client._config, "option1 should be in provider config"
        
    print("✓ Provider-specific config extracted correctly")
    print("=== test_get_provider_config completed successfully ===")

# -------------------------------------------------------------------------------- #
# Tests - Authentication
# -------------------------------------------------------------------------------- #


def test_auth_with_config_api_key():
    """Test authentication using API key from config."""
    print("\n=== Starting test_auth_with_config_api_key ===")
    
    config = {
        "mock_provider": {
            "api_key": "test_api_key_from_config",
            "cache_client": True
        }
    }
    
    print(f"\n--- Config: {config}")
    
    client = MockProviderClient(config)
    
    print(f"\n--- Client initialized, provider: {client._model_provider}")
    print(f"Client config: {client._config}")
    
    assert client.client is not None, "Client should be initialized"
    assert client.client._mock_name == "mock_client_from_api_key", "Should use API key auth"
    
    print("✓ Client authenticated successfully with API key from config")
    print("=== test_auth_with_config_api_key completed successfully ===")


def test_auth_with_env_api_key(mock_api_key_env):
    """Test authentication using API key from environment."""
    print("\n=== Starting test_auth_with_env_api_key ===")
    
    print(f"\n--- Environment: MOCK_API_KEY={os.environ.get('MOCK_API_KEY')}")
    
    with patch('astral_ai._auth.get_env_vars', return_value={"MOCK_API_KEY": "test_api_key_from_env"}):
        client = MockProviderClient({})
        
        print(f"\n--- Client initialized, provider: {client._model_provider}")
        print(f"Client config: {client._config}")
        
        assert client.client is not None, "Client should be initialized"
        assert client.client._mock_name == "mock_client_from_api_key", "Should use API key auth"
        
    print("✓ Client authenticated successfully with API key from environment")
    print("=== test_auth_with_env_api_key completed successfully ===")


def test_auth_method_selection():
    """Test selecting auth method based on config."""
    print("\n=== Starting test_auth_method_selection ===")
    
    # Create an AuthMethodConfig-like object
    class AuthMethodConfig:
        def __init__(self, method):
            self.auth_method = method
    
    config = {
        "mock_provider": {
            "auth_method": AuthMethodConfig("oauth"),  # Use an object with auth_method attribute
            "oauth_token": "test_oauth_token_from_config",
            "cache_client": True
        }
    }
    
    print(f"\n--- Config with OAuth method: {config}")
    
    client = MockProviderClient(config)
    
    print(f"\n--- Client initialized, provider: {client._model_provider}")
    print(f"Client config: {client._config}")
    
    assert client.client is not None, "Client should be initialized"
    assert client.client._mock_name == "mock_client_from_oauth", "Should use OAuth auth"
    
    print("✓ Client selected OAuth authentication method correctly")
    print("=== test_auth_method_selection completed successfully ===")


def test_auth_unknown_method():
    """Test error handling for unknown auth method."""
    print("\n=== Starting test_auth_unknown_method ===")
    
    # Create an AuthMethodConfig-like object
    class AuthMethodConfig:
        def __init__(self, method):
            self.auth_method = method
    
    config = {
        "mock_provider": {
            "auth_method": AuthMethodConfig("unknown_method"),  # Use an object with auth_method attribute
            "cache_client": True
        }
    }
    
    print(f"\n--- Config with unknown method: {config}")
    
    with pytest.raises(UnknownAuthMethodError) as excinfo:
        client = MockProviderClient(config)
        
    print(f"\n--- Error raised: {excinfo.value}")
    
    assert "unknown_method" in str(excinfo.value), "Error should mention the unknown method"
    
    print("✓ Client raised appropriate error for unknown auth method")
    print("=== test_auth_unknown_method completed successfully ===")


def test_auth_failure():
    """Test handling of authentication failure."""
    print("\n=== Starting test_auth_failure ===")
    
    config = {
        "mock_provider": {
            "api_key": "invalid_key",
            "cache_client": True
        }
    }
    
    print(f"\n--- Config with invalid API key: {config}")
    
    with pytest.raises(ProviderAuthenticationError) as excinfo:
        client = MockProviderClient(config)
        
    print(f"\n--- Error raised: {excinfo.value}")
    
    assert "Invalid API key provided" in str(excinfo.value), "Error should indicate invalid API key"
    
    print("✓ Client raised appropriate error for authentication failure")
    print("=== test_auth_failure completed successfully ===")

# -------------------------------------------------------------------------------- #
# Tests - Client Caching
# -------------------------------------------------------------------------------- #


def test_client_caching():
    """Test client caching behavior."""
    print("\n=== Starting test_client_caching ===")
    
    # Reset the client cache
    MockProviderClient._client_cache = {}
    
    config = {
        "mock_provider": {
            "api_key": "test_api_key",
            "cache_client": True
        }
    }
    
    print(f"\n--- Config with caching enabled: {config}")
    
    print("\n--- Initializing first client ---")
    client1 = MockProviderClient(config)
    
    print("\n--- Initializing second client ---")
    client2 = MockProviderClient(config)
    
    print(f"\n--- Client cache size: {len(MockProviderClient._client_cache)}")
    print(f"Client1 id: {id(client1.client)}")
    print(f"Client2 id: {id(client2.client)}")
    
    assert id(client1.client) == id(client2.client), "Clients should be the same instance when caching is enabled"
    
    print("✓ Client caching works correctly")
    print("=== test_client_caching completed successfully ===")


def test_client_no_caching():
    """Test behavior when caching is disabled."""
    print("\n=== Starting test_client_no_caching ===")
    
    # Reset the client cache
    MockProviderClient._client_cache = {}
    
    config = {
        "mock_provider": {
            "api_key": "test_api_key",
            "cache_client": False
        }
    }
    
    print(f"\n--- Config with caching disabled: {config}")
    
    print("\n--- Initializing first client ---")
    client1 = MockProviderClient(config)
    
    print("\n--- Initializing second client ---")
    client2 = MockProviderClient(config)
    
    print(f"\n--- Client cache size: {len(MockProviderClient._client_cache)}")
    print(f"Client1 id: {id(client1.client)}")
    print(f"Client2 id: {id(client2.client)}")
    
    assert id(client1.client) != id(client2.client), "Clients should be different instances when caching is disabled"
    
    print("✓ Client caching disabled correctly")
    print("=== test_client_no_caching completed successfully ===")

# -------------------------------------------------------------------------------- #
# Tests - API Methods
# -------------------------------------------------------------------------------- #


def test_completion_methods():
    """Test the completion methods."""
    print("\n=== Starting test_completion_methods ===")
    
    config = {
        "mock_provider": {
            "api_key": "test_api_key",
        }
    }
    
    print("\n--- Initializing client ---")
    client = MockProviderClient(config)
    
    print("\n--- Testing create_completion_chat ---")
    chat_request = {"messages": [{"role": "user", "content": "Hello"}]}
    chat_response = client.create_completion_chat(chat_request)
    
    print(f"Chat response: {chat_response}")
    assert chat_response["mock_chat_response"] is True, "Should return mock chat response"
    
    print("\n--- Testing create_completion_structured ---")
    structured_request = {"messages": [{"role": "user", "content": "Hello"}], "response_format": {"type": "json_object"}}
    structured_response = client.create_completion_structured(structured_request)
    
    print(f"Structured response: {structured_response}")
    assert structured_response["mock_structured_response"] is True, "Should return mock structured response"
    
    print("✓ Completion methods work correctly")
    print("=== test_completion_methods completed successfully ===")
