# -------------------------------------------------------------------------------- #
# DeepSeek Provider Client Tests
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
import traceback

# openai imports
from openai import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAIError,
)

# module imports
from astral_ai.errors.exceptions import (
    AstralProviderAuthenticationError,
    AstralProviderRateLimitError,
    AstralProviderConnectionError,
    AstralProviderStatusError,
    AstralUnexpectedError,
)
from astral_ai.providers.deepseek._client import DeepSeekProviderClient

# -------------------------------------------------------------------------------- #
# Test Setup Classes and Fixtures
# -------------------------------------------------------------------------------- #
class DummyRequest:
    pass

class DummyResponse:
    def __init__(self):
        self.request = DummyRequest()
        self.status_code = 400
        self.headers = {"x-request-id": "dummy-id"}

dummy_request = DummyRequest()
dummy_response = DummyResponse()

def make_dummy_client(error_instance: OpenAIError):
    class DummyCompletions:
        def create(self, **kwargs):
            raise error_instance

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    return DummyClient()

@pytest.fixture
def deepseek_client():
    client = DeepSeekProviderClient(config={})
    client._model_provider = "deepseek"
    return client

# -------------------------------------------------------------------------------- #
# Test Cases
# -------------------------------------------------------------------------------- #
def test_deepseek_authentication_error(deepseek_client):
    """Test authentication error handling with DeepSeek"""
    original_error = AuthenticationError("Original authentication error", response=dummy_response, body={})
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralProviderAuthenticationError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Authentication Error Verbose Message ---\n")
    print(error_message)
    assert "AUTHENTICATION" in error_message
    assert "deepseek" in error_message.lower()


def test_deepseek_rate_limit_error(deepseek_client):
    """Test rate limit error handling with DeepSeek"""
    original_error = RateLimitError("Original rate limit error", response=dummy_response, body={})
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralProviderRateLimitError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Rate Limit Error Verbose Message ---\n")
    print(error_message)
    assert "RATE LIMIT" in error_message
    assert "deepseek" in error_message.lower()


def test_deepseek_connection_error(deepseek_client):
    """Test connection error handling with DeepSeek"""
    original_error = APIConnectionError(message="Original connection error", request=dummy_request)
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralProviderConnectionError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Connection Error Verbose Message ---\n")
    print(error_message)
    assert "CONNECTION" in error_message
    assert "deepseek" in error_message.lower()


def test_deepseek_timeout_error(deepseek_client):
    """Test timeout error handling with DeepSeek"""
    original_error = APITimeoutError(request=dummy_request)
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralProviderConnectionError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Timeout Error Verbose Message ---\n")
    print(error_message)
    assert "CONNECTION" in error_message  # Timeout mapped to connection error
    assert "deepseek" in error_message.lower()


def test_deepseek_status_error(deepseek_client):
    """Test status error handling with DeepSeek"""
    original_error = APIStatusError("Original status error", response=dummy_response, body={})
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralProviderStatusError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Status Error Verbose Message ---\n")
    print(error_message)
    assert "STATUS" in error_message
    assert "deepseek" in error_message.lower()


def test_deepseek_unexpected_error(deepseek_client):
    """Test unexpected error handling with DeepSeek"""
    class DummyError(OpenAIError):
        pass
    original_error = DummyError("Original unexpected error")
    deepseek_client.client = make_dummy_client(original_error)
    
    with pytest.raises(AstralUnexpectedError) as exc_info:
        deepseek_client.create_completion_chat(request={})
    
    error_message = str(exc_info.value)
    print("\n--- Unexpected Error Verbose Message ---\n")
    print(error_message)
    assert "UNEXPECTED" in error_message
    assert "deepseek" in error_message.lower()
