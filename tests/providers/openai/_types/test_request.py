# -------------------------------------------------------------------------------- #
# Test OpenAI Request Models
# -------------------------------------------------------------------------------- #

"""
Tests for OpenAI Request Models in Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
from typing import Dict, List

# Pydantic
from pydantic import BaseModel

# OpenAI Types
from openai.types.chat import ChatCompletion

# Astral AI imports
from astral_ai.providers.openai._types import (
    OpenAIRequestChat,
    OpenAIRequestStreaming,
    OpenAIRequestStructured,
    OpenAIRequestType
)
from astral_ai.constants._models import OpenAIModels

# -------------------------------------------------------------------------------- #
# Test Data
# -------------------------------------------------------------------------------- #

@pytest.fixture
def basic_chat_request() -> Dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "max_tokens": 100
    }

@pytest.fixture
def streaming_request() -> Dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
        "temperature": 0.7
    }

class TestSchema(BaseModel):
    response: str
    confidence: float

@pytest.fixture
def structured_request() -> Dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
        "response_format": TestSchema(response="test", confidence=0.9)
    }

# -------------------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------------------- #

def test_chat_request_creation(basic_chat_request):
    """Test basic chat request creation and validation"""
    print("\n=== Starting Chat Request Creation Test ===")
    
    print("\n--- Creating Chat Request ---")
    request = OpenAIRequestChat(**basic_chat_request)
    
    print("\n--- Request Details ---")
    print(f"Model: {request['model']}")
    print(f"Messages: {request['messages']}")
    print(f"Temperature: {request['temperature']}")
    print(f"Max Tokens: {request['max_tokens']}")
    
    print("\n--- Running Assertions ---")
    assert request["model"] == "gpt-4o", "Model mismatch"
    print("✓ Model assertion passed")
    
    assert isinstance(request["messages"], list), "Messages not a list"
    print("✓ Messages type assertion passed")
    
    assert request["temperature"] == 0.7, "Temperature mismatch"
    print("✓ Temperature assertion passed")
    
    print("\n=== Chat Request Creation Test Completed Successfully ===")

# -------------------------------------------------------------------------------- #
# Streaming Request Creation
# -------------------------------------------------------------------------------- #

def test_streaming_request_creation(streaming_request):
    """Test streaming request creation and validation"""
    print("\n=== Starting Streaming Request Creation Test ===")
    
    print("\n--- Creating Streaming Request ---")
    request = OpenAIRequestStreaming(**streaming_request)
    
    print("\n--- Request Details ---")
    print(f"Model: {request['model']}")
    print(f"Stream: {request['stream']}")
    
    print("\n--- Running Assertions ---")
    assert request["stream"] is True, "Stream flag not set"
    print("✓ Stream flag assertion passed")
    
    print("\n=== Streaming Request Creation Test Completed Successfully ===")


# -------------------------------------------------------------------------------- #
# Structured Request Creation
# -------------------------------------------------------------------------------- #

def test_structured_request_creation(structured_request):
    """Test structured request creation with schema"""
    print("\n=== Starting Structured Request Creation Test ===")
    
    print("\n--- Creating Structured Request ---")
    request = OpenAIRequestStructured[TestSchema](**structured_request)
    
    print("\n--- Request Details ---")
    print(f"Model: {request['model']}")
    print(f"Response Format: {request['response_format']}")
    
    print("\n--- Running Assertions ---")
    assert isinstance(request["response_format"], TestSchema), "Response format not correct schema"
    print("✓ Response format type assertion passed")
    
    print("\n=== Structured Request Creation Test Completed Successfully ===")


# -------------------------------------------------------------------------------- #
# Request Type Union
# -------------------------------------------------------------------------------- #

def test_request_type_union():
    """Test that different request types can be used as OpenAIRequestType"""
    print("\n=== Starting Request Type Union Test ===")
    
    def process_request(request: OpenAIRequestType):
        return True
    
    print("\n--- Testing Different Request Types ---")
    chat_request = OpenAIRequestChat(model="gpt-4o", messages=[])
    streaming_request = OpenAIRequestStreaming(model="gpt-4o", messages=[], stream=True)
    structured_request = OpenAIRequestStructured[TestSchema](
        model="gpt-4o", 
        messages=[],
        response_format=TestSchema(response="test", confidence=0.9)
    )
    
    print("\n--- Running Assertions ---")
    assert process_request(chat_request), "Failed to process chat request"
    print("✓ Chat request processing passed")
    
    assert process_request(streaming_request), "Failed to process streaming request"
    print("✓ Streaming request processing passed")
    
    assert process_request(structured_request), "Failed to process structured request"
    print("✓ Structured request processing passed")
    
    print("\n=== Request Type Union Test Completed Successfully ===") 