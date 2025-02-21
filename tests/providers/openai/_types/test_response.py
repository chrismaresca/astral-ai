# -------------------------------------------------------------------------------- #
# Test OpenAI Response Models
# -------------------------------------------------------------------------------- #

"""
Tests for OpenAI Response Models in Astral AI
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
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# Astral AI imports
from astral_ai.providers.openai._types import (
    OpenAIChatResponseType,
    OpenAIStreamingResponseType,
    OpenAIStructuredResponseType,
    OpenAIResponseType
)

# -------------------------------------------------------------------------------- #
# Test Data
# -------------------------------------------------------------------------------- #

@pytest.fixture
def chat_completion_data() -> Dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-01-15-24",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

@pytest.fixture
def streaming_chunk_data() -> Dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o-01-15-24",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "Hello"
            },
            "finish_reason": None
        }]
    }

class TestSchema(BaseModel):
    response: str
    confidence: float

# -------------------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------------------- #

def test_chat_response_handling(chat_completion_data):
    """Test handling of standard chat completion responses"""
    print("\n=== Starting Chat Response Handling Test ===")
    
    print("\n--- Creating Chat Response ---")
    response: OpenAIChatResponseType = ChatCompletion(**chat_completion_data)
    
    print("\n--- Response Details ---")
    print(f"ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Choices: {response.choices}")
    print(f"Usage: {response.usage}")
    
    print("\n--- Running Assertions ---")
    assert response.id == "chatcmpl-123", "Response ID mismatch"
    print("✓ Response ID assertion passed")
    
    assert response.model == "gpt-4o-01-15-24", "Model mismatch"
    print("✓ Model assertion passed")
    
    assert len(response.choices) == 1, "Incorrect number of choices"
    print("✓ Choices length assertion passed")
    
    print("\n=== Chat Response Handling Test Completed Successfully ===")

def test_streaming_response_handling(streaming_chunk_data):
    """Test handling of streaming response chunks"""
    print("\n=== Starting Streaming Response Handling Test ===")
    
    print("\n--- Creating Streaming Response Chunk ---")
    chunk: OpenAIStreamingResponseType = ChatCompletionChunk(**streaming_chunk_data)
    
    print("\n--- Chunk Details ---")
    print(f"ID: {chunk.id}")
    print(f"Object: {chunk.object}")
    print(f"Choices: {chunk.choices}")
    
    print("\n--- Running Assertions ---")
    assert chunk.object == "chat.completion.chunk", "Incorrect object type"
    print("✓ Object type assertion passed")
    
    assert chunk.choices[0].delta.content == "Hello", "Incorrect chunk content"
    print("✓ Chunk content assertion passed")
    
    print("\n=== Streaming Response Handling Test Completed Successfully ===")

def test_structured_response_handling():
    """Test handling of structured responses with schema"""
    print("\n=== Starting Structured Response Handling Test ===")
    
    print("\n--- Creating Structured Response Data ---")
    structured_data = {
        "response": "test response",
        "confidence": 0.95
    }
    
    print("\n--- Creating Structured Response ---")
    response = TestSchema(**structured_data)
    
    print("\n--- Response Details ---")
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence}")
    
    print("\n--- Running Assertions ---")
    assert response.response == "test response", "Response content mismatch"
    print("✓ Response content assertion passed")
    
    assert response.confidence == 0.95, "Confidence value mismatch"
    print("✓ Confidence value assertion passed")
    
    print("\n=== Structured Response Handling Test Completed Successfully ===")

def test_response_type_union(chat_completion_data, streaming_chunk_data):
    """Test that different response types can be used as OpenAIResponseType"""
    print("\n=== Starting Response Type Union Test ===")
    
    def process_response(response: OpenAIResponseType):
        return True
    
    print("\n--- Creating Different Response Types ---")
    chat_response = ChatCompletion(**chat_completion_data)
    streaming_response = ChatCompletionChunk(**streaming_chunk_data)
    
    print("\n--- Running Assertions ---")
    assert process_response(chat_response), "Failed to process chat response"
    print("✓ Chat response processing passed")
    
    assert process_response(streaming_response), "Failed to process streaming response"
    print("✓ Streaming response processing passed")
    
    print("\n=== Response Type Union Test Completed Successfully ===")