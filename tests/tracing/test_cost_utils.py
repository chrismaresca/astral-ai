# -------------------------------------------------------------------------------- #
# Test Cost Utils
# Tests for the cost utility functions in astral_ai.tracing._cost_utils
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
from typing import Union

# Astral AI imports
from astral_ai.utilities.cost_utils import get_model_costs
from astral_ai._types._response._usage import ChatCost, EmbeddingCost
from astral_ai.constants._models import (
    ModelName,
    ModelProvider,
    ChatModels,
    EmbeddingModels,
)

# -------------------------------------------------------------------------------- #
# Test Cases
# -------------------------------------------------------------------------------- #

def test_get_model_costs_chat_models():
    """Test getting costs for chat models."""
    # Test cases for OpenAI chat models
    chat_models = [
        ("gpt-4o", "openai"),
        ("o1", "openai"),
        ("o1-mini", "openai"),
        ("o3-mini", "openai"),
    ]
    
    print("\n=== Testing Chat Models ===")
    for model_name, provider in chat_models:
        print(f"\nTesting {model_name} from {provider}")
        try:
            cost = get_model_costs(model_name, provider)
            print(f"✓ Successfully retrieved costs:")
            print(f"  - Input cost per million: ${cost.prompt_tokens}")
            print(f"  - Output cost per million: ${cost.output_tokens}")
            print(f"  - Cached input cost per million: ${cost.cached_prompt_tokens}")
            assert isinstance(cost, ChatCost)
        except Exception as e:
            print(f"✗ Error getting costs for {model_name}: {str(e)}")
            raise

def test_get_model_costs_embedding_models():
    """Test getting costs for embedding models."""
    embedding_models = [
        ("text-embedding-3-small", "openai"),
        ("text-embedding-3-large", "openai"),
    ]
    
    print("\n=== Testing Embedding Models ===")
    for model_name, provider in embedding_models:
        print(f"\nTesting {model_name} from {provider}")
        try:
            cost = get_model_costs(model_name, provider)
            print(f"✓ Successfully retrieved costs:")
            print(f"  - Input cost per million: ${cost.prompt_tokens}")
            assert isinstance(cost, EmbeddingCost)
        except Exception as e:
            print(f"✗ Error getting costs for {model_name}: {str(e)}")
            raise

def test_get_model_costs_anthropic_models():
    """Test getting costs for Anthropic models."""
    anthropic_models = [
        ("claude-3-5-sonnet", "anthropic"),
        ("claude-3-opus", "anthropic"),
        ("claude-3-haiku", "anthropic"),
    ]
    
    print("\n=== Testing Anthropic Models ===")
    for model_name, provider in anthropic_models:
        print(f"\nTesting {model_name} from {provider}")
        try:
            cost = get_model_costs(model_name, provider)
            print(f"✓ Successfully retrieved costs:")
            print(f"  - Input cost per million: ${cost.prompt_tokens}")
            print(f"  - Output cost per million: ${cost.output_tokens}")
            print(f"  - Cached input cost per million: ${cost.cached_prompt_tokens}")
            assert isinstance(cost, ChatCost)
        except Exception as e:
            print(f"✗ Error getting costs for {model_name}: {str(e)}")
            raise

def test_invalid_provider():
    """Test handling of invalid provider."""
    print("\n=== Testing Invalid Provider ===")
    with pytest.raises(ValueError) as exc_info:
        get_model_costs("gpt-4o", "invalid_provider")
    print(f"✓ Successfully caught invalid provider error: {str(exc_info.value)}")

def test_invalid_model():
    """Test handling of invalid model."""
    print("\n=== Testing Invalid Model ===")
    with pytest.raises(ValueError) as exc_info:
        get_model_costs("invalid_model", "openai")
    print(f"✓ Successfully caught invalid model error: {str(exc_info.value)}")

def test_cost_object_attributes():
    """Test that cost objects have the correct attributes."""
    print("\n=== Testing Cost Object Attributes ===")
    
    # Test ChatCost object
    print("\nTesting ChatCost object attributes")
    chat_cost = get_model_costs("gpt-4o", "openai")
    assert hasattr(chat_cost, 'prompt_tokens')
    assert hasattr(chat_cost, 'output_tokens')
    assert hasattr(chat_cost, 'cached_prompt_tokens')
    print("✓ ChatCost object has all required attributes")
    
    # Test EmbeddingCost object
    print("\nTesting EmbeddingCost object attributes")
    embedding_cost = get_model_costs("text-embedding-3-small", "openai")
    assert hasattr(embedding_cost, 'prompt_tokens')
    print("✓ EmbeddingCost object has all required attributes")

if __name__ == "__main__":
    print("\n🚀 Starting Cost Utils Tests 🚀\n")
    
    # Run all tests
    test_get_model_costs_chat_models()
    test_get_model_costs_embedding_models()
    test_get_model_costs_anthropic_models()
    test_invalid_provider()
    test_invalid_model()
    test_cost_object_attributes()
    
    print("\n✨ All tests completed successfully! ✨") 