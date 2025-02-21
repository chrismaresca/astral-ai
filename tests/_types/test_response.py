# -------------------------------------------------------------------------------- #
# Test Response Models
# -------------------------------------------------------------------------------- #

"""
Tests for Response Models in Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest

# Astral AI imports
from astral_ai._types import (
    ProviderResponseObject,
    AstralChatResponse,
)

# Astral AI Tracing Types
from astral_ai._types._response._usage import ChatUsage, ChatCost

# -------------------------------------------------------------------------------- #
# Test Data
# -------------------------------------------------------------------------------- #


@pytest.fixture
def provider_response():
    return ProviderResponseObject(
        provider_object="chat.completions",
        provider_response_id="123",
        provider_model_id="gpt-4o-mini",
        provider_request_id="456",
        provider_created=1713859200,
    )


@pytest.fixture
def chat_usage():
    return ChatUsage(
        prompt_tokens=100,
        completion_tokens=100,
        total_tokens=200,
    )


@pytest.fixture
def chat_cost():
    return ChatCost(
        input_cost=0.0001,
        output_cost=0.0002,
        total_cost=0.0003,
        total_tokens=200,
    )

# -------------------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------------------- #


def test_private_attrs_propagation(provider_response, chat_usage, chat_cost):
    """Test that private attributes are correctly propagated to usage and cost"""
    print("\n=== Starting Private Attrs Propagation Test ===")

    print("\n--- Creating Response Object ---")
    response = AstralChatResponse(
        model="gpt-4o",
        provider_response=provider_response,
        usage=chat_usage,
        cost=chat_cost,
    )

    print("\n--- Response Object Details ---")
    print(f"Response ID: {response.response_id}")
    print(f"Provider Name: {response.provider_name}")
    print(f"Model: {response.model}")
    print(f"Time Created: {response.time_created}")

    print("\n--- Usage Object Details ---")
    print(f"Usage Response ID: {response.usage.response_id}")
    print(f"Usage Model Provider: {response.usage.model_provider}")
    print(f"Usage Model Name: {response.usage.model_name}")
    print(f"Usage Prompt Tokens: {response.usage.prompt_tokens}")
    print(f"Usage Completion Tokens: {response.usage.completion_tokens}")
    print(f"Usage Total Tokens: {response.usage.total_tokens}")

    print("\n--- Cost Object Details ---")
    print(f"Cost Response ID: {response.cost.response_id}")
    print(f"Cost Model Provider: {response.cost.model_provider}")
    print(f"Cost Model Name: {response.cost.model_name}")
    print(f"Cost Input Cost: {response.cost.input_cost}")
    print(f"Cost Output Cost: {response.cost.output_cost}")
    print(f"Cost Total Cost: {response.cost.total_cost}")

    print("\n--- Running Assertions ---")
    # Test usage private attrs
    assert response.usage.response_id == response.response_id, "Usage response ID mismatch"
    print("✓ Usage response ID assertion passed")

    assert response.usage.model_provider == response.provider_name, "Usage model provider mismatch"
    print("✓ Usage model provider assertion passed")

    assert response.usage.model_name == response.model, "Usage model name mismatch"
    print("✓ Usage model name assertion passed")

    # Test cost private attrs
    assert response.cost.response_id == response.response_id, "Cost response ID mismatch"
    print("✓ Cost response ID assertion passed")

    assert response.cost.model_provider == response.provider_name, "Cost model provider mismatch"
    print("✓ Cost model provider assertion passed")

    assert response.cost.model_name == response.model, "Cost model name mismatch"
    print("✓ Cost model name assertion passed")

    print("\n=== Private Attrs Propagation Test Completed Successfully ===")


def test_response_creation(provider_response, chat_usage, chat_cost):
    """Test basic response creation and attributes"""
    print("\n=== Starting Response Creation Test ===")

    print("\n--- Creating Response Object ---")
    response = AstralChatResponse(
        model="gpt-4o",
        provider_response=provider_response,
        usage=chat_usage,
        cost=chat_cost,
    )

    print("\n--- Response Object Details ---")
    print(f"Model: {response.model}")
    print(f"Provider Response Object: {response.provider_response}")
    print(f"Usage Object: {response.usage}")
    print(f"Cost Object: {response.cost}")
    print(f"Response ID: {response.response_id}")
    print(f"Time Created: {response.time_created}")

    print("\n--- Provider Response Details ---")
    print(f"Provider Object: {provider_response.provider_object}")
    print(f"Provider Response ID: {provider_response.provider_response_id}")
    print(f"Provider Model ID: {provider_response.provider_model_id}")
    print(f"Provider Request ID: {provider_response.provider_request_id}")
    print(f"Provider Created: {provider_response.provider_created}")

    print("\n--- Running Assertions ---")
    assert response.model == "gpt-4o", "Model mismatch"
    print("✓ Model assertion passed")

    assert response.provider_response == provider_response, "Provider response mismatch"
    print("✓ Provider response assertion passed")

    assert response.usage == chat_usage, "Usage mismatch"
    print("✓ Usage assertion passed")

    assert response.cost == chat_cost, "Cost mismatch"
    print("✓ Cost assertion passed")

    assert isinstance(response.response_id, str), "Response ID not a string"
    print("✓ Response ID type assertion passed")

    assert isinstance(response.time_created, float), "Time created not a float"
    print("✓ Time created type assertion passed")

    print("\n=== Response Creation Test Completed Successfully ===")
