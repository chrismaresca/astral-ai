# ------------------------------------------------------------------------------
# Test Completions Resource
# ------------------------------------------------------------------------------
"""
This file contains tests for the Completions functionality in Astral AI.
It includes basic tests, advanced parameter tests, structured completion tests,
error and edge case tests, and integration tests to ensure proper functionality.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
# Built-in imports
import pytest
from typing import List
from unittest.mock import patch

# Third-party imports
from pydantic import BaseModel

# Astral AI imports
from astral_ai.resources.completions import completion, completion_structured, Completions

# Astral AI Types
from astral_ai._types import (
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,
    AstralParams,
    AstralClientParams,
    AstralChatResponse,
    AstralStructuredResponse,
    Metadata,
)

# Astral AI Request Params
from astral_ai._types._request import Metadata

# Astral AI Cost Strategies
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy

# Astral AI Exceptions
from astral_ai.errors.exceptions import ResponseModelMissingError
from astral_ai.constants._models import ModelName


# ------------------------------------------------------------------------------
# Test Models and Fixtures
# ------------------------------------------------------------------------------
class MockStructuredResponse(BaseModel):
    """
    A mock Pydantic model to represent structured responses in tests.
    """
    summary: str
    sentiment: str
    key_points: List[str]


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def astral_client_params():
    """
    Fixture for providing astral client parameters for tests.
    """
    return AstralClientParams(
        new_client=False
    )


@pytest.fixture
def astral_params():
    """
    Fixture for providing astral parameters for tests.
    """
    return AstralParams(
        astral_client=astral_client_params(),
        cost_strategy=ReturnCostStrategy()
    )


@pytest.fixture
def basic_messages():
    """
    Fixture for providing a basic set of messages for tests.
    """
    print("\n=== Creating basic_messages fixture ===")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


# ------------------------------------------------------------------------------
# Mock Completion Response
# ------------------------------------------------------------------------------


@pytest.fixture
def mock_completion_response():
    """
    Fixture for providing a mock completion response from a provider.
    """
    print("\n=== Creating mock_completion_response fixture ===")
    return {
        "id": "mock-completion-id",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking! How can I help you today?"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        }
    }


# ------------------------------------------------------------------------------
# Basic Completion Tests
# ------------------------------------------------------------------------------
def test_basic_completion(basic_messages, mock_completion_response):
    """
    Test basic chat completion with minimal parameters.
    """
    print("\n=== Starting test_basic_completion ===")
    with patch('astral_ai.providers.openai.OpenAIAdapter.to_provider_completion_request'), \
            patch('astral_ai.providers.openai.OpenAIAdapter.to_astral_completion_response'):

        # Basic Completion
        response = completion(
            model="gpt-4o",
            messages=basic_messages
        )

        # Assertions
        print(f"Response Type: {type(response)}")
        print(f"Response Model: {response.model}")
        print(f"Response Text: {response.response}")
        assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
        assert response.model == "gpt-4o", "Response model should match the provided model"
        print("=== test_basic_completion completed successfully ===")


def test_completion_with_temperature(basic_messages):
    """
    Test completion with temperature parameter.
    """
    print("\n=== Starting test_completion_with_temperature ===")
    response = completion(
        model="gpt-4o",
        messages=basic_messages,
        temperature=0.7
    )
    print(f"Response Temperature: 0.7 (set), Response Object: {type(response)}")
    assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
    print("=== test_completion_with_temperature completed successfully ===")


def test_completion_with_max_tokens(basic_messages):
    """
    Test completion with max tokens parameter.
    """
    print("\n=== Starting test_completion_with_max_tokens ===")
    response = completion(
        model="gpt-4o",
        messages=basic_messages,
        max_tokens=100
    )
    print(f"Response max_tokens: 100 (set), Response Object: {type(response)}")
    assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
    print("=== test_completion_with_max_tokens completed successfully ===")


# ------------------------------------------------------------------------------
# Advanced Parameter Tests
# ------------------------------------------------------------------------------
def test_completion_with_all_parameters(basic_messages):
    """
    Test completion with all available parameters.
    """
    print("\n=== Starting test_completion_with_all_parameters ===")
    metadata = Metadata(conversation_id="test-conv-123")
    response = completion(
        model="gpt-4o",
        messages=basic_messages,
        temperature=0.7,
        max_tokens=150,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        top_p=0.9,
        metadata=metadata,
        user="test-user",
        reasoning_effort="high",
        stream=False,
        n=1
    )
    print(f"Response Model: {response.model}")
    print(f"Response Metadata: {metadata}")
    assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
    print("=== test_completion_with_all_parameters completed successfully ===")


def test_completion_with_stop_sequence(basic_messages):
    """
    Test completion with stop sequence.
    """
    print("\n=== Starting test_completion_with_stop_sequence ===")
    response = completion(
        model="gpt-4o",
        messages=basic_messages,
        stop=["END", "STOP"]
    )
    print(f"Stop Sequences: ['END', 'STOP'], Response Object: {type(response)}")
    assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
    print("=== test_completion_with_stop_sequence completed successfully ===")


# ------------------------------------------------------------------------------
# Structured Completion Tests
# ------------------------------------------------------------------------------
def test_basic_structured_completion(basic_messages):
    """
    Test basic structured completion.
    """
    print("\n=== Starting test_basic_structured_completion ===")
    response = completion_structured(
        model="gpt-4o",
        messages=basic_messages,
        response_format=MockStructuredResponse
    )
    print(f"Structured Response Type: {type(response.response)}")
    assert isinstance(response, AstralStructuredResponse), (
        "Response should be an instance of AstralStructuredResponse"
    )
    assert isinstance(response.response, MockStructuredResponse), (
        "Inner response should be an instance of MockStructuredResponse"
    )
    print("=== test_basic_structured_completion completed successfully ===")


def test_structured_completion_missing_model():
    """
    Test structured completion with missing response model.
    """
    print("\n=== Starting test_structured_completion_missing_model ===")
    with pytest.raises(ResponseModelMissingError):
        completion_structured(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            response_format=None
        )
    print("=== test_structured_completion_missing_model completed successfully ===")


# ------------------------------------------------------------------------------
# Error and Edge Case Tests
# ------------------------------------------------------------------------------
def test_completion_with_invalid_model():
    """
    Test completion with invalid model name.
    """
    print("\n=== Starting test_completion_with_invalid_model ===")
    with pytest.raises(ValueError):
        completion(
            model="invalid-model",
            messages=[{"role": "user", "content": "Test"}]
        )
    print("=== test_completion_with_invalid_model completed successfully ===")


def test_completion_with_empty_messages():
    """
    Test completion with empty messages list.
    """
    print("\n=== Starting test_completion_with_empty_messages ===")
    with pytest.raises(ValueError):
        completion(
            model="gpt-4o",
            messages=[]
        )
    print("=== test_completion_with_empty_messages completed successfully ===")


def test_completion_with_invalid_temperature():
    """
    Test completion with invalid temperature value.
    """
    print("\n=== Starting test_completion_with_invalid_temperature ===")
    with pytest.raises(ValueError):
        completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            temperature=2.0  # Temperature should be between 0 and 1
        )
    print("=== test_completion_with_invalid_temperature completed successfully ===")


# ------------------------------------------------------------------------------
# Astral Parameters Tests
# ------------------------------------------------------------------------------
def test_completion_with_astral_params(basic_messages, astral_params):
    """
    Test completion with custom Astral parameters.
    """
    print("\n=== Starting test_completion_with_astral_params ===")

    response = completion(
        model="gpt-4o",
        messages=basic_messages,
        astral_params=astral_params
    )
    print(f"Astral Params: {astral_params}")
    print(f"Response Object: {type(response)}")
    assert isinstance(response, AstralChatResponse), "Response should be an instance of AstralChatResponse"
    print("=== test_completion_with_astral_params completed successfully ===")


# ------------------------------------------------------------------------------
# Completions Class Tests
# ------------------------------------------------------------------------------
def test_completions_class_initialization():
    """
    Test direct initialization of Completions class.
    """
    print("\n=== Starting test_completions_class_initialization ===")
    request = AstralCompletionRequest(
        model=ModelName("gpt-4o"),
        messages=[{"role": "user", "content": "Test"}]
    )

    completions = Completions(request)
    print(f"Completions Object: {type(completions)}")
    assert isinstance(completions, Completions), "Should be able to instantiate Completions class directly"
    print("=== test_completions_class_initialization completed successfully ===")


def test_completions_class_run_method():
    """
    Test run method of Completions class.
    """
    print("\n=== Starting test_completions_class_run_method ===")
    request = AstralCompletionRequest(
        model=ModelName("gpt-4o"),
        messages=[{"role": "user", "content": "Test"}]
    )

    completions = Completions(request)
    response = completions.run()
    print(f"Response from run method: {response.response}")
    assert isinstance(response, AstralChatResponse), (
        "Response should be an instance of AstralChatResponse after running Completions"
    )
    print("=== test_completions_class_run_method completed successfully ===")


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------
@pytest.mark.integration
def test_real_completion_integration():
    """
    Integration test with real API call.
    """
    print("\n=== Starting test_real_completion_integration ===")
    response = completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(f"Real Integration Response: {response.response}")
    assert isinstance(response, AstralChatResponse), (
        "Response should be an instance of AstralChatResponse"
    )
    assert response.response is not None, "Response text should not be None"
    assert "Paris" in response.response, "Response should contain 'Paris'"

    # Verify response attributes
    assert response.model == "gpt-4o", "Model name should match request"
    assert response.provider_response is not None, "Provider response should be present"
    assert response.usage is not None, "Usage information should be present"
    assert response.usage.total_tokens > 0, "Total tokens should be greater than 0"
    assert response.time_created > 0, "Creation timestamp should be present"
    print("=== test_real_completion_integration completed successfully ===")


@pytest.mark.integration
def test_real_structured_completion_integration():
    """
    Integration test for structured completion with real API call.
    """
    print("\n=== Starting test_real_structured_completion_integration ===")
    response = completion_structured(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Analyze this text: 'I really enjoyed the movie!'"}
        ],
        response_format=MockStructuredResponse
    )
    print(f"Structured Integration Response Type: {type(response.response)}")
    assert isinstance(response, AstralStructuredResponse), (
        "Response should be an instance of AstralStructuredResponse"
    )
    assert isinstance(response.response, MockStructuredResponse), (
        "Inner response should be an instance of MockStructuredResponse"
    )
    print(f"Structured Integration Response Content: {response.response}")

    # Verify structured response content
    assert response.response.sentiment in ["positive", "negative", "neutral"], (
        "Sentiment should be one of the expected values"
    )
    assert len(response.response.summary) > 0, "Summary should not be empty"
    assert isinstance(response.response.key_points, list), "Key points should be a list"
    assert len(response.response.key_points) > 0, "Should have at least one key point"

    # Verify response metadata
    assert response.model == "gpt-4o", "Model name should match request"
    assert response.provider_response is not None, "Provider response should be present"
    assert response.usage is not None, "Usage information should be present"
    assert response.usage.total_tokens > 0, "Total tokens should be greater than 0"
    print("=== test_real_structured_completion_integration completed successfully ===")


@pytest.mark.integration
def test_real_completion_with_parameters_integration():
    """
    Integration test for completion with various parameters.
    """
    print("\n=== Starting test_real_completion_with_parameters_integration ===")
    metadata = Metadata(conversation_id="test-integration-123")
    response = completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Write a very short story about a cat."}
        ],
        temperature=0.7,
        max_tokens=50,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        metadata=metadata,
        user="test-integration-user"
    )

    print(f"Response with Parameters: {response.response}")

    # Verify response content
    assert isinstance(response, AstralChatResponse), "Should be an AstralChatResponse"
    assert len(response.response) > 0, "Response should not be empty"
    assert response.response.count('.') >= 1, "Should have at least one complete sentence"

    # Verify token usage
    assert response.usage.total_tokens <= 50, "Should respect max_tokens parameter"
    assert response.usage.prompt_tokens > 0, "Should have prompt tokens"
    assert response.usage.completion_tokens > 0, "Should have completion tokens"

    # Verify metadata and model info
    assert response.model == "gpt-4o", "Model should match request"
    assert response.provider_name is not None, "Provider name should be present"
    print("=== test_real_completion_with_parameters_integration completed successfully ===")


@pytest.mark.integration
def test_real_completion_conversation_integration():
    """
    Integration test for multi-turn conversation.
    """
    print("\n=== Starting test_real_completion_conversation_integration ===")

    # First message
    response1 = completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Let's talk about space exploration. What's your favorite mission?"}
        ]
    )

    # Continue conversation
    response2 = completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Let's talk about space exploration. What's your favorite mission?"},
            {"role": "assistant", "content": response1.response},
            {"role": "user", "content": "Why is that mission particularly significant?"}
        ]
    )

    print(f"First Response: {response1.response}")
    print(f"Second Response: {response2.response}")

    # Verify conversation flow
    assert isinstance(response1, AstralChatResponse), "First response should be AstralChatResponse"
    assert isinstance(response2, AstralChatResponse), "Second response should be AstralChatResponse"
    assert len(response2.response) > 0, "Second response should not be empty"

    # Verify response attributes
    assert response1.usage is not None, "First response should have usage info"
    assert response2.usage is not None, "Second response should have usage info"
    assert response2.usage.total_tokens > response1.usage.total_tokens, (
        "Second response should use more tokens due to conversation history"
    )
    print("=== test_real_completion_conversation_integration completed successfully ===")
