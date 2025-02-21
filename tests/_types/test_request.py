# -------------------------------------------------------------------------------- #
# Test Request Models
# -------------------------------------------------------------------------------- #

"""
Creates a comprehensive set of tests for the Request models in Astral AI.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
import time
from typing import Any, Generator

# Astral AI imports
from astral_ai._types import (
    BaseRequest,
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,
    AstralParams,
)

# Astral AI Constants
from astral_ai.constants._models import ModelName


# -------------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------------- #


@pytest.fixture
def simple_request() -> AstralCompletionRequest:
    """
    A basic fixture that instantiates a generic AstralCompletionRequest object
    for use in tests.
    """
    print("\n=== Creating a simple_request fixture ===")
    return AstralCompletionRequest(model=ModelName("gpt-4o"))


@pytest.fixture
def time_marker() -> Generator[float, None, None]:
    """
    Provide a time marker before and after test execution to measure changes.
    """
    start_time = time.time()
    yield start_time
    end_time = time.time()
    print(f"\n=== Test duration: {end_time - start_time:.2f} seconds ===")

# -------------------------------------------------------------------------------- #
# Tests for BaseRequest
# -------------------------------------------------------------------------------- #


def test_base_request_private_attrs(time_marker: float) -> None:
    """
    Test the private attributes of BaseRequest to ensure they are set
    and accessible as expected.
    """
    print("\n=== Starting test_base_request_private_attrs ===")

    class DummyRequest(BaseRequest):
        """Concrete class extending the BaseRequest."""
        pass

    dummy_request = DummyRequest(model=ModelName("gpt-4o"))
    print(f"DummyRequest ID: {dummy_request.request_id}")
    print(f"DummyRequest Provider Name: {dummy_request.provider_name}")
    print(f"DummyRequest Time Created: {dummy_request._time_created}")

    # Assertions
    assert isinstance(dummy_request.request_id, str), "request_id should be a string"
    print("✓ request_id is a string")

    assert isinstance(dummy_request._time_created, float), "_time_created should be a float"
    print("✓ _time_created is a float")

    assert dummy_request.provider_name is not None, "provider_name should not be None"
    print("✓ provider_name is not None")

    print("\n=== test_base_request_private_attrs completed successfully ===")

# -------------------------------------------------------------------------------- #
# Tests for AstralCompletionRequest
# -------------------------------------------------------------------------------- #


def test_astral_completion_request_creation(simple_request: AstralCompletionRequest, time_marker: float) -> None:
    """
    Test basic AstralCompletionRequest creation and check default values
    and private attributes.
    """
    print("\n=== Starting test_astral_completion_request_creation ===")
    request_obj = simple_request

    print(f"Model: {request_obj.model}")
    print(f"Request ID: {request_obj.request_id}")
    print(f"Provider Name: {request_obj.provider_name}")
    print(f"Time Created: {request_obj._time_created}")

    # Assertions
    assert request_obj.model == "gpt-4o", "Model should be set to 'gpt-4o'"
    print("✓ model assertion passed")

    assert isinstance(request_obj.astral_params, AstralParams), "astral_params should be an instance of AstralParams"
    print("✓ astral_params is valid")

    assert request_obj.stream is False, "stream should default to False"
    print("✓ stream default assertion passed")

    print("\n=== test_astral_completion_request_creation completed successfully ===")


def test_astral_completion_request_to_provider_dict(simple_request: AstralCompletionRequest, time_marker: float) -> None:
    """
    Verifies that to_provider_dict omits fields that are set to NOT_GIVEN
    and includes only explicitly set or default values.
    """
    print("\n=== Starting test_astral_completion_request_to_provider_dict ===")
    request_obj = simple_request
    provider_dict = request_obj.to_provider_dict()

    print(f"Provider Dict: {provider_dict}")

    # Assertions
    assert "model" in provider_dict, "model should be in provider_dict"
    print("✓ 'model' found in provider_dict")

    assert provider_dict["model"] == request_obj.model, "model value should match"
    print("✓ model value matches in provider_dict")

    # Confirm that we do not see 'NOT_GIVEN' or any fields with that sentinel
    assert all(value != "NOT_GIVEN" for value in provider_dict.values()), (
        "No field should contain a NOT_GIVEN sentinel."
    )
    print("✓ No NOT_GIVEN sentinel found in provider_dict values")

    print("\n=== test_astral_completion_request_to_provider_dict completed successfully ===")

# -------------------------------------------------------------------------------- #
# Tests for AstralStructuredCompletionRequest
# -------------------------------------------------------------------------------- #


def test_astral_structured_completion_request(time_marker: float) -> None:
    """
    Test creation of AstralStructuredCompletionRequest and ensure that
    the model and response_format fields are set.
    """
    print("\n=== Starting test_astral_structured_completion_request ===")

    class MockResponseFormat(BaseRequest):
        """Mock class representing a custom BaseModel for structured output."""
        pass

    request_obj = AstralStructuredCompletionRequest(
        model=ModelName("gpt-4o"),
        response_format=MockResponseFormat(model=ModelName("mock-model"))
    )

    print(f"Model: {request_obj.model}")
    print(f"Response Format (type): {type(request_obj.response_format)}")
    print(f"Request ID: {request_obj.request_id}")
    print(f"Provider Name: {request_obj.provider_name}")

    # Assertions
    assert request_obj.model == "gpt-4o", "Model should be set to 'gpt-4o'"
    print("✓ model assertion passed")

    assert isinstance(request_obj.response_format, MockResponseFormat), (
        "response_format should be an instance of MockResponseFormat"
    )
    print("✓ response_format type assertion passed")

    print("\n=== test_astral_structured_completion_request completed successfully ===")
