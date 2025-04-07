# -------------------------------------------------------------------------------- #
# Tests for Completions Resource
# -------------------------------------------------------------------------------- #
"""
Pytest test suite for the Astral AI Completions resource.
Tests both standard and structured completions across different models and parameters.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import asyncio
import pytest
from typing import List, Dict, Optional, Union, Any, TypeVar, Type

# Pydantic
from pydantic import BaseModel, Field

# Astral AI Core Types
from astral_ai._types import NOT_GIVEN, ReasoningEffort, ToolChoice, Tool
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# Astral AI Model Constants
from astral_ai.constants._models import ModelName

# Astral AI Resources & Functions
from astral_ai.resources.completions.completions import (
    Completions,
    complete,
    complete_json,
    complete_structured,
    complete_async,
    complete_json_async,
    complete_structured_async
)

# Astral AI Tools
from astral_ai.tools.tool import function_tool

# -------------------------------------------------------------------------------- #
# Test Constants
# -------------------------------------------------------------------------------- #

# Sample messages for reuse in tests
BASIC_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the three laws of robotics?"}
]

SHORT_QUERY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant who responds very briefly."},
    {"role": "user", "content": "What's the capital of France?"}
]

COMPLEX_QUERY_MESSAGES = [
    {"role": "system", "content": "You are a knowledgeable assistant."},
    {"role": "user", "content": "Explain the concept of quantum entanglement and its implications for computing."}
]

# Supported models for testing
GPT_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet" 
DEEPSEEK_MODEL = "deepseek-chat"

# Define a common response model for structured output tests
class RoboticLaws(BaseModel):
    laws: List[str]
    author: str
    year_published: int

# For tool testing
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    conditions: str
    forecast: List[str]

# -------------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------------- #

@pytest.fixture
def calculate_area_tool():
    """Fixture providing a tool for calculating rectangle area."""
    @function_tool
    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle."""
        return length * width
    
    return calculate_area

@pytest.fixture
def weather_tool():
    """Fixture providing a tool for weather information."""
    @function_tool
    def get_weather(location: str, unit: str = "C") -> Dict[str, Any]:
        """
        Get the weather for a location.
        
        Args:
            location: City or location name
            unit: Temperature unit (C or F)
            
        Returns:
            Weather description
        """
        return {
            "location": location,
            "temperature": 25 if unit == "C" else 77,
            "conditions": "sunny",
            "forecast": ["Clear skies", "Low humidity", "Light breeze"]
        }
    
    return get_weather

@pytest.fixture
def basic_completion_client():
    """Fixture providing a basic Completions instance."""
    return Completions(
        model=GPT_MODEL,
        messages=BASIC_MESSAGES
    )

# -------------------------------------------------------------------------------- #
# Initialization Tests
# -------------------------------------------------------------------------------- #

def test_class_initialization_basic():
    """Test basic initialization of Completions class."""
    client = Completions(model=GPT_MODEL, messages=BASIC_MESSAGES)
    assert client.request is not None
    assert client.request.model == GPT_MODEL
    assert len(client.request.messages) == 2
    
def test_class_initialization_with_params():
    """Test Completions initialization with various parameters."""
    client = Completions(
        model=GPT_MODEL,
        messages=BASIC_MESSAGES,
        temperature=0.7,
        max_completion_tokens=100,
        top_p=0.8
    )
    
    assert client.request.temperature == 0.7
    assert client.request.max_completion_tokens == 100
    assert client.request.top_p == 0.8

def test_class_initialization_structured():
    """Test initialization with structured output format."""
    client = Completions(
        model=GPT_MODEL,
        messages=BASIC_MESSAGES,
        response_format=RoboticLaws
    )
    
    assert client.request.response_format == RoboticLaws

# -------------------------------------------------------------------------------- #
# Basic Completion Tests
# -------------------------------------------------------------------------------- #

def test_complete_basic_functionality():
    """Test basic complete functionality returns a proper response."""
    response = complete(
        model=GPT_MODEL,
        messages=SHORT_QUERY_MESSAGES
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.model == GPT_MODEL
    assert response.response is not None
    assert hasattr(response, 'usage')
    assert response.latency_ms > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0

def test_complete_with_temperature():
    """Test completion with temperature parameter."""
    response = complete(
        model=GPT_MODEL,
        messages=SHORT_QUERY_MESSAGES,
        temperature=0.2  # Low temperature for more deterministic output
    )
    
    assert isinstance(response, AstralChatResponse)
    assert "Paris" in response.response.content  # Should mention Paris as capital of France

def test_complete_with_max_tokens():
    """Test completion with max token limit."""
    # Set a very low token limit to test constraint
    response = complete(
        model=GPT_MODEL,
        messages=COMPLEX_QUERY_MESSAGES,
        max_completion_tokens=30  # Very short response
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.usage.completion_tokens <= 40  # Allow some flexibility in token counting

# -------------------------------------------------------------------------------- #
# Structured Output Tests
# -------------------------------------------------------------------------------- #

def test_complete_json_basic():
    """Test JSON structured output returns typed response."""
    response = complete_json(
        model=GPT_MODEL,
        messages=BASIC_MESSAGES,
        response_format=RoboticLaws
    )
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response, RoboticLaws)
    assert len(response.response.laws) == 3  # Should have 3 laws of robotics
    assert response.response.author == "Isaac Asimov"

def test_complete_structured():
    """Test structured output returns typed response."""
    response = complete_structured(
        model=GPT_MODEL,
        messages=BASIC_MESSAGES,
        response_format=RoboticLaws
    )
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response, RoboticLaws)
    assert len(response.response.laws) == 3
    assert response.response.author == "Isaac Asimov"
    assert response.response.year_published > 1940

@pytest.mark.parametrize("model", [GPT_MODEL, CLAUDE_MODEL])
def test_structured_output_across_models(model):
    """Test structured output across different models."""
    class SimpleAnswer(BaseModel):
        answer: str
        confidence: float
    
    messages = [
        {"role": "user", "content": "What's the capital of France?"}
    ]
    
    response = complete_json(
        model=model,
        messages=messages,
        response_format=SimpleAnswer
    )
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response, SimpleAnswer)
    assert "Paris" in response.response.answer
    assert 0 <= response.response.confidence <= 1

# -------------------------------------------------------------------------------- #
# Tool Usage Tests
# -------------------------------------------------------------------------------- #

def test_tools_basic(calculate_area_tool, weather_tool):
    """Test basic tool usage."""
    messages = [
        {"role": "user", "content": "What's the area of a 5x10 rectangle?"}
    ]
    
    response = complete(
        model=GPT_MODEL,
        messages=messages,
        tools=[calculate_area_tool],
        tool_choice="auto"
    )
    
    assert isinstance(response, AstralChatResponse)
    # Either the response includes "50" (the answer) or it used the tool
    assert "50" in response.response.content or hasattr(response.response, 'tool_calls')

def test_tools_with_specific_choice(weather_tool):
    """Test tools with specific tool choice."""
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
    
    response = complete(
        model=GPT_MODEL,
        messages=messages,
        tools=[weather_tool],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    
    assert isinstance(response, AstralChatResponse)
    assert hasattr(response.response, 'tool_calls')
    assert len(response.response.tool_calls) > 0
    assert response.response.tool_calls[0].function.name == "get_weather"

def test_tools_with_structured_output(weather_tool):
    """Test tools with structured output response."""
    messages = [
        {"role": "user", "content": "What's the weather in New York?"}
    ]
    
    response = complete_json(
        model=GPT_MODEL,
        messages=messages,
        tools=[weather_tool],
        response_format=WeatherResponse
    )
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response, WeatherResponse)
    assert response.response.location.lower() in ["new york", "ny", "nyc"]

# -------------------------------------------------------------------------------- #
# Async Tests
# -------------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_complete_async_basic():
    """Test basic async completion."""
    response = await complete_async(
        model=GPT_MODEL,
        messages=SHORT_QUERY_MESSAGES
    )
    
    assert isinstance(response, AstralChatResponse)
    assert "Paris" in response.response.content

@pytest.mark.asyncio
async def test_complete_json_async():
    """Test async JSON completion."""
    class CapitalInfo(BaseModel):
        city: str
        country: str
        population: int
    
    response = await complete_json_async(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        response_format=CapitalInfo
    )

    print(response.response.choices[0].message.content)
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response, CapitalInfo)
    assert response.response.city == "Paris"
    assert response.response.country == "France"
    assert response.response.population > 1000000

# -------------------------------------------------------------------------------- #
# Parameter Tests
# -------------------------------------------------------------------------------- #

def test_reasoning_effort():
    """Test reasoning_effort parameter with Claude model."""
    for effort in ["auto", "low", "medium", "high"]:
        response = complete(
            model=CLAUDE_MODEL,
            messages=COMPLEX_QUERY_MESSAGES,
            reasoning_effort=effort
        )
        
        assert isinstance(response, AstralChatResponse)
        assert response.response.content is not None

@pytest.mark.parametrize("temperature", [0.1, 0.5, 0.9])
def test_temperature_variations(temperature):
    """Test different temperature settings."""
    response = complete(
        model=GPT_MODEL,
        messages=SHORT_QUERY_MESSAGES,
        temperature=temperature
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.response.content is not None

@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9])
def test_top_p_variations(top_p):
    """Test different top_p settings."""
    response = complete(
        model=GPT_MODEL,
        messages=SHORT_QUERY_MESSAGES,
        top_p=top_p
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.response.content is not None

def test_presence_penalty():
    """Test presence_penalty parameter."""
    response = complete(
        model=GPT_MODEL,
        messages=COMPLEX_QUERY_MESSAGES,
        presence_penalty=0.8  # High presence penalty to discourage repetition
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.response.content is not None

def test_frequency_penalty():
    """Test frequency_penalty parameter."""
    response = complete(
        model=GPT_MODEL,
        messages=COMPLEX_QUERY_MESSAGES,
        frequency_penalty=0.8  # High frequency penalty to discourage repetition
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.response.content is not None

# -------------------------------------------------------------------------------- #
# Complex Integration Tests
# -------------------------------------------------------------------------------- #

def test_complex_request_integration():
    """Test complex request with multiple parameters set."""
    response = complete(
        model=GPT_MODEL,
        messages=COMPLEX_QUERY_MESSAGES,
        temperature=0.7,
        top_p=0.9,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        max_completion_tokens=200
    )
    
    assert isinstance(response, AstralChatResponse)
    assert response.response.content is not None
    assert response.usage.completion_tokens <= 250  # Allow some flexibility

def test_complex_structured_request(calculate_area_tool, weather_tool):
    """Test complex structured request with tools."""
    class TravelPlan(BaseModel):
        destination: str
        weather: Dict[str, Any]
        activities: List[str]
        budget_estimate: float
    
    messages = [
        {"role": "user", "content": "Plan a trip to Paris with details about weather and activities."}
    ]
    
    response = complete_json(
        model=GPT_MODEL,
        messages=messages,
        tools=[weather_tool],
        temperature=0.7,
        response_format=TravelPlan
    )
    
    assert isinstance(response, AstralStructuredResponse)
    assert isinstance(response.response.data, TravelPlan)
    assert response.response.data.destination.lower() == "paris"
    assert isinstance(response.response.data.weather, dict)
    assert len(response.response.data.activities) > 0
    assert response.response.data.budget_estimate > 0
