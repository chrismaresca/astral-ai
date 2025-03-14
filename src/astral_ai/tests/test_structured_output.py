# -------------------------------------------------------------------------------- #
# Test Structured Output Response
# -------------------------------------------------------------------------------- #
"""
Test file to demonstrate how to use the StructuredOutputCompletionResponse
class with BaseModel types.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import json
from typing import List, Optional

# Pydantic imports
from pydantic import BaseModel, Field

# OpenAI imports
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage, ParsedChoice

# Astral AI imports
from astral_ai._types._response.resources._completions_response import (
    StructuredOutputCompletionResponse,
    ChoiceLogprobs
)

# -------------------------------------------------------------------------------- #
# Example BaseModel for Structured Output
# -------------------------------------------------------------------------------- #

class Person(BaseModel):
    """Example BaseModel for structured output."""
    name: str
    age: int
    email: Optional[str] = None
    hobbies: List[str] = Field(default_factory=list)


# -------------------------------------------------------------------------------- #
# Test Function
# -------------------------------------------------------------------------------- #

def test_structured_output_response():
    """Test structured output response creation and usage."""
    
    # Create a structured output response with a Person model
    structured_response = StructuredOutputCompletionResponse[Person](
        id="test-completion-id",
        choices=[
            ParsedChoice[Person](
                finish_reason="stop",
                index=0,
                message=ParsedChatCompletionMessage[Person](
                    role="assistant",
                    content=json.dumps({
                        "name": "John Doe",
                        "age": 30,
                        "email": "john@example.com",
                        "hobbies": ["reading", "coding"]
                    }),
                    parsed=Person(
                        name="John Doe",
                        age=30,
                        email="john@example.com",
                        hobbies=["reading", "coding"]
                    )
                ),
                logprobs=None
            )
        ],
        created=1698765432,
        model="gpt-4",
        object="chat.completion",
        system_fingerprint="fp123456"
    )
    
    # Access the structured output
    person = structured_response.choices[0].message.parsed
    
    # Verify the parsed data
    assert person is not None
    assert person.name == "John Doe"
    assert person.age == 30
    assert person.email == "john@example.com"
    assert person.hobbies == ["reading", "coding"]
    
    print("Structured output test passed!")
    return structured_response


# -------------------------------------------------------------------------------- #
# Example Usage
# -------------------------------------------------------------------------------- #

if __name__ == "__main__":
    response = test_structured_output_response()
    print(f"Name: {response.choices[0].message.parsed.name}")
    print(f"Age: {response.choices[0].message.parsed.age}")
    print(f"Email: {response.choices[0].message.parsed.email}")
    print(f"Hobbies: {', '.join(response.choices[0].message.parsed.hobbies)}") 