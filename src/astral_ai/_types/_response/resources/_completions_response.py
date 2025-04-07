from __future__ import annotations
# ------------------------------------------------------------------------------
# Provider Agnostic Completions Response Models
# ------------------------------------------------------------------------------

"""
Provider Agnostic Completions Response Models
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in imports
from typing import List, Optional, Generic, Any
from typing_extensions import Literal

# Pydantic
from pydantic import BaseModel

# Base Resource Response
from ._base_resource_response import BaseProviderResourceResponse


# OpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.parsed_chat_completion import ParsedChoice

# TODO: Remove this once the openai types are updated
from openai._models import GenericModel

# Generic Types
from typing import TypeVar

# ------------------------------------------------------------------------------
# Generic Types
# ------------------------------------------------------------------------------

ContentType = TypeVar("ContentType")

# StructuredOutputResponse = TypeVar("StructuredOutputResponse", bound=BaseModel)


# ------------------------------------------------------------------------------
# Chat Completion
# ------------------------------------------------------------------------------


class ChatCompletionResponse(BaseProviderResourceResponse):
    id: str
    """A unique identifier for the chat completion."""

    choices: List[Choice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    # created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    # object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    # service_tier: Optional[Literal["scale", "default"]] = None
    """The service tier used for processing the request."""

    # system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """


# ------------------------------------------------------------------------------
# Structured Output Completion Response
# ------------------------------------------------------------------------------

class StructuredOutputCompletionResponse(ChatCompletionResponse, GenericModel, Generic[ContentType]):
    """
    Structured Output Completion Response

    A provider-agnostic completion response that contains structured output
    in the form of a Pydantic model specified by the StructuredOutputResponse type parameter.
    """
    choices: List[ParsedChoice[ContentType]]
    """A list of structured output choices with parsed content."""



# TODO: Figure out how to use the base model for the structured output response. but still have it work with Tools. 
# TODO: One solution could be to identify why the Tool fails with BaseModel. or have it bound to BaseModel and another non-BaseModel, have it pass appropriately.
# TODO: i.e. if request.tools is not empty, then use the non-BaseModel, if it is empty, then use the BaseModel.