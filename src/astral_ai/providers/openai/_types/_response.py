from __future__ import annotations
from typing import TypeVar
# ------------------------------------------------------------------------------
# OpenAI Response Types
# ------------------------------------------------------------------------------

"""
OpenAI Response Types for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in
from typing import (
    Union,
    TypeAlias,
    TypeVar,
)

# Pydantic
from pydantic import BaseModel


# OpenAI Types
from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion, ChatCompletion

# ------------------------------------------------------------------------------
# Generic Types
# ------------------------------------------------------------------------------

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# ------------------------------------------------------------------------------
# OpenAI Response Types
# ------------------------------------------------------------------------------


# Streaming Response
OpenAIStreamingResponseType: TypeAlias = ChatCompletionChunk

# Chat Response
OpenAIChatResponseType: TypeAlias = ChatCompletion

# # Structured Response
# OpenAIStructuredResponseType: TypeAlias = ParsedChatCompletion


OpenAIStructuredResponseType: TypeAlias = ParsedChatCompletion[_StructuredOutputT]


# Type Alias for OpenAI Response
OpenAIResponseType: TypeAlias = Union[OpenAIChatResponseType, OpenAIStructuredResponseType, OpenAIStreamingResponseType]
