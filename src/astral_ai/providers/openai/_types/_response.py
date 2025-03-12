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
from openai.types.completion_usage import CompletionUsage
# ------------------------------------------------------------------------------
# Generic Types
# ------------------------------------------------------------------------------

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# ------------------------------------------------------------------------------
# OpenAI Chat Response Types
# ------------------------------------------------------------------------------


# Streaming Response
OpenAIStreamingResponseType: TypeAlias = ChatCompletionChunk

# Chat Response
OpenAIChatResponseType: TypeAlias = ChatCompletion


# OpenAI Structured Response
OpenAIStructuredResponseType: TypeAlias = ParsedChatCompletion[_StructuredOutputT]

# OpenAI Completion Usage
OpenAICompletionUsageType: TypeAlias = CompletionUsage

# Type Alias for OpenAI Response
OpenAIResponseType: TypeAlias = Union[OpenAIChatResponseType, OpenAIStructuredResponseType, OpenAIStreamingResponseType]

# ------------------------------------------------------------------------------
# OpenAI Embedding Response
# ------------------------------------------------------------------------------

# TODO: Embedding Response
