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
DeepSeekStreamingResponseType: TypeAlias = ChatCompletionChunk

# Chat Response
DeepSeekChatResponseType: TypeAlias = ChatCompletion


# DeepSeek Structured Response
# TODO: This is because the DeepSeek API returns a ChatCompletion object for structured responses.
DeepSeekStructuredResponseType: TypeAlias = ChatCompletion

# DeepSeek Completion Usage
DeepSeekCompletionUsageType: TypeAlias = CompletionUsage

# Type Alias for DeepSeek Response
DeepSeekResponseType: TypeAlias = Union[DeepSeekChatResponseType, DeepSeekStructuredResponseType, DeepSeekStreamingResponseType]

# ------------------------------------------------------------------------------
# OpenAI Embedding Response
# ------------------------------------------------------------------------------

# TODO: Embedding Response
