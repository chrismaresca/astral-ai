from __future__ import annotations
# -------------------------------------------------------------------------------- #
# Anthropic Response Types
# -------------------------------------------------------------------------------- #

"""
Anthropic Response Types for Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in
from typing import (
    Union,
    TypeAlias,
    TypeVar,
)

# Pydantic
from pydantic import BaseModel

# Anthropic Types
from anthropic.types import (
    Message,
    RawMessageStreamEvent,
    Usage,
)

# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #
_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# -------------------------------------------------------------------------------- #
# Anthropic Response Types
# -------------------------------------------------------------------------------- #

# Streaming Response Types
AnthropicStreamingResponseType: TypeAlias = RawMessageStreamEvent


# Complete Response Type
AnthropicChatResponseType: TypeAlias = Message

# Structured Response Type
# IMPORTANT: Anthropic does not support structured responses.
AnthropicStructuredResponseType: TypeAlias = Message

# Usage Type
AnthropicCompletionUsageType: TypeAlias = Usage

# Type Alias for Anthropic Response (all types)
AnthropicResponseType: TypeAlias = Union[
    AnthropicChatResponseType,
    AnthropicStreamingResponseType,
]
