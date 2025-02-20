from __future__ import annotations
# ------------------------------------------------------------------------------
# Anthropic Response Types
# ------------------------------------------------------------------------------

"""
Anthropic Response Types for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in
from typing import (
    Union,
    TypeAlias,
)


# OpenAI Types
from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion, ChatCompletion


# ------------------------------------------------------------------------------
# OpenAI Response Types
# ------------------------------------------------------------------------------


# Streaming Response
AnthropicStreamingResponseType: TypeAlias = ChatCompletionChunk

# Chat Response
AnthropicChatResponseType: TypeAlias = ChatCompletion

# Structured Response
AnthropicStructuredResponseType: TypeAlias = ParsedChatCompletion

# Type Alias for Anthropic Response
AnthropicResponseType: TypeAlias = Union[AnthropicChatResponseType, AnthropicStructuredResponseType, AnthropicStreamingResponseType]
