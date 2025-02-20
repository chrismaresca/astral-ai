from __future__ import annotations
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
)


# OpenAI Types
from openai.types.chat import ChatCompletionChunk, ParsedChatCompletion, ChatCompletion


# ------------------------------------------------------------------------------
# OpenAI Response Types
# ------------------------------------------------------------------------------


# Streaming Response
OpenAIStreamingResponseType: TypeAlias = ChatCompletionChunk

# Chat Response
OpenAIChatResponseType: TypeAlias = ChatCompletion

# Structured Response
OpenAIStructuredResponseType: TypeAlias = ParsedChatCompletion

# Type Alias for OpenAI Response
OpenAIResponseType: TypeAlias = Union[OpenAIChatResponseType, OpenAIStructuredResponseType, OpenAIStreamingResponseType]
