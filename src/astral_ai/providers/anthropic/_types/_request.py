from __future__ import annotations
# ------------------------------------------------------------------------------
# Anthropic Request Models
# ------------------------------------------------------------------------------

"""
Anthropic Request Models for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in

from typing import (Literal,
                    Optional,
                    Dict,
                    List,
                    Iterable,
                    Union,
                    TypedDict,
                    Required,
                    TypeAlias,
                    TypeVar,
                    Generic,
                    NotRequired)

# HTTPX Timeout
from httpx import Timeout

# Pydantic
from pydantic import BaseModel

# Astral AI
from astral_ai.constants._models import AnthropicModels

# Anthropic Types
from anthropic.types import (
    MessageParam,
    MetadataParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUnionParam,
)

# Astral AI Types
from astral_ai._types._request._request import (
    ToolChoice,
    ReasoningEffort,
    ResponseFormat,
    Metadata,
    ResponsePrediction,
    Modality,
    StreamOptions,
    Tool,
)

# OpenAI Types
from ._message import AnthropicMessageType
# ------------------------------------------------------------------------------
# Generic Types
# ------------------------------------------------------------------------------

# Response Format Type
_ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)

# ------------------------------------------------------------------------------
# Reasoning Effort Type
# ------------------------------------------------------------------------------

class AnthropicReasoningEnabled(TypedDict):
    type: Literal["enabled"]
    budget_tokens: int


class AnthropicReasoningDisabled(TypedDict):
    type: Literal["disabled"]


AnthropicReasoning: TypeAlias = Union[AnthropicReasoningEnabled, AnthropicReasoningDisabled]

# ------------------------------------------------------------------------------
# Anthropic Request Objects
# ------------------------------------------------------------------------------


class AnthropicRequestBase(TypedDict):
    """
    Anthropic Request Base Model for Astral AI
    """
    model: Required[AnthropicModels]
    messages: Required[AnthropicMessageType]
    max_tokens: Required[int]
    """The maximum number of tokens to generate before stopping."""

    metadata: NotRequired[MetadataParam]
    """An object describing metadata about the request."""

    stop_sequences: NotRequired[List[str]]
    """Sequences that will cause the model to stop generating."""

    system: NotRequired[Union[str, Iterable[TextBlockParam]]]
    """System prompt that helps set the behavior of the assistant."""

    temperature: NotRequired[float]
    """Amount of randomness injected into the response. Defaults to 1.0."""

    thinking: NotRequired[ThinkingConfigParam]
    """Controls Claude's thinking capabilities."""

    tool_choice: NotRequired[ToolChoiceParam]
    """Controls which (if any) tool is called by the model."""

    tools: NotRequired[Iterable[ToolUnionParam]]
    """A list of tools the model may call."""

    top_k: NotRequired[int]
    """Only sample from the top K options for each subsequent token."""

    top_p: NotRequired[float]
    """Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p."""

    timeout: NotRequired[Union[float, Timeout]]
    """Override the client-level default timeout for this request (in seconds)."""


# ------------------------------------------------------------------------------
# Chat Request
# ------------------------------------------------------------------------------

class AnthropicRequestChat(AnthropicRequestBase, total=False):
    """
    Anthropic Request Chat Model for Astral AI

    This model is used for chat completions that don't use streaming.

    Parameters:
        model: The Anthropic model to use
        messages: A list of messages comprising the conversation so far
        max_tokens: The maximum number of tokens to generate
        metadata: An object describing metadata about the request
        stop_sequences: Sequences that will cause the model to stop generating
        system: System prompt that helps set the behavior of the assistant
        temperature: Amount of randomness injected into the response
        thinking: Controls Claude's thinking capabilities
        tool_choice: Controls which (if any) tool is called by the model
        tools: A list of tools the model may call
        top_k: Only sample from the top K options for each token
        top_p: Use nucleus sampling with this probability threshold
        timeout: Override the client-level default timeout
    """
    pass

# ------------------------------------------------------------------------------
# Streaming Request
# ------------------------------------------------------------------------------


class AnthropicRequestStreaming(AnthropicRequestBase):
    """
    Anthropic Request Streaming Model for Astral AI

    This model is used for streaming chat completions.
    """
    stream: Literal[True]

# ------------------------------------------------------------------------------
# Structured Request
# ------------------------------------------------------------------------------


class AnthropicRequestStructured(Generic[_ResponseFormatT], AnthropicRequestBase):
    """
    Anthropic Request Structured Model for Astral AI

    This model is used for structured responses.
    """
    # IMPORTANT: Anthropic does not support structured responses.
    response_format: _ResponseFormatT


# ------------------------------------------------------------------------------
# Embedding Request Types
# ------------------------------------------------------------------------------

class AnthropicRequestEmbedding(TypedDict, total=False):
    """
    Anthropic Request Embedding Model for Astral AI
    """
    model: Required[AnthropicModels]
    input: Required[str | List[str]]
    timeout: NotRequired[Union[float, Timeout]]


# ------------------------------------------------------------------------------
# Type Aliases
# ------------------------------------------------------------------------------


# Union of all request types
AnthropicRequestType: TypeAlias = Union[
    AnthropicRequestChat,
    AnthropicRequestStreaming,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding,
]
