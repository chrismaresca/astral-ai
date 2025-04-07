from __future__ import annotations
# -------------------------------------------------------------------------------- #
# OpenAI Request Models
# -------------------------------------------------------------------------------- #

"""
OpenAI Request Models for Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
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
from astral_ai.constants._models import OpenAIModels

# Astral AI Types
from astral_ai._types import (
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
from ._message import OpenAIMessageType
# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

# Response Format Type
_ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# OpenAI Request Objects
# -------------------------------------------------------------------------------- #


class OpenAIChatRequestBase(TypedDict, total=False):
    """
    OpenAI Request Base Model for Astral AI
    """
    model: Required[OpenAIModels]
    messages: Required[Union[List[OpenAIMessageType], OpenAIMessageType]]
    frequency_penalty: NotRequired[float]
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: NotRequired[Dict[str, int]]
    """Modify the likelihood of specified tokens appearing in the completion."""

    logprobs: NotRequired[bool]
    """Whether to return log probabilities of the output tokens."""

    max_completion_tokens: NotRequired[int]
    """An upper bound for the number of tokens that can be generated for a completion."""

    max_tokens: NotRequired[int]
    """The maximum number of tokens that can be generated in the chat completion (deprecated in favor of `max_completion_tokens`)."""

    metadata: NotRequired[Metadata]
    """Set of key-value pairs that can be attached to the object (max 16 pairs)."""

    modalities: NotRequired[List[Modality]]
    """Output types that you would like the model to generate for this request (e.g., ['text'] or ['text', 'audio'])."""

    n: NotRequired[int]
    """How many chat completion choices to generate for each input message."""

    parallel_tool_calls: NotRequired[bool]
    """Whether to enable parallel function calling during tool use."""

    prediction: NotRequired[ResponsePrediction]
    """Static predicted output content (e.g., content of a text file being regenerated)."""

    presence_penalty: NotRequired[float]
    """Number between -2.0 and 2.0. Positive values penalize tokens based on whether they appear in the text so far."""

    reasoning_effort: NotRequired[ReasoningEffort]
    """For reasoning models only. Constrains effort on reasoning (e.g., 'low', 'medium', 'high')."""

    response_format: NotRequired[ResponseFormat]
    """Specifies the format that the model must output."""

    seed: NotRequired[int]
    """If specified, attempts to make sampling deterministic for repeated requests."""

    service_tier: NotRequired[Literal["auto", "default"]]
    """Specifies the latency tier to use for processing the request."""

    stop: NotRequired[Union[str, List[str]]]
    """Up to 4 sequences where the API will stop generating further tokens."""

    store: NotRequired[bool]
    """Whether or not to store the output of this chat completion request."""

    stream_options: NotRequired[StreamOptions]
    """Options for streaming responses; only set when `stream` is true."""

    temperature: NotRequired[float]
    """Sampling temperature to use, between 0 and 2."""

    tool_choice: NotRequired[ToolChoice]
    """Controls which (if any) tool is called by the model."""

    tools: NotRequired[Iterable[Tool]]
    """A list of tools the model may call (e.g., functions)."""

    top_logprobs: NotRequired[int]
    """An integer (0 to 20) specifying the number of most likely tokens to return at each token position."""

    top_p: NotRequired[float]
    """Nucleus sampling parameter, where the model considers tokens with top_p probability mass."""

    user: NotRequired[str]
    """A unique identifier representing your end-user."""

    timeout: NotRequired[Union[float, Timeout]]
    """Override the client-level default timeout for this request (in seconds)."""


# -------------------------------------------------------------------------------- #
# Chat Request
# -------------------------------------------------------------------------------- #

class OpenAIRequestChat(OpenAIChatRequestBase, total=False):
    """
    OpenAI Request Chat Model for Astral AI
    """
    pass

# -------------------------------------------------------------------------------- #
# OpenAI Request Types
# -------------------------------------------------------------------------------- #


class OpenAIRequestStreaming(OpenAIChatRequestBase):
    """
    OpenAI Request Streaming Model for Astral AI
    """
    stream: Literal[True]

# -------------------------------------------------------------------------------- #
# OpenAI Request Types
# -------------------------------------------------------------------------------- #


class OpenAIRequestStructured(Generic[_ResponseFormatT], OpenAIChatRequestBase):
    """
    OpenAI Request Structured Model for Astral AI
    """
    response_format: _ResponseFormatT


# -------------------------------------------------------------------------------- #
# TODO: OpenAI Embedding Request Types
# -------------------------------------------------------------------------------- #


class OpenAIRequestEmbedding(TypedDict, total=False):
    """
    OpenAI Request Embedding Model for Astral AI
    """
    model: Required[OpenAIModels]
    input: Required[str | List[str]]
    user: NotRequired[str]
    timeout: NotRequired[Union[float, Timeout]]


# -------------------------------------------------------------------------------- #
# Type Aliases
# -------------------------------------------------------------------------------- #

OpenAIRequestType: TypeAlias = Union[
    OpenAIRequestChat,
    OpenAIRequestStreaming,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding,
]
