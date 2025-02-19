# -------------------------------------------------------------------------------- #
# Provider Generics
# -------------------------------------------------------------------------------- #
# This module contains:
#   - Provider-specific types (OpenAI, Anthropic, etc.).
#   - Generic (union) type variables for all providers.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
from typing import TypeAlias, TypeVar, Union

# Pydantic imports
from pydantic import BaseModel

# -------------------------
# OpenAI Imports
# -------------------------
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from astral_ai.providers.openai import (
    # Message Types
    OpenAIMessage,
    # Request Types
    OpenAIRequestChatT,
    OpenAIRequestStructuredT,
    OpenAIRequestStreamingT,
    # Response Types
    OpenAIResponse,
    OpenAIChatResponse,
    OpenAIStructuredResponse,
    OpenAIStreamingResponse,
)

# -------------------------
# Anthropic Imports
# -------------------------
from astral_ai.providers.anthropic import (
    # Message Types
    AnthropicMessage,
    # Request Types
    AnthropicRequest,
    # Response Types
    AnthropicResponse,
    AnthropicChatResponse,
    AnthropicStructuredResponse,
    AnthropicStreamingResponse,
)

# -------------------------------------------------------------------------------- #
# Provider-Specific Types: OpenAI
# -------------------------------------------------------------------------------- #
# OpenAI Message
OpenAIMessageType: TypeAlias = OpenAIMessage
# OpenAI Request
OpenAIRequestType: TypeAlias = Union[OpenAIRequestChatT, OpenAIRequestStructuredT, OpenAIRequestStreamingT]
# OpenAI Response Types
OpenAIResponseType: TypeAlias = OpenAIResponse
OpenAIChatResponseType: TypeAlias = OpenAIChatResponse
OpenAIStructuredResponseType: TypeAlias = OpenAIStructuredResponse
OpenAIStreamingResponseType: TypeAlias = OpenAIStreamingResponse

# OpenAI Client Types
OpenAIClients: TypeAlias = Union[OpenAI, AsyncOpenAI]
AzureOpenAIClients: TypeAlias = Union[AzureOpenAI, AsyncAzureOpenAI]
OpenAIClientT = TypeVar("OpenAIClientT", bound=OpenAIClients)

# -------------------------------------------------------------------------------- #
# Provider-Specific Types: Anthropic
# -------------------------------------------------------------------------------- #
# Anthropic Message
AnthropicMessageType: TypeAlias = AnthropicMessage
# Anthropic Request
AnthropicRequestType: TypeAlias = AnthropicRequest
# Anthropic Response Types
AnthropicResponseType: TypeAlias = AnthropicResponse
AnthropicChatResponseType: TypeAlias = AnthropicChatResponse
AnthropicStructuredResponseType: TypeAlias = AnthropicStructuredResponse
AnthropicStreamingResponseType: TypeAlias = AnthropicStreamingResponse


# -------------------------------------------------------------------------------- #
# Structured Output Generic
# -------------------------------------------------------------------------------- #

StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# Provider Message Types
# -------------------------------------------------------------------------------- #

# Union alias for any provider message.
ProviderMessage: TypeAlias = Union[OpenAIMessage, AnthropicMessage]
ProviderMessageT = TypeVar("ProviderMessageT", bound=ProviderMessage)

# -------------------------------------------------------------------------------- #
# Provider Client Types
# -------------------------------------------------------------------------------- #
# Provider Client Types (union of all supported provider clients)
ModelProviderClient: TypeAlias = Union[OpenAIClients, AzureOpenAIClients]
ModelProviderClientT = TypeVar("ModelProviderClientT", bound=ModelProviderClient)

# -------------------------------------------------------------------------------- #
# Provider Request Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider request.
ProviderRequest: TypeAlias = Union[OpenAIRequestType, AnthropicRequest]
ProviderRequestT = TypeVar("ProviderRequestT", bound=ProviderRequest)
ProviderRequestChatT = TypeVar("ProviderRequestChatT", bound=Union[OpenAIRequestChatT, AnthropicRequest])
ProviderRequestStructuredT = TypeVar("ProviderRequestStructuredT", bound=Union[OpenAIRequestStructuredT, AnthropicRequest])
ProviderRequestStreamingT = TypeVar("ProviderRequestStreamingT", bound=Union[OpenAIRequestStreamingT, AnthropicRequest])
# -------------------------------------------------------------------------------- #
# Provider Response Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider response.
ProviderResponse: TypeAlias = Union[OpenAIResponse, AnthropicResponse]
ProviderResponseT = TypeVar("ProviderResponseT", bound=ProviderResponse)
ProviderResponseChatT = TypeVar("ProviderResponseChatT", bound=Union[OpenAIChatResponse, AnthropicChatResponse])
ProviderResponseStructuredT = TypeVar("ProviderResponseStructuredT", bound=Union[OpenAIStructuredResponse, AnthropicStructuredResponse])
ProviderResponseStreamingT = TypeVar("ProviderResponseStreamingT", bound=Union[OpenAIStreamingResponse, AnthropicStreamingResponse])

