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

# -------------------------------------------------------------------------------- #
# OpenAI Imports
# -------------------------------------------------------------------------------- #

from astral_ai.providers.openai._types import (
    # Message Types
    OpenAIMessageType,

    # Request Types
    OpenAIRequestChatType,
    OpenAIRequestStructuredType,
    OpenAIRequestStreamingType,

    # Clients
    OpenAIClientsType,
    AzureOpenAIClientsType,

    # Response Types

    OpenAIChatResponseType,
    OpenAIStructuredResponseType,
    OpenAIStreamingResponseType,
)

# -------------------------------------------------------------------------------- #
# Anthropic Imports
# -------------------------------------------------------------------------------- #
from astral_ai.providers.anthropic._types import (

    # Message Types
    AnthropicMessageType,
    # Request Types

    AnthropicRequestChatType,
    AnthropicRequestStreamingType,
    AnthropicRequestStructuredType,

    # Response Types
    AnthropicChatResponseType,
    AnthropicStructuredResponseType,
    AnthropicStreamingResponseType,
)


# -------------------------------------------------------------------------------- #
# Provider-Specific Types: Anthropic
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Structured Output Generic
# -------------------------------------------------------------------------------- #

StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# Provider Message Types
# -------------------------------------------------------------------------------- #

# Union alias for any provider message.
ProviderMessageType: TypeAlias = Union[OpenAIMessageType, AnthropicMessageType]
ProviderMessageT = TypeVar("ProviderMessageT", bound=ProviderMessageType)

# -------------------------------------------------------------------------------- #
# Provider Client Types
# -------------------------------------------------------------------------------- #
# Provider Client Types (union of all supported provider clients)
ProviderClientType: TypeAlias = Union[OpenAIClientsType, AzureOpenAIClientsType]
ProviderClientT = TypeVar("ProviderClientT", bound=ProviderClientType)

# -------------------------------------------------------------------------------- #
# Provider Request Types
# -------------------------------------------------------------------------------- #

# Chat Request Types
ProviderRequestChatType: TypeAlias = Union[OpenAIRequestChatType, AnthropicRequestChatType]
ProviderRequestChatT = TypeVar("ProviderRequestChatT", bound=ProviderRequestChatType)

# Structured Request Types
ProviderRequestStructuredType: TypeAlias = Union[OpenAIRequestStructuredType, AnthropicRequestStructuredType]
ProviderRequestStructuredT = TypeVar("ProviderRequestStructuredT", bound=ProviderRequestStructuredType)

# Streaming Request Types
ProviderRequestStreamingType: TypeAlias = Union[OpenAIRequestStreamingType, AnthropicRequestStreamingType]
ProviderRequestStreamingT = TypeVar("ProviderRequestStreamingT", bound=ProviderRequestStreamingType)

# Union alias for any provider request.
ProviderRequestType: TypeAlias = Union[ProviderRequestChatType, ProviderRequestStructuredType, ProviderRequestStreamingType]
ProviderRequestT = TypeVar("ProviderRequestT", bound=ProviderRequestType)

# -------------------------------------------------------------------------------- #
# Provider Response Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider response.
# Chat Response Types
ProviderResponseChatType: TypeAlias = Union[OpenAIChatResponseType, AnthropicChatResponseType]
ProviderResponseChatT = TypeVar("ProviderResponseChatT", bound=ProviderResponseChatType)

# Structured Response Types
ProviderResponseStructuredType: TypeAlias = Union[OpenAIStructuredResponseType, AnthropicStructuredResponseType]
ProviderResponseStructuredT = TypeVar("ProviderResponseStructuredT", bound=ProviderResponseStructuredType)

# Streaming Response Types
ProviderResponseStreamingType: TypeAlias = Union[OpenAIStreamingResponseType, AnthropicStreamingResponseType]
ProviderResponseStreamingT = TypeVar("ProviderResponseStreamingT", bound=ProviderResponseStreamingType)

# Union alias for any provider response.
ProviderResponseType: TypeAlias = Union[ProviderResponseChatType, ProviderResponseStructuredType, ProviderResponseStreamingType]
ProviderResponseT = TypeVar("ProviderResponseT", bound=ProviderResponseType)
