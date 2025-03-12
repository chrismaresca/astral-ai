# -------------------------------------------------------------------------------- #
# Provider Generics
# -------------------------------------------------------------------------------- #
# This module defines generic type aliases and type variables for provider-specific
# types to enable type-safe interactions with different AI providers while maintaining
# a consistent interface throughout the codebase.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import TypeAlias, TypeVar, Union, TYPE_CHECKING
# Pydantic imports
from pydantic import BaseModel

# -------------------------------------------------------------------------------- #
# Type-checking only imports
# -------------------------------------------------------------------------------- #
if TYPE_CHECKING:
    # These imports will only be active during type checking and not at runtime,
    # breaking the circular dependency.

    # -------------------------------------------------------------------------------- #
    # OpenAI Types
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

        # Embedding Types
        OpenAIRequestEmbeddingType,
    )

    # -------------------------------------------------------------------------------- #
    # Anthropic Types
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

        # Embedding Types
        AnthropicRequestEmbeddingType,
    )

    # -------------------------------------------------------------------------------- #
    # DeepSeek Types
    # -------------------------------------------------------------------------------- #
    from astral_ai.providers.deepseek._types import (
        # Clients
        DeepSeekClientsType,
        # IMPORTANT: Verify this before implementing Azure DeepSeek
        # DeepSeekAzureClientsType,

        # Request Types
        DeepSeekRequestChatType,
        DeepSeekRequestStreamingType,
        DeepSeekRequestStructuredType,
        # Response Types
        DeepSeekChatResponseType,
        DeepSeekStructuredResponseType,
        DeepSeekStreamingResponseType,

        # # Embedding Types
        # DeepSeekRequestEmbeddingType,
    )

# -------------------------------------------------------------------------------- #
# Structured Output Generic
# -------------------------------------------------------------------------------- #
# Type variable for structured outputs that must be Pydantic models
StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# -------------------------------------------------------------------------------- #
# Provider Message Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider message format.
# This allows for type-safe handling of different message formats across providers.
ProviderMessageType: TypeAlias = Union[
    "OpenAIMessageType",  # type: ignore  # These names are only resolved during type checking.
    "AnthropicMessageType"
]
ProviderMessageT = TypeVar("ProviderMessageT", bound=ProviderMessageType)

# -------------------------------------------------------------------------------- #
# Provider Client Types
# -------------------------------------------------------------------------------- #
# Provider Client Types (union of all supported provider clients)
# This enables generic handling of different client implementations.
ProviderClientType: TypeAlias = Union["OpenAIClientsType", "AzureOpenAIClientsType", "DeepSeekClientsType"]
ProviderClientT = TypeVar("ProviderClientT", bound=ProviderClientType)

# -------------------------------------------------------------------------------- #
# Provider Request Types
# -------------------------------------------------------------------------------- #
# Chat Request Types - For standard chat completions
ProviderRequestChatType: TypeAlias = Union["OpenAIRequestChatType", "AnthropicRequestChatType", "DeepSeekRequestChatType"]
ProviderRequestChatT = TypeVar("ProviderRequestChatT", bound=ProviderRequestChatType)

# Structured Request Types - For requests that expect structured (e.g., JSON) responses
ProviderRequestStructuredType: TypeAlias = Union["OpenAIRequestStructuredType", "AnthropicRequestStructuredType", "DeepSeekRequestStructuredType"]
ProviderRequestStructuredT = TypeVar("ProviderRequestStructuredT", bound=ProviderRequestStructuredType)

# Streaming Request Types - For requests that use streaming responses
ProviderRequestStreamingType: TypeAlias = Union["OpenAIRequestStreamingType", "AnthropicRequestStreamingType", "DeepSeekRequestStreamingType"]
ProviderRequestStreamingT = TypeVar("ProviderRequestStreamingT", bound=ProviderRequestStreamingType)

# -------------------------------------------------------------------------------- #
# Provider Response Types
# -------------------------------------------------------------------------------- #
# Chat Response Types - For standard chat completion responses
ProviderResponseChatType: TypeAlias = Union["OpenAIChatResponseType", "AnthropicChatResponseType", "DeepSeekChatResponseType"]
ProviderResponseChatT = TypeVar("ProviderResponseChatT", bound=ProviderResponseChatType)

# Structured Response Types - For structured (e.g., JSON) responses
ProviderResponseStructuredType: TypeAlias = Union["OpenAIStructuredResponseType", "AnthropicStructuredResponseType", "DeepSeekStructuredResponseType"]
ProviderResponseStructuredT = TypeVar("ProviderResponseStructuredT", bound=ProviderResponseStructuredType)

# Streaming Response Types - For streaming response formats
ProviderResponseStreamingType: TypeAlias = Union["OpenAIStreamingResponseType", "AnthropicStreamingResponseType", "DeepSeekStreamingResponseType"]
ProviderResponseStreamingT = TypeVar("ProviderResponseStreamingT", bound=ProviderResponseStreamingType)

# -------------------------------------------------------------------------------- #
# Provider Combined Response Types
# -------------------------------------------------------------------------------- #
# Non-streaming completion response types (chat or structured)
# This combines both chat and structured response types for generic handling
ProviderCompletionResponseType: TypeAlias = Union[
    ProviderResponseChatType,
    ProviderResponseStructuredType
]

# -------------------------------------------------------------------------------- #
# Provider Request/Response Union Aliases
# -------------------------------------------------------------------------------- #
# Union alias for any chat-related provider request (standard or streaming).
# This allows for handling both standard and streaming requests with the same code.
ProviderChatRequestType: TypeAlias = Union[
    ProviderRequestChatType,
    ProviderRequestStreamingType
]
ProviderChatRequestT = TypeVar("ProviderChatRequestT", bound=ProviderChatRequestType)

# Structured Request Types - For requests that expect structured output
ProviderStructuredRequestType: TypeAlias = Union[
    ProviderRequestStructuredType,
]
ProviderStructuredRequestT = TypeVar("ProviderStructuredRequestT", bound=ProviderStructuredRequestType)

# -------------------------------------------------------------------------------- #
# Provider Embedding Request Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider embedding request.
# This enables generic handling of embedding requests across providers.
ProviderEmbeddingRequestType: TypeAlias = Union[
    'OpenAIRequestEmbeddingType',
    'AnthropicRequestEmbeddingType'
]
ProviderEmbeddingRequestT = TypeVar("ProviderEmbeddingRequestT", bound=ProviderEmbeddingRequestType)

# -------------------------------------------------------------------------------- #
# Provider Response Types
# -------------------------------------------------------------------------------- #
# Union alias for any provider response (chat, structured, or streaming).
# This allows for generic handling of all response types.
ProviderResponseType: TypeAlias = Union[
    ProviderResponseChatType,
    ProviderResponseStructuredType,
    ProviderResponseStreamingType
]
ProviderResponseT = TypeVar("ProviderResponseT", bound=ProviderResponseType)
