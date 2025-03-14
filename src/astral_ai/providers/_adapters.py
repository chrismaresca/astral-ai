# -------------------------------------------------------------------------------- #
# Provider Adapters
# -------------------------------------------------------------------------------- #
# This module contains:
#   - Provider-specific adapters for converting between Astral AI and provider formats
#   - A factory function for creating the appropriate adapter
#   - Helper functions for message conversions
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
from typing import (
    Literal,
    cast,
    Generic,
    Type,
    Optional,
    overload,
    TypeVar,
    Union,
    List,
    Any,
    Dict,
)

# Pydantic imports
from pydantic import BaseModel

# -------------------------------------------------------------------------------- #
# Astral AI Imports
# -------------------------------------------------------------------------------- #
# Constants
from astral_ai.constants._models import ModelProvider

# Types
from astral_ai._types import NOT_GIVEN
from astral_ai._types._request._request import (
    AstralCompletionRequest,
    AstralEmbeddingRequest,
    AstralStructuredCompletionRequest,
)
from astral_ai._types._response._response import (
    AstralChatResponse,
    AstralStructuredResponse,
)
from astral_ai._types._response._usage import ChatUsage, ChatCost

# Messaging
from astral_ai.messaging._models import Message

from astral_ai.providers._generics import (
    ProviderChatRequestType,
    ProviderStructuredRequestType,
    ProviderEmbeddingRequestType,
    ProviderCompletionResponseType,
    StructuredOutputT,
)

# Response Resources
from astral_ai._types._response.resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
    ParsedChoice,
)

# -------------------------------------------------------------------------------- #
# Provider-Specific Types
# -------------------------------------------------------------------------------- #

# -- OpenAI
from astral_ai.providers.openai._types._request import (
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding,
)
from astral_ai.providers.openai._types._response import (
    OpenAIChatResponseType,
    OpenAIStructuredResponseType,
    OpenAICompletionUsageType,
)
from astral_ai.providers.openai._types._message import OpenAIMessageType

# -- Anthropic
from astral_ai.providers.anthropic._types._request import (
    AnthropicRequestChat,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding,
)
from astral_ai.providers.anthropic._types._response import (
    AnthropicChatResponseType,
    AnthropicStructuredResponseType,
)

# -- DeepSeek
from astral_ai.providers.deepseek._types._request import (
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
)
from astral_ai.providers.deepseek._types._response import (
    DeepSeekChatResponseType,
    DeepSeekStructuredResponseType,
    DeepSeekCompletionUsageType,
)


# -------------------------------------------------------------------------------- #
# Message Conversion
# -------------------------------------------------------------------------------- #

def convert_messages_to_provider_format(
    messages: list[Message],
    provider: Literal["openai", "anthropic", "deepseek"] = "openai"
) -> Union[List[OpenAIMessageType], List[Dict[str, Any]]]:
    """
    Convert a list of Astral Messages to the provider-specific format.
    
    Args:
        messages: List of Astral Messages to convert
        provider: The provider to format messages for
        
    Returns:
        A list of messages in the provider-specific format
    """
    return [
        {
            "role": m.role,
            "content": m.content,
        }
        for m in messages
    ]


# -------------------------------------------------------------------------------- #
# Usage Data Conversion
# -------------------------------------------------------------------------------- #

def create_usage_data(
    usage: Optional[Union[OpenAICompletionUsageType, DeepSeekCompletionUsageType]]
) -> ChatUsage:
    """
    Create usage data from a provider's usage block, if present.
    
    Args:
        usage: The usage data from a provider response
        
    Returns:
        Standardized ChatUsage object with token counts
    """
    if not usage:
        return ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return ChatUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        # example nested tokens
        audio_tokens=(
            getattr(usage.prompt_tokens_details, "audio_tokens", None)
            if hasattr(usage, "prompt_tokens_details") else None
        ),
        cached_tokens=(
            getattr(usage.prompt_tokens_details, "cached_tokens", None)
            if hasattr(usage, "prompt_tokens_details") else None
        ),
        accepted_prediction_tokens=(
            getattr(usage.completion_tokens_details, "accepted_prediction_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
        rejected_prediction_tokens=(
            getattr(usage.completion_tokens_details, "rejected_prediction_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
        reasoning_tokens=(
            getattr(usage.completion_tokens_details, "reasoning_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
    )


# -------------------------------------------------------------------------------- #
# Base Provider Adapter (Generic)
# -------------------------------------------------------------------------------- #

_ModelProviderT = TypeVar("_ModelProviderT", bound=ModelProvider)

TChatReq = TypeVar("TChatReq", bound=ProviderChatRequestType)
TStructReq = TypeVar("TStructReq", bound=ProviderStructuredRequestType)
TEmbedReq = TypeVar("TEmbedReq", bound=ProviderEmbeddingRequestType)


class BaseProviderAdapter(
    Generic[_ModelProviderT, TChatReq, TStructReq, TEmbedReq]
):
    """
    Base adapter class with common functionality for all providers.
    
    Provides the interface for converting between Astral AI and provider-specific
    request and response formats. Each provider implements this interface
    with their specific conversion logic.
    
    Type Parameters:
        _ModelProviderT: The model provider type (e.g., "openai", "anthropic")
        TChatReq: The provider's chat request type
        TStructReq: The provider's structured request type
        TEmbedReq: The provider's embedding request type
    """

    def __init__(self, provider_name: _ModelProviderT):
        """
        Initialize the adapter with a provider name.
        
        Args:
            provider_name: The name of the provider this adapter is for
        """
        self.provider_name: _ModelProviderT = provider_name

    # -------------------------------------------------------------------------
    # Astral -> Provider
    # -------------------------------------------------------------------------
    def to_chat_request(self, request: AstralCompletionRequest) -> TChatReq:
        """
        Convert an AstralCompletionRequest into a provider-specific chat request.
        
        Args:
            request: The Astral completion request
            
        Returns:
            Provider-specific chat request
        """
        raise NotImplementedError()

    def to_structured_request(self, request: AstralStructuredCompletionRequest) -> TStructReq:
        """
        Convert an AstralStructuredCompletionRequest into a provider-specific structured request.
        
        Args:
            request: The Astral structured completion request
            
        Returns:
            Provider-specific structured request
        """
        raise NotImplementedError()

    def to_embedding_request(self, request: AstralEmbeddingRequest) -> TEmbedReq:
        """
        Convert an AstralEmbeddingRequest into a provider-specific embedding request.
        
        Args:
            request: The Astral embedding request
            
        Returns:
            Provider-specific embedding request
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Provider -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType,
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Convert a provider-specific response to an AstralChatResponse or AstralStructuredResponse.
        
        Args:
            response: The provider's response
            response_format: Optional type for structured output parsing
            
        Returns:
            An Astral response object (either chat or structured)
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _build_astral_chat_response(
        self,
        model: str,
        provider_response: ChatCompletionResponse,
        usage: ChatUsage,
        cost: Optional[ChatCost] = None
    ) -> AstralChatResponse:
        """
        Build an AstralChatResponse from a ChatCompletionResponse
        
        Args:
            model: The model name
            provider_response: Provider's chat completion response
            usage: Token usage information
            cost: Optional cost information
            
        Returns:
            Standardized Astral chat response
        """
        return AstralChatResponse(
            model=model,
            response=provider_response,
            usage=usage,
            cost=cost,
        )

    def _build_astral_structured_response(
        self,
        model: str,
        provider_response: StructuredOutputCompletionResponse[StructuredOutputT],
        usage: ChatUsage,
        cost: Optional[ChatCost] = None
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Build an AstralStructuredResponse from a StructuredOutputCompletionResponse
        
        Args:
            model: The model name
            provider_response: Provider's structured completion response
            usage: Token usage information
            cost: Optional cost information
            
        Returns:
            Standardized Astral structured response
        """
        return AstralStructuredResponse[StructuredOutputT](
            model=model,
            response=provider_response,
            usage=usage,
            cost=cost,
        )


# -------------------------------------------------------------------------------- #
# OpenAI Adapter
# -------------------------------------------------------------------------------- #

class OpenAIAdapter(
    BaseProviderAdapter[
        Literal["openai"],
        OpenAIRequestChat,
        OpenAIRequestStructured,
        OpenAIRequestEmbedding,
    ]
):
    """
    Adapter for OpenAI-specific request and response formats.
    
    Handles converting between Astral AI's standardized formats and
    OpenAI's API-specific formats for requests and responses.
    """
    
    def __init__(self):
        """Initialize the OpenAI adapter"""
        super().__init__("openai")

    # -------------------------------------------------------------------------
    # Astral -> OpenAI
    # -------------------------------------------------------------------------
    def to_chat_request(self, request: AstralCompletionRequest) -> OpenAIRequestChat:
        """
        Convert an Astral completion request to OpenAI's chat request format.
        
        Args:
            request: The Astral completion request
            
        Returns:
            OpenAI-compatible chat request
        """
        data = request.model_dump_without_astral_params()
        data["messages"] = convert_messages_to_provider_format(
            data["messages"], provider="openai"
        )
        # Filter out NOT_GIVEN
        filtered = {k: v for k, v in data.items() if v is not NOT_GIVEN}
        return cast(OpenAIRequestChat, filtered)

    def to_structured_request(
        self, request: AstralStructuredCompletionRequest
    ) -> OpenAIRequestStructured:
        """
        Convert an Astral structured request to OpenAI's structured request format.
        
        Args:
            request: The Astral structured completion request
            
        Returns:
            OpenAI-compatible structured request
        """
        data = request.model_dump_without_astral_params()
        data["messages"] = convert_messages_to_provider_format(
            data["messages"], provider="openai"
        )
        filtered = {k: v for k, v in data.items() if v is not NOT_GIVEN}
        return cast(OpenAIRequestStructured, filtered)

    def to_embedding_request(
        self, request: AstralEmbeddingRequest
    ) -> OpenAIRequestEmbedding:
        """
        Convert an Astral embedding request to OpenAI's embedding request format.
        
        Args:
            request: The Astral embedding request
            
        Returns:
            OpenAI-compatible embedding request
        """
        # Example placeholder
        raise NotImplementedError("OpenAI embeddings not implemented.")

    # -------------------------------------------------------------------------
    # OpenAI -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: Union[OpenAIChatResponseType, OpenAIStructuredResponseType],
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Convert an OpenAI response to an Astral response.
        
        Args:
            response: The OpenAI response
            response_format: Optional type for structured output parsing
            
        Returns:
            An Astral response (either chat or structured)
        """
        if response_format is None:
            return self._from_chat_response(cast(OpenAIChatResponseType, response))
        else:
            return self._from_structured_response(
                cast(OpenAIStructuredResponseType, response),
                response_format
            )

    # -------------------------------------------------------------------------
    # Response Converters
    # -------------------------------------------------------------------------
    def _from_chat_response(
        self, response: OpenAIChatResponseType
    ) -> AstralChatResponse:
        """
        Convert an OpenAI chat response to an Astral chat response.
        
        Args:
            response: The OpenAI chat response
            
        Returns:
            Standardized Astral chat response
        """
        provider_resp = ChatCompletionResponse(
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=response.service_tier,
            system_fingerprint=response.system_fingerprint,
        )
        usage_data = create_usage_data(response.usage)
        return self._build_astral_chat_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            cost=None,
        )

    def _from_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Convert an OpenAI structured response to an Astral structured response.
        
        Args:
            response: The OpenAI structured response
            response_model: Type for structured output parsing
            
        Returns:
            Standardized Astral structured response
        """
        provider_resp = StructuredOutputCompletionResponse[StructuredOutputT](
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=response.service_tier,
            system_fingerprint=response.system_fingerprint,
        )
        usage_data = create_usage_data(response.usage)
        return self._build_astral_structured_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            cost=None,
        )


# -------------------------------------------------------------------------------- #
# DeepSeek Adapter
# -------------------------------------------------------------------------------- #

class DeepSeekAdapter(
    BaseProviderAdapter[
        Literal["deepseek"],
        DeepSeekRequestChat,
        DeepSeekRequestStructured,
        Dict[str, Any],  # or a real "DeepSeekEmbeddingRequest" if you have one
    ]
):
    """
    Adapter for DeepSeek-specific request and response formats.
    
    Handles converting between Astral AI's standardized formats and
    DeepSeek's API-specific formats for requests and responses.
    """
    
    def __init__(self):
        """Initialize the DeepSeek adapter"""
        super().__init__("deepseek")

    # -------------------------------------------------------------------------
    # Astral -> DeepSeek
    # -------------------------------------------------------------------------
    def to_chat_request(self, request: AstralCompletionRequest) -> DeepSeekRequestChat:
        """
        Convert an Astral completion request to DeepSeek's chat request format.
        
        Args:
            request: The Astral completion request
            
        Returns:
            DeepSeek-compatible chat request
        """
        data = request.model_dump_without_astral_params()
        data["messages"] = convert_messages_to_provider_format(
            data["messages"], provider="deepseek"
        )
        filtered = {k: v for k, v in data.items() if v is not NOT_GIVEN}
        return cast(DeepSeekRequestChat, filtered)

    def to_structured_request(
        self, request: AstralStructuredCompletionRequest
    ) -> DeepSeekRequestStructured:
        """
        Convert an Astral structured request to DeepSeek's structured request format.
        
        Args:
            request: The Astral structured completion request
            
        Returns:
            DeepSeek-compatible structured request
        """
        data = request.model_dump_without_astral_params()
        data["messages"] = convert_messages_to_provider_format(
            data["messages"], provider="deepseek"
        )
        filtered = {k: v for k, v in data.items() if v is not NOT_GIVEN}
        return cast(DeepSeekRequestStructured, filtered)

    def to_embedding_request(
        self, request: AstralEmbeddingRequest
    ) -> Dict[str, Any]:
        """
        Convert an Astral embedding request to DeepSeek's embedding request format.
        
        Args:
            request: The Astral embedding request
            
        Returns:
            DeepSeek-compatible embedding request
        """
        # Placeholder
        raise NotImplementedError("DeepSeek embedding requests not implemented.")

    # -------------------------------------------------------------------------
    # DeepSeek -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: Union[DeepSeekChatResponseType, DeepSeekStructuredResponseType],
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Convert a DeepSeek response to an Astral response.
        
        Args:
            response: The DeepSeek response
            response_format: Optional type for structured output parsing
            
        Returns:
            An Astral response (either chat or structured)
        """
        if response_format is None:
            return self._from_chat_response(cast(DeepSeekChatResponseType, response))
        else:
            return self._from_structured_response(
                cast(DeepSeekStructuredResponseType, response),
                response_format
            )

    # -------------------------------------------------------------------------
    # Response Converters
    # -------------------------------------------------------------------------
    def _from_chat_response(
        self,
        response: DeepSeekChatResponseType
    ) -> AstralChatResponse:
        """
        Convert a DeepSeek chat response to an Astral chat response.
        
        Args:
            response: The DeepSeek chat response
            
        Returns:
            Standardized Astral chat response
        """
        provider_resp = ChatCompletionResponse(
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=getattr(response, "service_tier", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
        )
        usage_data = create_usage_data(response.usage)
        return self._build_astral_chat_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            cost=None,
        )

    def _from_structured_response(
        self,
        response: DeepSeekStructuredResponseType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Convert a DeepSeek structured response to an Astral structured response.
        
        Args:
            response: The DeepSeek structured response
            response_model: Type for structured output parsing
            
        Returns:
            Standardized Astral structured response
        """
        provider_resp = StructuredOutputCompletionResponse[StructuredOutputT](
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=getattr(response, "service_tier", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
        )
        usage_data = create_usage_data(response.usage)
        return self._build_astral_structured_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            cost=None,
        )


# -------------------------------------------------------------------------------- #
# Anthropic Adapter
# -------------------------------------------------------------------------------- #

class AnthropicAdapter(
    BaseProviderAdapter[
        Literal["anthropic"],
        AnthropicRequestChat,
        AnthropicRequestStructured,
        AnthropicRequestEmbedding,
    ]
):
    """
    Adapter for Anthropic-specific request and response formats.
    
    Handles converting between Astral AI's standardized formats and
    Anthropic's API-specific formats for requests and responses.
    """
    
    def __init__(self):
        """Initialize the Anthropic adapter"""
        super().__init__("anthropic")

    # -------------------------------------------------------------------------
    # Astral -> Anthropic
    # -------------------------------------------------------------------------
    def to_chat_request(self, request: AstralCompletionRequest) -> AnthropicRequestChat:
        """
        Convert an Astral completion request to Anthropic's chat request format.
        
        Args:
            request: The Astral completion request
            
        Returns:
            Anthropic-compatible chat request
        """
        raise NotImplementedError("Anthropic chat requests not yet implemented.")

    def to_structured_request(
        self, request: AstralStructuredCompletionRequest
    ) -> AnthropicRequestStructured:
        """
        Convert an Astral structured request to Anthropic's structured request format.
        
        Args:
            request: The Astral structured completion request
            
        Returns:
            Anthropic-compatible structured request
        """
        raise NotImplementedError("Anthropic structured requests not yet implemented.")

    def to_embedding_request(
        self, request: AstralEmbeddingRequest
    ) -> AnthropicRequestEmbedding:
        """
        Convert an Astral embedding request to Anthropic's embedding request format.
        
        Args:
            request: The Astral embedding request
            
        Returns:
            Anthropic-compatible embedding request
        """
        raise NotImplementedError("Anthropic embedding requests not yet implemented.")

    # -------------------------------------------------------------------------
    # Anthropic -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: Union[AnthropicChatResponseType, AnthropicStructuredResponseType],
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Convert an Anthropic response to an Astral response.
        
        Args:
            response: The Anthropic response
            response_format: Optional type for structured output parsing
            
        Returns:
            An Astral response (either chat or structured)
        """
        if response_format is None:
            return self._from_chat_response(cast(AnthropicChatResponseType, response))
        else:
            return self._from_structured_response(
                cast(AnthropicStructuredResponseType, response),
                response_format
            )

    # -------------------------------------------------------------------------
    # Response Converters
    # -------------------------------------------------------------------------
    def _from_chat_response(
        self,
        response: AnthropicChatResponseType
    ) -> AstralChatResponse:
        """
        Convert an Anthropic chat response to an Astral chat response.
        
        Args:
            response: The Anthropic chat response
            
        Returns:
            Standardized Astral chat response
        """
        raise NotImplementedError("Anthropic chat responses not yet implemented.")

    def _from_structured_response(
        self,
        response: AnthropicStructuredResponseType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Convert an Anthropic structured response to an Astral structured response.
        
        Args:
            response: The Anthropic structured response
            response_model: Type for structured output parsing
            
        Returns:
            Standardized Astral structured response
        """
        raise NotImplementedError("Anthropic structured responses not yet implemented.")


# -------------------------------------------------------------------------------- #
# Main Provider Adapter (the "umbrella" class)
# -------------------------------------------------------------------------------- #

class ProviderAdapter(
    Generic[_ModelProviderT, TChatReq, TStructReq, TEmbedReq]
):
    """
    A typed adapter interface that delegates to the appropriate provider-specific adapter.
    
    Provides a unified interface for handling different request types while maintaining
    proper type information through the use of overloads. Each request type (chat,
    structured, embedding) is dispatched to the appropriate adapter method.
    
    Type Parameters:
        _ModelProviderT: The model provider type (e.g., "openai", "anthropic")
        TChatReq: The provider's chat request type
        TStructReq: The provider's structured request type
        TEmbedReq: The provider's embedding request type
    """

    def __init__(
        self,
        provider_name: _ModelProviderT,
        adapter: BaseProviderAdapter[_ModelProviderT, TChatReq, TStructReq, TEmbedReq]
    ):
        """
        Initialize the provider adapter.
        
        Args:
            provider_name: The name of the provider
            adapter: The provider-specific adapter to delegate to
        """
        self.provider_name = provider_name
        self._adapter = adapter

    # -------------------------------------------------------------------------
    # Astral -> Provider (with Overloads)
    # -------------------------------------------------------------------------
    @overload
    def to_provider_request(
        self, request: AstralCompletionRequest
    ) -> TChatReq:
        ...

    @overload
    def to_provider_request(
        self, request: AstralStructuredCompletionRequest
    ) -> TStructReq:
        ...

    @overload
    def to_provider_request(
        self, request: AstralEmbeddingRequest
    ) -> TEmbedReq:
        ...

    def to_provider_request(
        self,
        request: Union[
            AstralCompletionRequest,
            AstralStructuredCompletionRequest,
            AstralEmbeddingRequest,
        ]
    ) -> Union[TChatReq, TStructReq, TEmbedReq]:
        """
        Convert an Astral request to the appropriate provider-specific request.
        
        Dispatches to the correct adapter method based on the request type.
        
        Args:
            request: The Astral request to convert
            
        Returns:
            Provider-specific request object
        """
        if isinstance(request, AstralCompletionRequest):
            return self._adapter.to_chat_request(request)
        elif isinstance(request, AstralStructuredCompletionRequest):
            return self._adapter.to_structured_request(request)
        else:
            return self._adapter.to_embedding_request(request)

    # -------------------------------------------------------------------------
    # Provider -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType,
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Convert a provider-specific response to an Astral response.
        
        Args:
            response: The provider response
            response_format: Optional type for structured output parsing
            
        Returns:
            An Astral response object (either chat or structured)
        """
        return self._adapter.to_astral_completion_response(
            response, response_format
        )


# -------------------------------------------------------------------------------- #
# Factory Function: create_adapter
# -------------------------------------------------------------------------------- #

# Overloads for each recognized provider
@overload
def create_adapter(
    provider: Literal["openai"]
) -> ProviderAdapter[
    Literal["openai"],
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding
]:
    ...


@overload
def create_adapter(
    provider: Literal["deepseek"]
) -> ProviderAdapter[
    Literal["deepseek"],
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
    AnthropicRequestEmbedding
]:
    ...


@overload
def create_adapter(
    provider: Literal["anthropic"]
) -> ProviderAdapter[
    Literal["anthropic"],
    AnthropicRequestChat,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding
]:
    ...


def create_adapter(
    provider: _ModelProviderT
) -> ProviderAdapter[_ModelProviderT, Any, Any, Any]:
    """
    Creates a typed ProviderAdapter for the given provider string.
    
    The overloads ensure that calling create_adapter with a specific provider
    returns an appropriately typed adapter for that provider.
    
    Args:
        provider: The name of the provider to create an adapter for
        
    Returns:
        A properly typed ProviderAdapter for the specified provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        return ProviderAdapter("openai", OpenAIAdapter())
    elif provider == "anthropic":
        return ProviderAdapter("anthropic", AnthropicAdapter())
    elif provider == "deepseek":
        return ProviderAdapter("deepseek", DeepSeekAdapter())
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# # -------------------------------------------------------------------------------- #
# # Example Usage
# # -------------------------------------------------------------------------------- #

# # Create an adapter for OpenAI
# openai_adapter = create_adapter("openai")

# deepseek_adapter = create_adapter("deepseek")

# anthropic_adapter = create_adapter("anthropic")


# chat_request = AstralCompletionRequest(
#     model="gpt-4o",
#     messages=[
#         Message(role="user", content="Hello, world!")
#     ]
# )

# from pydantic import BaseModel

# class MyStruct(BaseModel):
#     name: str
#     age: int

# structured_request = AstralStructuredCompletionRequest(
#     model="gpt-4o",
#     messages=[
#         Message(role="user", content="Hello, world!")
#     ],
#     response_format=MyStruct
# )

# embedding_request = AstralEmbeddingRequest(
#     model="text-embedding-3-small",
#     input="Hello, world!"
# )


# # Convert the requests to provider-specific requests
# openai_chat_request = openai_adapter.to_provider_request(chat_request)
# deepseek_chat_request = deepseek_adapter.to_provider_request(chat_request)
# anthropic_chat_request = anthropic_adapter.to_provider_request(chat_request)

# openai_structured_request = openai_adapter.to_provider_request(structured_request)
# deepseek_structured_request = deepseek_adapter.to_provider_request(structured_request)
# anthropic_structured_request = anthropic_adapter.to_provider_request(structured_request)

# openai_embedding_request = openai_adapter.to_provider_request(embedding_request)
# deepseek_embedding_request = deepseek_adapter.to_provider_request(embedding_request)
# anthropic_embedding_request = anthropic_adapter.to_provider_request(embedding_request)






