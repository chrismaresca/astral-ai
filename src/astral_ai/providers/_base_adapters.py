# -------------------------------------------------------------------------------- #
# Base Provider Adapters
# -------------------------------------------------------------------------------- #
# This module contains:
#   - Base adapter class with common functionality for all providers
#   - Shared utility functions for message and usage conversions
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import (
    Literal,
    cast,
    Generic,
    Type,
    Optional,
    TypeVar,
    Union,
    List,
    Any,
    Dict,
    overload,

)

from abc import ABC, abstractmethod

# Astral AI imports
from astral_ai.constants._models import ModelProvider
from astral_ai._types._response._usage import ChatUsage, ChatCost
from astral_ai._types._request._request import (
    AstralCompletionRequest,
    AstralEmbeddingRequest,
    AstralStructuredCompletionRequest,
    AstralCompletionRequestWithJSONResponse,
)
from astral_ai._types._response._response import (
    AstralCompletionResponse,
    AstralStructuredCompletionResponse,
)
from astral_ai.messages._models import Message
from astral_ai.providers._generics import (
    ProviderChatRequestType,
    ProviderStructuredRequestType,
    ProviderEmbeddingRequestType,
    ProviderCompletionResponseType,
    ProviderUsageType,
    StructuredOutputT,
)
from astral_ai._types._response.resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
)


# -------------------------------------------------------------------------------- #
# Base Provider Adapter (Generic)
# -------------------------------------------------------------------------------- #

_ModelProviderT = TypeVar("_ModelProviderT", bound=ModelProvider)

TChatReq = TypeVar("TChatReq", bound=ProviderChatRequestType)
TStructReq = TypeVar("TStructReq", bound=ProviderStructuredRequestType)
TEmbedReq = TypeVar("TEmbedReq", bound=ProviderEmbeddingRequestType)


class BaseProviderAdapter(ABC,
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

    # def to_completion_request_with_json_response(self, request: AstralCompletionRequestWithJSONResponse) -> TStructReq:
    #     """
    #     Convert an AstralCompletionRequestWithJSONResponse into a provider-specific structured request.

    #     Args:
    #         request: The Astral completion request with JSON response
    #     """
    #     raise NotImplementedError()

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
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[StructuredOutputT]]:
        """
        Convert a provider-specific response to an AstralCompletionResponse or AstralStructuredCompletionResponse.

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

    @abstractmethod
    def _build_astral_completion_usage(
        self,
        usage: Optional[Any]  # This will be properly typed in each implementation
    ) -> Optional[ChatUsage]:
        """
        Convert provider-specific usage data to standardized Astral AI usage data.

        This abstract method must be implemented by each provider adapter to handle
        the specific usage data format returned by that provider.

        Args:
            usage: The provider-specific usage data

        Returns:
            Standardized usage data or None if usage data is not available
        """
        pass

    def _build_astral_chat_response(
        self,
        model: str,
        provider_response: ChatCompletionResponse,
        usage: ChatUsage,
        content: Optional[str] = None,

        cost: Optional[ChatCost] = None
    ) -> AstralCompletionResponse:
        """
        Build an AstralCompletionResponse from a ChatCompletionResponse

        Args:
            model: The model name
            provider_response: Provider's chat completion response
            usage: Token usage information
            cost: Optional cost information

        Returns:
            Standardized Astral chat response
        """
        return AstralCompletionResponse(
            model=model,
            response=provider_response,
            usage=usage,
            content=content,
            cost=cost,
        )

    def _build_astral_structured_response(
        self,
        model: str,
        provider_response: StructuredOutputCompletionResponse[StructuredOutputT],
        usage: ChatUsage,
        content: Optional[StructuredOutputT] = None,
        cost: Optional[ChatCost] = None

    ) -> AstralStructuredCompletionResponse[StructuredOutputT]:
        """
        Build an AstralStructuredCompletionResponse from a StructuredOutputCompletionResponse

        Args:
            model: The model name
            provider_response: Provider's structured completion response
            usage: Token usage information
            cost: Optional cost information

        Returns:
            Standardized Astral structured response
        """
        return AstralStructuredCompletionResponse[StructuredOutputT](
            model=model,
            response=provider_response,
            usage=usage,
            content=content,
            cost=cost,
        )


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
        self,
        request: AstralCompletionRequest
    ) -> TChatReq:
        ...

    @overload
    def to_provider_request(
        self,
        request: AstralStructuredCompletionRequest
    ) -> TStructReq:
        ...

    @overload
    def to_provider_request(
        self,
        request: AstralEmbeddingRequest
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

    @overload
    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType
    ) -> AstralCompletionResponse:
        ...

    @overload
    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType,
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> AstralStructuredCompletionResponse[StructuredOutputT]:
        ...

    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType,
        response_format: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[StructuredOutputT]]:
        """
        Convert a provider-specific response to an Astral response.

        Args:
            response: The provider response
            response_format: Optional type for structured output parsing

        Returns:
            An Astral response object (either chat or structured)
        """
        return self._adapter.to_astral_completion_response(
            response,
            response_format
        )
