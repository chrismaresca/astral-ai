# -------------------------------------------------------------------------------- #
# OpenAI Provider Adapter
# -------------------------------------------------------------------------------- #
# This module contains the adapter implementation for the OpenAI provider.
# It converts between Astral AI formats and OpenAI-specific formats.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Literal, Optional, Type, Union, cast, Dict, Any

# Astral AI imports
from astral_ai._types import NOT_GIVEN
from astral_ai._types._request._request import (
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,
    AstralEmbeddingRequest,
)
from astral_ai._types._response._response import (
    AstralChatResponse,
    AstralStructuredResponse,
)
from astral_ai._types._response.resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
)

# Base adapter imports
from astral_ai.providers._base_adapters import (
    BaseProviderAdapter,
)

# OpenAI-specific imports
from astral_ai.providers.openai._types._request import (
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding,
)
from astral_ai.providers.openai._types._response import (
    OpenAIChatResponseType,
    OpenAIStructuredResponseType,
)

# Utility functions
from astral_ai.providers._usage import create_usage_data

# -------------------------------------------------------------------------------- #
# OpenAI Adapter Implementation
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
        data["messages"] = request.messages
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
        data["messages"] = request.messages
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
        response_format: Optional[Type[Any]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[Any]]:
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
        response_model: Type[Any]
    ) -> AstralStructuredResponse[Any]:
        """
        Convert an OpenAI structured response to an Astral structured response.

        Args:
            response: The OpenAI structured response
            response_model: Type for structured output parsing

        Returns:
            Standardized Astral structured response
        """
        provider_resp = StructuredOutputCompletionResponse[Any](
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
