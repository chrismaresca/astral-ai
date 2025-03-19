# -------------------------------------------------------------------------------- #
# DeepSeek Provider Adapter
# -------------------------------------------------------------------------------- #
# This module contains the adapter implementation for the DeepSeek provider.
# It converts between Astral AI formats and DeepSeek-specific formats.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Literal, Optional, Type, Union, cast, Dict, Any, TypeVar

# Pydantic imports
from pydantic import BaseModel

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
    BaseProviderAdapter)

# DeepSeek-specific imports
from astral_ai.providers.deepseek._types._request import (
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
    DeepSeekRequestEmbedding,
)
from astral_ai.providers.deepseek._types._response import (
    DeepSeekChatResponseType,
    DeepSeekStructuredResponseType,
)

# Utility functions

# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #


StructuredOutputResponse = TypeVar("StructuredOutputResponse", bound=BaseModel)



# -------------------------------------------------------------------------------- #
# DeepSeek Adapter Implementation
# -------------------------------------------------------------------------------- #

class DeepSeekAdapter(
    BaseProviderAdapter[
        Literal["deepseek"],
        DeepSeekRequestChat,
        DeepSeekRequestStructured,
        DeepSeekRequestEmbedding,
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
        # TODO: Convert messages to provider format
        data["messages"] = request.messages
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
        # TODO: Convert messages to provider format
        data["messages"] = request.messages
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
        response_format: Optional[Type[Any]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[Any]]:
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
        from astral_ai.providers._usage import create_usage_data

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
        response_model: Type[StructuredOutputResponse]
    ) -> AstralStructuredResponse[Any]:
        """
        Convert a DeepSeek structured response to an Astral structured response.
        
        Args:
            response: The DeepSeek structured response
            response_model: Type for structured output parsing
            
        Returns:
            Standardized Astral structured response
        """


        provider_resp = StructuredOutputCompletionResponse[StructuredOutputResponse](
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=getattr(response, "service_tier", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
        )

        # Avoid circular import
        from astral_ai.providers._usage import create_usage_data

        usage_data = create_usage_data(response.usage)
        return self._build_astral_structured_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            cost=None,
        ) 