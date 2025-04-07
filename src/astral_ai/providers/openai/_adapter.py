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
from typing import Literal, Optional, Type, Union, cast, Dict, Any, List, TypeVar, overload


# Astral AI Message Mapper
from astral_ai.providers.openai._mapper import to_openai_messages

# Astral AI imports
from astral_ai._types import NOT_GIVEN
from astral_ai._types._request._request import (
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,
    AstralEmbeddingRequest,
)
from astral_ai._types._response._response import (
    AstralCompletionResponse,
    AstralStructuredCompletionResponse,
)
from astral_ai._types._response.resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
)

# Base adapter imports
from astral_ai.providers._base_adapters import (
    BaseProviderAdapter,
)

# Constants
from astral_ai.constants._provider_mapping_keys import PROVIDER_REQUEST_MAPPING_KEYS

# OpenAI-specific imports
from ._types import (
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding,
    OpenAIChatResponseType,
    OpenAIStructuredResponseType,
    OpenAIMessageType,
    OpenAICompletionUsageType,
)

# Astral AI Types
from astral_ai._types._response._usage import ChatUsage, ChatCost

# Astral AI Utilities
from astral_ai.utilities import apply_key_mapping

# Usage
from astral_ai.providers._usage import create_usage_data

# Logger
from astral_ai.logger import logger

# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #
from pydantic import BaseModel

# Type variables for overloading
StructuredOutputResponse = TypeVar('StructuredOutputResponse', bound=BaseModel)
NonStructuredOutputResponse = TypeVar('NonStructuredOutputResponse')

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

        # Initialize the provider request mapping keys
        self._provider_request_mapping_keys = PROVIDER_REQUEST_MAPPING_KEYS["openai"]

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    def _prepare_request_data(
        self,
        request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest],
    ) -> Dict[str, Any]:
        """
        Prepare the request data for the OpenAI API.

        This method transforms Astral's standardized request format into OpenAI's 
        expected format by:
        1. Retrieving the appropriate key mapping for the request resource type
        2. Converting the request to a raw dictionary without Astral-specific parameters
        3. Transforming Astral message format to OpenAI's message format
        4. Removing the system_message after it's been incorporated into messages
        5. Applying key mapping to rename fields and filter out unsupported parameters

        Args:
            request: The Astral completion or structured completion request

        Returns:
            A dictionary containing the OpenAI-compatible request parameters
        """

        # Get the key map for the request resource type
        key_map = self._provider_request_mapping_keys[request.resource_type]

        # Get the raw request data
        raw = request.model_dump_without_astral_params()

        # Convert the messages to OpenAI messages
        raw["messages"] = to_openai_messages(
            model_name=request.model,
            messages=request.messages,
            system_message=getattr(request, "system_message", None),
        )

        # Remove the system_message after it's been incorporated into messages
        raw.pop("system_message", None)

        # Apply the key mapping to rename fields and filter out unsupported parameters
        filtered = apply_key_mapping(raw, key_map)
        return filtered

    def _build_astral_completion_usage(
        self,
        usage: Optional[OpenAICompletionUsageType]
    ) -> ChatUsage:
        """
        Convert OpenAI usage data to Astral AI usage data.
        """
        if usage is None:
            return ChatUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                audio_tokens=0,
                cached_tokens=0,
                accepted_prediction_tokens=0,
                rejected_prediction_tokens=0,
                reasoning_tokens=0,
            )
        
        # Extract nested token details
        audio_tokens = None
        cached_tokens = None
        accepted_prediction_tokens = None
        rejected_prediction_tokens = None
        reasoning_tokens = None
        
        # Extract prompt token details if available
        if usage.prompt_tokens_details is not None:
            audio_tokens = usage.prompt_tokens_details.audio_tokens
            cached_tokens = usage.prompt_tokens_details.cached_tokens
            
        # Extract completion token details if available
        if usage.completion_tokens_details is not None:
            accepted_prediction_tokens = usage.completion_tokens_details.accepted_prediction_tokens
            rejected_prediction_tokens = usage.completion_tokens_details.rejected_prediction_tokens
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
        
        # Build the usage data
        return ChatUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            audio_tokens=audio_tokens,
            cached_tokens=cached_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
            reasoning_tokens=reasoning_tokens,
        )

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
        filtered = self._prepare_request_data(request)
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
        filtered = self._prepare_request_data(request)


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
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[Any]]:
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
    ) -> AstralCompletionResponse:
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
        usage_data = self._build_astral_completion_usage(response.usage)
        return self._build_astral_chat_response(
            model=response.model,
            provider_response=provider_resp,
            content=response.choices[0].message.content,
            usage=usage_data,
            cost=None,
        )

    @overload
    def _from_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[StructuredOutputResponse]
    ) -> AstralStructuredCompletionResponse[StructuredOutputResponse]:
        """Overload for BaseModel responses"""
        ...

    @overload
    def _from_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[Dict[str, Any]]
    ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
        """Overload for Dict responses (typically with tools)"""
        ...

    def _from_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Union[Type[StructuredOutputResponse], Type[Any]]
    ) -> Union[AstralStructuredCompletionResponse[StructuredOutputResponse], AstralStructuredCompletionResponse[Any]]:
        """
        Convert an OpenAI structured response to an Astral structured response.

        Args:
            response: The OpenAI structured response
            response_model: Type for structured output parsing

        Returns:
            Standardized Astral structured response
        """
        # Check if response uses tools (which might be incompatible with BaseModel)
        has_tools = False
        if response.choices and hasattr(response.choices[0], 'message'):
            tool_calls = getattr(response.choices[0].message, 'tool_calls', [])
            if tool_calls is not None:
                from collections.abc import Iterable
                has_tools = len(tool_calls) > 0 if isinstance(tool_calls, Iterable) else False
        
        # Check if the response_model is a BaseModel subclass
        is_base_model = hasattr(response_model, '__bases__') and any('BaseModel' in str(base) for base in response_model.__bases__)
        
        # Extract the content - this is the actual structured output
        content = response.choices[0].message.parsed
        
        # Choose appropriate type for structured output
        if is_base_model:
            # When using a BaseModel subclass
            response_type = cast(Type[StructuredOutputResponse], response_model)
        elif has_tools:
            # When using tools, we use a more flexible type
            logger.debug("Using Dict structured response for tools")
            response_type = Dict[str, Any]
        else:
            # Fallback to Any type
            logger.debug("Using generic structured response type")
            response_type = Any
        
        # Create provider_resp with the proper type
        provider_resp = StructuredOutputCompletionResponse[response_type](
            id=response.id,
            choices=response.choices,
            created=response.created,
            model=response.model,
            object=response.object,
            service_tier=response.service_tier,
            system_fingerprint=response.system_fingerprint,
        )
        
        # Create usage data
        usage_data = self._build_astral_completion_usage(response.usage)

        # Build and return the structured response with the proper type
        result = self._build_astral_structured_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            content=content,
            cost=None,
        )
        
        # Return the concrete response with the appropriate model type
        return AstralStructuredCompletionResponse[response_type](
            model=result.model,
            response=result.response,
            usage=result.usage,
            cost=result.cost,
            latency_ms=result.latency_ms,
            content=content
        )
