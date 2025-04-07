# -------------------------------------------------------------------------------- #
# Anthropic Provider Adapter
# -------------------------------------------------------------------------------- #
# This module contains the adapter implementation for the Anthropic provider.
# It converts between Astral AI formats and Anthropic-specific formats.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Literal, Optional, Type, Union, cast, Dict, Any, TypeVar
import time

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

# Astral AI Types
from astral_ai._types._response._usage import ChatUsage

# Base adapter imports
from astral_ai.providers._base_adapters import (
    BaseProviderAdapter,
)

# Constants
from astral_ai.constants._provider_mapping_keys import PROVIDER_REQUEST_MAPPING_KEYS

# Astral AI Utilities
from astral_ai.utilities import apply_key_mapping

# Anthropic-specific imports
from ._types import (
    AnthropicRequestChat,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding,
    AnthropicCompletionUsageType,
    AnthropicChatResponseType,
    AnthropicStructuredResponseType,
)

# Utility functions
from astral_ai.providers._usage import create_usage_data

# Anthropic-specific mapper
from astral_ai.providers.anthropic._mapper import (
    to_anthropic_messages,
    to_anthropic_thinking,
    to_anthropic_max_tokens,
    to_anthropic_system_message,
)

# -------------------------------------------------------------------------------- #
# Anthropic Adapter Implementation
# -------------------------------------------------------------------------------- #

T = TypeVar('T')


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

        # Initialize the provider request mapping keys
        self._provider_request_mapping_keys = PROVIDER_REQUEST_MAPPING_KEYS["anthropic"]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _prepare_request_data(
        self,
        request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest]
    ) -> Dict[str, Any]:
        """
        Prepare the request data for the Anthropic API.

        This method transforms Astral's standardized request format into Anthropic's 
        expected format by:
        1. Retrieving the appropriate key mapping for the request resource type
        2. Converting the request to a raw dictionary without Astral-specific parameters
        3. Applying key mapping to rename fields and filter out unsupported parameters

        Args:
            request: The Astral completion or structured completion request

        Returns:
            A dictionary containing the Anthropic-compatible request parameters
        """
        # Get the key map for the request resource type
        key_map = self._provider_request_mapping_keys[request.resource_type]

        # Get the raw request data
        raw = request.model_dump_without_astral_params()

        # Copy messages directly - Anthropic has different message handling than OpenAI
        # Unlike OpenAI, Anthropic handles system messages separately, not as part of messages
        raw["messages"] = to_anthropic_messages(messages=request.messages, system_message=request.system_message)

        # Set max tokens to the model's max tokens
        raw["max_tokens"] = to_anthropic_max_tokens(request.model, request.max_tokens)

        # Add thinking to the request
        raw["thinking"] = to_anthropic_thinking(model=request.model, reasoning_effort=request.reasoning_effort, max_tokens=request.max_tokens)

        raw["system_message"] = to_anthropic_system_message(request.system_message)


        # Apply the key mapping to rename fields and filter out unsupported parameters
        filtered = apply_key_mapping(raw, key_map)
        return filtered

    def _build_astral_completion_usage(
        self,
        usage: Optional[AnthropicCompletionUsageType]
    ) -> ChatUsage:
        """
        Convert Anthropic usage data to Astral AI usage data.
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

        # Build the usage data
        return ChatUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            audio_tokens=0,
            cached_tokens=0,
            accepted_prediction_tokens=0,
            rejected_prediction_tokens=0,
            reasoning_tokens=0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        )

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
        filtered = self._prepare_request_data(request)
        return cast(AnthropicRequestChat, filtered)

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
        # TODO: Implement structured request conversion when Anthropic supports it
        raise NotImplementedError("Anthropic does not support structured requests.")

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
        # TODO: Implement embedding request conversion when Anthropic embedding API is integrated
        raise NotImplementedError("Anthropic embedding requests not yet implemented.")

    # -------------------------------------------------------------------------
    # Anthropic -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: Union[AnthropicChatResponseType, AnthropicStructuredResponseType],
        response_format: Optional[Type[T]] = None
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[T]]:
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
    ) -> AstralCompletionResponse:
        """
        Convert an Anthropic chat response to an Astral chat response.

        Args:
            response: The Anthropic chat response

        Returns:
            Standardized Astral chat response
        """
        # Extract the plain text content from the content blocks
        # TODO: Handle different content block types if Anthropic adds more in the future
        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text

        # Create a choice object similar to OpenAI format
        choice = {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": content_text,
                "role": "assistant",
            }
        }

        # Create the provider response object
        provider_resp = ChatCompletionResponse(
            id=response.id,
            choices=[choice],
            created=int(time.time()),  # Anthropic doesn't provide a timestamp, use current time
            model=response.model,
            object="chat.completion",
            service_tier=None,  # Anthropic doesn't provide this
            system_fingerprint=None,  # Anthropic doesn't provide this
        )

        usage_data = self._build_astral_completion_usage(response.usage)
        return self._build_astral_chat_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            content=content_text,
            cost=None,
        )

    def _from_structured_response(
        self,
        response: AnthropicStructuredResponseType,
        response_model: Type[T]
    ) -> AstralStructuredCompletionResponse[T]:
        """
        Convert an Anthropic structured response to an Astral structured response.

        Args:
            response: The Anthropic structured response
            response_model: Type for structured output parsing

        Returns:
            Standardized Astral structured response
        """
        # TODO: Implement when Anthropic adds structured response support
        raise NotImplementedError("Anthropic does not support structured responses.")
