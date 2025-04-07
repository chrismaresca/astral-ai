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
    AstralCompletionResponse,
    AstralStructuredCompletionResponse,
)
from astral_ai._types._response.resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
)

# Base adapter imports
from astral_ai.providers._base_adapters import (
    BaseProviderAdapter)

# Message Mapper
from astral_ai.providers.openai._mapper import to_openai_messages
from astral_ai.providers.deepseek._mapper import to_deepseek_max_tokens

# DeepSeek-specific imports
from astral_ai.providers.deepseek._types._request import (
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
    DeepSeekRequestEmbedding,
)
from astral_ai.providers.deepseek._types._response import (
    DeepSeekChatResponseType,
    DeepSeekStructuredResponseType,
    DeepSeekCompletionUsageType,
)

# Usage
from astral_ai.providers._usage import ChatUsage

# Constants
from astral_ai.constants._provider_mapping_keys import PROVIDER_REQUEST_MAPPING_KEYS

# Astral AI Utilities
from astral_ai.utilities import apply_key_mapping

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
        
        # Initialize the provider request mapping keys
        self._provider_request_mapping_keys = PROVIDER_REQUEST_MAPPING_KEYS["deepseek"]

    # -------------------------------------------------------------------------
    # Astral -> DeepSeek
    # -------------------------------------------------------------------------
    def _prepare_request_data(self, request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest]) -> Dict[str, Any]:
        """
        Prepare the request data for the DeepSeek API.
        
        This method transforms Astral's standardized request format into DeepSeek's 
        expected format by:
        1. Retrieving the appropriate key mapping for the request resource type
        2. Converting the request to a raw dictionary without Astral-specific parameters
        3. Transforming Astral message format to DeepSeek's message format
        4. Removing the system_message after it's been incorporated into messages
        5. Applying key mapping to rename fields and filter out unsupported parameters
        
        Args:
            request: The Astral completion or structured completion request
            
        Returns:
            A dictionary containing the DeepSeek-compatible request parameters
        """
        # Get the key map for the request resource type
        key_map = self._provider_request_mapping_keys[request.resource_type]
        
        # Get the raw request data
        raw = request.model_dump_without_astral_params()
        
        # Convert the messages to DeepSeek messages (using OpenAI mapper since format is the same)
        # TODO: Consider creating a DeepSeek-specific mapper if message format diverges from OpenAI
        raw["messages"] = to_openai_messages(
            model_name=request.model,
            messages=request.messages,
            system_message=getattr(request, "system_message", None),
        )
        
        # Remove the system_message after it's been incorporated into messages
        raw.pop("system_message", None)

        # Set max tokens to the model's max tokens
        raw["max_tokens"] = to_deepseek_max_tokens(request.model, request.max_tokens)
        
        # Apply the key mapping to rename fields and filter out unsupported parameters
        filtered = apply_key_mapping(raw, key_map)
        return filtered

    def _build_astral_completion_usage(
        self,
        usage: Optional[DeepSeekCompletionUsageType]
    ) -> ChatUsage:
        """
        Convert DeepSeek usage data to Astral AI usage data.
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
        filtered = self._prepare_request_data(request)
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
        filtered = self._prepare_request_data(request)
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
        # TODO: Implement embedding request conversion when DeepSeek embedding API is available
        # This would use a similar pattern with key_mapping for embedding resource type
        raise NotImplementedError("DeepSeek embedding requests not implemented.")

    # -------------------------------------------------------------------------
    # DeepSeek -> Astral
    # -------------------------------------------------------------------------
    def to_astral_completion_response(
        self,
        response: Union[DeepSeekChatResponseType, DeepSeekStructuredResponseType],
        response_format: Optional[Type[Any]] = None
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[Any]]:
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
    ) -> AstralCompletionResponse:
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
        # Avoid circular import
        from astral_ai.providers._usage import create_usage_data

        usage_data = create_usage_data(response.usage)
        return self._build_astral_chat_response(
            model=response.model,
            provider_response=provider_resp,
            usage=usage_data,
            content=response.choices[0].message.content,
            cost=None,
        )

    # @overload
    # def _from_structured_response(
    #     self,
    #     response: DeepSeekStructuredResponseType,
    #     response_model: Type[StructuredOutputResponse]
    # ) -> AstralStructuredCompletionResponse[StructuredOutputResponse]:
    #     """Overload for BaseModel responses"""
    #     ...

    # @overload
    # def _from_structured_response(
    #     self,
    #     response: DeepSeekStructuredResponseType,
    #     response_model: Type[Dict[str, Any]]
    # ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
    #     """Overload for Dict responses (typically with tools)"""
    #     ...

    def _from_structured_response(
        self,
        response: DeepSeekStructuredResponseType,
        response_model: Union[Type[StructuredOutputResponse], Type[Any]]
    ) -> Union[AstralStructuredCompletionResponse[StructuredOutputResponse], AstralStructuredCompletionResponse[Any]]:
        """
        Convert a DeepSeek structured response to an Astral structured response.

        Args:
            response: The DeepSeek structured response
            response_model: Type for structured output parsing

        Returns:
            Standardized Astral structured response
        """
        raise NotImplementedError("DeepSeek does not support structured responses currently.")

