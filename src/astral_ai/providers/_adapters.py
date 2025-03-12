# Provider Adapters
# -------------------------------------------------------------------------------- #
# This module contains:
#   - The mapping and functions for converting project messages into
#     provider-specific messages.
#   - Overloads for the get_provider_message_converter function.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in imports
from typing import (Literal,
                    cast,
                    Generic,
                    Type,
                    Optional,
                    overload,
                    TypeVar,
                    Union,
                    List,
                    Any,
                    Dict)

# Pydantic imports
from pydantic import BaseModel

# Astral AI Constants
from astral_ai.constants._models import ModelProvider

# Astral AI Types
from astral_ai._types import NOT_GIVEN
from astral_ai._types._request._request import AstralCompletionRequest, AstralEmbeddingRequest
from astral_ai._types._response._response import AstralChatResponse, AstralStructuredResponse

# Provider Generic Types
from astral_ai.providers._generics import (
    ProviderRequestChatType,
    ProviderRequestStructuredType,
    ProviderResponseChatType,
    ProviderResponseStructuredType,
    ProviderCompletionResponseType,
    ProviderChatRequestType,
    ProviderStructuredRequestType,
    ProviderEmbeddingRequestType,
    StructuredOutputT,
)

# OpenAI Types
from astral_ai.providers.openai._types._request import OpenAIRequestChat, OpenAIRequestEmbedding
from astral_ai.providers.openai._types._response import OpenAIChatResponseType, OpenAIStructuredResponseType, OpenAICompletionUsageType
# Anthropic Types
from astral_ai.providers.anthropic._types._request import AnthropicRequestChat, AnthropicRequestEmbedding
from astral_ai.providers.anthropic._types._response import AnthropicChatResponseType, AnthropicStructuredResponseType

# Astral AI Usage Types
from astral_ai._types._response._usage import ChatUsage, ChatCost

# Message Types
from astral_ai.messaging._models import Message

# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Message Conversion
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #

from astral_ai.providers.openai._types._message import OpenAIMessageType


@overload
def convert_messages_to_provider_format(
    messages: list[Message],
    provider: Literal["openai"]
) -> List[OpenAIMessageType]:
    ...


@overload
def convert_messages_to_provider_format(
    messages: list[Message],
    provider: Literal["anthropic"]
    # TODO: add anthropic message type
) -> str:
    ...


def convert_messages_to_provider_format(
    messages: list[Message],
    provider: Literal["openai", "anthropic"]
    # TODO: replace with generic provider message type
) -> Union[List[OpenAIMessageType], str]:
    """
    Convert a list of AstralMessages to the provider-specific format.
    Overloads ensure we return the correct typed structure for each provider.
    """
    pass


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Usage Data Conversion
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


def create_usage_data(usage: Optional[OpenAICompletionUsageType]) -> ChatUsage:
    """Create usage data from an OpenAI ChatCompletion usage."""
    if not usage:
        return ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    # Create usage data matching available fields in OpenAI's CompletionUsage
    return ChatUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        # Get nested audio_tokens values if they exist
        audio_tokens=(
            usage.prompt_tokens_details.audio_tokens if
            usage.prompt_tokens_details and
            usage.prompt_tokens_details.audio_tokens is not None
            else None
        ),
        # Get cached_tokens if it exists
        cached_tokens=(
            usage.prompt_tokens_details.cached_tokens if
            usage.prompt_tokens_details and
            usage.prompt_tokens_details.cached_tokens is not None
            else None
        ),
        # Get prediction tokens if they exist
        accepted_prediction_tokens=(
            usage.completion_tokens_details.accepted_prediction_tokens if
            usage.completion_tokens_details and
            usage.completion_tokens_details.accepted_prediction_tokens is not None
            else None
        ),
        rejected_prediction_tokens=(
            usage.completion_tokens_details.rejected_prediction_tokens if
            usage.completion_tokens_details and
            usage.completion_tokens_details.rejected_prediction_tokens is not None
            else None
        ),
        reasoning_tokens=(
            usage.completion_tokens_details.reasoning_tokens if
            usage.completion_tokens_details and
            usage.completion_tokens_details.reasoning_tokens is not None
            else None
        )
    )


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Provider Adapter Generic Types
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #

_ModelProviderT = TypeVar("_ModelProviderT", bound=ModelProvider)

# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Provider Adapter
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #


class ProviderAdapter(Generic[_ModelProviderT]):
    """
    A single adapter that is type-safe for a specific provider. 
    `provider_name` is a literal that Mypy tracks: "openai" or "anthropic".
    """

    def __init__(self, provider_name: _ModelProviderT):
        self.provider_name: _ModelProviderT = provider_name

    # --------------------------------------------------------------------------
    # Overloads for COMPLETION requests
    # --------------------------------------------------------------------------
    @overload
    def to_provider_request(
        self: "ProviderAdapter[Literal['openai']]",
        request: AstralCompletionRequest
    ) -> OpenAIRequestChat:
        ...

    @overload
    def to_provider_request(
        self: "ProviderAdapter[Literal['anthropic']]",
        request: AstralCompletionRequest
    ) -> AnthropicRequestChat:
        ...

    # --------------------------------------------------------------------------
    # Overloads for EMBEDDING requests
    # --------------------------------------------------------------------------
    @overload
    def to_provider_request(
        self: "ProviderAdapter[Literal['openai']]",
        request: AstralEmbeddingRequest
    ) -> OpenAIRequestEmbedding:
        ...

    @overload
    def to_provider_request(
        self: "ProviderAdapter[Literal['anthropic']]",
        request: AstralEmbeddingRequest
    ) -> AnthropicRequestEmbedding:
        ...

    # --------------------------------------------------------------------------
    # Implementation
    # --------------------------------------------------------------------------
    def to_provider_request(
        self,
        request: Union[AstralCompletionRequest, AstralEmbeddingRequest],
    ) -> Union[ProviderChatRequestType, ProviderEmbeddingRequestType]:
        """
        Single implementation that checks `self.provider_name`
        at runtime and returns the correct typed-dict.
        However, from a caller's perspective, Mypy sees only the 
        correct overload (based on self's type + request type).
        """
        if self.provider_name == "openai":
            if isinstance(request, AstralCompletionRequest):
                # Build an OpenAIRequestChat typed-dict
                return self._to_openai_chat_request(request)
            else:
                # Build an OpenAIRequestEmbedding typed-dict
                return self._to_openai_embedding_request(request)
        else:
            if isinstance(request, AstralCompletionRequest):
                # Build an AnthropicRequestChat typed-dict
                return self._to_anthropic_chat_request(request)
            else:
                # Build an AnthropicRequestEmbedding typed-dict
                return self._to_anthropic_embedding_request(request)

    # --------------------------------------------------------------------------
    # TODO: Implement the to_*_request methods
    # --------------------------------------------------------------------------

    def _to_openai_chat_request(self, request: AstralCompletionRequest) -> Dict[str, Any]:
        """
        Build an OpenAIRequestChat typed-dict from an AstralCompletionRequest.
        """
        request_data = request.model_dump_without_astral_params()

        # convert messages to provider format
        # TODO: Implement this
        # messages = convert_messages_to_provider_format(request_data["messages"], provider="openai")
        request_data["messages"] = request_data["messages"]

        # Filter out any None values
        request_with_defaults = {k: v for k, v in request_data.items() if v is not NOT_GIVEN}

        return cast(OpenAIRequestChat, request_with_defaults)

    def _to_openai_embedding_request(self, request: AstralEmbeddingRequest) -> Dict[str, Any]:
        """
        Build an OpenAIRequestEmbedding typed-dict from an AstralEmbeddingRequest.
        """
        raise NotImplementedError("OpenAI embedding requests are not yet implemented.")

    def _to_anthropic_chat_request(self, request: AstralCompletionRequest) -> Dict[str, Any]:
        """
        Build an AnthropicRequestChat typed-dict from an AstralCompletionRequest.
        """
        raise NotImplementedError("Anthropic chat requests are not yet implemented.")

    def _to_anthropic_embedding_request(self, request: AstralEmbeddingRequest) -> Dict[str, Any]:
        """
        Build an AnthropicRequestEmbedding typed-dict from an AstralEmbeddingRequest.
        """
        raise NotImplementedError("Anthropic embedding requests are not yet implemented.")

    # --------------------------------------------------------------------------
    # COMPLETION RESPONSE Overloads
    # --------------------------------------------------------------------------

    @overload
    def to_astral_completion_response(
        self: "ProviderAdapter[Literal['openai']]",
        response: ProviderResponseChatType
    ) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(
        self: "ProviderAdapter[Literal['openai']]",
        response: ProviderResponseStructuredType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        ...

    @overload
    def to_astral_completion_response(
        self: "ProviderAdapter[Literal['anthropic']]",
        response: ProviderResponseChatType
    ) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(
        self: "ProviderAdapter[Literal['anthropic']]",
        response: ProviderResponseStructuredType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        ...

    def to_astral_completion_response(
        self,
        response: ProviderCompletionResponseType,
        response_model: Optional[Type[StructuredOutputT]] = None
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputT]]:
        """
        Single runtime implementation that checks provider & response_model.
        """
        if self.provider_name == "openai":
            if response_model is None:
                # parse normal chat response
                return self._from_openai_chat_response(response)
            else:
                # parse structured
                return self._from_openai_structured_response(response, response_model)
        else:
            if response_model is None:
                return self._from_anthropic_chat_response(response)
            else:
                return self._from_anthropic_structured_response(response, response_model)

    # --------------------------------------------------------------------------
    # TODO: Implement the parse methods
    # --------------------------------------------------------------------------

    def _from_openai_chat_response(self, response: OpenAIChatResponseType) -> AstralChatResponse:
        """
        Parse an OpenAIChatResponseType into an AstralChatResponse.
        """
        # Extract the content.
        content = response.choices[0].message.content if response.choices else ""

        # Create usage data
        usage_data = create_usage_data(response.usage)

        # Extract the cost data.
        # TODO: We may not need this here.
        cost: Optional[ChatCost] = None

        # Construct and return the chat response.
        # TODO: Fix provider response
        return AstralChatResponse(
            model_provider="openai",
            provider_response=None,
            model=response.model,
            response=content,
            usage=usage_data,
            cost=cost
        )

    def _from_openai_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Parse an OpenAIStructuredResponseType into an AstralStructuredResponse.
        """
        # Extract the parsed content.
        parsed_content_data = response.choices[0].message.parsed

        # Check if the parsed content is None.
        if parsed_content_data is None:
            raise ValueError("Structured response missing parsed content")

        # Parse the structured output using the provided BaseModel subclass.
        parsed_content: StructuredOutputT = response_model.model_validate(parsed_content_data)

        # Extract the usage data.
        usage_data = create_usage_data(response.usage)
        # TODO: Do I need to extract the cost data?
        from astral_ai._types._response._response import ChatCost
        cost: Optional[ChatCost] = None
        # Construct and return the structured response.
        return AstralStructuredResponse[StructuredOutputT](response=parsed_content, usage=usage_data, cost=cost)

    def _from_anthropic_chat_response(self, response: AnthropicChatResponseType) -> AstralChatResponse:
        # TODO: Implement
        raise NotImplementedError("Anthropic chat responses are not yet implemented.")

    def _from_anthropic_structured_response(
        self,
        response: AnthropicStructuredResponseType,
        model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse[StructuredOutputT]:
        """
        Parse an AnthropicStructuredResponseType into an AstralStructuredResponse.
        """
        # TODO: Implement
        raise NotImplementedError("Anthropic structured responses are not yet implemented.")


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Factory function
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #

def create_adapter(provider: _ModelProviderT) -> ProviderAdapter[_ModelProviderT]:
    """
    Factory function that preserves the literal type. 
    So `create_adapter("openai")` -> `ProviderAdapter[Literal["openai"]]`
    """
    return ProviderAdapter(provider)
