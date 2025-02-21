from __future__ import annotations
from typing import overload, Union, Optional, TypeVar, Type, cast
from pydantic import BaseModel

# Astral Base Adapters
from astral_ai.providers._base_adapters import BaseCompletionAdapter

# Astral AI Types
from astral_ai._types import (
    AstralCompletionRequest,
    AstralChatResponse,
    AstralStructuredResponse,
)

# OpenAI Types
from ._types import (
    OpenAIRequestType,
    OpenAIResponseType,
    OpenAIChatResponseType,
    OpenAIStructuredResponseType,
)

# Astral Usage Types
from astral_ai._types._response._usage import ChatUsage, ChatCost

# Generic type variable for structured output (parsed content must be a BaseModel)
_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# OpenAICompletionAdapter Implementation
# -------------------------------------------------------------------------------- #

class OpenAICompletionAdapter(BaseCompletionAdapter):
    """
    Adapter for the OpenAI provider.
    """

    @overload
    def to_astral_completion_response(
        self,
        response: OpenAIChatResponseType,
        response_model: None = None,
    ) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[_StructuredOutputT],
    ) -> AstralStructuredResponse[_StructuredOutputT]:
        ...

    def to_provider_completion_request(self, request: AstralCompletionRequest) -> OpenAIRequestType:
        """
        Convert an AstralCompletionRequest into an OpenAIRequest.
        For demonstration, we just return the request.
        """
        ...


    def to_astral_completion_response(
        self,
        response: OpenAIResponseType,
        response_model: Optional[Type[_StructuredOutputT]] = None,
    ) -> Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]:
        """
        Convert an OpenAIResponse into either an AstralChatResponse or
        an AstralStructuredResponse based on whether a response_model is provided.
        """
        if response_model is None:
            # Process as a chat response.
            response_chat: OpenAIChatResponseType = response  # type: ignore
            astral_chat_response = self._convert_chat_response(response_chat)
            return astral_chat_response
        else:
            # Process as a structured response.
            response_structured = cast(OpenAIStructuredResponseType, response)
            astral_structured_response = self._convert_structured_response(response_structured, response_model)
            return astral_structured_response

    def _convert_chat_response(self, response: OpenAIChatResponseType) -> AstralChatResponse:
        """Convert the OpenAI ChatCompletion to an AstralChatResponse."""
        # Extract the content.
        content = response.choices[0].message.content if response.choices else ""
        # Extract the usage data.
        usage_data = response.usage if response.usage else ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        # Extract the cost data.
        cost: Optional[ChatCost] = None
        # Construct the chat response.
        return AstralChatResponse(response=content, usage=usage_data, cost=cost)

    def _convert_structured_response(
        self,
        response: OpenAIStructuredResponseType,
        response_model: Type[_StructuredOutputT],
    ) -> AstralStructuredResponse[_StructuredOutputT]:
        """
        Convert the OpenAIResponse (assumed to be structured) to an AstralStructuredResponse.
        This uses the provided response_model to parse the structured data.
        """
        # Extract the parsed content.
        parsed_content_data = response.choices[0].message.parsed
        if parsed_content_data is None:
            raise ValueError("Structured response missing parsed content")
        # Parse the structured output using the provided BaseModel subclass.
        parsed_content: _StructuredOutputT = response_model.model_validate(parsed_content_data)
        # Extract the usage data.
        usage_data = response.usage if response.usage else ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        # Extract the cost data.
        cost: Optional[ChatCost] = None
        # Construct and return the structured response.
        return AstralStructuredResponse[_StructuredOutputT](response=parsed_content, usage=usage_data, cost=cost)



