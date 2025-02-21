from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Completions Resource
# -------------------------------------------------------------------------------- #

"""

Astral AI Completions Resource

Handles both chat and structured completion requests by providing:
- Type-safe request handling
- Provider-specific request/response adaptation
- Cost calculation and tracking
- Response validation and parsing

"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import Dict, List, Optional, Union, Iterable, Literal, overload, TypeVar, Type
from abc import ABC

# Pydantic
from pydantic import BaseModel

# HTTPX Timeout
from httpx import Timeout

# Astral AI Types
from astral_ai._types import (
    # Base
    NotGiven,
    NOT_GIVEN,

    # Request
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,

    # Response
    Metadata,

    # Request Params
    Modality,
    ResponsePrediction,
    ReasoningEffort,
    ResponseFormat,
    StreamOptions,
    ToolChoice,
    Tool,
    AstralParams,
    AstralChatResponse,
    AstralStructuredResponse,
)

# Astral AI Exceptions
from astral_ai.exceptions import ResponseModelMissingError

# Astral AI Decorators
from astral_ai._decorators import required_parameters

# Astral AI Messaging Models
from astral_ai.messaging._models import Messages

# Astral AI Resources
from astral_ai.resources._base_resource import AstralResource

# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# -------------------------------------------------------------------------------- #
# Completions Resource Class
# -------------------------------------------------------------------------------- #


class Completions(AstralResource):
    """
    Astral AI Completions Resource.

    Handles both chat and structured completion requests by providing:
    - Type-safe request handling
    - Provider-specific request/response adaptation
    - Cost calculation and tracking
    - Response validation and parsing
    """

    def __init__(
        self,
        request: AstralCompletionRequest,
        astral_params: Optional[AstralParams] = None,
    ) -> None:
        super().__init__(request, astral_params)

    # -------------------------------------------------------------------------------- #
    # Run Method Overloads & Implementation
    # -------------------------------------------------------------------------------- #

    @overload
    def run(self) -> AstralChatResponse:
        ...

    @overload
    def run(self, response_format: Type[_StructuredOutputT]) -> AstralStructuredResponse[_StructuredOutputT]:
        ...

    def run(
        self, response_format: Optional[Type[_StructuredOutputT]] = None,
    ) -> Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]:
        """
        Execute the completion request.

        This method handles both chat and structured completion requests by:
        1. Converting the Astral request to provider-specific format
        2. Executing the request with the appropriate provider client
        3. Converting and validating the provider response
        4. Calculating and attaching cost metrics if enabled

        Args:
            response_format: Optional structured output model for parsing responses.
                           If provided, the response will be parsed into this model type.

        Returns:
            AstralChatResponse: For standard chat completions when response_format is None
            AstralStructuredResponse: For structured outputs when response_format is provided

        Note:
            Cost calculation is only performed if a cost_strategy is configured.
            The cost metrics will be attached to the response object.
        """
        # Convert the request to provider-specific format
        provider_request = self.adapter.to_provider_completion_request(self.request)

        # Execute request and convert response based on type
        if response_format is None:
            provider_response = self.client.create_completion_chat(provider_request)
            astral_response = self.adapter.to_astral_completion_response(provider_response)
        else:
            provider_response = self.client.create_completion_structured(provider_request)
            astral_response = self.adapter.to_astral_completion_response(
                provider_response,
                response_model=response_format
            )

        # Calculate and attach cost metrics if enabled
        if self.cost_strategy is not None:
            astral_response = self.cost_strategy.run_cost_strategy(
                response=astral_response,
                model_name=self.model,
                model_provider=self.model_provider,
            )

        return astral_response


# -------------------------------------------------------------------------------- #
# Top-level Functions
# -------------------------------------------------------------------------------- #


@required_parameters("model", "messages")
def completion(
    *,
    model: str,
    messages: Messages,
    astral_params: AstralParams | NotGiven = NOT_GIVEN,
    stream: Optional[bool] | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
    modalities: Optional[List[Modality]] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
    prediction: Optional[ResponsePrediction] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    response_format: ResponseFormat | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    service_tier: Literal["auto", "default"] | NotGiven = NOT_GIVEN,
    stop: Optional[str] | List[str] | NotGiven = NOT_GIVEN,
    store: Optional[bool] | NotGiven = NOT_GIVEN,
    stream_options: Optional[StreamOptions] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[Tool] | NotGiven = NOT_GIVEN,
    top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    timeout: Union[float, Timeout, None] | NotGiven = NOT_GIVEN,
) -> AstralChatResponse:
    """
    Top-level function for a chat completion request.

    Args:
        model: The model to use for completion
        messages: The conversation history
        astral_params: Optional Astral-specific parameters
        **kwargs: Additional model-specific parameters

    Returns:
        AstralChatResponse: The chat completion response
    """
    request_data = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": store,
        "stream_options": stream_options,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "user": user,
        "timeout": timeout,
    }
    request = AstralCompletionRequest(**request_data)
    comp = Completions(request, astral_params=astral_params)
    return comp.run()

# -------------------------------------------------------------------------------- #
# Structured Completion
# -------------------------------------------------------------------------------- #


StructuredOutputResponseT = TypeVar('StructuredOutputResponseT', bound=BaseModel)


@required_parameters("model", "messages", "response_model")
def completion_structured(
    *,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] | NotGiven = NOT_GIVEN,
    stream: bool | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
    modalities: Optional[List[Modality]] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
    prediction: Optional[ResponsePrediction] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    service_tier: Literal["auto", "default"] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    store: Optional[bool] | NotGiven = NOT_GIVEN,
    stream_options: Optional[StreamOptions] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[Tool] | NotGiven = NOT_GIVEN,
    top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    timeout: Union[float, Timeout, None] | NotGiven = NOT_GIVEN,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level function for a structured completion request.

    Args:
        model: The model to use for completion
        messages: The conversation history
        response_format: The Pydantic model to parse the response into
        astral_params: Optional Astral-specific parameters
        **kwargs: Additional model-specific parameters

    Returns:
        AstralStructuredResponse: The structured response, with its inner `response`
        field parsed using the provided `response_model`

    Raises:
        ResponseModelMissingError: If response_format is None
    """

    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    # Mark the request as structured.
    request_data = {
        "model": model,
        "response_format": response_format,
        "messages": messages,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": store,
        "stream_options": stream_options,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "user": user,
        "timeout": timeout,
        # Additional flag to signal a structured response.
        "structured": True,
    }

    request = AstralStructuredCompletionRequest(**request_data)
    comp = Completions(request, astral_params=astral_params)
    return comp.run(response_format=response_format)
