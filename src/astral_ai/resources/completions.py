# ------------------------------------------------------------------------------
# Completions Resource
# ------------------------------------------------------------------------------

"""
Completions Resource for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Typing
from typing import Any, Dict, List, Optional, Union, Iterable, Type, Literal, overload, TypeVar

# Pydantic
from pydantic import BaseModel

# Astral AI Types
from astral_ai._types._base import NOT_GIVEN, NotGiven
from astral_ai._types._request import AstralCompletionRequest, AstralStructuredCompletionRequest
from astral_ai._types._astral import AstralParams
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# Models
from astral_ai.constants._models import ModelName, get_provider_from_model_name

# Exceptions
from astral_ai.exceptions import ProviderNotFoundForModelError, ModelNameError, ResponseModelMissingError

# Providers
from astral_ai.providers._mappings import get_provider_client

# Additional types from your modules.
from astral_ai._types._request import (
    Metadata,
    Modality,
    ResponsePrediction,
    ReasoningEffort,
    ResponseFormat,
    StreamOptions,
    ToolChoice,
    Tool,
    Timeout,
)

# Astral AI Decorators
# TODO: implement required_parameters
from astral_ai._decorators import required_parameters


# Mappings
from astral_ai.providers._mappings import get_provider_adapter

# Generic Types
from astral_ai.providers._generics import StructuredOutputT


# ------------------------------------------------------------------------------
# Completions Resource
# ------------------------------------------------------------------------------


class Completions:
    """
    Astral AI Completions Resource.
    """

    def __init__(
        self,
        request: AstralCompletionRequest,
        astral_params: Optional[AstralParams] = None,
    ) -> None:
        """
        Initialize the completions resource.
        """
        self.request = request

        # Validate model
        self.model = request.model
        if not isinstance(self.model, ModelName):
            raise ModelNameError(model_name=self.model)

        # Validate provider
        model_provider = get_provider_from_model_name(self.model)
        if not model_provider:
            raise ProviderNotFoundForModelError(model_name=self.model)

        # Retrieve (or create) the provider client.
        self.client = get_provider_client(model_provider, astral_client=astral_params.astral_client)

        model_provider = "openai"

        # Get the provider adapter.
        self.adapter = get_provider_adapter(model_provider)

    # --------------------------------------------------------------------------
    # Run Chat Overload
    # --------------------------------------------------------------------------

    @overload
    def run(self) -> AstralChatResponse:
        ...

    # --------------------------------------------------------------------------
    # Run Structured Overload
    # --------------------------------------------------------------------------

    @overload
    def run(self, response_format: StructuredOutputT) -> AstralStructuredResponse:
        ...

    # --------------------------------------------------------------------------
    # Run Implementation
    # --------------------------------------------------------------------------

    def run(self, response_format: Optional[StructuredOutputT] = None) -> Union[AstralChatResponse, AstralStructuredResponse]:
        """
        Execute the completion request.

        If `response_model` is provided, the provider response is assumed to be a structured response.
        The `response` field of the structured response is parsed into the given model.
        Otherwise, a standard chat response is returned.
        """

        # Step One: Convert the request to the provider request.
        provider_request = self.adapter.to_provider_completion_request(self.request)

        if response_format is None:
            # Step Two: Execute the request.
            provider_response = self.client.create_completion_chat(provider_request)
        else:
            provider_response = self.client.create_completion_structured(provider_request)

        # Step Three: Convert the provider response to the Astral response.
        astral_response = self.adapter.to_astral_completion_response(provider_response)

        return astral_response


# ------------------------------------------------------------------------------
# Top-level Functions
# ------------------------------------------------------------------------------


@required_parameters("model", "messages")
def completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
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

    Returns:
        AstralChatResponse: The chat completion response.
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


# ------------------------------------------------------------------------------
# Structured Completion
# ------------------------------------------------------------------------------

StructuredOutputResponseT = TypeVar('StructuredOutputResponseT', bound=BaseModel)


@required_parameters("model", "messages", "response_model")
def completion_structured(
    *,
    model: str,
    messages: List[Dict[str, str]],
    response_format: StructuredOutputResponseT,
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
) -> AstralStructuredResponse:
    """
    Top-level function for a structured completion request.

    Returns:
        AstralStructuredResponse: The structured response, with its inner `response`
        field parsed using the provided `response_model`.
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
    return comp.run(response_model=response_format)
