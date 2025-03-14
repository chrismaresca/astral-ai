from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Enhanced Completions Resource (New Approach)
# -------------------------------------------------------------------------------- #

"""
Astral AI Enhanced Completions Resource

Provides a more flexible interface for completions with:
- Direct initialization or top-level convenience usage
- Single pass request validation (NOT_GIVEN vs None vs actual values)
- Tools + Reasoning Effort validation if the model supports them
- Support for JSON output and structured output, with fallback logic
- Both synchronous and asynchronous execution

CHANGE HIGHLIGHT:
- By the time we call the adapter, we always know whether we have a standard
  (AstralCompletionRequest) or a structured (AstralStructuredCompletionRequest)
  request. This is handled in `_validate_request` by constructing the correct
  request type and ensuring `response_format` is set appropriately if structured.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Literal,
    TypeVar,
    Type,
    Any,
    cast,
    overload
)
from abc import ABC

# Pydantic
from pydantic import BaseModel

# HTTPX Timeout
from httpx import Timeout

# Astral AI Types
from astral_ai._types import (
    NotGiven,
    NOT_GIVEN,
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,
    Metadata,
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

# Astral AI Model Constants
from astral_ai.constants._models import ModelName, ChatModels

# Astral AI Model Support
from astral_ai.constants._supported_models import (
    supports_feature,
    get_specific_model_id,
    MODEL_CAPABILITIES
)

# Astral AI Exceptions
from astral_ai.errors.model_support_exceptions import (
    ModelNameError,
    ResponseModelMissingError,
    StructuredOutputNotSupportedError,
    ReasoningEffortNotSupportedError,
    ToolsNotSupportedError,
    ImageIngestionNotSupportedError,
    SystemMessageNotSupportedError,
    DeveloperMessageNotSupportedError,
)

# Astral AI Decorators
from astral_ai._decorators import required_parameters

# Astral AI Messaging Models
from astral_ai.messaging._models import Messages

# Astral AI Resources
from astral_ai.resources._base_resource import AstralResource

# Astral AI Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

StructuredOutputResponseT = TypeVar("StructuredOutputResponseT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# Completions Resource Class (New Approach)
# -------------------------------------------------------------------------------- #


class Completions(AstralResource):
    """
    Enhanced Astral AI Completions Resource (New Approach).

    This class supports two main usage patterns:
      1. Direct initialization:
         >>> c = Completions(model="gpt-4o", messages=[{"role":"user","content":"Hi"}])
         >>> response = c.complete()

      2. Using the top-level convenience methods:
         >>> from astral_ai.resources._enhanced_completions import complete
         >>> response = complete(model="gpt-4o", messages=[{"role":"user","content":"Hi"}])

    Key Features:
      - Single-pass validation of model, messages, reasoning effort, tools, etc.
      - Differentiation between `None` and `NOT_GIVEN` for partial or default usage.
      - Additional methods to handle structured output or JSON output, with fallback logic
        depending on the model's capabilities.
      - Both synchronous and asynchronous execution.
    """

    def __init__(
        self,
        request: Optional[Union[AstralCompletionRequest, AstralStructuredCompletionRequest]] = None,
        *,
        model: ModelName | None = None,
        messages: Union[Messages, List[Dict[str, str]]] | None = None,
        astral_params: AstralParams | None = None,
        tools: List[Tool] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
        response_format: Type[StructuredOutputResponseT] | NotGiven = NOT_GIVEN,
        # If user explicitly calls a JSON-based method, we can track that
        # so `_validate_request` knows to treat it as a "json" request
        _is_json_request: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Completions resource with either:
          - A fully formed request object, OR
          - Individual parameters (model, messages, etc.).

        The `request` parameter is primarily for internal use or advanced scenarios where you're
        working with pre-configured request objects. For most use cases, provide individual 
        parameters instead.

        If `request` is provided alongside other parameters (like `model`), a ValueError is raised.

        Args:
            request: A complete AstralCompletionRequest or AstralStructuredCompletionRequest.
                     This is primarily for internal use or advanced scenarios.
            model: The model name (only used if `request` is None).
            messages: The conversation messages (only used if `request` is None).
            astral_params: Astral-specific parameters for special usage.
            tools: List of function calling tools (if supported by the model).
            tool_choice: Setting to manage usage of tools (auto, none, or explicit).
            reasoning_effort: The chain-of-thought (or similar) setting, if supported.
            response_format: A Pydantic model for structured output (if structured is desired).
            _is_json_request: An internal flag to indicate the user explicitly wants JSON mode.
            **kwargs: Additional fields to pass into the request if creating a new one.
        """
        self.request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest, None] = None

        if request is not None:
            # If a request object is provided, ensure no other direct param is set
            non_request_params = [
                model,
                messages,
                astral_params,
                tools,
                tool_choice,
                reasoning_effort,
                response_format,
                _is_json_request,
            ]
            if any(param is not None and param is not NOT_GIVEN for param in non_request_params) or kwargs:
                raise ValueError(
                    "Cannot provide both 'request' and other parameters (model, messages, etc.)"
                )
            self.request = request
            super().__init__(request)
        else:
            # Prepare for building a new request
            self._model = model
            self._messages = messages
            self._astral_params = astral_params
            self._tools = tools
            self._tool_choice = tool_choice
            self._reasoning_effort = reasoning_effort
            self._response_format = response_format
            self._kwargs = kwargs
            self._is_json_request = _is_json_request

            # Validate and build self.request
            self.request = self._validate_request()
            super().__init__(self.request)

    def _validate_request(self) -> Union[AstralCompletionRequest, AstralStructuredCompletionRequest]:
        """
        Validate the user's inputs and build a corresponding request object.

        This includes:
          - Validating model
          - Validating messages
          - Validating reasoning effort (if supported by the model)
          - Validating tools (if function calls are supported by the model)
          - Deciding between standard vs. structured request types
          - Setting `response_format` on the structured request so it's type safe
            by the time the adapter sees it.

        Returns:
            A completed request object (AstralCompletionRequest or AstralStructuredCompletionRequest).
        """

        # 1. Validate model
        if self._model is None:
            raise ValueError("`model` must be provided when not passing a complete request.")
        validated_model = self._validate_model()

        # 2. Validate messages
        if self._messages is None:
            raise ValueError("`messages` must be provided when not passing a complete request.")
        validated_messages = self._set_messages(self._messages, validated_model)

        # 3. Validate reasoning effort
        validated_reasoning_effort = self._set_reasoning_effort(self._reasoning_effort, validated_model)

        # 4. Validate and set tools
        validated_tools = self._set_tools(self._tools, validated_model)
        validated_tool_choice = self._set_tool_choice(self._tool_choice, validated_tools)

        # 5. Determine if we need a structured request or a standard request
        #    Two conditions create a structured request:
        #    (a) The user explicitly provided a `response_format`, or
        #    (b) The user explicitly wants JSON mode (`_is_json_request`).
        if (
            (self._response_format is not NOT_GIVEN and self._response_format is not None)
            or self._is_json_request
        ):
            # The user wants structured or JSON output
            request_cls = AstralStructuredCompletionRequest

            # If user gave an actual `response_format`, use it
            if self._response_format is not NOT_GIVEN and self._response_format is not None:
                # Use the actual model class as the response_format - not an instance
                response_format = self._response_format
                print("The response format HERE is: ", response_format)
            else:
                # For JSON requests without a specific model, this needs to be handled differently
                # This might need adjustment based on your implementation
                raise ValueError("JSON requests require a response model")

            # Validate the model can do EITHER structured output or JSON
            if not (supports_feature(validated_model, "structured_output") or
                    supports_feature(validated_model, "json_mode")):
                raise StructuredOutputNotSupportedError(validated_model)

            logger.debug(
                f"Constructing AstralStructuredCompletionRequest for model={validated_model}, "
                f"is_json_request={self._is_json_request}, response_format={response_format}"
            )

            request_data = {
                "model": validated_model,
                "messages": validated_messages,
                "astral_params": self._astral_params,
                "tools": validated_tools if validated_tools is not None else NOT_GIVEN,
                "tool_choice": validated_tool_choice if validated_tool_choice is not None else NOT_GIVEN,
                "reasoning_effort": validated_reasoning_effort if validated_reasoning_effort is not None else NOT_GIVEN,
                "response_format": response_format,
                **self._kwargs,
            }


            return request_cls(**request_data)
        else:
            # The user wants a standard chat request
            logger.debug(f"Constructing AstralCompletionRequest for model={validated_model}")
            request_cls = AstralCompletionRequest
            request_data = {
                "model": validated_model,
                "messages": validated_messages,
                "astral_params": self._astral_params,
                "tools": validated_tools if validated_tools is not None else NOT_GIVEN,
                "tool_choice": validated_tool_choice if validated_tool_choice is not None else NOT_GIVEN,
                "reasoning_effort": validated_reasoning_effort if validated_reasoning_effort is not None else NOT_GIVEN,
                # For standard requests, do not set `response_format`
                **self._kwargs,
            }
            return request_cls(**request_data)

    # -------------------------------------------------------------------------------- #
    # Validate Model
    # -------------------------------------------------------------------------------- #

    def _validate_model(self) -> ModelName:
        """
        Validate the model and return the model name.
        If a model alias is provided, convert it to a specific model ID.
        """
        
        # Validate the model
        from typing import get_args
        from astral_ai.constants._models import ModelAlias, ModelId
        
        valid_models = get_args(ModelAlias) + get_args(ModelId)
        if self._model not in valid_models:
            raise ModelNameError(model_name=self._model)
        
        # Convert any model alias to the specific model ID
        specific_model_id = get_specific_model_id(self._model)
        logger.debug(f"Model validation: '{self._model}' -> '{specific_model_id}'")
        
        return specific_model_id

    # -------------------------------------------------------------------------------- #
    # Set Messages
    # -------------------------------------------------------------------------------- #

    def _set_messages(
        self,
        messages: Union[Messages, List[Dict[str, str]]],
        model_name: ModelName
    ) -> Messages:
        """
        Validate and convert messages into the internal `Messages` format if needed.
        """
        # TODO: Implement this
        return self._messages

    # -------------------------------------------------------------------------------- #
    # Set Reasoning Effort
    # -------------------------------------------------------------------------------- #

    def _set_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort | NotGiven,
        model_name: ModelName
    ) -> ReasoningEffort | NotGiven:
        """
        Validate the reasoning effort. If it's None, we keep it as is. If it's an actual
        ReasoningEffort, check that the model supports it. Otherwise, return NOT_GIVEN.
        """
        if reasoning_effort is NOT_GIVEN:
            # The user didn't give it => do not set
            return NOT_GIVEN

        # Check if the model supports reasoning effort
        supports_reasoning = supports_feature(model_name, "reasoning_effort")

        if reasoning_effort is None:
            logger.info("No reasoning effort provided, returning None")
            if supports_reasoning:
                logger.info(f"Model {model_name} supports reasoning effort, changing ReasoningEffort from None to NOT_GIVEN")
                return NOT_GIVEN

        # If we got here, effort is an actual value
        if not supports_reasoning:
            logger.info(f"Model {model_name} does not support reasoning effort and it's set to {reasoning_effort}. Raising error.")
            raise ReasoningEffortNotSupportedError(model_name)

        return reasoning_effort

    # -------------------------------------------------------------------------------- #
    # Set Tools
    # -------------------------------------------------------------------------------- #

    def _set_tools(
        self,
        tools: List[Tool] | NotGiven,
        model_name: ModelName
    ) -> List[Tool] | NotGiven:
        """
        Validate that the model can handle function calls if tools are given.
        If the user provided no tools, return None.
        If the model doesn't support function calls, raise an error.
        """
        if tools is NOT_GIVEN:
            logger.info(f"No tools provided for model {model_name}.")
            return NOT_GIVEN

        # If the user did provide tools, check if the model supports function calls
        if not supports_feature(model_name, "function_calls"):
            logger.info(f"Model {model_name} does not support function calls. Raising error.")
            raise ToolsNotSupportedError(model_name)

        if not isinstance(tools, list):
            logger.info(f"Tools must be provided as a list. Raising error.")
            raise ValueError("Tools must be provided as a list")

        # Optionally ensure each tool is properly formatted
        # TODO: Implement this
        # for t in tools:
        #     # Check if the tool has a name and description
        #     if not hasattr(t.function, "name") or not hasattr(t.function, "description"):
        #         raise InvalidToolError(f"Tool {t.function.name if hasattr(t.function, 'name') else t.function} must have both a 'name' and 'description'.")

        logger.info(f"Successfully validated {len(tools)} tools ({', '.join([t.function.name for t in tools])}) for model {model_name}")
        return tools

    # -------------------------------------------------------------------------------- #
    # Set Tool Choice
    # -------------------------------------------------------------------------------- #

    def _set_tool_choice(
        self,
        tool_choice: ToolChoice | NotGiven,
        tools: List[Tool] | NotGiven
    ) -> ToolChoice | NotGiven:
        """
        Determine the final tool_choice based on presence of tools. If no tools,
        the tool_choice is forced to None. Otherwise, default to "auto" if not specified.
        """
        if tools is NOT_GIVEN or len(tools) == 0:
            if tool_choice is not NOT_GIVEN and tool_choice != 'auto':
                logger.warning("`tool_choice` was explicitly set but no tools were provided. Ignoring. We recommend not setting `tool_choice` or setting it to `auto` if no tools are provided.")
            return NOT_GIVEN

        if tool_choice is NOT_GIVEN:
            logger.info("No tool choice specified, defaulting to 'auto' since tools are present")
            # Default to "auto"
            return "auto"

        logger.info(f"Setting explicit tool_choice to {tool_choice}")
        return tool_choice

    # -------------------------------------------------------------------------------- #
    # Internal Helpers
    # -------------------------------------------------------------------------------- #

    @overload
    def _apply_cost(
        self,
        response: AstralChatResponse
    ) -> AstralChatResponse:
        ...
        
    @overload
    def _apply_cost(
        self,
        response: AstralStructuredResponse[StructuredOutputResponseT]
    ) -> AstralStructuredResponse[StructuredOutputResponseT]:
        ...
        
    def _apply_cost(
        self,
        response: Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputResponseT]]
    ) -> Union[AstralChatResponse, AstralStructuredResponse[StructuredOutputResponseT]]:
        """
        Apply cost calculation to the given response if a cost strategy is configured.
        """
        if self.cost_strategy is not None:
            # Identify the correct model name to use
            model_name = self._model if hasattr(self, '_model') else (
                self.request.model if self.request is not None else None
            )
            
            # Apply cost calculation
            return self.cost_strategy.run_cost_strategy(
                response=response,
                model_name=model_name,
                model_provider=self._model_provider,
            )
        return response

    # -------------------------------------------------------------------------------- #
    # Primary Sync Methods
    # -------------------------------------------------------------------------------- #

    def complete(self, **kwargs: Any) -> AstralChatResponse:
        """
        Synchronous completion entrypoint for standard chat completion only (no structured output).

        This method strictly handles non-structured, non-JSON responses. For structured output
        or JSON output, use `complete_structured()` or `complete_json()` respectively.

        Args:
            **kwargs: Optional runtime parameters to update/override the existing request

        Returns:
            AstralChatResponse: The standard chat completion response.
        """
        # This request must be an AstralCompletionRequest. If it's not, user is mixing usage.
        if not isinstance(self.request, AstralCompletionRequest) or isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for structured/JSON usage. "
                "Please call `complete_structured` or `complete_json` instead."
            )

        # Update request with any provided parameters
        request = self._update_request_with_params(**kwargs)

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)
        # Execute request
        provider_response = self.client.create_completion_chat(provider_request)
        # Convert to Astral response
        astral_response = self.adapter.to_astral_completion_response(provider_response)
        astral_response_with_cost = self._apply_cost(astral_response)

        return astral_response_with_cost

    def complete_json(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredResponse[StructuredOutputResponseT]:
        """
        Synchronous method requesting a JSON-based structured output, falling back to
        native structured output if available. By the time this method is called,
        `self.request` should already be an AstralStructuredCompletionRequest.

        Args:
            response_format: The Pydantic model to structure the JSON response into
            **kwargs: Optional runtime params

        Returns:
            AstralStructuredResponse with the provided response model.
        """
        if not isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for a standard chat usage. "
                "Please call `complete` instead, or re-initialize with a structured/JSON focus."
            )

        # Update request with new parameters (notably, possibly new response_format)
        request = self._update_request_with_params(response_format=response_format, **kwargs)

        provider_request = self.adapter.to_provider_request(request)
        # Because this is a "json" style request in concept,
        # we call the adapter method that handles structured or JSON.
        # The adapter itself can check if the model has structured_output or just json_mode
        provider_response = self.client.create_completion_structured(provider_request)

        astral_response = self.adapter.to_astral_completion_response(
            provider_response,
            response_format=response_format
        )
        astral_response_with_cost = self._apply_cost(astral_response)

        return cast(AstralStructuredResponse[StructuredOutputResponseT], astral_response_with_cost)

    def complete_structured(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredResponse[StructuredOutputResponseT]:
        """
        Synchronous method for requesting structured output with schema enforcement
        if available. By the time this method is called, `self.request` is an
        AstralStructuredCompletionRequest.

        Args:
            response_format: The Pydantic model to structure the response into
            **kwargs: Optional runtime params

        Returns:
            AstralStructuredResponse[response_format]
        """
        if not isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for standard chat usage. "
                "Please call `complete` instead, or re-initialize with a structured focus."
            )

        request = self._update_request_with_params(response_format=response_format, **kwargs)
        provider_request = self.adapter.to_provider_request(request)
        provider_response = self.client.create_completion_structured(provider_request)
        astral_response = self.adapter.to_astral_completion_response(
            provider_response,
            response_format=response_format
        )
        astral_response_with_cost = self._apply_cost(astral_response)
        return cast(AstralStructuredResponse[StructuredOutputResponseT], astral_response_with_cost)

    # -------------------------------------------------------------------------------- #
    # Primary Async Methods
    # -------------------------------------------------------------------------------- #

    async def complete_async(self, **kwargs: Any) -> AstralChatResponse:
        """
        Asynchronous version of `complete()` for standard chat completion only.

        This method strictly handles non-structured, non-JSON responses. For structured or JSON
        output, use `complete_structured_async()` or `complete_json_async()` respectively.

        Args:
            **kwargs: Optional runtime parameters to update/override the existing request

        Returns:
            AstralChatResponse: The standard chat completion response.
        """
        if not isinstance(self.request, AstralCompletionRequest) or isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for structured/JSON usage. "
                "Please call `complete_structured_async` or `complete_json_async` instead."
            )

        # Update request
        request = self._update_request_with_params(**kwargs)
        provider_request = self.adapter.to_provider_request(request)
        provider_response = await self.async_client.create_completion_chat_async(provider_request)
        astral_response = self.adapter.to_astral_completion_response(provider_response)
        astral_response_with_cost = self._apply_cost(astral_response)
        return astral_response_with_cost

    async def complete_json_async(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredResponse[StructuredOutputResponseT]:
        """
        Asynchronous method requesting a JSON-based structured output. By the time this
        method is called, `self.request` is an AstralStructuredCompletionRequest.

        Args:
            response_format: The Pydantic model to structure the JSON response into
            **kwargs: Optional runtime params

        Returns:
            AstralStructuredResponse[response_form  at]
        """
        if not isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for standard chat usage. "
                "Please call `complete_async` instead, or re-initialize with a structured/JSON focus."
            )

        request = self._update_request_with_params(response_format=response_format, **kwargs)
        provider_request = self.adapter.to_provider_request(request)
        provider_response = await self.async_client.create_completion_structured_async(provider_request)
        astral_response = self.adapter.to_astral_completion_response(
            provider_response,
            response_format=response_format
        )
        astral_response_with_cost = self._apply_cost(astral_response)
        return cast(AstralStructuredResponse[StructuredOutputResponseT], astral_response_with_cost)

    async def complete_structured_async(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredResponse[StructuredOutputResponseT]:
        """
        Asynchronous method for requesting structured output with schema enforcement
        if available.

        Args:
            response_format: The Pydantic model to structure the response into
            **kwargs: Optional runtime params

        Returns:
            AstralStructuredResponse[response_format]
        """
        if not isinstance(self.request, AstralStructuredCompletionRequest):
            raise ValueError(
                "This Completions instance was configured for standard chat usage. "
                "Please call `complete_async` instead, or re-initialize with a structured focus."
            )

        request = self._update_request_with_params(response_format=response_format, **kwargs)
        provider_request = self.adapter.to_provider_request(request)
        provider_response = await self.async_client.create_completion_structured_async(provider_request)
        astral_response = self.adapter.to_astral_completion_response(
            provider_response,
            response_format=response_format
        )
        return cast(AstralStructuredResponse[StructuredOutputResponseT], self._apply_cost(astral_response))

    # -------------------------------------------------------------------------------- #
    # Parameter Merging
    # -------------------------------------------------------------------------------- #

    def _merge_parameters(
        self,
        runtime_params: Dict[str, Any],
        merge_strategy: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Merge runtime parameters with existing request parameters using sensible defaults.

        Default behavior:
        - For 'messages' and 'tools': Additive (lists are combined)
        - For all other parameters: Overwrite (runtime values replace existing)

        Args:
            runtime_params: Parameters provided at runtime
            merge_strategy: Optional dict to override default merging behavior
                            {param_name: should_merge} where True = merge, False = overwrite

        Returns:
            Dict[str, Any]: The merged parameters
        """
        # Start with current request parameters
        base_params = self.request.model_dump(exclude_unset=True)
        result = base_params.copy()

        # Default merge strategy: messages and tools are merged, everything else is overwritten
        default_merge = {
            "messages": True,
            "tools": True,
        }

        if merge_strategy:
            default_merge.update(merge_strategy)

        for param, value in runtime_params.items():
            if value is None:
                # Skip None values for convenience
                continue

            should_merge = default_merge.get(param, False)

            if param not in base_params or not should_merge:
                result[param] = value
                continue

            base_value = base_params.get(param)
            if isinstance(base_value, list) and isinstance(value, list):
                result[param] = base_value + value
            elif isinstance(base_value, dict) and isinstance(value, dict):
                merged_dict = base_value.copy()
                merged_dict.update(value)
                result[param] = merged_dict
            else:
                result[param] = value

        return result

    # -------------------------------------------------------------------------------- #
    # Dynamic Parameter Updates
    # -------------------------------------------------------------------------------- #

    def _update_request_with_params(
        self,
        response_format: Type[StructuredOutputResponseT] | None | NotGiven = NOT_GIVEN,
        **kwargs: Any
    ) -> Union[AstralCompletionRequest, AstralStructuredCompletionRequest]:
        """
        Create an updated request object with the provided parameters merged
        with the original request parameters.

        - If we're dealing with an AstralStructuredCompletionRequest and the user
          provides a new response_format, we update the `response_format` field.

        Args:
            response_format: Possibly updated Pydantic model for structured output
            **kwargs: Parameters to update/add to the request

        Returns:
            A new request object of the same type as the original.
        """
        if not kwargs and response_format is NOT_GIVEN:
            return self.request

        merged_params = self._merge_parameters(kwargs)

        request_cls = type(self.request)

        # If we are structured, update `response_format` if needed
        if issubclass(request_cls, AstralStructuredCompletionRequest):
            if response_format is not NOT_GIVEN and response_format is not None:
                merged_params["response_format"] = response_format

        return request_cls(**merged_params)


# -------------------------------------------------------------------------------- #
# Convenience Function Helpers
# -------------------------------------------------------------------------------- #

def _prepare_convenience_params(
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    response_format: Type[StructuredOutputResponseT] | None = None,
    _is_json_request: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Helper function to prepare parameters for convenience methods.
    This centralizes the parameter handling logic used across all convenience functions.

    Args:
        model: The model to use for the completion
        messages: The conversation messages
        astral_params: Optional Astral-specific parameters
        reasoning_effort: Optional reasoning effort setting
        tools: Optional list of tools for function calling
        tool_choice: Optional tool choice setting
        response_format: Optional Pydantic model for structured or JSON output
        _is_json_request: Internal flag indicating explicit JSON usage
        **kwargs: Additional parameters to include in the request

    Returns:
        Dict[str, Any]: Prepared parameters for creating a Completions instance
    """
    req_data = {
        "model": model,
        "messages": messages,
        "astral_params": astral_params,
        "_is_json_request": _is_json_request,
    }

    if reasoning_effort is not NOT_GIVEN:
        req_data["reasoning_effort"] = reasoning_effort
    if tools is not NOT_GIVEN:
        req_data["tools"] = tools
    if tool_choice is not NOT_GIVEN:
        req_data["tool_choice"] = tool_choice
    if response_format is not None:
        req_data["response_format"] = response_format

    req_data.update(kwargs)

    return req_data


# -------------------------------------------------------------------------------- #
# Top-level Convenience Functions
# -------------------------------------------------------------------------------- #

@required_parameters("model", "messages")
def complete(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralChatResponse:
    """
    Top-level synchronous function for a standard chat completion (non-structured, non-JSON).

    Example:
        >>> resp = complete(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    Returns:
        AstralChatResponse: A standard chat completion response.
    """
    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        _is_json_request=False,
        **kwargs
    )

    c = Completions(**params)
    return c.complete()


@required_parameters("model", "messages", "response_format")
def complete_structured(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level synchronous function for a structured chat completion. 
    The response is parsed into the provided Pydantic model.

    Example:
        class MyOutputModel(BaseModel):
            answer: str
            confidence: float

        >>> resp = complete_structured(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=MyOutputModel
            )

    Returns:
        AstralStructuredResponse[response_format]: A structured response using the provided model.
    """
    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        _is_json_request=False,
        **kwargs
    )

    c = Completions(**params)
    return c.complete_structured(response_format=response_format)


@required_parameters("model", "messages", "response_format")
def complete_json(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level synchronous function requesting a JSON response with flexible model support.

    Example:
        >>> class CatInfo(BaseModel):
        >>>     name: str
        >>>     age: int
        >>>     breeds: List[str]
        >>>
        >>> resp = complete_json(
        >>>     model="gpt-4o",
        >>>     messages=[{"role": "user", "content": "Give me JSON about a cat"}],
        >>>     response_format=CatInfo
        >>> )

    Returns:
        AstralStructuredResponse[response_format].
    """
    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        _is_json_request=True,
        **kwargs
    )

    c = Completions(**params)
    return c.complete_json(response_format=response_format)


# -------------------------------------------------------------------------------- #
# Async Convenience Functions
# -------------------------------------------------------------------------------- #

@required_parameters("model", "messages")
async def complete_async(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralChatResponse:
    """
    Top-level asynchronous function for a standard chat completion (non-structured, non-JSON).

    Example:
        >>> resp = await complete_async(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}]
            )

    Returns:
        AstralChatResponse: A standard chat completion response.
    """
    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        _is_json_request=False,
        **kwargs
    )

    c = Completions(**params)
    return await c.complete_async()


@required_parameters("model", "messages", "response_format")
async def complete_structured_async(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level asynchronous function for a structured chat completion. 
    The response is parsed into the provided Pydantic model.

    Example:
        class MyOutputModel(BaseModel):
            answer: str
            confidence: float

        >>> resp = await complete_structured_async(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=MyOutputModel
            )

    Returns:
        AstralStructuredResponse[response_format].
    """
    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        _is_json_request=False,
        **kwargs
    )

    c = Completions(**params)
    return await c.complete_structured_async(response_format=response_format)


@required_parameters("model", "messages", "response_format")
async def complete_json_async(
    *,
    model: ModelName,
    messages: Union[Messages, List[Dict[str, str]]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] = None,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    **kwargs: Any,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level asynchronous function requesting a JSON response with flexible model support.

    Example:
        >>> class CatInfo(BaseModel):
        >>>     name: str
        >>>     age: int
        >>>     breeds: List[str]
        >>>
        >>> resp = await complete_json_async(
        >>>     model="gpt-4o",
        >>>     messages=[{"role": "user", "content": "Give me JSON about a cat"}],
        >>>     response_format=CatInfo
        >>> )

    Returns:
        AstralStructuredResponse[response_format].
    """
    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        _is_json_request=True,
        **kwargs
    )

    c = Completions(**params)
    return await c.complete_json_async(response_format=response_format)


# -------------------------------------------------------------------------------- #
# Simple Test Functions
# -------------------------------------------------------------------------------- #

def run_simple_tests() -> str:
    """
    Run a series of simple tests with the new Completions approach and print the results.

    This function demonstrates various usage patterns:
    1. Class initialization with different completion methods
    2. Direct convenience methods

    Each test displays:
    - Response content
    - Latency
    - Token usage
    - Cost information

    Returns:
        str: A message indicating all tests have completed
    """
    import time
    from pydantic import BaseModel
    from typing import List

    # Sample messages for testing
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]

    # Define a simple Pydantic model for structured output tests
    class RoboticLaws(BaseModel):
        laws: List[str]
        author: str
        year_published: int

    print("\n# -------------------------------------------------------------------------------- #")
    print("# Running Simple Tests for Completions (New Approach)")
    print("# -------------------------------------------------------------------------------- #\n")

    # # Test 1: Class initialization with chat completion
    # print("\n[TEST 1] Class initialization with chat completion")
    # print("-" * 70)

    # start_time = time.time()
    # c = Completions(model="gpt-4o", messages=messages)
    # response = c.complete()
    # latency = time.time() - start_time

    # print_response_details(response, latency)

    # Test 2: Class initialization with structured completion (JSON)
    print("\n[TEST 2] Class initialization with structured completion (JSON mode)")
    print("-" * 70)

    start_time = time.time()
    c_json = Completions(
        model="gpt-4o",
        messages=messages,
        response_format=RoboticLaws          # Potential structured parse
    )
    response2 = c_json.complete_json(response_format=RoboticLaws)
    latency = time.time() - start_time

    print_response_details(response2, latency)

    # Test 3: Direct convenience method - complete
    print("\n[TEST 3] Direct convenience method - complete")
    print("-" * 70)

    start_time = time.time()
    response3 = complete(model="gpt-4o", messages=messages)
    latency = time.time() - start_time

    print_response_details(response3, latency)

    # Test 4: Direct convenience method - complete_json
    print("\n[TEST 4] Direct convenience method - complete_json")
    print("-" * 70)

    start_time = time.time()
    response4 = complete_json(model="gpt-4o", messages=messages, response_format=RoboticLaws)
    latency = time.time() - start_time

    print_response_details(response4, latency)

    # Test 5: Async methods (run in sync context for simplicity)
    print("\n[TEST 5] Async methods (run in sync context)")
    print("-" * 70)

    import asyncio

    async def run_async_tests() -> None:
        # Async class method for standard chat
        start_time_a = time.time()
        c_async = Completions(model="gpt-4o", messages=messages)
        response_a = await c_async.complete_async()
        latency_a = time.time() - start_time_a

        print("\n[TEST 5.1] Class initialization with async chat completion")
        print_response_details(response_a, latency_a)

        # Async convenience method for JSON
        start_time_b = time.time()
        response_b = await complete_json_async(
            model="gpt-4o",
            messages=messages,
            response_format=RoboticLaws
        )
        latency_b = time.time() - start_time_b

        print("\n[TEST 5.2] Direct async convenience method - complete_json_async")
        print_response_details(response_b, latency_b)

    asyncio.run(run_async_tests())

    return "All tests completed"


def print_response_details(response: Union[AstralChatResponse, AstralStructuredResponse], latency: float) -> None:
    """
    Helper function to print response details in a consistent format.

    Args:
        response: The response object from a completion call
        latency: The time taken to receive the response in seconds
    """
    if hasattr(response, 'response'):
        print(f"Response:\n{'-' * 40}")
        print(response.response)
        print(f"{'-' * 40}")
    elif hasattr(response, 'data'):
        print(f"Structured Response:\n{'-' * 40}")
        print(response.data)
        print(f"{'-' * 40}")

    print(f"\nLatency: {latency:.3f}s")

    # Print usage information
    if hasattr(response, 'usage') and response.usage:
        print("\nToken Usage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

    # Print cost information
    if hasattr(response, 'cost') and response.cost:
        print("\nCost Information:")
        print(f"  Input cost: ${response.cost.input_cost:.6f}")
        print(f"  Output cost: ${response.cost.output_cost:.6f}")
        print(f"  Total cost: ${response.cost.total_cost:.6f}")


if __name__ == "__main__":
    run_simple_tests()


# -------------------------------------------------------------------------------- #
# Table of Changes
# -------------------------------------------------------------------------------- #
#
# 1) In `_validate_request`, we now explicitly create either:
#    - `AstralCompletionRequest` (if no structured/JSON usage),
#    - `AstralStructuredCompletionRequest` (if user requests a `response_format` or `_is_json_request`).
#
# 2) For the structured case, we set `response_format` on the request, satisfying the
#    requirement for `AstralStructuredCompletionRequest`.
#
# 3) We added guard clauses in `complete()`, `complete_async()`, etc., ensuring we only call
#    them if we actually hold the matching request type. This provides the type safety
#    needed to know exactly which request type is being sent to the adapter.
#
# 4) The `_is_json_request` flag is introduced to differentiate explicit JSON usage from
#    explicit structured output usage. Both produce an `AstralStructuredCompletionRequest`,
#    but `_is_json_request` clarifies how the request is *intended* to be used by the adapter.
#
# 5) The fallback logic for structured vs. JSON is effectively handled up front in
#    `_validate_request`. We either create a standard request or a structured request,
#    ensuring the adapter sees a fully realized request type with `response_format`.
