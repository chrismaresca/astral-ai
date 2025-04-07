from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Enhanced Completions Resource (New Approach)
# -------------------------------------------------------------------------------- #

"""
Astral AI Completions Resource
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
import json
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
    AstralBaseCompletionRequest,
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
    AstralCompletionResponse,
    AstralStructuredCompletionResponse,
)

# Astral AI Model Constants
from astral_ai.constants._models import ModelName, CompletionModels

# Astral AI Model Support
from astral_ai.constants._model_capabilities import (
    supports_feature,
    get_specific_model_id,
)

# Astral AI Exceptions
from astral_ai.errors.model_support_exceptions import (
    ModelNameError,
    ResponseModelMissingError,
    StructuredOutputNotSupportedError,
    ReasoningEffortNotSupportedError,
    ToolsNotSupportedError,
    InvalidResponseFormatError,

)

# Astral AI Decorators
from astral_ai._decorators import required_parameters, timeit, atimeit
from astral_ai.tracing._observability import log_event, log_event_async

# Astral AI Messaging Models
from astral_ai.messages._models import Messages, Message, ValidatedMessageList

# Astral AI Response Types
from astral_ai.resources._base_resource import AstralResource
from astral_ai._types._response.resources import (
    StructuredOutputCompletionResponse,
)

# Astral AI Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Logging Helper Functions
# -------------------------------------------------------------------------------- #

# Global emoji for completion-related logs
COMPLETION_EMOJI = "⚡"


def format_model_name(model_name: str) -> str:
    """Format a model name for pretty logging display."""
    return f"`{model_name}`"


def format_response_schema(schema_class: Type[BaseModel]) -> str:
    """
    Format a response schema class for human-readable logging.
    Returns the class name and top-level fields.
    """
    if not hasattr(schema_class, "model_json_schema"):
        return schema_class.__name__

    try:
        schema = schema_class.model_json_schema()
        properties = schema.get("properties", {})
        field_names = list(properties.keys())

        if len(field_names) <= 3:
            fields_str = ", ".join(field_names)
            return f"{schema_class.__name__} ({fields_str})"
        else:
            fields_preview = ", ".join(field_names[:3])
            return f"{schema_class.__name__} ({fields_preview}, ...)"
    except Exception:
        return schema_class.__name__


def format_tool_names(tools: List[Tool]) -> str:
    """
    Format a list of tools into a concise human-readable string of tool names.
    """
    if not tools:
        return "no tools"

    tool_names = []
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            name = tool["function"].get("name", "unnamed")
            tool_names.append(name)
        else:
            tool_names.append("unnamed_tool")

    if len(tool_names) <= 3:
        return ", ".join(f"`{name}`" for name in tool_names)
    else:
        preview = ", ".join(f"`{name}`" for name in tool_names[:3])
        return f"{preview} and {len(tool_names) - 3} more"


def format_tool_choice(choice: ToolChoice) -> str:
    """Format tool choice setting for human-readable logs."""
    if isinstance(choice, dict) and choice.get("type") == "function":
        function_name = choice.get("function", {}).get("name", "unknown")
        return f"function `{function_name}`"
    return f"`{choice}`"


# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

StructuredOutputResponseT = TypeVar("StructuredOutputResponseT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
# Completions Resource Class (New Approach)
# -------------------------------------------------------------------------------- #

class Completions(AstralResource[AstralBaseCompletionRequest, AstralCompletionResponse]):
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
      - Additional methods to handle structured output or JSON output, with no fallback
        from one to the other. Structured calls remain structured, JSON calls remain JSON.
      - Both synchronous and asynchronous execution.
    """

    def __init__(
        self,
        request: Optional[Union[AstralCompletionRequest, AstralStructuredCompletionRequest]] = None,
        *,
        model: ModelName | None = None,
        messages: Messages | None = None,
        astral_params: AstralParams | None = None,
        tools: List[Tool] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
        response_format: Type[StructuredOutputResponseT] | NotGiven = NOT_GIVEN,
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
            **kwargs: Additional fields to pass into the request if creating a new one.
        """
        self.request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest, None] = None

        if request is not None:
            non_request_params = [
                model,
                messages,
                astral_params,
                tools,
                tool_choice,
                reasoning_effort,
                response_format,
            ]
            if any(param is not None and param is not NOT_GIVEN for param in non_request_params) or kwargs:
                raise ValueError(
                    "Cannot provide both 'request' and other parameters (model, messages, etc.)"
                )

            # Store request temporarily
            self._model = request.model
            self._messages = request.messages
            self._astral_params = request.astral_params
            self._tools = getattr(request, 'tools', NOT_GIVEN)
            self._tool_choice = getattr(request, 'tool_choice', NOT_GIVEN)
            self._reasoning_effort = getattr(request, 'reasoning_effort', NOT_GIVEN)
            self._response_format = getattr(request, 'response_format', NOT_GIVEN)
            self._kwargs = {}

        else:
            self._model = model
            self._messages = messages
            self._astral_params = astral_params
            self._tools = tools
            self._tool_choice = tool_choice
            self._reasoning_effort = reasoning_effort
            self._response_format = response_format
            self._kwargs = kwargs

        self.request = self._validate_request()
        super().__init__(self.request)

    # -------------------------------------------------------------------------------- #
    # Request Validation
    # -------------------------------------------------------------------------------- #

    def _validate_request(self) -> Union[AstralCompletionRequest, AstralStructuredCompletionRequest]:
        """
        Validate the user's inputs and construct the appropriate request object.

        - This method performs a comprehensive validation of the inputs provided by the user.
        - It checks for the presence and correctness of required parameters such as model and messages.
        - Optional parameters like tools and reasoning effort are also validated.
        - Based on the validation results, it constructs and returns either an AstralCompletionRequest or an AstralStructuredCompletionRequest.
        - The decision to create a standard or structured request is determined by the presence of a response format.
        - If a response format is specified, a structured request is created; otherwise, a standard request is constructed.

        Returns:
        - Union[AstralCompletionRequest, AstralStructuredCompletionRequest]: The validated and constructed request object
          that encapsulates all necessary information for processing the user's request.
        """

        # Validate the model
        validated_model = self._validate_model()

        # Validate the messages
        validated_messages = self._standardize_messages(self._messages)

        # Validate the reasoning effort
        validated_reasoning_effort = self._set_reasoning_effort(self._reasoning_effort, validated_model)

        # Validate the tools
        validated_tools = self._set_tools(self._tools, validated_model)

        # Validate the tool choice
        validated_tool_choice = self._set_tool_choice(self._tool_choice, validated_tools)

        # Create and return the appropriate request object
        return self._create_request_object(
            validated_model=validated_model,
            validated_messages=validated_messages,
            validated_tools=validated_tools,
            validated_tool_choice=validated_tool_choice,
            validated_reasoning_effort=validated_reasoning_effort,
        )

    # -------------------------------------------------------------------------------- #
    # Request Object Creation
    # -------------------------------------------------------------------------------- #

    def _create_request_object(
        self,
        validated_model: ModelName,
        validated_messages: List[Message],
        validated_tools: List[Tool] | NotGiven,
        validated_tool_choice: ToolChoice | NotGiven,
        validated_reasoning_effort: ReasoningEffort | NotGiven,
    ) -> Union[AstralCompletionRequest, AstralStructuredCompletionRequest]:
        """
        Create the appropriate request object based on whether structured output is requested.
        """
        # Decide standard vs structured
        if self._response_format not in (None, NOT_GIVEN):
            # Explicitly validate and type the response format
            validated_response_format = self._validate_response_format(self._response_format, validated_model)

            logger.debug(
                f"{COMPLETION_EMOJI} Creating structured request with model {format_model_name(validated_model)} "
                f"using schema: {format_response_schema(validated_response_format)}"
            )
            request_cls = AstralStructuredCompletionRequest
            request_data = {
                "model": validated_model,
                "messages": validated_messages,
                "astral_params": self._astral_params,
                "tools": validated_tools if validated_tools is not None else NOT_GIVEN,
                "tool_choice": validated_tool_choice if validated_tool_choice is not None else NOT_GIVEN,
                "reasoning_effort": validated_reasoning_effort if validated_reasoning_effort is not None else NOT_GIVEN,
                "response_format": validated_response_format,
                **self._kwargs,
            }
            return request_cls(**request_data)
        else:
            logger.debug(
                f"{COMPLETION_EMOJI} Creating standard chat request with model {format_model_name(validated_model)}"
            )
            request_cls = AstralCompletionRequest
            request_data = {
                "model": validated_model,
                "messages": validated_messages,
                "astral_params": self._astral_params,
                "tools": validated_tools if validated_tools is not None else NOT_GIVEN,
                "tool_choice": validated_tool_choice if validated_tool_choice is not None else NOT_GIVEN,
                "reasoning_effort": validated_reasoning_effort if validated_reasoning_effort is not None else NOT_GIVEN,
                **self._kwargs,
            }
            return request_cls(**request_data)

    # -------------------------------------------------------------------------------- #
    # Validation Helpers
    # -------------------------------------------------------------------------------- #

    def _validate_model(self) -> ModelName:
        """
        Validate the model and return the model name.
        If a model alias is provided, convert it to a specific model ID.
        """
        if self._model is None:
            raise ValueError("`model` must be provided when not passing a complete request.")

        from typing import get_args
        from astral_ai.constants._models import ModelAlias, ModelId

        valid_models = get_args(ModelAlias) + get_args(ModelId)
        if self._model not in valid_models:
            raise ModelNameError(model_name=self._model)

        specific_model_id = get_specific_model_id(self._model)
        logger.debug(f"{COMPLETION_EMOJI} Model resolved: '{self._model}' → '{format_model_name(specific_model_id)}'")
        return specific_model_id

    def _standardize_messages(
        self,
        messages: Messages,
    ) -> ValidatedMessageList:
        """
        Validate and normalize message formats to ensure type compliance using the standardize_messages utility.

        Args:
            messages: Input messages in various allowed formats
            model_name: The model name for potential format-specific handling

        Returns:
            Messages: Validated and normalized messages
        """
        from astral_ai.messages._message_utils import standardize_messages

        if messages is None or len(messages) == 0:
            raise ValueError("`messages` must be provided when not passing a complete request.")

        # Use the standardize_messages function to validate and normalize messages
        standardized_messages = standardize_messages(messages)

        if len(standardized_messages) == 0:
            raise ValueError("No messages provided")

        return standardized_messages

    def _validate_response_format(
        self,
        response_format: Type[StructuredOutputResponseT] | NotGiven | None,
        model_name: ModelName
    ) -> Type[StructuredOutputResponseT]:
        """
        Validate the response format, if any, for the given model.
        """

        logger.debug(f"{COMPLETION_EMOJI} Validating response format for model {format_model_name(model_name)}")

        supports_structured_output = supports_feature(model_name, "structured_output")

        if not supports_structured_output:
            logger.warning(f"{COMPLETION_EMOJI} Model {format_model_name(model_name)} does not support structured output, killing the request. Please use a model that supports structured output.")
            raise StructuredOutputNotSupportedError(model_name)

        if not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
            raise InvalidResponseFormatError(model_name, response_format)

        return response_format

    def _set_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort | NotGiven,
        model_name: ModelName
    ) -> ReasoningEffort | NotGiven:
        """
        Validate the reasoning effort, if any, for the given model.
        """
        if reasoning_effort is NOT_GIVEN:
            return NOT_GIVEN

        supports_reasoning = supports_feature(model_name, "reasoning_effort")

        if reasoning_effort is None:
            logger.info(f"{COMPLETION_EMOJI} No reasoning effort specified for model {format_model_name(model_name)}")
            if supports_reasoning:
                logger.debug(f"{COMPLETION_EMOJI} Model {format_model_name(model_name)} supports reasoning effort, defaulting to model's choice")
                return NOT_GIVEN

        if not supports_reasoning and reasoning_effort is not NOT_GIVEN:
            logger.warning(
                f"{COMPLETION_EMOJI} Model {format_model_name(model_name)} does not support reasoning effort level: {reasoning_effort}"
            )
            raise ReasoningEffortNotSupportedError(model_name)

        logger.debug(f"{COMPLETION_EMOJI} Setting reasoning effort to {reasoning_effort} for model {format_model_name(model_name)}. "
                     "As of April 3, 2025, only OpenAI models natively support reasoning effort, for Anthropic, we will enable 'thinking' instead and define a maximum number of tokens this model can handle.")

        return reasoning_effort

    def _set_tools(
        self,
        tools: List[Tool] | NotGiven,
        model_name: ModelName
    ) -> List[Tool] | NotGiven:
        """
        Validate if the model can handle function calls if tools are provided.
        """
        if tools is NOT_GIVEN:
            logger.debug(f"{COMPLETION_EMOJI} No tools provided for model {format_model_name(model_name)}")
            return NOT_GIVEN

        if not supports_feature(model_name, "function_calling"):
            logger.warning(
                f"{COMPLETION_EMOJI} Model {format_model_name(model_name)} does not support function calling - cannot use tools"
            )
            raise ToolsNotSupportedError(model_name)

        if not isinstance(tools, list):
            logger.warning(f"{COMPLETION_EMOJI} Tools must be provided as a list")
            raise ValueError("Tools must be provided as a list")

        from astral_ai.tools.tool import get_tool

        processed_tools = []
        for tool in tools:
            tool_obj = get_tool(tool, enforce_strict=False)
            if tool_obj is None:
                logger.warning(f"{COMPLETION_EMOJI} Invalid tool definition skipped: {str(tool)[:40]}...")
                continue
            processed_tools.append(tool_obj)

        if not processed_tools:
            logger.warning(f"{COMPLETION_EMOJI} No valid tools found in provided list - all were skipped")
            return NOT_GIVEN

        logger.debug(f"{COMPLETION_EMOJI} Validated {len(processed_tools)} tools: {format_tool_names(processed_tools)}")
        return processed_tools

    def _set_tool_choice(
        self,
        tool_choice: ToolChoice | NotGiven,
        tools: List[Tool] | NotGiven
    ) -> ToolChoice | NotGiven:
        """
        Determine final tool_choice if any, based on presence of tools.
        """
        if tools is NOT_GIVEN or len(tools) == 0:
            if tool_choice is not NOT_GIVEN and tool_choice != 'auto':
                logger.warning(f"{COMPLETION_EMOJI} tool_choice was set but no tools were provided - ignoring tool_choice")
            return NOT_GIVEN

        if tool_choice is NOT_GIVEN:
            logger.debug(f"{COMPLETION_EMOJI} No tool choice specified, defaulting to 'auto' since tools are present")
            return "auto"

        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name", "unknown")
            logger.debug(f"{COMPLETION_EMOJI} Setting tool_choice to use specific function: {function_name}")
        else:
            logger.debug(f"{COMPLETION_EMOJI} Setting tool_choice to: {format_tool_choice(tool_choice)}")

        return tool_choice

    # -------------------------------------------------------------------------------- #
    # Cost Calculation
    # -------------------------------------------------------------------------------- #

    @overload
    def _apply_cost(
        self,
        response: AstralCompletionResponse
    ) -> AstralCompletionResponse:
        ...

    @overload
    def _apply_cost(
        self,
        response: AstralStructuredCompletionResponse[StructuredOutputResponseT]
    ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
        ...

    def _apply_cost(
        self,
        response: Union[AstralCompletionResponse, AstralStructuredCompletionResponse[StructuredOutputResponseT]]
    ) -> Union[AstralCompletionResponse, AstralStructuredCompletionResponse[StructuredOutputResponseT]]:
        """
        Apply cost calculation if a cost strategy is configured.
        """
        if self.cost_strategy is not None:
            model_name = response.model
            logger.debug(f"{COMPLETION_EMOJI} Calculating cost for model {format_model_name(model_name)}")
            return self.cost_strategy.run_cost_strategy(
                response=response,
                model_name=model_name,
                model_provider=self._model_provider,
            )
        return response

    # -------------------------------------------------------------------------------- #
    # JSON Parsing
    # -------------------------------------------------------------------------------- #

    # @overload
    # def _parse_json_response(
    #     self,
    #     astral_response: AstralCompletionResponse,
    #     response_format: None = None
    # ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
    #     """Overload for when no response format is provided (returns Dict)"""
    #     ...

    # @overload
    # def _parse_json_response(
    #     self,
    #     astral_response: AstralCompletionResponse,
    #     response_format: Type[StructuredOutputResponseT]
    # ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
    #     """Overload for when a BaseModel response format is provided"""
    #     ...

    # def _parse_json_response(
    #     self,
    #     astral_response: AstralCompletionResponse,
    #     response_format: Optional[Type[StructuredOutputResponseT]] = None
    # ) -> Union[AstralStructuredCompletionResponse[Dict[str, Any]], AstralStructuredCompletionResponse[StructuredOutputResponseT]]:
    #     """
    #     Parse the JSON response from the provider.

    #     This method extracts JSON content from the LLM's response text and optionally
    #     converts it to a structured Pydantic model.

    #     Args:
    #         astral_response: The standard completion response to parse
    #         response_format: Optional Pydantic model type to parse the JSON into

    #     Returns:
    #         A structured completion response with either a Dict or the specified model
    #     """
    #     # Log what we're trying to do
    #     if response_format is None:
    #         logger.debug(f"{COMPLETION_EMOJI} No response format provided, parsing as Dict[str, Any]")
    #     else:
    #         logger.debug(f"{COMPLETION_EMOJI} Parsing as {response_format.__name__}")

    #     # Get the raw text from the response
    #     raw_text = astral_response.response.choices[0].message.content or ""

    #     # Extract JSON if it's wrapped in a code block
    #     if "```" in raw_text:
    #         # Try to extract JSON from code blocks
    #         import re
    #         code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    #         matches = re.findall(code_block_pattern, raw_text)
    #         if matches:
    #             raw_text = matches[0].strip()
    #             logger.debug(f"{COMPLETION_EMOJI} Extracted JSON from code block")

    #     # Parse the JSON text into a dictionary
    #     parsed_dict: Dict[str, Any] = {}
    #     try:
    #         parsed_dict = json.loads(raw_text)
    #         logger.debug(f"{COMPLETION_EMOJI} Successfully parsed JSON")
    #     except Exception as e:
    #         logger.warning(f"{COMPLETION_EMOJI} Failed to parse chat response as JSON: {str(e)}")
    #         parsed_dict = {}

    #     # Convert to the requested model type if provided
    #     if response_format is not None:
    #         try:
    #             # Create structured output from the dictionary
    #             parsed_content = response_format.model_validate(parsed_dict)
    #             logger.debug(f"{COMPLETION_EMOJI} Successfully converted JSON to {response_format.__name__}")

    #             # Create the structured response with the specified model type
    #             return AstralStructuredCompletionResponse[response_format](
    #                 model=astral_response.model,
    #                 response=StructuredOutputCompletionResponse[response_format].model_construct(
    #                     original_message=raw_text,
    #                     parsed=parsed_content,
    #                 ),
    #                 content=parsed_content,
    #                 usage=astral_response.usage,
    #                 cost=astral_response.cost,
    #                 latency_ms=None
    #             )
    #         except Exception as e:
    #             logger.warning(f"{COMPLETION_EMOJI} Failed to convert JSON to {response_format.__name__}: {str(e)}")
    #             # Fall back to dictionary response if conversion fails

    #     # If no model provided or conversion failed, return dictionary response
    #     return AstralStructuredCompletionResponse[Dict[str, Any]](
    #         model=astral_response.model,
    #         response=StructuredOutputCompletionResponse[Dict[str, Any]].model_construct(
    #             original_message=raw_text,
    #             parsed=parsed_dict,
    #         ),
    #         content=parsed_dict,
    #         usage=astral_response.usage,
    #         cost=astral_response.cost,
    #         latency_ms=None
    #     )

    # -------------------------------------------------------------------------------- #
    # Primary Sync Helpers (Standard, Structured, JSON)
    # -------------------------------------------------------------------------------- #

    @log_event("standard")
    @timeit
    def _complete_standard(self, new_request: Optional[AstralCompletionRequest] = None) -> AstralCompletionResponse:
        """
        Synchronous standard chat completion. This is a plain text completion with no
        structured or JSON post-processing.
        """

        request = new_request if new_request else self.request

        if not isinstance(request, AstralCompletionRequest):
            raise ValueError(f"Invalid request type provided to _complete_standard: {type(request)}")

        logger.debug(f"{COMPLETION_EMOJI} Executing standard chat completion with model {format_model_name(request.model)}")

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = self.client.create_completion_chat(provider_request)
        astral_response = self.adapter.to_astral_completion_response(provider_response)

        # Apply cost calculation
        return self._apply_cost(astral_response)

    @log_event("structured")
    @timeit
    def _complete_structured(self, new_request: Optional[AstralStructuredCompletionRequest[StructuredOutputResponseT]] = None, response_format: Optional[Type[StructuredOutputResponseT]] = None) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
        """
        Synchronous structured output completion. This is a structured output completion
        with a Pydantic model for the response.
        """

        if response_format is None:
            raise ResponseModelMissingError()

        request = new_request if new_request else self.request

        if not isinstance(request, AstralStructuredCompletionRequest):
            raise ValueError(f"Invalid request type provided to _complete_structured: {type(request)}")

        logger.debug(
            f"{COMPLETION_EMOJI} Executing structured completion with model {format_model_name(request.model)} "
            f"using schema: {format_response_schema(response_format)}"
        )

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = self.client.create_completion_structured(provider_request)

        # Convert to astral response
        astral_response = self.adapter.to_astral_completion_response(provider_response, response_format=response_format)

        # Apply cost calculation
        return self._apply_cost(astral_response)

    # @log_event("complete_json")
    # @timeit
    # @overload
    # def _complete_json(
    #     self,
    #     new_request: Optional[AstralCompletionRequest] = None,
    #     response_format: None = None
    # ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
    #     ...

    # @log_event("complete_json")
    # @timeit
    # @overload
    # def _complete_json(
    #     self,
    #     new_request: Optional[AstralCompletionRequest],
    #     response_format: Type[StructuredOutputResponseT]
    # ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
    #     ...

    @log_event("json")
    @timeit
    def _complete_json(
        self,
        new_request: Optional[AstralCompletionRequest] = None
    ) -> AstralCompletionResponse:
        """
        Synchronous JSON completion. This is a JSON completion with no structured
        or JSON post-processing.
        """

        request = new_request if new_request else self.request

        request.response_format = {"type": "json_object"}

        if not isinstance(request, AstralCompletionRequest):
            raise ValueError(f"Invalid request type provided to _complete_json: {type(request)}")

        # if response_format is None:
        #     logger.debug(f"{COMPLETION_EMOJI} Executing JSON completion with standard chat approach on model {format_model_name(request.model)}")
        # else:
        #     logger.debug(f"{COMPLETION_EMOJI} Executing JSON completion and attempting to parse as {response_format.__name__} on model {format_model_name(request.model)}")
        # # Convert to provider request

        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = self.client.create_completion_chat(provider_request)
        astral_response = self.adapter.to_astral_completion_response(provider_response)

        # Parse the response as JSON
        # astral_json_response = self._parse_json_response(astral_response, response_format)

        return self._apply_cost(astral_response)

    # -------------------------------------------------------------------------------- #
    # Primary Async Helpers (Standard, Structured, JSON)
    # -------------------------------------------------------------------------------- #

    @log_event_async("standard")
    @atimeit
    async def _complete_standard_async(self, new_request: Optional[AstralCompletionRequest] = None) -> AstralCompletionResponse:
        """
        Asynchronous standard chat completion.
        """

        request = new_request if new_request else self.request

        if not isinstance(request, AstralCompletionRequest):
            raise ValueError(f"Invalid request type provided to _complete_standard_async: {type(request)}")

        logger.debug(f"{COMPLETION_EMOJI} Executing async standard chat completion with model {format_model_name(request.model)}")

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = await self.async_client.create_completion_chat_async(provider_request)

        # Convert to astral response
        astral_response = self.adapter.to_astral_completion_response(provider_response)

        # Apply cost calculation
        return self._apply_cost(astral_response)

    @log_event_async("structured")
    @atimeit
    async def _complete_structured_async(self, new_request: Optional[AstralStructuredCompletionRequest] = None, response_format: Optional[Type[StructuredOutputResponseT]] = None) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
        """
        Asynchronous structured output completion.
        """

        if response_format is None:
            raise ResponseModelMissingError()

        request = new_request if new_request else self.request

        if not isinstance(request, AstralStructuredCompletionRequest[StructuredOutputResponseT]):
            raise ValueError(f"Invalid request type provided to _complete_structured_async: {type(request)}")

        logger.debug(
            f"{COMPLETION_EMOJI} Executing async structured completion with model {format_model_name(request.model)} "
            f"using schema: {format_response_schema(response_format)}"
        )

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = await self.async_client.create_completion_structured_async(provider_request)

        # Convert to astral response
        astral_response = self.adapter.to_astral_completion_response(provider_response, response_format=response_format)

        # Apply cost calculation
        return self._apply_cost(astral_response)

    # @log_event_async("complete_json")
    # @atimeit
    # @overload
    # async def _complete_json_async(
    #     self,
    #     new_request: Optional[AstralCompletionRequest] = None,
    #     response_format: None = None
    # ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
    #     ...

    # @log_event_async("complete_json")
    # @atimeit
    # @overload
    # async def _complete_json_async(
    #     self,
    #     new_request: Optional[AstralCompletionRequest],
    #     response_format: Type[StructuredOutputResponseT]
    # ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
    #     ...

    @log_event_async("json")
    @atimeit
    async def _complete_json_async(self, new_request: Optional[AstralCompletionRequest] = None) -> AstralCompletionResponse:
        """
        Asynchronous JSON completion.
        """

        request = new_request if new_request else self.request
        request.response_format = {"type": "json_object"}

        if not isinstance(request, AstralCompletionRequest):
            raise ValueError(f"Invalid request type provided to _complete_json_async: {type(request)}")

        # # Log appropriate message based on model capabilities and response format
        # if supports_feature(request.model, "structured_outputs"):
        #     logger.warning(f"{COMPLETION_EMOJI} Model {format_model_name(request.model)} supports structured outputs, which is typically more reliable than JSON parsing. We recommend using structured outputs whenever possible. See docs for more information.")
        # else:
        #     message = f"{COMPLETION_EMOJI} Executing async JSON completion with standard chat approach on model {format_model_name(request.model)}"
        #     if response_format is not None:
        #         message = f"{COMPLETION_EMOJI} Executing async JSON completion and attempting to parse as {response_format.__name__} on model {format_model_name(request.model)}"
        #     logger.debug(message)

        # Convert to provider request
        provider_request = self.adapter.to_provider_request(request)

        # Create the response
        provider_response = await self.async_client.create_completion_chat_async(provider_request)

        # Convert to astral response
        astral_response = self.adapter.to_astral_completion_response(provider_response)

        # Parse the response as JSON
        # astral_json_response = self._parse_json_response(astral_response, response_format)

        # Apply cost calculation
        return self._apply_cost(astral_response)

    # -------------------------------------------------------------------------------- #
    # Primary Sync Methods
    # -------------------------------------------------------------------------------- #

    def complete(self, **kwargs: Any) -> AstralCompletionResponse:
        """
        Synchronous standard chat completion. This is a plain text completion with no
        structured or JSON post-processing.

        Args:
            **kwargs: Additional parameters to modify or override in the request at runtime.

        Returns:
            AstralCompletionResponse
        """

        # Update the request with new parameters
        new_request = self._update_and_validate_runtime_request("standard")

        # Complete the request
        return self._complete_standard(new_request)

    def complete_structured(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
        """
        Synchronous structured output request. Uses the provider's structured approach.

        Args:
            response_format: The Pydantic model for structured output.
            **kwargs: Additional parameters to modify or override in the request at runtime.

        Returns:
            AstralStructuredCompletionResponse[response_format]
        """
        # Replaced the lines with a call to the new helper function
        new_request = self._update_and_validate_runtime_request("structured")

        # Complete the request
        return self._complete_structured(new_request, response_format)

    # TODO: Reimplement this once we have a way to handle the response format
    # def complete_json(
    #     self,
    #     **kwargs: Any
    # ) -> AstralCompletionResponse:
    #     """
    #     Synchronous JSON request. We do NOT rely on structured outputs or fallback logic.
    #     We simply call the chat endpoint and parse the text as JSON into a dictionary.

    #     If the model supports structured outputs, we log a warning indicating that
    #     JSON is typically less reliable than the structured approach.

    #     Args:
    #         **kwargs: Additional parameters to modify or override in the request at runtime.

    #     Returns:
    #         AstralCompletionResponse
    #     """
    #     # Replaced the lines with a call to the new helper function
    #     new_request = self._update_and_validate_runtime_request("json")

    #     # Complete the request
    #     return self._complete_json(new_request)

    # -------------------------------------------------------------------------------- #
    # Primary Async Methods
    # -------------------------------------------------------------------------------- #

    async def complete_async(self, **kwargs: Any) -> AstralCompletionResponse:
        """
        Asynchronous standard chat completion.

        Args:
            **kwargs: Additional parameters to modify or override in the request.

        Returns:
            AstralCompletionResponse
        """
        # Replaced the lines with a call to the new helper function
        new_request = self._update_and_validate_runtime_request("standard")

        # Complete the request
        return self._complete_standard_async(new_request)

    async def complete_structured_async(
        self,
        response_format: Type[StructuredOutputResponseT],
        **kwargs: Any
    ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
        """
        Asynchronous structured output request.

        Args:
            response_format: The Pydantic model for structured output.
            **kwargs: Additional parameters to modify or override in the request.

        Returns:
            AstralStructuredCompletionResponse[response_format]
        """
        # Update the request with new parameters
        new_request = self._update_and_validate_runtime_request("structured")

        # Complete the request
        return self._complete_structured_async(new_request, response_format)

    # TODO: Reimplement this once we have a way to handle the response format
    # async def complete_json_async(
    #     self,
    #     **kwargs: Any
    # ) -> AstralCompletionResponse:
    #     """
    #     Asynchronous JSON request. We do NOT rely on structured outputs.
    #     We simply call the chat endpoint and parse the text as JSON into a dictionary.

    #     If the model supports structured outputs, we log a warning indicating that
    #     JSON is typically less reliable than the structured approach.

    #     Args:
    #         **kwargs: Additional parameters to modify or override in the request.

    #     Returns:
    #         AstralCompletionResponse
    #     """
    #     # Replaced the lines with a call to the new helper function
    #     new_request = self._update_and_validate_runtime_request("json")

    #     # Complete the request
    #     return self._complete_json_async(new_request)

    # -------------------------------------------------------------------------------- #
    # Runtime Helpers
    # -------------------------------------------------------------------------------- #
    # These are helpers for runtime validation and updates to the request
    # Only useful when working with the Completions class directly, not needed
    # when using the top-level convenience methods which handle validation automatically
    # -------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------- #
    # Update and Validate Request
    # -------------------------------------------------------------------------------- #
    @overload
    def _update_and_validate_runtime_request(
        self,
        mode: Literal["standard", "json"],
        response_format: Type[StructuredOutputResponseT] | None | NotGiven = NOT_GIVEN,
        **kwargs: Any
    ) -> AstralCompletionRequest:
        ...

    @overload
    def _update_and_validate_runtime_request(
        self,
        mode: Literal["structured"],
        response_format: Type[StructuredOutputResponseT] | None | NotGiven = NOT_GIVEN,
        **kwargs: Any
    ) -> AstralStructuredCompletionRequest:
        ...

    def _update_and_validate_runtime_request(
        self,
        mode: Literal["standard", "structured", "json"],
        response_format: Type[StructuredOutputResponseT] | None | NotGiven = NOT_GIVEN,
        **kwargs: Any
    ) -> Union[AstralCompletionRequest, AstralStructuredCompletionRequest]:
        """
        Update the request with new parameters and validate the request.

        This method performs a full validation cycle similar to _validate_request:
        1. Merges runtime parameters with existing request
        2. Re-validates critical components (model, tools, reasoning_effort)
        3. Creates a new request object with validated parameters
        """
        logger.debug(f"{COMPLETION_EMOJI} Preparing to update and validate request for {mode} completion mode")

        # Get current request parameters
        current_params = self.request.model_dump(exclude_unset=True)

        # Merge with runtime parameters
        merged_params = self._merge_parameters(kwargs)

        # Store temporarily for validation
        self._model = merged_params.get("model", current_params.get("model"))
        self._messages: Messages = merged_params.get("messages", current_params.get("messages"))
        self._tools = merged_params.get("tools", current_params.get("tools", NOT_GIVEN))
        self._tool_choice = merged_params.get("tool_choice", current_params.get("tool_choice", NOT_GIVEN))
        self._reasoning_effort = merged_params.get("reasoning_effort", current_params.get("reasoning_effort", NOT_GIVEN))
        self._response_format = response_format if response_format is not NOT_GIVEN else current_params.get("response_format")
        self._kwargs = {k: v for k, v in merged_params.items() if k not in {
            "model", "messages", "tools", "tool_choice", "reasoning_effort", "response_format"
        }}

        # Perform full validation cycle
        validated_model = self._validate_model()
        validated_messages = self._standardize_messages(self._messages)
        validated_reasoning_effort = self._set_reasoning_effort(self._reasoning_effort, validated_model)
        validated_tools = self._set_tools(self._tools, validated_model)
        validated_tool_choice = self._set_tool_choice(self._tool_choice, validated_tools)

        # Create new request object with validated parameters
        new_request = self._create_request_object(
            validated_model=validated_model,
            validated_messages=validated_messages,
            validated_tools=validated_tools,
            validated_tool_choice=validated_tool_choice,
            validated_reasoning_effort=validated_reasoning_effort,
        )

        # Validate the mode-specific requirements
        self._validate_runtime_request(mode, new_request)

        logger.debug(f"{COMPLETION_EMOJI} Request successfully updated and validated for {mode} completion")
        return new_request

    # -------------------------------------------------------------------------------- #
    # Runtime Validation
    # -------------------------------------------------------------------------------- #

    def _validate_runtime_request(self, mode: Literal["standard", "structured", "json"], request: Union[AstralCompletionRequest, AstralStructuredCompletionRequest]) -> None:
        """
        Centralized helper function to handle runtime checks for request usage.

        Args:
            mode: Can be "standard", "structured", or "json".
        """
        logger.debug(f"{COMPLETION_EMOJI} Validating request for {mode} completion mode")

        if mode == "standard":
            # Check if we have a structured request trying to be used as standard
            if isinstance(request, AstralStructuredCompletionRequest):
                model_name = request.model
                if supports_feature(model_name, "structured_output"):
                    logger.warning(
                        f"{COMPLETION_EMOJI} ⚠️ Detected structured request but using standard completion - this is not allowed. "
                        f"Model {format_model_name(model_name)} supports structured output."
                    )
                    raise ValueError(
                        "Cannot use structured request with standard completion. "
                        f"Model {model_name} supports structured output. Use complete_structured instead:\n\n"
                        "Example:\n"
                        "complete_structured(model='gpt-4', messages=[...], response_format=MyPydanticModel)\n"
                    )
                else:
                    logger.warning(
                        f"{COMPLETION_EMOJI} ⚠️ Detected structured request but using standard completion - this is not allowed. "
                        f"Model {format_model_name(model_name)} does not support structured output."
                    )
                    raise ValueError(
                        "Cannot use structured request with standard completion. "
                        f"Model {model_name} does not support structured output. Please view documentation for supported models."
                    )

            # Verify we have a standard request
            if not isinstance(request, AstralCompletionRequest):
                logger.error(f"{COMPLETION_EMOJI} ❌ Request type mismatch: Expected standard request")
                raise ValueError(
                    "This Completions instance was configured for structured usage. "
                    "Please call `complete_structured` if you want structured output, or re-initialize properly."
                )

            logger.debug(f"{COMPLETION_EMOJI} ✅ Request validated for standard completion with model {format_model_name(request.model)}")

        elif mode == "structured":
            # Original check from complete_structured / complete_structured_async
            if not isinstance(request, AstralStructuredCompletionRequest):
                logger.error(f"{COMPLETION_EMOJI} ❌ Request type mismatch: Expected structured request but found standard configuration")
                raise ValueError(
                    "This Completions instance was configured for standard usage. "
                    "Please call `complete` if you want plain text output, or re-initialize properly."
                )

            logger.debug(
                f"{COMPLETION_EMOJI} ✅ Request validated for structured completion with model "
                f"{format_model_name(request.model)} and schema: "
                f"{format_response_schema(request.response_format) if request.response_format else 'None'}"
            )

        elif mode == "json":

            if isinstance(request, AstralStructuredCompletionRequest) and request.response_format is not None:
                model_name = request.model
                if supports_feature(model_name, "structured_output"):
                    logger.warning(
                        f"{COMPLETION_EMOJI} ⚠️ Detected response_format in request but using JSON completion. "
                        f"Model {format_model_name(model_name)} supports structured outputs which would be more reliable."
                    )
                    raise ValueError(
                        "A response_format was detected, but JSON mode ignores any response model. "
                        f"Model {model_name} supports structured outputs. Are you sure you didn't want to call `complete_structured`?\n\n"
                        "Example:\n"
                        "complete_structured(model='gpt-4', messages=[...], response_format=MyPydanticModel)\n"
                    )
                else:
                    logger.warning(
                        f"{COMPLETION_EMOJI} ⚠️ Detected response_format in request but using JSON completion. "
                        f"Model {format_model_name(model_name)} does not support structured outputs."
                    )
                    raise ValueError(
                        "A response_format was detected, but JSON mode ignores any response model. "
                        "If you want structured outputs, pick a model that supports them.\n\n"
                        "Example:\n"
                        "complete_structured(model='structured-model', messages=[...], response_format=MyPydanticModel)\n"
                    )

            if not isinstance(request, AstralCompletionRequest):
                logger.error(f"{COMPLETION_EMOJI} ❌ Request type mismatch: Expected standard request for JSON completion")
                raise ValueError(
                    "This Completions instance was configured for structured usage. "
                    "Please re-initialize for plain/JSON usage."
                )

            logger.debug(f"{COMPLETION_EMOJI} ✅ Request validated for JSON completion with model {format_model_name(request.model)}")

    # -------------------------------------------------------------------------------- #
    # Parameter Merging
    # -------------------------------------------------------------------------------- #

    def _merge_parameters(
        self,
        runtime_params: Dict[str, Any],
        merge_behavior: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Merge runtime parameters with existing request parameters according to defined merge behavior.

        Parameters that are lists (like messages and tools) are combined by default, while other parameters
        override existing values. This behavior can be customized using merge_behavior.

        Args:
            runtime_params: New parameters provided at execution time
            merge_behavior: Optional dict specifying merge behavior for specific parameters
                          True = combine values, False = override values
                          Example: {"messages": False} makes messages override instead of combine

        Returns:
            Dict[str, Any]: Merged parameters dictionary
        """
        existing_params = self.request.model_dump(exclude_unset=True)
        merged_result = existing_params.copy()

        # Define which parameters should be combined vs overridden
        default_combine_behavior = {
            "messages": True,  # Combine lists
            "tools": True,     # Combine lists
        }

        if merge_behavior:
            default_combine_behavior.update(merge_behavior)

        for param_name, new_value in runtime_params.items():
            if new_value is None:
                continue

            # Determine if we should combine values or override
            should_combine = default_combine_behavior.get(param_name, False)
            logger.debug(f"Should combine {param_name}: {should_combine}")

            # If parameter doesn't exist or shouldn't be combined, just override
            if param_name not in existing_params or not should_combine:
                merged_result[param_name] = new_value
                continue

            # Get the existing value
            existing_value = existing_params.get(param_name)

            # Combine the values
            merged_result[param_name] = self._combine_values(param_name, existing_value, new_value)

        return merged_result

    # -------------------------------------------------------------------------------- #
    # Combine Values
    # -------------------------------------------------------------------------------- #

    def _combine_values(self, param_name: str, existing_value: Any, new_value: Any) -> Any:
        """
        Combine two parameter values based on their types.

        Args:
            param_name: Name of the parameter being combined (for logging)
            existing_value: Current value in the request
            new_value: New value to combine with existing

        Returns:
            Combined value, or new value if combination not possible
        """
        if isinstance(existing_value, list) and isinstance(new_value, list):
            logger.debug(
                f"{COMPLETION_EMOJI} Combining {len(new_value)} {param_name} items with "
                f"existing {len(existing_value)} items"
            )
            return existing_value + new_value

        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            logger.debug(f"{COMPLETION_EMOJI} Merging {param_name} dictionaries")
            merged = existing_value.copy()
            merged.update(new_value)
            return merged

        logger.debug(f"{COMPLETION_EMOJI} Overriding existing {param_name} with new value")
        return new_value


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
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Helper function to prepare parameters for convenience methods.
    """
    req_data = {
        "model": model,
        "messages": messages,
        "astral_params": astral_params,
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
    **kwargs: Any
) -> AstralCompletionResponse:
    """
    Convenience function for a standard chat completion (non-structured, non-JSON).
    """
    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs
    )
    c = Completions(**params)
    return c._complete_standard(c.request)


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
    **kwargs: Any
) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
    """
    Convenience function for a structured chat completion.
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
        **kwargs
    )
    c = Completions(**params)
    return c._complete_structured(c.request, response_format)


# @required_parameters("model", "messages")
# @overload
# def complete_json(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     # TODO: Remove this once we have a way to handle the response format
#     # response_format: None = None,
#     **kwargs: Any
# ) -> AstralCompletionResponse:
#     """Overload for dict-typed response when response_format is None"""
#     ...


# @required_parameters("model", "messages")
# @overload
# def complete_json(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     # TODO: Remove this once we have a way to handle the response format
#     # response_format: Type[StructuredOutputResponseT],
#     **kwargs: Any
# ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
#     """Overload for model-typed response when response_format is specified"""
#     ...


# @required_parameters("model", "messages")
# def complete_json(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     # TODO: Remove this once we have a way to handle the response format
#     # response_format: Optional[Type[StructuredOutputResponseT]] = None,
#     **kwargs: Any
# ) -> AstralCompletionResponse:
#     """
#     Convenience function to request a JSON completion, ignoring any structured output support.
#     Always uses the standard chat approach and parses the result as JSON.

#     Args:
#         model: The model to use
#         messages: The messages to use
#         astral_params: Optional Astral-specific parameters
#         reasoning_effort: Optional reasoning effort level
#         tools: Optional tools to use
#         tool_choice: Optional tool choice
#         # TODO: Remove this once we have a way to handle the response format
#         # response_format: Optional Pydantic model to parse the JSON into
#         **kwargs: Additional keyword arguments

#     Returns:
#         An AstralCompletionResponse
#     """
#     params = _prepare_convenience_params(
#         model=model,
#         messages=messages,
#         astral_params=astral_params,
#         reasoning_effort=reasoning_effort,
#         tools=tools,
#         tool_choice=tool_choice,
#         # TODO: Remove this once we have a way to handle the response format
#         # response_format=response_format,
#         **kwargs
#     )
#     c = Completions(**params)
#     return c._complete_json(c.request)


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
    **kwargs: Any
) -> AstralCompletionResponse:
    """
    Asynchronous convenience function for a standard chat completion.
    """
    params = _prepare_convenience_params(
        model=model,
        messages=messages,
        astral_params=astral_params,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs
    )
    c = Completions(**params)
    return await c._complete_standard_async(c.request)


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
    **kwargs: Any
) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
    """
    Asynchronous convenience function for a structured chat completion.
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
        **kwargs
    )
    c = Completions(**params)
    return await c._complete_structured_async(c.request, response_format)


# @required_parameters("model", "messages")
# @overload
# async def complete_json_async(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     response_format: None = None,
#     **kwargs: Any
# ) -> AstralStructuredCompletionResponse[Dict[str, Any]]:
#     """Overload for dict-typed response when response_format is None"""
#     ...


# @required_parameters("model", "messages")
# @overload
# async def complete_json_async(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     response_format: Type[StructuredOutputResponseT],
#     **kwargs: Any
# ) -> AstralStructuredCompletionResponse[StructuredOutputResponseT]:
#     """Overload for model-typed response when response_format is specified"""
#     ...


# @required_parameters("model", "messages")
# async def complete_json_async(
#     *,
#     model: ModelName,
#     messages: Union[Messages, List[Dict[str, str]]],
#     astral_params: Optional[AstralParams] = None,
#     reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
#     tools: Optional[List[Tool]] | NotGiven = NOT_GIVEN,
#     tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
#     # TODO: Remove this once we have a way to handle the response format
#     # response_format: Optional[Type[StructuredOutputResponseT]] = None,
#     **kwargs: Any
# ) -> AstralCompletionResponse:
#     """
#     Asynchronous convenience function to request a JSON completion.
#     Always uses the standard chat approach and parses the result as JSON.

#     Args:
#         model: The model to use
#         messages: The messages to use
#         astral_params: Optional Astral-specific parameters
#         reasoning_effort: Optional reasoning effort level
#         tools: Optional tools to use
#         tool_choice: Optional tool choice
#         **kwargs: Additional keyword arguments

#     Returns:
#         An AstralCompletionResponse
#     """
#     params = _prepare_convenience_params(
#         model=model,
#         messages=messages,
#         astral_params=astral_params,
#         reasoning_effort=reasoning_effort,
#         tools=tools,
#         tool_choice=tool_choice,
#         **kwargs
#     )
#     c = Completions(**params)
#     return await c._complete_json_async(c.request)


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #

# # Tests

# if __name__ == "__main__":

#     from pydantic import BaseModel

#     class TestModel(BaseModel):
#         name: str
#         age: int

#     # # -------------------------------------------------------------------------------- #
#     # # Standard Completion with OpenAI
#     # # -------------------------------------------------------------------------------- #

#     result = complete(
#         model="gpt-4o",
#         messages=[
#             {"role": "user", "content": "Hello, how are you?"}
#         ]
#     )

#     # result_1 = complete(
#     #     model="gpt-4o",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you?"}
#     #     ]
#     # )
#     # result_2 = complete(
#     #     model="gpt-4o",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you?"}
#     #     ]
#     # )
#     # print(result)
#     # print(result_1)
#     # print(result_2)

#     # # -------------------------------------------------------------------------------- #
#     # # Standard Completion with OpenAI No store
#     # # -------------------------------------------------------------------------------- #

#     # result = complete(
#     #     model="gpt-4o",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you?"}
#     #     ],
#     #     astral_params=AstralParams(store=False)
#     # )
#     # print(result)

#     # # -------------------------------------------------------------------------------- #
#     # # Standard Completion with Anthropic
#     # # -------------------------------------------------------------------------------- #

#     result = complete(
#         model="claude-3-5-sonnet-20240620",
#         messages=[
#             {"role": "user", "content": "Hello, how are you?"}
#         ]
#     )

#     # # --------------------------------------------------------------------------------
#     # # Structured Completion with OpenAI
#     # # --------------------------------------------------------------------------------

#     print("=" * 100)
#     print("Structured Completion with OpenAI")
#     print("=" * 100)
#     result = complete_structured(
#         model="gpt-4o",
#         messages=[
#             {"role": "user", "content": "Hello, how are you? What is your name and age?"}
#         ],
#         response_format=TestModel
#     )
#     print("=" * 100)
#     print(result)
#     print("=" * 100)

#     # # --------------------------------------------------------------------------------
#     # # Reasoning Model with OpenAI
#     # # --------------------------------------------------------------------------------

#     # print("=" * 100)
#     # print("Reasoning Model with OpenAI")
#     # print("=" * 100)
#     # result = complete(
#     #     model="o3-mini",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you?"}
#     #     ],
#     #     reasoning_effort="high"
#     # )
#     # print("=" * 100)
#     # print(result)
#     # print("=" * 100)

#     # # --------------------------------------------------------------------------------
#     # # Reasoning Model with Anthropic
#     # # --------------------------------------------------------------------------------

#     # print("=" * 100)
#     # print("Reasoning Model with Anthropic")
#     # print("=" * 100)
#     # result = complete(
#     #     model="claude-3-7-sonnet-20250219",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you? Really think hard about life and explain your answer in a way that is easy to understand."}
#     #     ],
#     #     reasoning_effort="high"
#     # )
#     # print("=" * 100)
#     # print(result)
#     # print("=" * 100)

#     # # --------------------------------------------------------------------------------
#     # # Structured Compeltion with Reasoning Model For OpenAI
#     # # --------------------------------------------------------------------------------

#     # print("=" * 100)
#     # print("Structured Compeltion with Reasoning Model For OpenAI")
#     # print("=" * 100)
#     # result = complete_structured(
#     #     model="o3-mini",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you? What is your name and age? Really come up with a creative answer."}
#     #     ],
#     #     response_format=TestModel,
#     #     reasoning_effort="high"
#     # )
#     # print("=" * 100)
#     # print(result)
#     # print("=" * 100)

#     # --------------------------------------------------------------------------------
#     # Structured Compeltion with Reasoning Model For DeepSeek
#     # --------------------------------------------------------------------------------

#     # print("=" * 100)
#     # print("Structured Compeltion with Reasoning Model For DeepSeek")
#     # print("=" * 100)
#     # result = complete_structured(
#     #     model="deepseek-chat",
#     #     messages=[
#     #         {"role": "user", "content": "Hello, how are you? What is your name and age? Really come up with a creative answer."}
#     #     ],
#     #     response_format=TestModel,
#     # )
#     # print("=" * 100)
#     # print(result)
#     # print("=" * 100)
