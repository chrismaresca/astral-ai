from __future__ import annotations
# -------------------------------------------------------------------------------- #
# DeepSeek Request Models
# -------------------------------------------------------------------------------- #

"""
DeepSeek Request Models for Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in

from typing import Any, Dict, List, Union, Literal, TypedDict, Required, TypeAlias, NotRequired


# Astral AI
from astral_ai.constants._models import DeepSeekModels

# Astral AI Types
from astral_ai._types import (
    ToolChoice,
    ReasoningEffort,
    ResponseFormat,
    Metadata,
    ResponsePrediction,
    Modality,
    StreamOptions,
    Tool,
)

# OpenAI Types
from astral_ai.providers.openai._types._message import OpenAIMessageType

# -------------------------------------------------------------------------------- #
# DeepSeek Response Format For Structured Requests
# -------------------------------------------------------------------------------- #


class DeepseekResponseFormat(TypedDict, total=False):
    """
    DeepseekResponseFormat specifies the output format for a structured chat request.

    Attributes:
        type (Literal["json_object"]):
            [Required] Specifies the type of the output format.
            - Possible value: "json_object"
            - Ensures that the model's generated output is valid JSON.
    """
    type: Literal["json_object"]


# -------------------------------------------------------------------------------- #
# DeepSeek Chat Request
# -------------------------------------------------------------------------------- #

class DeepSeekRequestChat(TypedDict, total=False):
    """
    DeepSeekChatRequest represents a standard chat request for the Deepseek model.

    This request object is used to create a model response for a given chat conversation.
    It includes a list of messages representing the conversation history and various parameters
    that control the response generation behavior of the model.
    """

    messages: Required[Union[List[OpenAIMessageType], OpenAIMessageType]]
    """
    [Required] A list of messages comprising the conversation so far.
    - Each message must be an object (dictionary) with the following keys:
        - content (str): The text content of the message.
        - role (str): The role of the message's author (e.g., "system", "user", "assistant", "tool").
        - name (NotRequired[str]): An optional name to differentiate between participants with the same role.
    - The list must contain at least one message.
    """

    model: Required[DeepSeekModels]
    """
    [Required] The identifier of the model to be used.
    - Possible values: "deepseek-chat" or "deepseek-reasoner".
    - Specifies which Deepseek model should process the chat request.
    """

    frequency_penalty: NotRequired[float]
    """
    [NotRequired] Number between -2.0 and 2.0.
    - Positive values penalize new tokens based on their existing frequency in the text so far,
      decreasing the likelihood that the model will repeat the same line verbatim.
    - Possible values: >= -2 and <= 2.
    - Default value: 0.
    """

    max_tokens: NotRequired[int]
    """
    [NotRequired] Maximum number of tokens that can be generated for the chat completion.
    - Must be an integer greater than 1.
    - The total length of both input and generated tokens is limited by the model's context length.
    - If not specified, the default value is 4096.
    """

    presence_penalty: NotRequired[float]
    """
    [NotRequired] Number between -2.0 and 2.0.
    - Positive values penalize tokens based on whether they appear in the text so far,
      thus increasing the model's likelihood to discuss new topics.
    - Possible values: >= -2 and <= 2.
    - Default value: 0.
    """

    response_format: NotRequired[Dict[str, Any]]
    """
    [NotRequired] An object specifying the format that the model must output.
    - Should include a key "type" with possible values "text" or "json_object".
    - For basic chat requests, if not specified, the default is "text".
    - When set to "json_object", it instructs the model to produce valid JSON output.
    """

    stop: NotRequired[Union[str, List[str]]]
    """
    [NotRequired] One or more sequences where the API will stop generating further tokens.
    - Can be provided as a single string or as a list of strings.
    - Supports up to 16 distinct stop sequences.
    """

    stream: NotRequired[bool]
    """
    [NotRequired] Flag indicating whether partial message deltas will be streamed as they become available.
    - When set to true, tokens are streamed as server-sent events (SSE), ending with a final "data: [DONE]" message.
    """

    stream_options: NotRequired[Dict[str, Any]]
    """
    [NotRequired] An object providing options for streaming responses.
    - Only applicable when the 'stream' field is set to true.
    - Allows configuration of streaming behavior (such as chunk sizes or timeouts).
    """

    include_usage: NotRequired[bool]
    """
    [NotRequired] When enabled, an additional chunk containing token usage statistics will be sent
    before the termination of the stream.
    - The usage field in this chunk shows the token consumption for the entire request.
    - The choices field in this chunk will always be an empty array.
    """

    temperature: NotRequired[float]
    """
    [NotRequired] Sampling temperature to control the randomness of the generated output.
    - Acceptable values range between 0 and 2.
    - Higher values (e.g., 0.8) yield more random outputs, while lower values (e.g., 0.2) produce more focused responses.
    - Default value: 1.
    """

    top_p: NotRequired[float]
    """
    [NotRequired] Nucleus sampling parameter that limits the tokens considered for generation.
    - Represents the cumulative probability threshold (e.g., 0.1 means only tokens comprising the top 10% probability mass are considered).
    - Default value: 1.
    """

    tools: NotRequired[List[Dict[str, Any]]]
    """
    [NotRequired] A list of tools that the model may call during response generation.
    - Each tool must be an object with the following structure:
        - type (str): Must be "function".
        - function (dict): An object describing the function, including:
            - description (str): A detailed explanation of what the function does.
            - name (str): The name of the function to be called.
            - parameters (dict): An object defining the parameters accepted by the function.
    - A maximum of 128 tools are supported.
    """

    tool_choice: NotRequired[Union[str, Dict[str, Any]]]
    """
    [NotRequired] Controls which (if any) tool is called by the model.
    - Can be specified as a string with possible values: "none", "auto", or "required".
    - Alternatively, it can be an object that forces the model to call a specific tool,
      e.g., {"type": "function", "function": {"name": "my_function"}}.
    - Default behavior is "none" if no tools are provided, or "auto" if tools are available.
    """

    logprobs: NotRequired[bool]
    """
    [NotRequired] When enabled (set to true), the API returns the log probabilities of each generated token.
    - This is useful for analyzing the model's token generation process.
    """

    top_logprobs: NotRequired[int]
    """
    [NotRequired] Specifies the number of most likely tokens to return at each token position.
    - Must be an integer between 0 and 20.
    - Requires that 'logprobs' is enabled (set to true) for this field to have an effect.
    """


# -------------------------------------------------------------------------------- #
# DeepSeek Chat Streaming Request
# -------------------------------------------------------------------------------- #

class DeepSeekRequestStreaming(DeepSeekRequestChat, total=False):
    """
    DeepSeekChatStreamingRequest represents a streaming chat request for the Deepseek model.

    This request extends the basic DeepSeekChatRequest by enabling streaming of partial message deltas.
    When used, the model will send tokens as they become available, until a final termination signal is sent.
    """

    stream: Required[Literal[True]]
    """
    [Required] A boolean flag that must be set to True to enable streaming responses.
    - When set, the model will send partial message deltas as server-sent events (SSE).
    - The streaming ends with a final "data: [DONE]" message.
    """


# -------------------------------------------------------------------------------- #
# DeepSeek Chat Structured Request
# -------------------------------------------------------------------------------- #

class DeepSeekRequestStructured(DeepSeekRequestChat, total=False):
    """
    DeepseekChatRequestStructured represents a structured chat request for the Deepseek model.

    This request is intended for cases where the output must adhere to a specific structure (JSON).
    It extends the basic DeepSeekChatRequest by enforcing that the response format is set to JSON.
    """

    response_format: Required[DeepseekResponseFormat]
    """
    [Required] An object specifying the output format for the response.
    - Must include a key "type" with the value "json_object".
    - This ensures that the model's output is a valid JSON object.
    - In structured requests, this setting is critical to avoid issues like unending whitespace or incomplete responses.
    """

# -------------------------------------------------------------------------------- #
# DeepSeek Type Aliases
# -------------------------------------------------------------------------------- #

# Embedding Request
class DeepSeekRequestEmbedding(TypedDict, total=False):
    """
    DeepSeekEmbeddingRequest represents an embedding request for the Deepseek model.

    This request object is used to create an embedding for a given text input.
    """
    model: Required[DeepSeekModels]
    """
    [Required] The identifier of the model to be used.
    - Possible values: "deepseek-chat" or "deepseek-reasoner".
    - Specifies which Deepseek model should process the chat request.
    """
    input: Required[str]
    """
    [Required] The text input to be embedded.
    """

# Chat Request
DeepSeekRequestChatType: TypeAlias = DeepSeekRequestChat

# Streaming Request
DeepSeekRequestStreamingType: TypeAlias = DeepSeekRequestStreaming

# Structured Request
DeepSeekRequestStructuredType: TypeAlias = DeepSeekRequestStructured

# Embedding Request
DeepSeekRequestEmbeddingType: TypeAlias = DeepSeekRequestEmbedding

# Union of all request types
DeepSeekRequestType: TypeAlias = Union[DeepSeekRequestChatType, DeepSeekRequestStreamingType, DeepSeekRequestStructuredType, DeepSeekRequestEmbeddingType]
