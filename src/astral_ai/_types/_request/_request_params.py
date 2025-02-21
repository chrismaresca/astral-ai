# -------------------------------------------------------------------------------- #
# Request Params
# -------------------------------------------------------------------------------- #

"""
Request Params for Astral AI
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import (
    TypeAlias,
    Optional,
    Literal,
    Dict,
    Any,
    Union,
    Iterable,
    TypedDict,
    Required,
)


# ------------------------------------------------------------------------------
# Modality
# ------------------------------------------------------------------------------


Modality: TypeAlias = Optional[Literal["text", "audio"]]


# ------------------------------------------------------------------------------
# Stream Options
# ------------------------------------------------------------------------------

class StreamOptions(TypedDict, total=False):
    include_usage: bool
    """If set, an additional chunk will be streamed before the `data: [DONE]` message.

    The `usage` field on this chunk shows the token usage statistics for the entire
    request, and the `choices` field will always be an empty array. All other chunks
    will also include a `usage` field, but with a null value.
    """


# ------------------------------------------------------------------------------
# Reasoning Effort
# ------------------------------------------------------------------------------
ReasoningEffort: TypeAlias = Optional[Literal["low", "medium", "high"]]


# ------------------------------------------------------------------------------
# Response Format
# ------------------------------------------------------------------------------


class ResponseFormatJSONSchema(TypedDict, total=False):
    type: Required[Literal["json_schema"]]
    json_schema: Required[Dict[str, Any]]


class ResponseFormatJSONObject(TypedDict, total=False):
    type: Required[Literal["json_object"]]


class ResponseFormatText(TypedDict, total=False):
    type: Required[Literal["text"]]


ResponseFormat: TypeAlias = Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema]


# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------


class ToolDefinition(TypedDict, total=False):
    name: Required[str]
    """The name of the function to be called.

    Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length
    of 64.
    """

    description: str
    """
    A description of what the function does, used by the model to choose when and
    how to call the function.
    """

    parameters: Dict[str, object]
    """The parameters the functions accepts, described as a JSON Schema object.

    See the [guide](https://platform.openai.com/docs/guides/function-calling) for
    examples, and the
    [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
    documentation about the format.

    Omitting `parameters` defines a function with an empty parameter list.
    """

    strict: Optional[bool]
    """Whether to enable strict schema adherence when generating the function call.

    If set to true, the model will follow the exact schema defined in the
    `parameters` field. Only a subset of JSON Schema is supported when `strict` is
    `true`. Learn more about Structured Outputs in the
    [function calling guide](docs/guides/function-calling).
    """


class Tool(TypedDict, total=False):
    function: Required[ToolDefinition]
    """The function to call."""

    type: Required[Literal["function"]]
    """The type of the tool. Currently, only `function` is supported."""


# ------------------------------------------------------------------------------
# Tool Choice
# ------------------------------------------------------------------------------


class ToolAlias(TypedDict, total=True):
    name: Required[str]
    """The name of the tool to call."""


class SpecificToolSelection(TypedDict, total=False):
    function: Required[ToolAlias]

    type: Required[Literal["function"]]
    """The type of the tool. Currently, only `function` is supported."""


ToolChoice: TypeAlias = Union[
    Literal["none", "auto", "required"], SpecificToolSelection
]


# ------------------------------------------------------------------------------
# Prediction Content
# ------------------------------------------------------------------------------


class TextContentPart(TypedDict, total=False):
    text: Required[str]
    """The text content."""

    type: Required[Literal["text"]]
    """The type of the content part."""


class ResponsePrediction(TypedDict, total=False):
    content: Required[Union[str, Iterable[TextContentPart]]]
    """
    The content that should be matched when generating a model response. If
    generated tokens would match this content, the entire model response can be
    returned much more quickly.
    """

# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------


Metadata: TypeAlias = Dict[str, str]
