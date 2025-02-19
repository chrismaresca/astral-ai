"""
Provider Message Conversion

This module contains:
  - Overloads and implementation for converting project messages
    into provider-specific messages.
"""

# TODO: MOVE THIS

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

from typing import Union, List, Iterable, overload

# Project Models
from astral_ai._models import (
    ModelName,
    OpenAIModels,
    AnthropicModels,
    get_provider_from_model_name,
)

# Project Utils
from astral_ai.messaging._utils import handle_no_messages, standardize_messages

# Project Message Types
from astral_ai.messaging._models import Message, MessageList

# Provider Message Types and Converter (from our converters module)
from astral_ai.providers._generics import (
    OpenAIMessage,
    AnthropicMessage,
    ProviderMessage,
)

# ------------------------------------------------------------------------------
# Provider Message Converters From Model Name
# ------------------------------------------------------------------------------

"""
Overloads for converting project messages into provider-specific messages.
"""

# ------------------------------------------------------------------------------
# Overloads
# ------------------------------------------------------------------------------


@overload
def to_provider_messages(
    model_name: OpenAIModels,
    messages: Union[MessageList, List[Message], Message],
    init_call: bool = True,
) -> List[OpenAIMessage]:
    ...


@overload
def to_provider_messages(
    model_name: AnthropicModels,
    messages: Union[MessageList, List[Message], Message],
    init_call: bool = True,
) -> List[AnthropicMessage]:
    ...


# ------------------------------------------------------------------------------
# Implementation
# ------------------------------------------------------------------------------

# TODO: probably do not need this function
def to_provider_messages(
    model_name: ModelName,
    messages: Union[MessageList, List[Message], Message],
    init_call: bool = True,
) -> Iterable[ProviderMessage]:
    """
    Convert a list of project messages to a provider-specific request.
    The return type is determined by the provider associated with the model_name.
    """
    if messages is None:
        handle_no_messages(model_name=model_name, init_call=init_call)

    # Standardize the messages to a list.
    standardized_messages = standardize_messages(messages)

    # Determine the provider from the model name.
    provider_name = get_provider_from_model_name(model_name)

    # Select the correct message converter based on the provider.
    message_converter = get_provider_message_converter(provider_name)

    # Convert each message using the selected converter.
    return [message_converter(message) for message in standardized_messages]
