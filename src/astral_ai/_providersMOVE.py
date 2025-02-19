# Project Types
from typing import TypeAlias, Union, TypeVar, Mapping, Callable, Any, overload, Literal, Iterable, List

# Project Models
from astral_ai._models import ModelProvider, ModelName, OpenAIModels, AnthropicModels, get_provider_from_model_name

# Project Utils
from astral_ai.messaging._utils import handle_no_messages, standardize_messages

# Project Message Types
from astral_ai.messaging._models import Message, MessageList, TextMessage, ImageMessage

# ------------------------------------------------------------------------------
# Provider Imports
# ------------------------------------------------------------------------------

# OpenAI Clients
from openai import OpenAI, AsyncOpenAI

# Azure OpenAI Clients
from openai import AzureOpenAI, AsyncAzureOpenAI

# TODO: Add support for Anthropic
# from anthropic import Anthropic, AsyncAnthropic

# ------------------------------------------------------------------------------
# Provider Message Types
# ------------------------------------------------------------------------------
# OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam as OpenAIMessage

# TODO: Add support for Anthropic

# ------------------------------------------------------------------------------
# Provider Message Converters
# ------------------------------------------------------------------------------
# OpenAI
from astral_ai.providers.openai import convert_to_openai_message

# TODO: Add support for Anthropic
# from astral_ai.providers.anthropic.messages import convert_to_anthropic_message


# ------------------------------------------------------------------------------
# Model Provider Clients
# ------------------------------------------------------------------------------

# OpenAI Clients
OpenAIClients: TypeAlias = Union[OpenAI, AsyncOpenAI]

# Azure OpenAI Clients
AzureOpenAIClients: TypeAlias = Union[AzureOpenAI, AsyncAzureOpenAI]

# Model Provider Clients
ModelProviderClient: TypeAlias = Union[OpenAIClients, AzureOpenAIClients]


# ------------------------------------------------------------------------------
# Model Provider Client Types
# ------------------------------------------------------------------------------

# OpenAI Clients Types
OpenAIClientT = TypeVar("OpenAIClientT", bound=OpenAIClients)

# Azure OpenAI Clients Types
AzureOpenAIClientT = TypeVar("AzureOpenAIClientT", bound=AzureOpenAIClients)

# Model Provider Clients Types
ModelProviderClientT = TypeVar("ModelProviderClientT", bound=ModelProviderClient)

# ------------------------------------------------------------------------------
# Provider Message Converters Types
# ------------------------------------------------------------------------------


# TODO: Add support for Anthropic. This is a placeholder for now.
# Anthropic Message Converter Type
# AnthropicMessageConverter = TypeVar("AnthropicMessageConverter", bound=Callable[[Message], AnthropicMessage])
AnthropicMessage: TypeAlias = dict

# Union alias for any provider message.
ProviderMessage = Union[OpenAIMessage, AnthropicMessage]

# Provider Message Type
ProviderMessageT = TypeVar("ProviderMessageT", bound=ProviderMessage)


def convert_to_anthropic_message(message: Message) -> AnthropicMessage:
    """
    Convert a project message to an Anthropic message.
    """
    return {}

# ------------------------------------------------------------------------------
# Provider Message Converters
# ------------------------------------------------------------------------------


PROVIDER_MESSAGE_CONVERTERS: Mapping[ModelProvider, Callable[[Message], ProviderMessage]] = {
    "openai": convert_to_openai_message,
    "azureOpenAI": convert_to_openai_message,
    "anthropic": convert_to_anthropic_message,
}


# ------------------------------------------------------------------------------
# Get Provider Message Converter From Model Provider
# ------------------------------------------------------------------------------


@overload
def get_provider_message_converter(
    model_provider: Literal["openai"]
) -> Callable[[Message], OpenAIMessage]:
    ...


@overload
def get_provider_message_converter(
    model_provider: Literal["azureOpenAI"]
) -> Callable[[Message], OpenAIMessage]:
    ...


@overload
def get_provider_message_converter(
    model_provider: Literal["anthropic"]
) -> Callable[[Message], AnthropicMessage]:
    ...


def get_provider_message_converter(model_provider: ModelProvider) -> Callable[[Message], ProviderMessage]:
    """
    Get the provider message converter for a given model provider.
    """
    return PROVIDER_MESSAGE_CONVERTERS[model_provider]


# ------------------------------------------------------------------------------
# Provider Message Converters From Model Name
# ------------------------------------------------------------------------------

# TODO: MOVE THIS

@overload
def astral_messages_to_provider_messages(
    model_name: OpenAIModels,
    messages: Union[MessageList, List[Message], Message],
    init_call: bool = True,
) -> List[OpenAIMessage]:
    ...


@overload
def astral_messages_to_provider_messages(
    model_name: AnthropicModels,
    messages: Union[MessageList, List[Message], Message],
    init_call: bool = True,
) -> List[AnthropicMessage]:
    ...


def astral_messages_to_provider_messages(
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

    # Determine the provider.
    provider_name = get_provider_from_model_name(model_name)

    # Select the correct message converter based on the provider.
    message_converter = get_provider_message_converter(provider_name)
    # Convert each message using the selected converter.
    return [message_converter(message) for message in standardized_messages]


