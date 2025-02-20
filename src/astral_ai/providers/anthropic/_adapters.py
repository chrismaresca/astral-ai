# # ------------------------------------------------------------------------------
# # Anthropic Message Handlers
# # ------------------------------------------------------------------------------

# """
# Message Handlers for Anthropic.
# """

# # ------------------------------------------------------------------------------
# # Imports
# # ------------------------------------------------------------------------------

# # Built-in Types
# from typing import overload, Literal, Union, List

# # Astral Models
# from astral_ai.constants._models import OpenAIModels

# # Astral Messaging Types
# from astral_ai.messaging._models import MessageList, Message, TextMessage, ImageMessage


# # Anthropic Message Types
# from astral_ai._providers import AnthropicMessage

# # Astral AI Types
# from astral_ai._types._request import AstralCompletionRequest
# from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# # Anthropic Request Types
# from ._types import AnthropicRequest, AnthropicChatResponse, AnthropicStructuredResponse


# # ------------------------------------------------------------------------------
# # OpenAI Message List to Provider Request
# # ------------------------------------------------------------------------------

# def convert_to_anthropic_message(message: Message) -> AnthropicMessage:
#     """
#     Convert a project message to an Anthropic message format.
#     For example, a TextMessage is mapped to a dict with 'role' and 'content' keys.
#     """
#     if isinstance(message, TextMessage):
#         # For OpenAI, we assume a text message has a 'role' and 'content'
#         return AnthropicMessage({
#             "role": message.role,
#             "content": {
#                 "type": "text",
#                 "text": message.text,
#             }
#         })
#     elif isinstance(message, ImageMessage):
#         # For image messages, you might want to include additional keys
#         return AnthropicMessage({
#             "role": message.role,
#             "image_url": message.image_url,
#             "image_detail": message.image_detail,
#         })
#     else:
#         raise TypeError("Unsupported project message type for Anthropic conversion.")


# # ------------------------------------------------------------------------------
# # Anthropic Request to Provider Request
# # ------------------------------------------------------------------------------


# def convert_to_anthropic_request(request: AstralCompletionRequest) -> AnthropicRequest:
#     """
#     Convert a project request to an Anthropic request.
#     """
#     # Placeholder conversion logic; implement as needed.
#     return {}


# # ------------------------------------------------------------------------------
# # Anthropic Response to Project Response
# # ------------------------------------------------------------------------------

# def convert_to_anthropic_chat_response(response: AnthropicChatResponse) -> AstralChatResponse:
#     """
#     Convert an Anthropic chat response to a project response.
#     """
#     return {}


# def convert_to_anthropic_structured_response(response: AnthropicStructuredResponse) -> AstralStructuredResponse:
#     """
#     Convert an Anthropic structured response to a project response.
#     """
#     return {}


# ------------------------------------------------------------------------------
# OpenAI Message Handlers
# ------------------------------------------------------------------------------

"""
Message Handlers for OpenAI.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in Types
from typing import overload, Literal, Union, List, Optional, TypeAlias


# Astral Messaging Types
from astral_ai.messaging._models import (MessageList,
                                         Message,
                                         TextMessage,
                                         ImageMessage)

# Astral Providers
from astral_ai.providers._generics import (AnthropicRequestType,
                                           AnthropicResponseType,
                                           StructuredOutputT)

# Astral Base Adapters
from astral_ai.providers._base_adapters import BaseCompletionAdapter

# Astral AI Types
from astral_ai._types._request import AstralCompletionRequest
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# OpenAI Types
# TODO: is this right?
from ._types import (
    AnthropicRequestType,
    AnthropicResponseType,

)


class Anthropic:
    pass


class AsyncAnthropic:
    pass


AnthropicClients: TypeAlias = Union[Anthropic, AsyncAnthropic]


class AnthropicCompletionAdapter(BaseCompletionAdapter[AnthropicClients, AnthropicRequestType, AnthropicResponseType]):
    """
    Adapter for the Anthropic provider.
    """

    # -------------------------------------------------------------------------- #
    # Overloads
    # -------------------------------------------------------------------------- #

    @overload
    def to_astral_completion_response(self, response: AnthropicResponseType) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(self, response: AnthropicResponseType, response_model: StructuredOutputT) -> AstralStructuredResponse:
        ...

    # -------------------------------------------------------------------------- #
    # Implementation
    # -------------------------------------------------------------------------- #

    def to_provider_completion_request(self, request: AstralCompletionRequest) -> AnthropicRequestType:
        """
        Convert an AstralCompletionRequest into an AnthropicRequest.
        """
        pass

    def to_astral_completion_response(self, response: AnthropicResponseType, response_model: Optional[StructuredOutputT] = None) -> Union[AstralChatResponse, AstralStructuredResponse]:
        """
        Convert an AnthropicResponse into an AstralChatResponse or AstralStructuredResponse.
        """
        pass
