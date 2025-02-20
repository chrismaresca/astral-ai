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
from typing import overload, Literal, Union, List, Optional, TypeVar

# Pydantic Types
from pydantic import BaseModel


# Astral Messaging Types
from astral_ai.messaging._models import (MessageList,
                                         Message,
                                         TextMessage,
                                         ImageMessage)


# Astral Base Adapters
from astral_ai.providers._base_adapters import BaseCompletionAdapter

# Astral AI Types
from astral_ai._types._request import AstralCompletionRequest
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# OpenAI Types
from ._types import (
    OpenAIRequestType,
    OpenAIResponseType,
)

# ------------------------------------------------------------------------------
# Generic Types
# ------------------------------------------------------------------------------

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


# ------------------------------------------------------------------------------
# OpenAI Message List to Provider Request
# ------------------------------------------------------------------------------

# def convert_to_openai_message(message: Message) -> OpenAIMessageType:
#     """
#     Convert a project message to an OpenAI message format.
#     For example, a TextMessage is mapped to a dict with 'role' and 'content' keys.
#     """
#     if isinstance(message, TextMessage):
#         # For OpenAI, we assume a text message has a 'role' and 'content'
#         return OpenAIMessage({
#             "role": message.role,
#             "content": {
#                 "type": "text",
#                 "text": message.text,
#             }
#         })
#     elif isinstance(message, ImageMessage):
#         # For image messages, you might want to include additional keys
#         return OpenAIMessage({
#             "role": message.role,
#             "image_url": message.image_url,
#             "image_detail": message.image_detail,
#         })
#     else:
#         raise TypeError("Unsupported project message type for OpenAI conversion.")


# ------------------------------------------------------------------------------
# OpenAI Completion Adapter
# ------------------------------------------------------------------------------

class OpenAICompletionAdapter(BaseCompletionAdapter[OpenAIRequestType, OpenAIResponseType]):
    """
    Adapter for the OpenAI provider.
    """

    # -------------------------------------------------------------------------- #
    # Overloads
    # -------------------------------------------------------------------------- #

    @overload
    def to_astral_completion_response(self, response: OpenAIResponseType) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(self, response: OpenAIResponseType, response_model: _StructuredOutputT) -> AstralStructuredResponse:
        ...

    # -------------------------------------------------------------------------- #
    # Implementation
    # -------------------------------------------------------------------------- #

    def to_provider_completion_request(self, request: AstralCompletionRequest) -> OpenAIRequestType:
        """
        Convert an AstralCompletionRequest into an OpenAIRequest.
        """
        pass

    def to_astral_completion_response(self, response: OpenAIResponseType, response_model: Optional[_StructuredOutputT] = None) -> Union[AstralChatResponse, AstralStructuredResponse]:
        """
        Convert an OpenAIResponse into an AstralChatResponse or AstralStructuredResponse.
        """
        pass
