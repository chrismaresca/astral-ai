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
from typing import overload, Literal, Union, List, Optional


# Astral Messaging Types
from astral_ai.messaging._models import MessageList, Message, TextMessage, ImageMessage

# Astral Providers
from astral_ai.providers._generics import OpenAIClients, OpenAIResponse, StructuredOutputT, OpenAIRequestType, OpenAIResponseType
from astral_ai.providers._adapters import BaseCompletionAdapter

# OpenAI Message Types
from astral_ai._providersMOVE import OpenAIMessage

# Astral AI Types
from astral_ai._types._request import AstralCompletionRequest
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse




# ------------------------------------------------------------------------------
# OpenAI Message List to Provider Request
# ------------------------------------------------------------------------------

def convert_to_openai_message(message: Message) -> OpenAIMessage:
    """
    Convert a project message to an OpenAI message format.
    For example, a TextMessage is mapped to a dict with 'role' and 'content' keys.
    """
    if isinstance(message, TextMessage):
        # For OpenAI, we assume a text message has a 'role' and 'content'
        return OpenAIMessage({
            "role": message.role,
            "content": {
                "type": "text",
                "text": message.text,
            }
        })
    elif isinstance(message, ImageMessage):
        # For image messages, you might want to include additional keys
        return OpenAIMessage({
            "role": message.role,
            "image_url": message.image_url,
            "image_detail": message.image_detail,
        })
    else:
        raise TypeError("Unsupported project message type for OpenAI conversion.")


# # ------------------------------------------------------------------------------
# # OpenAI Request to Provider Request
# # ------------------------------------------------------------------------------


# def convert_to_openai_request(request: AstralCompletionRequest) -> OpenAIRequest:
#     """
#     Convert a project request to an OpenAI request.
#     """
#     # Placeholder conversion logic; implement as needed.
#     return {}


# # ------------------------------------------------------------------------------
# # OpenAI Response to Project Response
# # ------------------------------------------------------------------------------

# def convert_to_openai_chat_response(response: OpenAIChatResponse) -> AstralChatResponse:
#     """
#     Convert an OpenAI chat response to a project response.
#     """
#     return {}


# def convert_to_openai_structured_response(response: OpenAIStructuredResponse) -> AstralStructuredResponse:
#     """
#     Convert an OpenAI structured response to a project response.
#     """
#     return {}


# ------------------------------------------------------------------------------
# OpenAI Adapter Implementation
# ------------------------------------------------------------------------------


class OpenAICompletionAdapter(BaseCompletionAdapter[OpenAIRequestType, OpenAIResponseType]):
    """
    Adapter for the OpenAI provider.
    """

    # -------------------------------------------------------------------------- #
    # Overloads
    # -------------------------------------------------------------------------- #

    @overload
    def to_astral_completion_response(self, response: OpenAIResponse) -> AstralChatResponse:
        ...

    @overload
    def to_astral_completion_response(self, response: OpenAIResponse, response_model: StructuredOutputT) -> AstralStructuredResponse:
        ...

    # -------------------------------------------------------------------------- #
    # Implementation
    # -------------------------------------------------------------------------- #

    def to_provider_completion_request(self, request: AstralCompletionRequest) -> OpenAIRequestType:
        """
        Convert an AstralCompletionRequest into an OpenAIRequest.
        """
        pass

    def to_astral_completion_response(self, response: OpenAIResponse, response_model: Optional[StructuredOutputT] = None) -> Union[AstralChatResponse, AstralStructuredResponse]:
        """
        Convert an OpenAIResponse into an AstralChatResponse or AstralStructuredResponse.
        """
        pass
