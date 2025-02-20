# ------------------------------------------------------------------------------
# Anthropic Message Handlers
# ------------------------------------------------------------------------------

"""
Message Handlers for Anthropic.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in Types
from typing import overload, Literal, Union, List

# Astral Models
from astral_ai.constants._models import AnthropicModels

# Astral Messaging Types
from astral_ai.messaging._models import MessageList, Message, TextMessage, ImageMessage

# Anthropic Message Types
from astral_ai._providersMOVE import AnthropicMessage


# ------------------------------------------------------------------------------
# Anthropic Message Converters
# ------------------------------------------------------------------------------


def convert_to_anthropic_message(message: Message) -> AnthropicMessage:
    """
    Convert a project message to an Anthropic message format.
    For example, a TextMessage is mapped to a dict with 'role' and 'content' keys.
    """
    if isinstance(message, TextMessage):
        # For OpenAI, we assume a text message has a 'role' and 'content'
        return AnthropicMessage({
            "role": message.role,
            "content": {
                "type": "text",
                "text": message.text,
            }
        })
    elif isinstance(message, ImageMessage):
        # For image messages, you might want to include additional keys
        return AnthropicMessage({
            "role": message.role,
            "image_url": message.image_url,
            "image_detail": message.image_detail,
        })
    else:
        raise TypeError("Unsupported project message type for Anthropic conversion.")
