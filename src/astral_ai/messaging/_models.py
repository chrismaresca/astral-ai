# -------------------------------------------------------------------------------- #
# message_template.py
# -------------------------------------------------------------------------------- #

# Built-in imports
from typing import Literal, Optional, List, Union, TypeAlias, Dict

# Pydantic imports
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------------- #
# Base Types
# -------------------------------------------------------------------------------- #

MessageRole = Literal["system", "user"]
ImageDetail = Literal["high", "low", "auto"]

MessageListType: TypeAlias = Union['MessageList', List['Message'], 'Message']


# -------------------------------------------------------------------------------- #
# Message Models
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Text Message
# -------------------------------------------------------------------------------- #

class TextMessage(BaseModel):
    """
    A text message model.
    """
    role: MessageRole = Field(
        default="user",
        description="The role of the message sender."
    )
    content: str = Field(
        ...,
        description="Plain text content for the message."
    )

# -------------------------------------------------------------------------------- #
# Image Message
# -------------------------------------------------------------------------------- #


class ImageMessage(BaseModel):
    """
    An image message model.
    """
    role: MessageRole = Field(
        default="user",
        description="The role of the message sender."
    )
    image_url: str = Field(
        ...,
        description="The URL of an image."
    )
    image_detail: Optional[ImageDetail] = Field(
        default="auto",
        description="The detail level of the image."
    )

# -------------------------------------------------------------------------------- #
# Audio Message
# -------------------------------------------------------------------------------- #

# TODO: implement audio message


class AudioMessage(BaseModel):
    """
    An audio message model.
    """
    pass


# -------------------------------------------------------------------------------- #
# Message Type Alias
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Message Type Alias
# -------------------------------------------------------------------------------- #

# Define the base message type
Message: TypeAlias = Union[TextMessage, ImageMessage, Dict[str, str]]

# -------------------------------------------------------------------------------- #
# Message List
# -------------------------------------------------------------------------------- #


class MessageList(BaseModel):
    """
    A list of messages.
    """
    messages: List[Message] = Field(
        ...,
        description="A list of messages."
    )

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def __len__(self):
        return len(self.messages)


# -------------------------------------------------------------------------------- #
# Messages Type Alias
# -------------------------------------------------------------------------------- #

Messages: TypeAlias = Union[MessageList, List[Message], List[Dict[str, str]], Message]
