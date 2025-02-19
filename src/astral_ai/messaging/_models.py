# -------------------------------------------------------------------------------- #
# message_template.py
# -------------------------------------------------------------------------------- #

# Built-in imports
from typing import Literal, Optional, List, Union, TypeAlias

# Pydantic imports
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------------- #
# Base Types
# -------------------------------------------------------------------------------- #

MessageRole = Literal["system", "user", "developer"]
ImageDetail = Literal["high", "low", "auto"]

MessageListType: TypeAlias = Union['MessageList', List['Message'], 'Message']


# -------------------------------------------------------------------------------- #
# Message Models
# -------------------------------------------------------------------------------- #

class TextMessage(BaseModel):
    """
    A text message model.
    """
    role: MessageRole = Field(
        default="user",
        description="The role of the message sender."
    )
    text: str = Field(
        ...,
        description="Plain text content for the message."
    )


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


Message: TypeAlias = Union[TextMessage, ImageMessage]


# -------------------------------------------------------------------------------- #
# Message List
# -------------------------------------------------------------------------------- #


class MessageList(BaseModel):
    """
    A list of messages.
    """
    messages: List[Union[TextMessage, ImageMessage]] = Field(
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
# Message List Type Alias
# -------------------------------------------------------------------------------- #
