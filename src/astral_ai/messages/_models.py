# -------------------------------------------------------------------------------- #
# message_template.py
# -------------------------------------------------------------------------------- #

# Built-in imports
from typing import Literal, Optional, List, Union, TypeAlias, Dict, TypedDict, Any, Annotated
import logging

# Pydantic imports
from pydantic import BaseModel, Field, BeforeValidator, ValidationError

# Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Base Types
# -------------------------------------------------------------------------------- #

MessageRole = Literal["assistant", "user"]
ImageDetail = Literal["high", "low", "auto"]

RawImageType = Literal["base64"]
RawImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

MessageListType: TypeAlias = Union['MessageList', List['Message'], 'Message']


# -------------------------------------------------------------------------------- #
# System Message
# -------------------------------------------------------------------------------- #

DEFAULT_SYSTEM_MESSAGE = None

SystemMessage: TypeAlias = Optional[str]

# -------------------------------------------------------------------------------- #
# Message Models
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Base Message
# -------------------------------------------------------------------------------- #


def handle_system_role(v: Any) -> Any:
    """Handle system role conversion."""

    if v == "system":
        logger.warning("The 'system' role is no longer supported in messages. "
                       "Please use the system_message parameter directly in your request instead. ")
        raise ValidationError("The 'system' role is no longer supported in messages. "
                              "Please use the system_message parameter directly in your request instead.")
    return v


class BaseMessageModel(BaseModel):
    """
    Base message model with common role field and initialization logic.
    """
    role: MessageRole = Annotated[MessageRole, Field(
        default="user",
        description="The role of the message sender. Must be either 'assistant' or 'user'."
    ), BeforeValidator(handle_system_role)]


# -------------------------------------------------------------------------------- #
# Text Message
# -------------------------------------------------------------------------------- #

class TextMessage(BaseMessageModel):
    """
    A text message model.
    """
    content: str = Field(
        ...,
        description="Plain text content for the message."
    )

# -------------------------------------------------------------------------------- #
# Remote Image Message
# -------------------------------------------------------------------------------- #


class RemoteImageMessage(BaseMessageModel):
    """
    A remote image message model.
    """
    image_url: str = Field(
        ...,
        description="The URL of an image."
    )
    image_detail: Optional[ImageDetail] = Field(
        default="auto",
        description="The detail level of the image."
    )


class LocalImageSource(BaseModel):
    """
    A local image source model.
    """
    type: RawImageType = Field(
        default="base64",
        description="The type of the image source."
    )
    media_type: RawImageMediaType = Field(
        default="image/jpeg",
        description="The media type of the image source."
    )
    data: str = Field(
        default="",
        description="The data of the base64 encoded image source."
    )
    image_detail: Optional[ImageDetail] = Field(
        default="auto",
        description="The detail level of the image."
    )


class LocalImageMessage(BaseMessageModel):
    """
    A local image message model.
    """
    source: LocalImageSource = Field(
        default_factory=LocalImageSource,
        description="The source of the image, including type, media type, and data."
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
# Message Dictionary
# -------------------------------------------------------------------------------- #

class MessageDict(TypedDict):
    """
    A TypedDict for message dictionaries that enforces the role field.
    """
    role: MessageRole
    content: str


# -------------------------------------------------------------------------------- #
# Message Type Alias
# -------------------------------------------------------------------------------- #

# Define the base message type
Message: TypeAlias = Union[TextMessage, RemoteImageMessage, LocalImageMessage, MessageDict]

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
# Validated Message Dict and List
# -------------------------------------------------------------------------------- #

# Message type literals
MessageType = Literal["text", "image_url", "image_base64"]

# Base class with common fields
class BaseValidatedMessageDict(TypedDict):
    """Base TypedDict for all message types with common fields."""
    role: Literal["assistant", "user", "system", "developer"]
    type: MessageType

class TextValidatedMessageDict(BaseValidatedMessageDict):
    """A TypedDict for text message dictionaries."""
    type: Literal["text"]
    content: str

class RemoteImageValidatedMessageDict(BaseValidatedMessageDict):
    """A TypedDict for remote image message dictionaries."""
    type: Literal["image_url"]
    image_url: str
    image_detail: ImageDetail

class LocalImageValidatedMessageDict(BaseValidatedMessageDict):
    """A TypedDict for local image message dictionaries."""
    type: Literal["image_base64"]
    image_data: str
    media_type: RawImageMediaType
    image_detail: ImageDetail

# Union type for all validated message types
ValidatedMessageDict = Union[
    TextValidatedMessageDict,
    RemoteImageValidatedMessageDict, 
    LocalImageValidatedMessageDict
]

class ValidatedMessageList(BaseModel):
    """
    A validated message list model.
    """
    messages: List[ValidatedMessageDict] = Field(
        ...,
        description="A list of messages."
    )

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def __len__(self):
        return len(self.messages)

    def to_validated_message_list(self) -> List[ValidatedMessageDict]:
        """
        Convert the ValidatedMessageList to a list of ValidatedMessageDict objects.
        """
        return self.messages

# -------------------------------------------------------------------------------- #
# Messages Type Alias
# -------------------------------------------------------------------------------- #


Messages: TypeAlias = Union[MessageList, List[Message], Message, ValidatedMessageList]
