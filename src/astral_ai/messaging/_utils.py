# -------------------------------------------------------------------------------- #
# Agent Utils
# -------------------------------------------------------------------------------- #

"""
Utils for the agents.
"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import Optional, List, Union, Tuple

# Astral AI Constants
from astral_ai.constants._models import ModelName

# Astral AI Messaging
from astral_ai.messaging._models import Message, MessageList

# Astral AI Exceptions
from astral_ai.exceptions import MessagesNotProvidedError, InvalidMessageError

# Astral AI Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Message Handling Utils
# -------------------------------------------------------------------------------- #


def handle_no_messages(model_name: ModelName, init_call: bool) -> Optional[None]:
    """
    Handle the case where no messages are provided.
    """
    if init_call:
        logger.warning(f"No messages provided for model during initialization of model {model_name}.\n"
                       "You must provide messages in each call to the model.")
        return None
    else:
        raise MessagesNotProvidedError(f"Messages must be provided to run the model {model_name}.")


def standardize_messages(messages: Union[MessageList, List[Message], Message]) -> List[Message]:
    """
    Standardize the messages to a list of Message instances.
    """

    log_message = "Standardizing messages to a list of messages.\nCurrent Message Type: {message_type}"

    # Standardize to a list of messages.
    if isinstance(messages, Message):
        logger.debug(log_message.format(message_type="`Message`"))
        return [messages]
    elif isinstance(messages, MessageList):
        logger.debug(log_message.format(message_type="`MessageList`"))
        return list(messages)
    elif isinstance(messages, list):
        logger.debug(f"Messages are already a list. Will validate each message later.")
        return messages
    else:
        raise InvalidMessageError(message_type=f"`{type(messages)}`")


def count_message_roles(messages: List[Message]) -> Tuple[int, int]:
    """
    Helper function to count system and developer messages in a single pass.

    Args:
        messages (List[Message]): List of messages to count roles for

    Returns:
        Tuple[int, int]: Count of (system_messages, developer_messages)
    """
    system_count = 0
    developer_count = 0

    for msg in messages:
        if msg.role == "system":
            system_count += 1
        elif msg.role == "developer":
            developer_count += 1

    return system_count, developer_count


def convert_message_roles(messages: List[Message], target_role: str, model_name: ModelName) -> None:
    """
    Helper function to convert message roles in-place.

    Args:
        messages (List[Message]): List of messages to convert.
        target_role (str): Role to convert messages to ("system", "developer", or "user").
    """
    # When converting to system or developer, we define the expected source role as the opposite.
    if target_role in ("system", "developer"):
        source_role = "developer" if target_role == "system" else "system"
        for msg in messages:
            if msg.role == source_role:
                logger.warning(
                    f"Incorrect message role provided for model {model_name}. "
                    f"{model_name} does not support {source_role} messages. "
                    f"Converting message role from {source_role} to {target_role}."
                )
                msg.role = target_role

    # When converting to "user", convert any message that isn't already a user message.
    elif target_role == "user":
        for msg in messages:
            if msg.role in ("system", "developer"):
                logger.warning(
                    f"Incorrect message role provided for model {model_name}. "
                    f"{model_name} does not support {msg.role} messages. "
                    f"Converting message role from {msg.role} to user."
                )
                msg.role = "user"
