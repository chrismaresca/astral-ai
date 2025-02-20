# ------------------------------------------------------------------------------
# Anthropic Message Types
# ------------------------------------------------------------------------------

"""
Anthropic Message Types for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in

from typing import (
    TypeAlias,
)

# OpenAI Types
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


# ------------------------------------------------------------------------------
# OpenAI Message Types
# ------------------------------------------------------------------------------


# Message Type
AnthropicMessageType: TypeAlias = ChatCompletionMessageParam
