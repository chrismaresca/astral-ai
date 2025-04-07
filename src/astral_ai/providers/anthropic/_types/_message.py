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

from anthropic.types.message_param import MessageParam

# ------------------------------------------------------------------------------
# OpenAI Message Types
# ------------------------------------------------------------------------------


# Message Type
AnthropicMessageType: TypeAlias = MessageParam
