# ------------------------------------------------------------------------------
# Anthropic Clients
# ------------------------------------------------------------------------------

# Built-in
from typing import TypeAlias, Union

# Anthropic Imports
from openai import OpenAI as Anthropic, AsyncOpenAI as AsyncAnthropic

# ------------------------------------------------------------------------------
# Anthropic Clients
# ------------------------------------------------------------------------------


AnthropicClientsType: TypeAlias = Union[Anthropic, AsyncAnthropic]
