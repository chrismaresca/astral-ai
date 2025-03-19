# ------------------------------------------------------------------------------
# Anthropic Clients
# ------------------------------------------------------------------------------

# Built-in
from typing import TypeAlias, Union

# Anthropic Imports
from anthropic import Anthropic as AnthropicSync, AsyncAnthropic as AnthropicAsync

# ------------------------------------------------------------------------------
# Anthropic Clients
# ------------------------------------------------------------------------------

AnthropicSyncClientType: TypeAlias = AnthropicSync
AnthropicAsyncClientType: TypeAlias = AnthropicAsync

AnthropicClientsType: TypeAlias = Union[AnthropicSync, AnthropicAsync]

