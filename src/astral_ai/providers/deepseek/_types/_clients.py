# ------------------------------------------------------------------------------
# DeepSeek Clients
# ------------------------------------------------------------------------------

# Built-in
from typing import TypeAlias, Union

# OpenAI Imports
# IMPORTANT: We use the OpenAI client for DeepSeek
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

# ------------------------------------------------------------------------------
# DeepSeek Clients
# ------------------------------------------------------------------------------

DeepSeekSyncClientType: TypeAlias = Union[OpenAI, AzureOpenAI]
DeepSeekAsyncClientType: TypeAlias = Union[AsyncOpenAI, AsyncAzureOpenAI]
