# ------------------------------------------------------------------------------
# OpenAI Clients
# ------------------------------------------------------------------------------

# Built-in
from typing import TypeAlias, Union

# OpenAI Imports
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

# ------------------------------------------------------------------------------
# OpenAI Clients
# ------------------------------------------------------------------------------

OpenAISyncClientType: TypeAlias = OpenAI
OpenAIAsyncClientType: TypeAlias = AsyncOpenAI

OpenAIClientsType: TypeAlias = Union[OpenAISyncClientType, OpenAIAsyncClientType]

AzureOpenAISyncClientType: TypeAlias = AzureOpenAI
AzureOpenAIAsyncClientType: TypeAlias = AsyncAzureOpenAI

AzureOpenAIClientsType: TypeAlias = Union[AzureOpenAISyncClientType, AzureOpenAIAsyncClientType]
