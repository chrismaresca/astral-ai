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


OpenAIClientsType: TypeAlias = Union[OpenAI, AsyncOpenAI]

AzureOpenAIClientsType: TypeAlias = Union[AzureOpenAI, AsyncAzureOpenAI]
