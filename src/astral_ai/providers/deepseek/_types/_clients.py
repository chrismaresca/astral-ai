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


DeepSeekClientsType: TypeAlias = Union[OpenAI, AsyncOpenAI]

# TODO: Verify if this is correct??
DeepSeekAzureClientsType: TypeAlias = Union[AzureOpenAI, AsyncAzureOpenAI]
