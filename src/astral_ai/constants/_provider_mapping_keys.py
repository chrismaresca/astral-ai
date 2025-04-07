# I need to import provider names, I need to import resource types.


# Type Aliases
from typing import TypeAlias, Dict

# Constants
from astral_ai.constants._models import ModelProvider, ResourceType

# Type Aliases
ProviderRequestMappingType: TypeAlias = Dict[ModelProvider, Dict[ResourceType, Dict[str, str]]]

AstralKey: TypeAlias = str
ProviderKey: TypeAlias = str

# -------------------------------------------------------------------------------- #
# Provider Request Mapping Keys
# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
# Provider Request Mapping Keys
# -------------------------------------------------------------------------------- #

PROVIDER_REQUEST_MAPPING_KEYS: ProviderRequestMappingType = {
    "openai": {
        "completion": {
            "model": "model",
            "messages": "messages",
            "frequency_penalty": "frequency_penalty",
            "logit_bias": "logit_bias",
            "logprobs": "logprobs",
            "max_completion_tokens": "max_completion_tokens",
            "max_tokens": "max_tokens",
            "metadata": "metadata",
            "modalities": "modalities",
            "n": "n",
            "parallel_tool_calls": "parallel_tool_calls",
            "prediction": "prediction",
            "presence_penalty": "presence_penalty",
            "reasoning_effort": "reasoning_effort",
            "response_format": "response_format",
            "seed": "seed",
            "service_tier": "service_tier",
            "stop": "stop",
            "store": "store",
            "stream_options": "stream_options",
            "temperature": "temperature",
            "tool_choice": "tool_choice",
            "tools": "tools",
            "top_logprobs": "top_logprobs",
            "top_p": "top_p",
            "user": "user",
            "timeout": "timeout",
        },
    },
    "anthropic": {
        "completion": {
            "model": "model",
            "system_message": "system",
            "messages": "messages",
            "max_tokens": "max_tokens",
            "stop": "stop_sequences",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "metadata": "metadata",
            "tool_choice": "tool_choice",
            "tools": "tools",
            "user": "user",
            "timeout": "timeout",
        },
    },
    "deepseek": {
        "completion": {
            "model": "model",
            "messages": "messages",
            "frequency_penalty": "frequency_penalty",
            "max_tokens": "max_tokens",
            "presence_penalty": "presence_penalty",
            "response_format": "response_format",
            "stop": "stop",
            "stream": "stream",
            "stream_options": "stream_options",
            "include_usage": "include_usage",
            "temperature": "temperature",
            "top_p": "top_p",
            "tools": "tools",
            "tool_choice": "tool_choice",
            "logprobs": "logprobs",
            "top_logprobs": "top_logprobs",
        },
    },
}

# -------------------------------------------------------------------------------- #
# Provider Response Mapping Keys
# -------------------------------------------------------------------------------- #


