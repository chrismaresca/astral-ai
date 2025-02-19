# Built-in Types
from typing import Literal, Dict, TypeAlias, Mapping, Callable

# Message Types
from astral_ai.messaging._models import Message


# Auto-generated types and constants

ModelProvider = Literal[
    "openai",
    "anthropic",
    "azureOpenAI",
]


ModelAlias = Literal[
    "gpt-4o",
    "o1",
    "o1-mini",
    "o3-mini",
    "claude-3-5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
]

ModelId = Literal[
    "gpt-4o-01-15-24",
    "gpt-4o-12-17-24",
    "gpt-4o-01-10-24",
    "o1-01-15-24",
    "o1-12-17-24",
    "o1-01-10-24",
    "o1-mini-01-15-24",
    "o1-mini-12-17-24",
    "o1-mini-01-10-24",
    "o3-mini-2025-01-31",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
]


MODEL_DEFINITIONS = {
    "gpt-4o": {
        "provider": "openai",
        "model_type": "llm",
        "model_ids": ['gpt-4o-01-15-24', 'gpt-4o-12-17-24', 'gpt-4o-01-10-24'],
        "pricing": {'prompt_tokens': 2.5, 'cached_prompt_tokens': 1.25, 'output_tokens': 10, 'per_million': 1000000},
        "most_recent_model": "gpt-4o-01-15-24",
    },
    "o1": {
        "provider": "openai",
        "model_type": "llm",
        "model_ids": ['o1-01-15-24', 'o1-12-17-24', 'o1-01-10-24'],
        "pricing": {'prompt_tokens': 15, 'cached_prompt_tokens': 7.5, 'output_tokens': 60, 'per_million': 1000000},
        "most_recent_model": "o1-01-15-24"
    },
    "o1-mini": {
        "provider": "openai",
        "model_type": "llm",
        "model_ids": ['o1-mini-01-15-24', 'o1-mini-12-17-24', 'o1-mini-01-10-24'],
        "pricing": {'prompt_tokens': 3.0, 'cached_prompt_tokens': 1.5, 'output_tokens': 12, 'per_million': 1000000},
        "most_recent_model": "o1-mini-01-15-24"
    },
    "o3-mini": {
        "provider": "openai",
        "model_type": "llm",
        "model_ids": ['o3-mini-2025-01-31'],
        "pricing": {'prompt_tokens': 0.0, 'cached_prompt_tokens': 0.0, 'output_tokens': 0.0, 'per_million': 1000000},
        "most_recent_model": "o3-mini-2025-01-31"
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "model_type": "llm",
        "model_ids": ['claude-3-5-sonnet-20241022'],
        "pricing": {'prompt_tokens': 0.0, 'cached_prompt_tokens': 0.0, 'output_tokens': 0.0, 'per_million': 1000000},
        "most_recent_model": "claude-3-5-sonnet-20241022"
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "model_type": "llm",
        "model_ids": ['claude-3-opus-20240229'],
        "pricing": {'prompt_tokens': 0.0, 'cached_prompt_tokens': 0.0, 'output_tokens': 0.0, 'per_million': 1000000},
        "most_recent_model": "claude-3-opus-20240229"
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "model_type": "llm",
        "model_ids": ['claude-3-5-haiku-20241022'],
        "pricing": {'prompt_tokens': 0.0, 'cached_prompt_tokens': 0.0, 'output_tokens': 0.0, 'per_million': 1000000},
        "most_recent_model": "claude-3-5-haiku-20241022"
    },
}


OpenAIModels = Literal[
    "gpt-4o-01-15-24",
    "gpt-4o-12-17-24",
    "gpt-4o-01-10-24",
    "o1-01-15-24",
    "o1-12-17-24",
    "o1-01-10-24",
    "o1-mini-01-15-24",
    "o1-mini-12-17-24",
    "o1-mini-01-10-24",
    "o3-mini-2025-01-31",
    "gpt-4o",
    "o1",
    "o1-mini",
    "o3-mini",
]

AnthropicModels = Literal[
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
]

ModelName: TypeAlias = Literal[OpenAIModels, AnthropicModels]

PROVIDER_MODEL_NAMES: Dict[ModelName, ModelProvider] = {
    "gpt-4o-01-15-24": "openai",
    "gpt-4o-12-17-24": "openai",
    "gpt-4o-01-10-24": "openai",
    "o1-01-15-24": "openai",
    "o1-12-17-24": "openai",
    "o1-01-10-24": "openai",
    "o1-mini-01-15-24": "openai",
    "o1-mini-12-17-24": "openai",
    "o1-mini-01-10-24": "openai",
    "o3-mini-2025-01-31": "openai",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "gpt-4o": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o3-mini": "openai",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-opus": "anthropic",
    "claude-3-haiku": "anthropic",
}

# ------------------------------------------------------------
# Get Provider from Model Name
# ------------------------------------------------------------


def get_provider_from_model_name(model_name: ModelName) -> ModelProvider:
    """
    Get the provider from a model name.
    """
    return PROVIDER_MODEL_NAMES[model_name]

# ------------------------------------------------------------
# Is Model Alias
# ------------------------------------------------------------


def is_model_alias(model_name: ModelName) -> bool:
    """
    Check if a model name is a model alias.
    """
    return model_name in MODEL_DEFINITIONS

# ------------------------------------------------------------
# Get Model from Model Alias
# ------------------------------------------------------------


def get_model_from_model_alias(model_alias: ModelAlias) -> ModelId:
    """
    Get the model from a model alias.
    """
    return MODEL_DEFINITIONS[model_alias]["most_recent_model"]
