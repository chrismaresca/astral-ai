# Built-in Types
from typing import Literal, Dict, TypeAlias


# ------------------------------------------------------------------------------
# Model Provider
# ------------------------------------------------------------------------------

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
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
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
        "model_ids": ['gpt-4o-2024-11-20', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13'],
        "pricing": {'prompt_tokens': 2.5, 'cached_prompt_tokens': 1.25, 'output_tokens': 10, 'per_million': 1000000},
        "most_recent_model": "gpt-4o-2024-11-20",
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

# ------------------------------------------------------------------------------
# Model Breakdown by Provider
# ------------------------------------------------------------------------------

OpenAIModels = Literal[
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
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

# ------------------------------------------------------------------------------
# Model Breakdown by Function
# ------------------------------------------------------------------------------


# OpenAI
ChatModels = Literal[
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
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
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
]

EmbeddingModels = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
]

# ------------------------------------------------------------------------------
# ALL MODEL NAMES
# Because every model is within the scope of the provider, we can use the
# union of the two types to create a single type that encompasses all models.
# ------------------------------------------------------------------------------

ModelName: TypeAlias = Literal[OpenAIModels, AnthropicModels]

# TODO: Run asserts to ensure that the model name is valid via Provider Mapping and Function Mapping.
# ------------------------------------------------------------------------------
# Model Name to Provider Mapping
# ------------------------------------------------------------------------------

PROVIDER_MODEL_NAMES: Dict[ModelName, ModelProvider] = {
    "gpt-4o-2024-11-20": "openai",
    "gpt-4o-2024-08-06": "openai",
    "gpt-4o-2024-05-13": "openai",
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
