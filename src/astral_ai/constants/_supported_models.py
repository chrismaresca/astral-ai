# -------------------------------------------------------------------------------- #
# Supported Models Constants
# -------------------------------------------------------------------------------- #

"""
Constants for supported models.

This module contains data structures for efficiently checking model capabilities
with O(1) lookup time.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Dict, TypedDict, Optional, Union, Literal

# module imports
from astral_ai.constants._models import ChatModels, ModelId, ModelAlias


FEATURE_NAME = Literal[
    "reasoning_effort",
    "structured_output",
    "json_mode",
    "image_ingestion",
    "function_calls",
    "system_message",
    "developer_message",
    "only_user_message",
]

# -------------------------------------------------------------------------------- #
# Type Definitions
# -------------------------------------------------------------------------------- #


class ModelCapabilities(TypedDict, total=False):
    """TypedDict for capturing model capabilities."""
    supports_reasoning_effort: bool
    supports_structured_output: bool
    supports_json_mode: bool
    supports_image_ingestion: bool
    supports_function_calls: bool
    supports_system_message: bool
    supports_developer_message: bool
    supports_only_user_message: bool

# -------------------------------------------------------------------------------- #
# Model Mappings
# -------------------------------------------------------------------------------- #


# Map from specific model IDs to their aliases
MODEL_ALIASES: Dict[ModelId, ModelAlias] = {
    # Anthropic models
    "claude-3-5-haiku-20241022": "claude-3-haiku",
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
    "claude-3-opus-20240229": "claude-3-opus",

    # OpenAI models
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o",
    "o1-01-15-24": "o1",
    "o1-12-17-24": "o1",
    "o1-01-10-24": "o1",
    "o1-mini-01-15-24": "o1-mini",
    "o1-mini-12-17-24": "o1-mini",
    "o1-mini-01-10-24": "o1-mini",
    "o3-mini-2025-01-31": "o3-mini",
}

# Map from aliases to their most recent specific model IDs
ALIAS_TO_MODEL_ID: Dict[ModelAlias, ModelId] = {
    # Anthropic models
    "claude-3-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",

    # OpenAI models
    "gpt-4o": "gpt-4o-2024-11-20",
    "o1": "o1-01-15-24",
    "o1-mini": "o1-mini-01-15-24",
    "o3-mini": "o3-mini-2025-01-31",

    # DeepSeek models
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}

# -------------------------------------------------------------------------------- #
# Model Capabilities
# -------------------------------------------------------------------------------- #

# Main dictionary for O(1) capability lookups
MODEL_CAPABILITIES: Dict[ChatModels, ModelCapabilities] = {
    # Anthropic models
    "claude-3-5-haiku-20241022": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "claude-3-haiku": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "claude-3-5-sonnet-20241022": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "claude-3-5-sonnet": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "claude-3-opus-20240229": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "claude-3-opus": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },

    # OpenAI models
    "gpt-4o-2024-11-20": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "gpt-4o-2024-08-06": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "gpt-4o-2024-05-13": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "gpt-4o": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": True,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "o1-01-15-24": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "o1-12-17-24": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "o1-01-10-24": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "o1": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": True,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "o1-mini-01-15-24": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "o1-mini-12-17-24": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "o1-mini-01-10-24": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "o1-mini": {
        "supports_reasoning_effort": False,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": True,
    },
    "o3-mini-2025-01-31": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "o3-mini": {
        "supports_reasoning_effort": True,
        "supports_structured_output": True,
        "supports_json_mode": True,
        "supports_image_ingestion": False,
        "supports_function_calls": True,
        "supports_system_message": False,
        "supports_developer_message": True,
        "supports_only_user_message": False,
    },
    "deepseek-chat": {
        "supports_reasoning_effort": False,
        "supports_structured_output": False,
        "supports_json_mode": False,
        "supports_image_ingestion": False,
        "supports_function_calls": False,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
    "deepseek-reasoner": {
        "supports_reasoning_effort": False,
        "supports_structured_output": False,
        "supports_json_mode": False,
        "supports_image_ingestion": False,
        "supports_function_calls": False,
        "supports_system_message": False,
        "supports_developer_message": False,
        "supports_only_user_message": False,
    },
}

# -------------------------------------------------------------------------------- #
# Helper Functions
# -------------------------------------------------------------------------------- #


def get_specific_model_id(model: Union[ModelAlias, ModelId]) -> ModelId:
    """
    Convert a model alias to its specific model ID.
    If a specific model ID is provided, return it directly.

    Args:
        model: The model alias or ID to convert

    Returns:
        ModelId: The specific model ID
    """
    if model in ALIAS_TO_MODEL_ID:
        return ALIAS_TO_MODEL_ID[model]
    return model  # It's already a specific model ID


def supports_feature(model: Union[ModelAlias, ModelId], feature: FEATURE_NAME) -> bool:
    """
    Check if a model supports a specific feature with O(1) lookup.
    Works with both aliases and specific model IDs.

    Args:
        model: The model alias or ID to check
        feature: The feature to check for (e.g., 'reasoning_effort')

    Returns:
        bool: True if the model supports the feature, False otherwise
    """
    model_id = get_specific_model_id(model)
    if model_id not in MODEL_CAPABILITIES:
        return False
    
    # Prepend 'supports_' to the feature name for lookup
    capability_key = f"supports_{feature}"
    return MODEL_CAPABILITIES[model_id].get(capability_key, False)


