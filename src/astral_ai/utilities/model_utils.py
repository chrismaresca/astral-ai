# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

"""
Astral AI Utilities.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------


# Astral AI Models Constants
from astral_ai.constants._models import (
    ModelName,
    ModelProvider,
    ModelAlias,
    ModelId,
    PROVIDER_MODEL_NAMES,
    MODEL_DEFINITIONS,
)


# Astral AI Exceptions
from astral_ai.errors.exceptions import ProviderNotFoundForModelError


# ------------------------------------------------------------------------------
# Get Provider from Model Name
# ------------------------------------------------------------------------------


def get_provider_from_model_name(model_name: ModelName) -> ModelProvider:
    """
    Get the provider from a model name.
    """
    if model_name not in PROVIDER_MODEL_NAMES:
        raise ProviderNotFoundForModelError(model_name=model_name)
    else:
        return PROVIDER_MODEL_NAMES[model_name]

# ------------------------------------------------------------------------------
# Is Model Alias
# ------------------------------------------------------------------------------


def is_model_alias(model_name: ModelName) -> bool:
    """
    Check if a model name is a model alias.
    """
    return model_name in MODEL_DEFINITIONS

# ------------------------------------------------------------------------------
# Get Model from Model Alias
# ------------------------------------------------------------------------------


def get_model_from_model_alias(model_alias: ModelAlias) -> ModelId:
    """
    Get the model from a model alias.
    """
    return MODEL_DEFINITIONS[model_alias]["most_recent_model"]

