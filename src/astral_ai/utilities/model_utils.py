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
    PROVIDER_MODEL_NAMES,
    MODEL_NAMES_TO_RESOURCE_TYPE,
    ResourceType,
)


# Astral AI Exceptions
from astral_ai.errors.exceptions import (
    ProviderNotFoundForModelError,
    ResourceTypeNotFoundForModelError,
)


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
# Get Resource Type from Model Name
# ------------------------------------------------------------------------------


def get_resource_type_from_model_name(model_name: ModelName) -> ResourceType:
    """
    Get the resource type from a model name.
    """
    if model_name not in MODEL_NAMES_TO_RESOURCE_TYPE:
        raise ResourceTypeNotFoundForModelError(model_name=model_name)
    else:
        return MODEL_NAMES_TO_RESOURCE_TYPE[model_name]


