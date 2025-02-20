# ------------------------------------------------------------------------------
# Cost Utils
# ------------------------------------------------------------------------------


"""
Cost Utils for Astral AI.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in
from typing import Tuple, overload, Literal, Union

# Astral AI
from astral_ai._types._usage import ChatCost, EmbeddingCost
from astral_ai.constants._models import ModelName, ModelProvider, ChatModels, EmbeddingModels

# Constants
from astral_ai.constants._costs import model_cost_mapping


# ------------------------------------------------------------------------------
# Cost Utils
# ------------------------------------------------------------------------------

# TODO: Move this to the constants module.


@overload
def get_model_costs(model_name: ChatModels, model_provider: ModelProvider) -> ChatCost:
    ...


@overload
def get_model_costs(model_name: EmbeddingModels, model_provider: ModelProvider) -> EmbeddingCost:
    ...


def get_model_costs(model_name: ModelName, model_provider: ModelProvider) -> Union[ChatCost, EmbeddingCost]:
    """
    Get the costs for a model.
    """
    if model_provider not in model_cost_mapping:
        raise ValueError(f"Provider {model_provider} not found in cost mapping")

    if model_name not in model_cost_mapping[model_provider]:
        raise ValueError(f"Model {model_name} not found in {model_provider} cost mapping")

    model_cost_dict = model_cost_mapping[model_provider][model_name]

    return ChatCost(**model_cost_dict) if model_name in ChatModels else EmbeddingCost(**model_cost_dict)
