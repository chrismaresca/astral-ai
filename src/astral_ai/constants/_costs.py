# -------------------------------------------------------------------------------- #
# Model Cost Constants
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import TypedDict, TypeAlias, Dict

# Module imports
from astral_ai.constants._models import ModelName, ModelProvider

# -------------------------------------------------------------------------------- #
# Model Cost Types
# -------------------------------------------------------------------------------- #


class ModelSpecificCosts(TypedDict):
    completion_token_cost: float
    cached_completion_token_cost: float
    prompt_token_cost: float
    cached_prompt_token_cost: float
    anthropic_cache_creation_token_cost: float | None


ModelSpecificCostsMapping: TypeAlias = Dict[ModelProvider, Dict[ModelName, ModelSpecificCosts]]

# -------------------------------------------------------------------------------- #
# Model Cost Constants
# -------------------------------------------------------------------------------- #

model_specific_cost_mapping: ModelSpecificCostsMapping = {
    "openai": {
        "gpt-4o": {
            "completion_token_cost": 0.03,
            "cached_completion_token_cost": 0.03,
            "prompt_token_cost": 0.06,
            "cached_prompt_token_cost": 0.06,
            "anthropic_cache_creation_token_cost": None
        }
    },
    "anthropic": {
        "claude-3-sonnet": {
            "completion_token_cost": 0.015,
            "cached_completion_token_cost": 0.015,
            "prompt_token_cost": 0.075,
            "cached_prompt_token_cost": 0.075,
            "anthropic_cache_creation_token_cost": 0.015
        }
    },
    "deepseek": {
        "deepseek-chat": {
            "completion_token_cost": 0.0,
            "cached_completion_token_cost": 0.0,
            "prompt_token_cost": 0.0,
            "cached_prompt_token_cost": 0.0,
        }
    }
}


