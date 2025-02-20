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


class ModelCosts(TypedDict):
    prompt_tokens: float
    cached_prompt_tokens: float
    output_tokens: float
    # Anthropic ONLY
    cache_creation_tokens: float | None


ModelCostMapping: TypeAlias = Dict[ModelProvider, Dict[ModelName, ModelCosts]]

# -------------------------------------------------------------------------------- #
# Model Cost Constants
# -------------------------------------------------------------------------------- #

model_cost_mapping: ModelCostMapping = {
    "openai": {
        "gpt-4o": {
            "prompt_tokens": 0.03,
            "cached_prompt_tokens": 0.03,
            "output_tokens": 0.06,
            "cache_creation_tokens": None
        }
    },
    "anthropic": {
        "claude-3-sonnet": {
            "prompt_tokens": 0.015,
            "cached_prompt_tokens": 0.015,
            "output_tokens": 0.075,
            "cache_creation_tokens": 0.015
        }
    }
}


