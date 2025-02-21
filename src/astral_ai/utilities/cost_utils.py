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
from functools import singledispatch
from typing import Union

# Astral AI Types
from astral_ai._types._usage import (
    ChatUsage,
    EmbeddingUsage,
    BaseUsage,
    BaseCost,
    ChatCost,
    EmbeddingCost,
)

# Astral AI Model Constants
from astral_ai.constants._models import (
    ModelName,
    ModelProvider,
)

# Astral AI Models Costs
from astral_ai.constants._costs import model_specific_cost_mapping, ModelSpecificCosts


# ------------------------------------------------------------------------------
# Get Model Costs
# ------------------------------------------------------------------------------


def get_model_costs(model_name: ModelName, model_provider: ModelProvider) -> ModelSpecificCosts:
    """
    Get the costs for a model.
    """
    if model_provider not in model_specific_cost_mapping:
        raise ValueError(f"Provider {model_provider} not found in cost mapping")

    if model_name not in model_specific_cost_mapping[model_provider]:
        raise ValueError(f"Model {model_name} not found in {model_provider} cost mapping")

    # TODO: make keys identical for easy lookup
    model_cost_dict = model_specific_cost_mapping[model_provider][model_name]

    return model_cost_dict


# ------------------------------------------------------------------------------
# Calculate Cost Single Dispatch
# ------------------------------------------------------------------------------


@singledispatch
def calculate_cost(usage: BaseUsage, model_name: ModelName, model_provider: ModelProvider) -> BaseCost:
    raise NotImplementedError(f"No cost calculator registered for {type(usage)}")


# ------------------------------------------------------------------------------
# Register Chat Cost Calculator
# ------------------------------------------------------------------------------

@calculate_cost.register
def _(usage: ChatUsage, model_name: ModelName, model_provider: ModelProvider) -> ChatCost:
    """
    Calculate the cost for chat usage.
    """
    # Look up the model costs via the model name and provider.
    model_costs = get_model_costs(model_name=model_name, model_provider=model_provider)

    # Calculate individual cost components.
    prompt_cost = model_costs["prompt_token_cost"] * usage.prompt_tokens
    cached_prompt_cost = model_costs["cached_prompt_token_cost"] * getattr(usage, "cached_tokens", 0)
    completion_cost = model_costs["completion_token_cost"] * usage.completion_tokens
    cached_completion_cost = model_costs["cached_completion_token_cost"] * usage.completion_tokens

    # Anthropic ONLY Cache Creation Cost (if applicable)
    anthropic_cache_creation_cost = None
    if model_provider == "anthropic" and getattr(usage, "cache_creation_input_tokens", None) and model_costs.get("anthropic_cache_creation_token_cost"):
        anthropic_cache_creation_cost = model_costs["anthropic_cache_creation_token_cost"] * usage.cache_creation_input_tokens

    # Calculate the total cost.
    total_cost = prompt_cost + cached_prompt_cost + completion_cost + cached_completion_cost
    if model_provider == "anthropic" and anthropic_cache_creation_cost is not None:
        total_cost += anthropic_cache_creation_cost

    return ChatCost(
        input_cost=prompt_cost + cached_prompt_cost,
        output_cost=completion_cost + cached_completion_cost,
        anthropic_cache_creation_cost=anthropic_cache_creation_cost,
        total_cost=total_cost
    )


# ------------------------------------------------------------------------------
# Embedding Cost Utils
# ------------------------------------------------------------------------------

@calculate_cost.register
def _(usage: EmbeddingUsage, model_name: ModelName, model_provider: ModelProvider) -> EmbeddingCost:
    """
    Calculate the cost for embedding usage.
    """
    pass
