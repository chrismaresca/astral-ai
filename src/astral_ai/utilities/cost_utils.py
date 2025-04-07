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
from typing import Union, Optional

# Astral AI Types
from astral_ai._types._response._usage import (
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
from astral_ai.constants._costs import (
    MODEL_COST_MAPPING,
    is_current_time_in_window,
    TimePeriodCosts,
    ModelSpecificCosts
)


# ------------------------------------------------------------------------------
# Cost Constants
# ------------------------------------------------------------------------------

# OpenAI
COST_MULTIPLIER = 1_000_000



# ------------------------------------------------------------------------------
# Get Model Costs
# ------------------------------------------------------------------------------


def get_model_costs(model_name: ModelName, model_provider: ModelProvider) -> TimePeriodCosts:
    """
    Get the costs for a model.
    """
    if model_provider not in MODEL_COST_MAPPING:
        raise ValueError(f"Provider {model_provider} not found in cost mapping")

    if model_name not in MODEL_COST_MAPPING[model_provider]:
        raise ValueError(f"Model {model_name} not found in {model_provider} cost mapping")

    model_cost_dict = MODEL_COST_MAPPING[model_provider][model_name]

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

    # Handle time-based costs
    cost_data = model_costs["standard"]
    if model_costs["time_based"]:
        # If it's time-based, determine whether to use standard or discount pricing
        if model_costs["discount"] and not is_current_time_in_window(model_costs["standard"]["time_period"]):
            # Outside standard period, use discount pricing
            cost_data = model_costs["discount"]  # type: ignore
        else:
            # Within standard period, use standard pricing
            cost_data = model_costs["standard"]

    # Calculate individual cost components.
    prompt_cost: float = cost_data["input_base_cost"] * (usage.prompt_tokens / COST_MULTIPLIER)
    cached_prompt_cost: float = cost_data["input_cache_hit_cost"] * (usage.cached_tokens / COST_MULTIPLIER) if usage.cached_tokens else 0.0
    completion_cost: float = cost_data["output_base_cost"] * (usage.completion_tokens / COST_MULTIPLIER)

    # Anthropic ONLY Cache Creation Cost (if applicable)
    anthropic_cache_creation_cost: Optional[float] = None
    if model_provider == "anthropic" and usage.cache_creation_input_tokens:
        cache_write_cost: float = cost_data.get("input_cache_write_cost", 0.0) or 0.0
        anthropic_cache_creation_cost = cache_write_cost * (usage.cache_creation_input_tokens / COST_MULTIPLIER)

    # Calculate the total cost.
    total_cost = prompt_cost + cached_prompt_cost + completion_cost
    if model_provider == "anthropic" and anthropic_cache_creation_cost is not None:
        total_cost += anthropic_cache_creation_cost

    return ChatCost(
        input_cost=prompt_cost + cached_prompt_cost,
        cached_input_cost=cached_prompt_cost,
        output_cost=completion_cost,
        anthropic_cache_creation_cost=anthropic_cache_creation_cost,
        total_cost=total_cost
    )

# ------------------------------------------------------------------------------
# Embedding Cost Utils
# ------------------------------------------------------------------------------

# @calculate_cost.register
# def _(usage: EmbeddingUsage, model_name: ModelName, model_provider: ModelProvider) -> EmbeddingCost:
#     """
#     Calculate the cost for embedding usage.
#     """
#     # Look up the model costs via the model name and provider.
#     model_costs = get_model_costs(model_name=model_name, model_provider=model_provider)

#     # Handle time-based costs
#     cost_data = model_costs
#     if model_costs.get("time_based", False):
#         # If it's time-based, determine whether to use standard or discount pricing
#         if model_costs.get("discount") and not is_current_time_in_window(model_costs["standard"]["time_period"]):
#             # Outside standard period, use discount pricing
#             cost_data = model_costs["discount"]
#         else:
#             # Within standard period, use standard pricing
#             cost_data = model_costs["standard"]

#     # Calculate individual cost components.
#     input_cost = cost_data["input_base_cost"] * (usage.token_count / COST_MULTIPLIER)

#     return EmbeddingCost(
#         input_cost=input_cost,
#         total_cost=input_cost
#     )
