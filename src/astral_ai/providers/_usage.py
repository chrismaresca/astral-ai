from __future__ import annotations
# -------------------------------------------------------------------------------- #
# Usage Data Utilities
# -------------------------------------------------------------------------------- #
# Utilities for handling usage data from various providers
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Optional, Union, Any, TYPE_CHECKING

# Astral AI imports
from astral_ai._types._response._usage import ChatUsage

# Provider usage types
if TYPE_CHECKING:
    from astral_ai.providers.openai._types._response import OpenAICompletionUsageType
    from astral_ai.providers.deepseek._types._response import DeepSeekCompletionUsageType
    from astral_ai.providers.anthropic._types._response import AnthropicCompletionUsageType

# -------------------------------------------------------------------------------- #
# Usage Data Conversion
# -------------------------------------------------------------------------------- #

def create_usage_data(
    usage: Optional[Union[OpenAICompletionUsageType, DeepSeekCompletionUsageType, AnthropicCompletionUsageType, Any]]
) -> ChatUsage:
    """
    Create usage data from a provider's usage block, if present.
    
    Args:
        usage: The usage data from a provider response
        
    Returns:
        Standardized ChatUsage object with token counts
    """

    if not usage:
        return ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return ChatUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        # example nested tokens
        audio_tokens=(
            getattr(usage.prompt_tokens_details, "audio_tokens", None)
            if hasattr(usage, "prompt_tokens_details") else None
        ),
        cached_tokens=(
            getattr(usage.prompt_tokens_details, "cached_tokens", None)
            if hasattr(usage, "prompt_tokens_details") else None
        ),
        accepted_prediction_tokens=(
            getattr(usage.completion_tokens_details, "accepted_prediction_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
        rejected_prediction_tokens=(
            getattr(usage.completion_tokens_details, "rejected_prediction_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
        reasoning_tokens=(
            getattr(usage.completion_tokens_details, "reasoning_tokens", None)
            if hasattr(usage, "completion_tokens_details") else None
        ),
    ) 