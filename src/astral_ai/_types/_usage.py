# -------------------------------------------------------------------------------- #
# Usage and Cost Types
# -------------------------------------------------------------------------------- #

"""
Usage and cost type definitions for Astral AI.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Optional

# Third-party imports
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------------- #
# Base Usage Types
# -------------------------------------------------------------------------------- #

class BaseUsage(BaseModel):
    """Base usage class for all usage types."""
    pass

class ChatUsage(BaseUsage):
    """Chat usage information."""
    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    cached_tokens: Optional[int] = Field(default=0, description="Number of cached tokens")
    cache_creation_input_tokens: Optional[int] = Field(default=None, description="Number of tokens used in cache creation")

class EmbeddingUsage(BaseUsage):
    """Embedding usage information."""
    total_tokens: int = Field(description="Total number of tokens processed")

# -------------------------------------------------------------------------------- #
# Cost Types
# -------------------------------------------------------------------------------- #

class BaseCost(BaseModel):
    """Base cost class for all cost types."""
    total_cost: float = Field(description="Total cost in USD")

class ChatCost(BaseCost):
    """Chat cost information."""
    input_cost: float = Field(description="Cost of input tokens in USD")
    output_cost: float = Field(description="Cost of output tokens in USD")
    anthropic_cache_creation_cost: Optional[float] = Field(default=None, description="Cost of cache creation for Anthropic models")

class EmbeddingCost(BaseCost):
    """Embedding cost information."""
    pass 