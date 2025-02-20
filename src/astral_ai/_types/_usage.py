# ------------------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------------------

"""
Response Models for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Standard Library
from typing import Protocol
import time
import uuid

# Pydantic
from pydantic import BaseModel, Field, PrivateAttr

# Types
from typing import Optional

# Astral AI
from astral_ai.constants._models import ModelProvider, ModelName

# ------------------------------------------------------------------------------
# Usage Details
# ------------------------------------------------------------------------------


class ChatUsageDetails(BaseModel):
    """
    Chat Completion Usage Details Model for Astral AI
    """
    accepted_prediction_tokens: Optional[int] = Field(description="The accepted prediction tokens for the request")
    audio_tokens: Optional[int] = Field(description="The audio tokens for the request")
    reasoning_tokens: Optional[int] = Field(description="The reasoning tokens for the request")
    rejected_prediction_tokens: Optional[int] = Field(description="The rejected prediction tokens for the request")

    # Anthropic ONLY
    cache_creation_input_tokens: Optional[int] = Field(description="The number of input tokens")


class PromptUsageDetails(BaseModel):
    """
    Prompt Usage Details Model for Astral AI
    """
    audio_tokens: Optional[int] = Field(description="The audio tokens for the request")
    cached_tokens: Optional[int] = Field(description="The cached tokens for the request")


# ------------------------------------------------------------------------------
# Usage and Cost Protocol
# ------------------------------------------------------------------------------

class UsageAndCostProtocolMixin:
    _response_id: str
    _model_provider: ModelProvider
    _model_name: ModelName

    @property
    def response_id(self) -> str:
        ...

    @property
    def model_provider(self) -> ModelProvider:
        ...

    @property
    def model_name(self) -> ModelName:
        ...


# ------------------------------------------------------------------------------
# Base Usage Model
# ------------------------------------------------------------------------------

# TODO: Different for now even though same implementation. More flexible this way.

class BaseUsage(BaseModel, UsageAndCostProtocolMixin):
    """
    Base Usage Model for Astral AI
    """
    _response_id: str = PrivateAttr(default=None)
    _model_provider: ModelProvider = PrivateAttr(default=None)
    _model_name: ModelName = PrivateAttr(default=None)

    @property
    def response_id(self) -> str:
        """
        Get the response ID
        """
        return self._response_id

    @property
    def model_provider(self) -> ModelProvider:
        """
        Get the model provider
        """
        return self._model_provider

    @property
    def model_name(self) -> ModelName:
        """
        Get the model name
        """
        return self._model_name


# ------------------------------------------------------------------------------
# Base Cost Model
# ------------------------------------------------------------------------------

class BaseCost(BaseModel, UsageAndCostProtocolMixin):
    """
    Base Cost Model for Astral AI
    """
    _response_id: str = PrivateAttr(default=None)
    _model_provider: ModelProvider = PrivateAttr(default=None)
    _model_name: ModelName = PrivateAttr(default=None)

    @property
    def response_id(self) -> str:
        """
        Get the response ID
        """
        return self._response_id

    @property
    def model_provider(self) -> ModelProvider:
        """
        Get the model provider
        """
        return self._model_provider

    @property
    def model_name(self) -> ModelName:
        """
        Get the model name
        """
        return self._model_name


# ------------------------------------------------------------------------------
# Chat Usage and Cost Models
# ------------------------------------------------------------------------------


class ChatUsage(BaseUsage):
    """
    Chat Usage Model for Astral AI
    """
    completion_tokens: int = Field(description="The completion tokens for the request")
    prompt_tokens: int = Field(description="The prompt tokens for the request")
    total_tokens: int = Field(description="The total tokens for the request")

    # Details
    completion_tokens_details: Optional[ChatUsageDetails] = Field(description="The completion tokens details for the request", default=None)
    prompt_tokens_details: Optional[PromptUsageDetails] = Field(description="The prompt tokens details for the request", default=None)


class ChatCost(BaseCost):
    """
    Chat Cost Model for Astral AI
    """
    input_cost: Optional[float] = Field(description="The input cost for the request", default=None)
    cached_input_cost: Optional[float] = Field(description="The cached input cost for the request", default=None)
    output_cost: Optional[float] = Field(description="The output cost for the request", default=None)

    # Anthropic ONLY
    cached_output_cost: Optional[float] = Field(description="The cached output cost for the request", default=None)

    # Total
    total_cost: Optional[float] = Field(description="The total cost for the request", default=None)


# ------------------------------------------------------------------------------
# Embedding Usage and Cost Models
# ------------------------------------------------------------------------------


class EmbeddingUsage(BaseUsage):
    """
    Embedding Usage Model for Astral AI
    """
    pass


class EmbeddingCost(BaseCost):
    """
    Embedding Cost Model for Astral AI
    """
    pass
