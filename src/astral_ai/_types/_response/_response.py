from __future__ import annotations
# ------------------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------------------

"""
Response Models for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in imports
from typing import Optional, Iterable, TypeVar, Generic, Dict, Any


# Pydantic
from pydantic import BaseModel, PrivateAttr, Field, model_validator

# Astral AI


# Astral AI Resource
from astral_ai._types._resource import AstralBaseResource

# Astral AI Response Types
from .resources import (
    ChatCompletionResponse,
    StructuredOutputCompletionResponse,
    BaseProviderResourceResponse,
)

# Astral AI Types
from ._usage import (
    ChatUsage,
    ChatCost,
    BaseUsage,
    BaseCost,
)

# ------------------------------------------------------------------------------
# Rate Limit Types
# ------------------------------------------------------------------------------


class RateLimit(BaseModel):
    """
    Rate Limit Model for Astral AI
    """
    tier: str = Field(description="The tier for the rate limit")
    limit: int = Field(description="The limit for the rate limit")
    remaining: int = Field(description="The remaining for the rate limit")
    reset: int = Field(description="The reset for the rate limit")


# ------------------------------------------------------------------------------
# Base Response
# ------------------------------------------------------------------------------

class AstralBaseResponse(AstralBaseResource):
    """
    Base Response Model for Astral AI
    """
    # Response-specific identifier (uses resource_id from base class)
    @property
    def response_id(self) -> str:
        """
        Get the response ID
        """
        return self.resource_id

    # Rate Limits
    _rate_limits: Optional[RateLimit] = PrivateAttr(default=None)

    # Response
    response: BaseProviderResourceResponse = Field(description="The response for the chat completion")


    # Usage
    usage: BaseUsage = Field(description="The usage for the response")
    cost: Optional[BaseCost] = Field(description="The cost for the response", default=None)
    latency_ms: Optional[float] = Field(description="The latency in milliseconds for the request", default=0.0)

    # --------------------------------------------------------------------------
    # Rate Limits
    # --------------------------------------------------------------------------

    @property
    def rate_limits(self) -> Optional[RateLimit]:
        """
        Get the rate limits
        """
        return self._rate_limits

    @rate_limits.setter
    def rate_limits(self, value: Optional[RateLimit]) -> None:
        """
        Set the rate limits
        """
        self._rate_limits = value


# ------------------------------------------------------------------------------
# Generic Mixin for Private Field Propagation
# ------------------------------------------------------------------------------
ResponseT = TypeVar('ResponseT', bound=AstralBaseResponse)
UsageT = TypeVar('UsageT', bound=BaseUsage)
CostT = TypeVar('CostT', bound=BaseCost)


class PrivatePropagationMixin(BaseModel, Generic[UsageT, CostT]):
    """
    Mixin to propagate private fields from a response to its usage.
    Assumes that the class has:
      - a 'usage' field with private attributes: _response_id, _model_provider, _model_name
      - properties 'response_id', 'provider_name', and 'model'
    """

    @model_validator(mode="after")
    def propagate_private_usage_fields(self: ResponseT) -> ResponseT:
        self.usage._response_id = self.response_id
        self.usage._model_provider = self.provider_name
        self.usage._model_name = self.model
        return self

    @model_validator(mode="after")
    def propagate_private_cost_fields(self: ResponseT) -> ResponseT:
        if self.cost is not None:
            self.cost._response_id = self.response_id
            self.cost._model_provider = self.provider_name
            self.cost._model_name = self.model
        return self


# ------------------------------------------------------------------------------
# Chat Response
# ------------------------------------------------------------------------------


class AstralCompletionResponse(PrivatePropagationMixin[ChatUsage, ChatCost], AstralBaseResponse):
    """
    Chat Response Model for Astral AI
    """
    response: ChatCompletionResponse = Field(description="The response for the chat completion")
    content: Optional[str] = Field(description="The content for the response. This is the text that is returned from the model.")
    usage: ChatUsage = Field(description="The usage for the response")
    cost: Optional[ChatCost] = Field(description="The cost for the response", default=None)

# ------------------------------------------------------------------------------
# AI Structured Response Objects
# ------------------------------------------------------------------------------


StructuredOutputResponse = TypeVar('StructuredOutputResponse', bound=BaseModel)

# ------------------------------------------------------------------------------
# Structured Response
# ------------------------------------------------------------------------------


class AstralStructuredCompletionResponse(AstralCompletionResponse, Generic[StructuredOutputResponse]):
    """
    Structured Response Model for Astral AI
    """
    response: StructuredOutputCompletionResponse[StructuredOutputResponse] = Field(description="The response for the structured response")
    content: StructuredOutputResponse = Field(description="The structured output for the response. This is the object that is returned from the model.")
