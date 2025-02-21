from __future__ import annotations
# -------------------------------------------------------------------------------- #
# Cost Strategies
# -------------------------------------------------------------------------------- #

"""
Cost Strategies for Astral AI.

This module provides a flexible framework for handling cost tracking and processing
in Astral AI. It includes:

- A generic base cost strategy class that can be extended for different use cases
- Type-safe cost processing with static type checking support
- Example implementations for common scenarios like S3 and DataDog integration
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from astral_ai._types import (
        AstralBaseResponse,
        AstralChatResponse,
        AstralStructuredResponse,
        BaseUsage,
        BaseCost,
    )

from astral_ai.constants._models import ModelName, ModelProvider
from astral_ai.utilities.cost_utils import calculate_cost

# -------------------------------------------------------------------------------- #
# Type Variables
# -------------------------------------------------------------------------------- #
_ResponseT = TypeVar("_ResponseT", bound="AstralBaseResponse")
_CostT = TypeVar("_CostT", bound="BaseCost")

# -------------------------------------------------------------------------------- #
# Base Cost Strategy
# -------------------------------------------------------------------------------- #
class BaseCostStrategy(ABC, Generic[_ResponseT, _CostT]):
    """
    A generic base cost strategy for processing costs and responses.

    This abstract base class provides a framework for implementing cost tracking strategies.
    It uses generics to ensure type safety and provides overloaded methods for different
    response/cost type combinations.

    Type Parameters:
        _ResponseT: The type of response being processed (must inherit from AstralBaseResponse)
        _CostT: The type of cost being processed (must inherit from BaseCost)
    """

    def _calculate_cost(self, usage: BaseUsage, model_name: ModelName, model_provider: ModelProvider) -> _CostT:
        """
        Calculate the cost using the appropriate cost calculator.
        """
        return calculate_cost(usage=usage, model_name=model_name, model_provider=model_provider)

    @overload
    def run_cost_strategy(
        self, response: AstralChatResponse, model_name: ModelName, model_provider: ModelProvider
    ) -> AstralChatResponse:
        ...

    @overload
    def run_cost_strategy(
        self, response: AstralStructuredResponse, model_name: ModelName, model_provider: ModelProvider
    ) -> AstralStructuredResponse:
        ...

    def run_cost_strategy(
        self, response: _ResponseT, model_name: ModelName, model_provider: ModelProvider
    ) -> _ResponseT:
        """
        Process the response by attaching cost and executing additional logic.
        """
        cost = self._calculate_cost(usage=response.usage, model_name=model_name, model_provider=model_provider)
        self._add_to_response(response, cost)
        self._additional_logic(response, cost)
        return response

    def _add_to_response(self, response: _ResponseT, cost: _CostT) -> None:
        """
        Attaches the cost information to the response.
        """
        response.cost = cost

    @abstractmethod
    def _additional_logic(self, response: _ResponseT, cost: _CostT) -> None:
        """
        Hook for implementing strategy-specific processing logic.
        """
        pass

# -------------------------------------------------------------------------------- #
# Concrete Strategy Implementations
# -------------------------------------------------------------------------------- #
class ReturnCostStrategy(BaseCostStrategy["AstralBaseResponse", "BaseCost"]):
    """
    A simple pass-through cost strategy.
    """
    def _additional_logic(self, response: "AstralBaseResponse", cost: "BaseCost") -> None:
        # No additional processing needed.
        pass

class S3CostStrategy(BaseCostStrategy["AstralBaseResponse", "BaseCost"]):
    """
    A cost strategy that persists cost information to Amazon S3.
    """
    def __init__(self, bucket_name: str, s3_client: Any):
        self.bucket_name = bucket_name
        self.s3_client = s3_client

    def _additional_logic(self, response: "AstralBaseResponse", cost: "BaseCost") -> None:
        # TODO: Implement S3 upload logic.
        pass

class DataDogCostStrategy(BaseCostStrategy["AstralBaseResponse", "BaseCost"]):
    """
    A cost strategy that sends metrics to DataDog.
    """
    def __init__(self, datadog_client: Any):
        self.datadog_client = datadog_client

    def _additional_logic(self, response: "AstralBaseResponse", cost: "BaseCost") -> None:
        # TODO: Implement DataDog metric submission.
        pass
