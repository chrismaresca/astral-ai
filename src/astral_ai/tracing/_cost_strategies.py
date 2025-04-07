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
from typing import Any, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from astral_ai._types import (
        AstralBaseResponse,
        BaseUsage,
        BaseCost,
    )

from astral_ai.constants._models import ModelName, ModelProvider
from astral_ai.utilities.cost_utils import calculate_cost

# -------------------------------------------------------------------------------- #
# Type Variables
# -------------------------------------------------------------------------------- #
TResponse = TypeVar("TResponse", bound="AstralBaseResponse")
TCost = TypeVar("TCost", bound="BaseCost")
StructuredOutputResponseT = TypeVar("StructuredOutputResponseT", bound=BaseModel)

# -------------------------------------------------------------------------------- #
# Base Cost Strategy
# -------------------------------------------------------------------------------- #


class BaseCostStrategy(ABC, Generic[TResponse, TCost]):
    """
    A generic base cost strategy for processing costs and responses.

    This abstract base class provides a framework for implementing cost tracking strategies.
    It uses generics to ensure type safety and provides overloaded methods for different
    response/cost type combinations.

    Type Parameters:
        TResponse: The type of response being processed (must inherit from AstralBaseResponse)
        TCost: The type of cost being processed (must inherit from BaseCost)
    """

    def _calculate_cost(self, usage: BaseUsage, model_name: ModelName, model_provider: ModelProvider) -> TCost:
        """
        Calculate the cost using the appropriate cost calculator.
        """
        return calculate_cost(usage, model_name=model_name, model_provider=model_provider)

    def run_cost_strategy(
        self,
        response: TResponse,
        model_name: ModelName,
        model_provider: ModelProvider
    ) -> TResponse:
        """
        Process the response by attaching cost and executing additional logic.
        """
        cost = self._calculate_cost(response.usage, model_name=model_name, model_provider=model_provider)
        self._add_to_response(response, cost)
        self._additional_logic(response, cost)
        return response

    def _add_to_response(self, response: TResponse, cost: TCost) -> None:
        """
        Attaches the cost information to the response.
        """
        response.cost = cost

    @abstractmethod
    def _additional_logic(self, response: TResponse, cost: TCost) -> None:
        """
        Hook for implementing strategy-specific processing logic.
        """
        pass

# -------------------------------------------------------------------------------- #
# Concrete Strategy Implementations
# -------------------------------------------------------------------------------- #


class ReturnCostStrategy(BaseCostStrategy[TResponse, TCost]):
    """
    A simple pass-through cost strategy.
    """

    def _additional_logic(self, response: TResponse, cost: TCost) -> None:
        # No additional processing needed.
        pass


class S3CostStrategy(BaseCostStrategy[TResponse, TCost]):
    """
    A cost strategy that persists cost information to Amazon S3.
    """

    def __init__(self, bucket_name: str, s3_client: Any):
        self.bucket_name = bucket_name
        self.s3_client = s3_client

    def _additional_logic(self, response: TResponse, cost: TCost) -> None:
        # TODO: Implement S3 upload logic.
        pass


class DataDogCostStrategy(BaseCostStrategy[TResponse, TCost]):
    """
    A cost strategy that sends metrics to DataDog.
    """

    def __init__(self, datadog_client: Any):
        self.datadog_client = datadog_client

    def _additional_logic(self, response: TResponse, cost: TCost) -> None:
        # TODO: Implement DataDog metric submission.
        pass
