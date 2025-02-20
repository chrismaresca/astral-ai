# ------------------------------------------------------------------------------
# Cost Strategies
# ------------------------------------------------------------------------------

"""
Cost Strategies for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Astral AI
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse

# ------------------------------------------------------------------------------
# Cost Strategies
# ------------------------------------------------------------------------------

# Built-in imports
from abc import ABC, abstractmethod
from typing import Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from astral_ai._types._response import BaseResponse
    from astral_ai._types._usage import BaseUsage

# ------------------------------------------------------------------------------
# Base Cost Strategy
# ------------------------------------------------------------------------------


class CostStrategy(ABC):
    @abstractmethod
    def handle_cost(self, response: BaseResponse, cost: float) -> Tuple[BaseResponse, float]:
        """
        Process the cost along with the response.
        This method should return either the unmodified response or an enhanced result.
        """
        pass


# ------------------------------------------------------------------------------
# Return Cost Strategy
# ------------------------------------------------------------------------------


class ReturnCostStrategy(CostStrategy):
    """Return the response and cost as a tuple."""

    def handle_cost(self, response: BaseResponse, cost: float) -> Tuple[BaseResponse, float]:
        return response, cost

# ------------------------------------------------------------------------------
# S3 Cost Strategy
# ------------------------------------------------------------------------------


class S3CostStrategy(CostStrategy):
    """Upload cost information to S3 and return only the response."""

    def __init__(self, bucket_name: str, s3_client: Any):
        self.bucket_name = bucket_name
        self.s3_client = s3_client

    def handle_cost(self, response: BaseResponse, cost: float) -> Tuple[BaseResponse, float]:
        pass

# ------------------------------------------------------------------------------
# DataDog Cost Strategy
# ------------------------------------------------------------------------------


class DataDogCostStrategy(CostStrategy):
    """Send cost metrics to DataDog and return only the response."""

    def __init__(self, datadog_client: Any):
        self.datadog_client = datadog_client

    def handle_cost(self, response: BaseResponse, cost: float) -> Tuple[BaseResponse, float]:
        pass
