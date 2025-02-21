# -------------------------------------------------------------------------------- #
# Base Resource
# -------------------------------------------------------------------------------- #

"""
Base Resource module for Astral AI.
Provides the abstract base class for all Astral AI resources.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from abc import ABC, abstractmethod
from typing import Optional

# Astral AI Types
from astral_ai._types._request._request import AstralCompletionRequest
from astral_ai._types._astral import AstralParams
from astral_ai._types._response._response import AstralBaseResponse

# Models
from astral_ai.constants._models import ModelName, ModelProvider

# Utilities
from astral_ai.utilities import get_provider_from_model_name

# Exceptions
from astral_ai.exceptions import ModelNameError

# Providers
from astral_ai.providers._mappings import get_provider_client, get_provider_adapter

# Cost Strategies
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy


# -------------------------------------------------------------------------------- #
# Base Resource Class
# -------------------------------------------------------------------------------- #


class AstralResource(ABC):
    """
    Abstract base class for all Astral AI resources.

    Provides common initialization and validation logic for model providers,
    clients and adapters. All resource implementations should inherit from this class.

    Args:
        request (AstralCompletionRequest): The request configuration
        astral_params (Optional[AstralParams]): Optional Astral-specific parameters

    Raises:
        ModelNameError: If the model name is invalid
        ProviderNotFoundForModelError: If no provider is found for the given model
    """

    def __init__(self,
                 request: AstralCompletionRequest,
                 astral_params: Optional[AstralParams] = None) -> None:
        """
        Initialize the AstralResource.

        Args:
            request (AstralCompletionRequest): The request configuration
            astral_params (Optional[AstralParams]): Optional Astral-specific parameters

        Raises:
            ModelNameError: If the model name is invalid
        """
        self.request = request
        self.astral_params = astral_params

        # Validate model
        if not isinstance(request.model, ModelName):
            raise ModelNameError(model_name=request.model)
        else:
            self.model: ModelName = request.model

        # Set the model provider.
        self.model_provider: ModelProvider = get_provider_from_model_name(self.model)

        # TODO: Add support for multiple cost strategies.
        self.cost_strategy = self._set_cost_strategy()

        # TODO: Remove this for production.
        self.model_provider = "openai"

        # Retrieve (or create) the provider client.
        self.client = get_provider_client(self.model_provider, astral_client=astral_params.astral_client)

        # Get the provider adapter.
        self.adapter = get_provider_adapter(self.model_provider)

    def _set_cost_strategy(self) -> BaseCostStrategy:
        """
        Set the cost strategy.
        """
        return self.astral_params.cost_strategy or ReturnCostStrategy()

    @abstractmethod
    def run(self) -> AstralBaseResponse:
        """
        Execute the resource synchronously.

        Returns:
            AstralBaseResponse: The response from the provider
        """
        pass

    @abstractmethod
    async def run_async(self) -> AstralBaseResponse:
        """
        Execute the resource asynchronously.

        Returns:
            AstralBaseResponse: The response from the provider
        """
        pass
