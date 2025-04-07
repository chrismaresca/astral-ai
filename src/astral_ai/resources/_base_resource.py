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
from typing import Optional, get_args, Generic, TypeVar, Type, Union

# Astral AI Types
from astral_ai._types._request._request import AstralCompletionRequest, AstralBaseRequest
from astral_ai._types._astral import AstralParams
from astral_ai._types._response._response import AstralBaseResponse

# Models
from astral_ai.constants._models import ModelName

# Utilities
from astral_ai.utilities import get_provider_from_model_name

# Providers
from astral_ai.providers._client_registry import ProviderClientRegistry
from astral_ai.providers._adapters import create_adapter

# Cost Strategies
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy

# Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Generic Type Variables
# -------------------------------------------------------------------------------- #

TRequest = TypeVar('TRequest', bound=AstralBaseRequest)
TResponse = TypeVar('TResponse', bound=AstralBaseResponse)


# -------------------------------------------------------------------------------- #
# Base Resource Class
# -------------------------------------------------------------------------------- #


class AstralResource(Generic[TRequest, TResponse], ABC):
    """
    Abstract base class for all Astral AI resources.

    Provides common initialization and validation logic for model providers,
    clients and adapters. All resource implementations should inherit from this class.

    Args:
        request (TRequest): The request configuration

    Raises:
        ModelNameError: If the model name is invalid
        ProviderNotFoundForModelError: If no provider is found for the given model
    """

    def __init__(
        self,
        request: TRequest,
    ) -> None:
        """
        Initialize the AstralResource.

        Args:
            request (TRequest): The request configuration

        Raises:
            ModelNameError: If the model name is invalid
        """

        # Set the request
        self.request = request

        # Extract core parameters
        self.astral_params = self.request.astral_params
        self.astral_client = self.astral_params.astral_client

        # Determine model provider
        self._model_provider = get_provider_from_model_name(self.request.model)

        # Initialize clients
        self._client = ProviderClientRegistry.get_client(
            self._model_provider,
            astral_client=self.astral_client,
            async_client=False
        )
        self._async_client = ProviderClientRegistry.get_client(
            self._model_provider,
            astral_client=self.astral_client,
            async_client=True
        )

        # Initialize adapter and cost strategy directly
        self._adapter = create_adapter(self._model_provider)
        self._cost_strategy = self.astral_params.cost_strategy or ReturnCostStrategy()

    @property
    def client(self):
        """Synchronous client."""
        return self._client

    @property
    def async_client(self):
        """Asynchronous client."""
        return self._async_client

    @property
    def adapter(self):
        """Provider adapter."""
        return self._adapter

    @property
    def cost_strategy(self):
        """Cost strategy."""
        return self._cost_strategy

    @cost_strategy.setter
    def cost_strategy(self, value):
        """Setter for cost strategy."""
        self._cost_strategy = value

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    @abstractmethod
    def _validate_model(self, model: Optional[ModelName] = None) -> None:
        """
        Validate the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
