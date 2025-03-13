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
from astral_ai._types._request._request import AstralCompletionRequest, BaseRequest
from astral_ai._types._astral import AstralParams
from astral_ai._types._response._response import AstralBaseResponse

# Models
from astral_ai.constants._models import ModelName

# Utilities
from astral_ai.utilities import get_provider_from_model_name


# Providers
from astral_ai.providers._client_registry import ProviderClientRegistry

# Cost Strategies
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy


# -------------------------------------------------------------------------------- #
# Generic Type Variables
# -------------------------------------------------------------------------------- #

TRequest = TypeVar('TRequest', bound=BaseRequest)
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

        # Setup the resource
        self._setup_resource()

    def _setup_resource(self) -> None:
        """
        Setup the resource.
        """
        # Extract core parameters
        self.astral_params = self.request.astral_params
        self.astral_client = self.astral_params.astral_client

        self._model_provider = get_provider_from_model_name(self.request.model)

        # TODO: remove this in production
        # self.model_provider = "openai"

        self.client = ProviderClientRegistry.get_client(
            self._model_provider,
            astral_client=self.astral_client
        )

        # Set up provider client and adapter
        from astral_ai.providers._adapters import create_adapter
        self.adapter = create_adapter(self._model_provider)

        # Set cost strategy from astral params
        self.cost_strategy = self.astral_params.cost_strategy or ReturnCostStrategy()

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    @abstractmethod
    def _validate_model(self, model: Optional[ModelName] = None) -> None:
        """
        Validate the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")
