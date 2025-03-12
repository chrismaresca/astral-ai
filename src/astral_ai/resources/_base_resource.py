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
from astral_ai.constants._models import ModelProvider, ModelAlias, ModelId

# Utilities
from astral_ai.utilities import get_provider_from_model_name

# Exceptions
from astral_ai.exceptions import ModelNameError

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

        # Validate the model and set up provider
        # TODO: make this an entire validation step for what features it supports vs whats passed in the request.
        self._validate_request()

        self.model = self.request.model
        self.model_provider = get_provider_from_model_name(self.model)

        # TODO: remove this in production
        self.model_provider = "openai"

        self.client = ProviderClientRegistry.get_client(
            self.model_provider,
            astral_client=self.astral_client
        )

        # Set up provider client and adapter
        from astral_ai.providers._adapters import create_adapter
        self.adapter = create_adapter(self.model_provider)

        # Set cost strategy from astral params
        self.cost_strategy = self.astral_params.cost_strategy or ReturnCostStrategy()


    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def _validate_request(self) -> None:
        """
        Validate the request.
        """
        valid_models = get_args(ModelAlias) + get_args(ModelId)
        if self.request.model not in valid_models:
            raise ModelNameError(model_name=self.request.model)
        
        # Validate the request data
        # TODO: Implement this. Look at earlier Astral AI implementations

    # --------------------------------------------------------------------------
    # Abstract Methods to Run the Resource
    # Has approaches for sync, async, and streaming execution
    # --------------------------------------------------------------------------

    @abstractmethod
    def run(self, *args, **kwargs) -> TResponse:
        """
        Execute the resource synchronously.

        Returns:
            TResponse: The response from the provider
        """
        pass

    @abstractmethod
    async def run_async(self, *args, **kwargs) -> TResponse:
        """
        Execute the resource asynchronously.

        Returns:
            TResponse: The response from the provider
        """
        pass

    # TODO: Implement this
    # @abstractmethod
    # def run_stream(self, *args, **kwargs) -> AsyncGenerator[TResponse, None]:
    #     """
    #     Execute the resource asynchronously.
    #     """
    #     pass