from __future__ import annotations

# Astral Types
from astral_ai._types import AstralBaseResponse

# Astral Resources
from astral_ai.resources._base_resource import AstralResource

# -------------------------------------------------------------------------------- #
# Decorators
# -------------------------------------------------------------------------------- #
"""
This file contains decorators for timing and logging.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import (List,
                    Dict,
                    Optional,
                    Callable, Tuple, Union, TYPE_CHECKING, ParamSpec, TypeVar,
                    Concatenate)

# Astral AI
from astral_ai.utilities.cost_utils import get_model_costs

# Astral imports
from astral_ai._types._request._request import NOT_GIVEN

# Standard library imports
import time
import functools

# Types
from typing import Callable

# Exceptions
from astral_ai.exceptions import MissingParameterError


# -------------------------------------------------------------------------------- #
# Type Checking Imports
# -------------------------------------------------------------------------------- #

if TYPE_CHECKING:
    from astral_ai.providers._base_client import BaseProviderClient
    from astral_ai.providers._generics import ProviderRequestType, ProviderResponseType
    from astral_ai.constants._models import ModelName, ModelProvider
    from astral_ai.tracing._cost_strategies import CostStrategy


# -------------------------------------------------------------------------------- #
# Timer Decorator
# -------------------------------------------------------------------------------- #


def timeit(func: Callable) -> Callable:
    """
    A decorator that times the execution of a function and returns both the result
    and execution time.

    Args:
        func: The function to time

    Returns:
        The result of the function and its execution time in seconds
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return result, run_time
    return wrapper_timer


# -------------------------------------------------------------------------------- #
# Required Parameters Decorator
# -------------------------------------------------------------------------------- #


def required_parameters(*required_args: str) -> Callable:
    """
    A decorator that checks if required parameters are provided. 
    Astral's Sentinel type of 'NOT_GIVEN' is used to indicate that a parameter
    is not provided. 

    Args:
        *required_args: The required parameters

    Returns:
        The decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg in required_args:
                if arg not in kwargs or kwargs[arg] == NOT_GIVEN:
                    raise MissingParameterError(parameter_name=arg, function_name=func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# -------------------------------------------------------------------------------- #
# Cost Decorator
# -------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------- #
# Generics
# -------------------------------------------------------------------------------- #
# P = ParamSpec("P")
# R = TypeVar("R")


P = ParamSpec("P")
R = TypeVar("R", bound=AstralBaseResponse)


class ModelNameError(Exception):
    """
    Exception raised when a model name is not valid.
    """

    def __init__(self, model_name: str):
        self.message = f"Invalid model name: {model_name}"
        super().__init__(self.message)


class ProviderNotFoundForModelError(Exception):
    """
    Exception raised when a provider is not found for a model.
    """

    def __init__(self, model_name: str):
        self.message = f"Provider not found for model: {model_name}"
        super().__init__(self.message)


# -------------------------------------------------------------------------------- #
# Cost Decorator
# -------------------------------------------------------------------------------- #

from astral_ai.tracing._cost_strategies import BaseCostStrategy
from astral_ai.utilities.cost_utils import get_model_costs


# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

from pydantic import BaseModel

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


# -------------------------------------------------------------------------------- #
def calculate_cost_decorator(
    func: Callable[Concatenate[AstralResource, P], R]
) -> Callable[Concatenate[AstralResource, P], Union[R, Tuple[R, float]]]:
    """
    A decorator that calculates the cost of a function call.
    If a cost_strategy is provided, it will be used to process the cost.
    """
    @functools.wraps(func)
    def wrapper(
        self: AstralResource,
        response_format: Optional[_StructuredOutputT] = None,    
        *args,
        **kwargs,
    ) -> Union[R, Tuple[R, float]]:

        model_name = self.model
        model_provider = self.model_provider

        if not isinstance(model_name, ModelName):
            raise ModelNameError(model_name=model_name)

        if not isinstance(model_provider, ModelProvider):
            raise ProviderNotFoundForModelError(model_name=model_name)
        

        cost_strategy = self.cost_strategy

        if response_format is None:
            # Execute the wrapped function to get the response.
            result = func(self, *args, **kwargs)
        else:
            result = func(self, response_format, *args, **kwargs)

        # Retrieve cost details for the model.
        costs = get_model_costs(model_name=model_name, model_provider=model_provider)

        output = cost_strategy.handle_costs(costs=costs) if cost_strategy else result
        if not isinstance(model_name, ModelName):
            raise ModelNameError(model_name=model_name)

        if not isinstance(model_provider, ModelProvider):
            raise ProviderNotFoundForModelError(model_name=model_name)

            # Execute the wrapped function to get the response.
        result = func(self, *args, **kwargs)

        # Retrieve cost details for the model.
        costs = get_model_costs(model_name=model_name, model_provider=model_provider, usage=result.usage)

        output = cost_strategy.handle_costs(costs=costs) if cost_strategy else result

        return output

 
