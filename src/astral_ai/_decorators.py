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
from typing import List, Dict, Optional, Callable, Tuple, Union, TYPE_CHECKING, ParamSpec, TypeVar

# Astral imports
from astral_ai._types._request import NOT_GIVEN

# Standard library imports
import time
import functools

# Types
from typing import Callable

# Exceptions
from astral_ai.exceptions import MissingParameterError

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


if TYPE_CHECKING:
    from astral_ai.providers._base_client import BaseProviderClient
    from astral_ai.providers._generics import ProviderRequest
    from astral_ai.tracing._cost_strategies import CostStrategy


# -------------------------------------------------------------------------------- #
# Generics
# -------------------------------------------------------------------------------- #

P = ParamSpec("P")
R = TypeVar("R")


def calculate_cost_decorator(
    func: Callable[..., R]
) -> Callable[..., Union[R, Tuple[R, float]]]:
    """
    A decorator that calculates the cost of a function call.
    If a cost_strategy is provided, it will be used to process the cost.
    """
    @functools.wraps(func)
    def wrapper(
        self: BaseProviderClient,
        request: ProviderRequest,
        cost_strategy: Optional[CostStrategy] = None,
        *args,
        **kwargs,
    ) -> Union[R, Tuple[R, float]]:
        model_provider = self._model_provider
        model_name = request.model

        # Retrieve cost details for the model.
        costs = get_model_costs(model_name, model_provider)

        # Execute the wrapped function to get the response.
        result = func(self, request, *args, **kwargs)

        # Calculate the cost using usage details and cost configuration.
        cost = calculate_cost(result, costs)

        # If a cost_strategy is specified, use it; otherwise, return the response.
        if cost_strategy is not None:
            return cost_strategy.handle_cost(result, cost)
        else:
            return result

    return wrapper
