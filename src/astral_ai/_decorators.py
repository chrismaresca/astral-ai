from __future__ import annotations
from pydantic import BaseModel
from astral_ai.tracing._cost_strategies import BaseCostStrategy

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
import time
import functools
from typing import (
    Optional,
    Callable,
    Tuple,
    Union,
    ParamSpec,
    TypeVar,
    Concatenate,
)

# Astral AI Utilities
from astral_ai.utilities import get_model_costs

# Astral AI Types
from astral_ai._types import NOT_GIVEN

# Astral AI Constants
from astral_ai.constants._models import ModelName, ModelProvider

# Astral AI Exceptions
from astral_ai.errors.exceptions import MissingParameterError


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
