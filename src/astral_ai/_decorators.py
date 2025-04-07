from __future__ import annotations


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
import inspect
from typing import (
    Callable,
    TypeVar,
    Protocol,
    runtime_checkable,
    Any,
    Union,
    Awaitable,
    ParamSpec
)

# Astral AI Types
from astral_ai._types import NOT_GIVEN

# Astral AI Exceptions
from astral_ai.errors.exceptions import MissingParameterError


# -------------------------------------------------------------------------------- #
# Timer Decorators
# -------------------------------------------------------------------------------- #

# Define a Protocol for response types that have a latency_ms field
@runtime_checkable
class HasLatency(Protocol):
    latency_ms: float

T = TypeVar('T')
P = ParamSpec('P')

def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """
    A decorator that times the execution of a synchronous function and sets the latency_ms field
    on the returned response object.
    
    This decorator works with any response object that has a latency_ms attribute,
    such as AstralChatResponse or AstralStructuredResponse.

    Args:
        func: The synchronous function to time

    Returns:
        The original result with latency_ms set to the execution time in milliseconds
    """
    @functools.wraps(func)
    def sync_wrapper_timer(self, *args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        # Set the latency in milliseconds if the response object has latency_ms field
        if isinstance(result, HasLatency):
            result.latency_ms = run_time * 1000
        
        return result
    return sync_wrapper_timer

def atimeit(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    A decorator that times the execution of an asynchronous function and sets the latency_ms field
    on the returned response object.
    
    This decorator works with any response object that has a latency_ms attribute,
    such as AstralChatResponse or AstralStructuredResponse.

    Args:
        func: The asynchronous function to time

    Returns:
        The original result with latency_ms set to the execution time in milliseconds
    """
    @functools.wraps(func)
    async def async_wrapper_timer(self, *args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        result = await func(self, *args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        
        # Set the latency in milliseconds if the response object has latency_ms field
        if isinstance(result, HasLatency):
            result.latency_ms = run_time * 1000
        
        return result
    return async_wrapper_timer

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
