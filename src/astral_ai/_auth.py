from __future__ import annotations
# ------------------------------------------------------------------------------
# _Auth
# ------------------------------------------------------------------------------

"""
This module contains the authentication registry and strategies.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
from functools import wraps, lru_cache
from typing import Callable, TYPE_CHECKING, Any, Dict, Literal, Union, ClassVar, Tuple, TypeAlias

# Pydantic
from pydantic import BaseModel, Field

# Astral AI Models
from astral_ai.constants._models import ModelProvider, ModelName

# Astral AI Providers
from astral_ai.providers._types import BaseProviderClient


# ------------------------------------------------------------------------------
# Auth Method Names
# ------------------------------------------------------------------------------


AUTH_METHOD_NAMES = Literal["api_key", "ad_token", "bearer_token", "oauth", "service_account"]


# ------------------------------------------------------------------------------
# Auth Environment Variables
# ------------------------------------------------------------------------------


AUTH_ENV_VARS: TypeAlias = Dict[Union[str, AUTH_METHOD_NAMES], str]

# ------------------------------------------------------------------------------
# Auth Method Config
# ------------------------------------------------------------------------------


class AuthMethodConfig(BaseModel):
    """
    Base configuration for an authentication method.

    Extend this class for provider-specific authentication configurations.
    """
    auth_method: AUTH_METHOD_NAMES = Field(
        default="api_key", description="The name of the authentication method to use."
    )
    environment_variables: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables to use for the authentication method."
    )

# ------------------------------------------------------------------------------
# Auth Config Type
# ------------------------------------------------------------------------------


AUTH_CONFIG_TYPE: TypeAlias = Dict[AUTH_METHOD_NAMES, AuthMethodConfig]

# ------------------------------------------------------------------------------
# Auth Callable
# ------------------------------------------------------------------------------

AuthCallable = Callable[
    [BaseProviderClient, AUTH_CONFIG_TYPE, AUTH_ENV_VARS],
    BaseProviderClient,
]

# ------------------------------------------------------------------------------
# Auth Registry Meta
# ------------------------------------------------------------------------------


class AuthRegistryMeta(type):
    """
    Metaclass that collects methods decorated with @auth_method.
    """
    def __new__(
        mcls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any
    ) -> type:
        cls = super().__new__(mcls, name, bases, namespace)
        # Merge auth strategies from base classes.
        auth_strategies: Dict[str, AuthCallable] = {}
        for base in bases:
            auth_strategies.update(getattr(base, "_auth_strategies", {}))
        # Register strategies from this class.
        auth_strategies.update({
            getattr(attr, "_auth_name"): attr
            for attr in namespace.values()
            if callable(attr) and hasattr(attr, "_auth_name")
        })
        cls._auth_strategies = auth_strategies
        return cls

# ------------------------------------------------------------------------------
# Auth Registry Base Class
# ------------------------------------------------------------------------------


class AuthRegistry(metaclass=AuthRegistryMeta):
    """
    Registry for authentication strategies.
    """
    # This annotation ensures that type checkers know that every subclass has _auth_strategies.
    _auth_strategies: ClassVar[Dict[str, AuthCallable]] = {}

# ------------------------------------------------------------------------------
# Auth Decorator
# ------------------------------------------------------------------------------


def auth_method(name: str) -> Callable[[AuthCallable], AuthCallable]:
    """
    Decorator to register an authentication strategy under a given name.

    Args:
        name (str): The name to register the authentication strategy under.

    Returns:
        Callable: A decorator function that registers the auth strategy.
    """
    def decorator(func: AuthCallable) -> AuthCallable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ModelProvider:
            return func(*args, **kwargs)
        setattr(wrapper, "_auth_name", name)
        return wrapper
    return decorator

# ------------------------------------------------------------------------------
# Environment Variables Caching
# ------------------------------------------------------------------------------


@lru_cache(maxsize=1, typed=True)
def get_env_vars() -> AUTH_ENV_VARS:
    """
    Reads and caches environment variables.
    """
    return dict(os.environ)
