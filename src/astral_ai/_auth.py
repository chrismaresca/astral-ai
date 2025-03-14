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
import traceback

# Pydantic
from pydantic import BaseModel, Field

# Astral AI Models
from astral_ai.constants._models import ModelProvider, ModelName

# Astral AI Providers
from astral_ai.providers._types import BaseProviderClient

# Astral AI Logger
from astral_ai.logger import logger


# ------------------------------------------------------------------------------
# Auth Method Names
# ------------------------------------------------------------------------------


AUTH_METHOD_NAMES = Literal["api_key", "api_key_with_base_url", "ad_token", "bearer_token", "oauth", "service_account"]


# ------------------------------------------------------------------------------
# Auth Environment Variables
# ------------------------------------------------------------------------------


AUTH_ENV_VARS: TypeAlias = Dict[Union[str, AUTH_METHOD_NAMES], str]

# ------------------------------------------------------------------------------
# Auth Method Required Credentials and Environment Variables
# ------------------------------------------------------------------------------

# Hierarchical mapping by provider, then auth method
# This allows for more intuitive organization and easier lookup
AUTH_CONFIG: Dict[ModelProvider, Dict[AUTH_METHOD_NAMES, Dict[str, Any]]] = {
    # OpenAI configurations
    "openai": {
        "api_key": {
            "required": ["api_key"],
            "env_vars": {"api_key": "OPENAI_API_KEY"}
        },
        "api_key_with_base_url": {
            "required": ["api_key"],
            "env_vars": {"api_key": "OPENAI_API_KEY"}
        }
    },

    # Azure configurations
    "azureOpenAI": {
        "api_key": {
            "required": ["api_key", "api_version"],
            "env_vars": {"api_key": "AZURE_OPENAI_API_KEY", "api_version": "AZURE_OPENAI_API_VERSION"}
        }
    },

    # Anthropic configurations
    "anthropic": {
        "api_key": {
            "required": ["api_key"],
            "env_vars": {"api_key": "ANTHROPIC_API_KEY"}
        }
    },

    # DeepSeek configurations
    "deepseek": {
        "api_key": {
            "required": ["api_key"],
            "env_vars": {"api_key": "DEEPSEEK_API_KEY"}
        },

    },
}

# Default configurations used as fallbacks
DEFAULT_AUTH_CONFIG: Dict[AUTH_METHOD_NAMES, Dict[str, Any]] = {
    "api_key": {
        "required": ["api_key"],
        "env_vars": {"api_key": "{provider}_API_KEY"}
    },
    "api_key_with_base_url": {
        "required": ["api_key", "base_url"],
        "env_vars": {"api_key": "{provider}_API_KEY", "base_url": "{provider}_BASE_URL"}
    },
    "ad_token": {
        "required": ["tenant_id", "client_id", "client_secret"],
        "env_vars": {
            "tenant_id": "{provider}_TENANT_ID",
            "client_id": "{provider}_CLIENT_ID",
            "client_secret": "{provider}_CLIENT_SECRET"
        }
    },
    "bearer_token": {
        "required": ["token"],
        "env_vars": {"token": "{provider}_TOKEN"}
    },
    "oauth": {
        "required": ["client_id", "client_secret", "scope"],
        "env_vars": {
            "client_id": "{provider}_CLIENT_ID",
            "client_secret": "{provider}_CLIENT_SECRET",
            "scope": "{provider}_SCOPE"
        }
    },
    "service_account": {
        "required": ["service_account_file"],
        "env_vars": {"service_account_file": "{provider}_SERVICE_ACCOUNT_FILE"}
    }
}

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


# Base configuration dictionary for a single authentication method
AuthMethodConfigDict = Dict[str, Any]

# Configuration for all available authentication methods for a provider
ProviderAuthConfigDict = Dict[AUTH_METHOD_NAMES, AuthMethodConfigDict]

# Top-level type: Configuration dictionary for a single provider
AUTH_CONFIG_TYPE: TypeAlias = Dict[AUTH_METHOD_NAMES, AuthMethodConfig]

# Full configuration with provider mapping
AUTH_CONFIG_TYPE_WITH_PROVIDER: TypeAlias = Dict[ModelProvider, AUTH_CONFIG_TYPE]

# ------------------------------------------------------------------------------
# Auth Callable
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Auth Callable Protocol
# ------------------------------------------------------------------------------

from typing import Protocol, runtime_checkable, Any

# -------------------------------------------------------------------------------- #
# Auth Callable Protocol
# -------------------------------------------------------------------------------- #
from astral_ai.providers._generics import ProviderClientType
@runtime_checkable
class AuthCallable(Protocol):
    """
    Protocol defining the interface for authentication strategy callables.
    
    This protocol ensures that all authentication methods follow the same signature,
    making them interchangeable as strategies in the authentication system.
    """
    def __call__(
        self, 
        instance: 'BaseProviderClient',
        config: AUTH_CONFIG_TYPE, 
        env: AUTH_ENV_VARS,
        async_client: bool = False
    ) -> ProviderClientType: ...


# TODO: Remove this
# # Type definition for auth callables that return provider clients
# AuthCallable = Callable[
#     ['BaseProviderClient', AUTH_CONFIG_TYPE, AUTH_ENV_VARS],
#     ProviderClientType,
# ]

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
        # Debug: Log class creation
        logger.debug(f"Creating class with AuthRegistryMeta: {name}")

        # Merge auth strategies from base classes.
        auth_strategies: Dict[str, AuthCallable] = {}
        for base in bases:
            base_strategies = getattr(base, "_auth_strategies", {})
            logger.debug(f"Base class {base.__name__} has strategies: {list(base_strategies.keys())}")
            auth_strategies.update(base_strategies)

        # Register strategies from this class.
        class_strategies = {
            getattr(attr, "_auth_name"): attr
            for attr in namespace.values()
            if callable(attr) and hasattr(attr, "_auth_name")
        }
        logger.debug(f"Found decorated methods in {name}: {list(class_strategies.keys())}")

        auth_strategies.update(class_strategies)
        cls._auth_strategies = auth_strategies

        logger.debug(f"Final auth strategies for {name}: {list(auth_strategies.keys())}")
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
    logger.debug(f"Registering authentication method: '{name}'")

    # Define the decorator function
    def decorator(func: AuthCallable) -> AuthCallable:
        logger.debug(f"Decorating function '{func.__name__}' with auth_method '{name}'")

        # Wrap the function to preserve its metadata
        # @auth_error_handler
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Extract provider information for better error messages
            provider_name: ModelProvider = getattr(self, "_model_provider", "unknown")
            logger.debug(f"Auth method '{name}' called for provider '{provider_name}'")

            # Call the original authentication method without any error handling
            # Let errors bubble up to __init__ in BaseProviderClient
            logger.debug(f"Calling original auth method function '{func.__name__}'")
            result = func(self, *args, **kwargs)
            logger.debug(f"Auth method '{name}' for provider '{provider_name}' completed successfully")
            return result

        setattr(wrapper, "_auth_name", name)
        logger.debug(f"Successfully registered auth method '{name}' with function '{func.__name__}'")
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# Environment Variables Caching
# ------------------------------------------------------------------------------


@lru_cache(maxsize=1, typed=True)
def get_env_vars() -> AUTH_ENV_VARS:
    """
    Reads and caches environment variables.

    Returns:
        Dict: Dictionary containing all environment variables
    """
    logger.debug("Loading environment variables (cached)")
    env_vars = dict(os.environ)

    # Log environment variables related to authentication, masking sensitive values
    auth_related_vars = {k: "********" for k in env_vars if any(
        pattern in k.upper() for pattern in
        ["API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
    )}

    if auth_related_vars:
        logger.debug(f"Found authentication-related environment variables: {list(auth_related_vars.keys())}")
    else:
        logger.debug("No authentication-related environment variables found")

    return env_vars
