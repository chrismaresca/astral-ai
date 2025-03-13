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

# Astral AI Exceptions
from astral_ai.errors.exceptions import (
    AstralAuthConfigurationError,
    AstralMissingCredentialsError,
    AstralInvalidCredentialsError,
    AstralEnvironmentVariableError,
    AstralAuthMethodFailureError,
    AstralAuthError,
    AstralUnknownAuthMethodError,
)

# Astral AI Error Decorators and Formatters
from astral_ai.errors.error_decorators import auth_error_handler
from astral_ai.errors.error_formatter import format_error_message

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
        # Debug: Log class creation
        logger.info(f"Creating class with AuthRegistryMeta: {name}")

        # Merge auth strategies from base classes.
        auth_strategies: Dict[str, AuthCallable] = {}
        for base in bases:
            base_strategies = getattr(base, "_auth_strategies", {})
            logger.info(f"Base class {base.__name__} has strategies: {list(base_strategies.keys())}")
            auth_strategies.update(base_strategies)

        # Register strategies from this class.
        class_strategies = {
            getattr(attr, "_auth_name"): attr
            for attr in namespace.values()
            if callable(attr) and hasattr(attr, "_auth_name")
        }
        logger.info(f"Found decorated methods in {name}: {list(class_strategies.keys())}")

        auth_strategies.update(class_strategies)
        cls._auth_strategies = auth_strategies

        logger.info(f"Final auth strategies for {name}: {list(auth_strategies.keys())}")
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
    logger.info(f"Registering authentication method: '{name}'")

    # Define the decorator function
    def decorator(func: AuthCallable) -> AuthCallable:
        logger.info(f"Decorating function '{func.__name__}' with auth_method '{name}'")

        # Wrap the function to preserve its metadata
        # @auth_error_handler
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Extract provider information for better error messages
            provider_name: ModelProvider = getattr(self, "_model_provider", "unknown")
            logger.info(f"Auth method '{name}' called for provider '{provider_name}'")

            # Call the original authentication method without any error handling
            # Let errors bubble up to __init__ in BaseProviderClient
            logger.info(f"Calling original auth method function '{func.__name__}'")
            result = func(self, *args, **kwargs)
            logger.info(f"Auth method '{name}' for provider '{provider_name}' completed successfully")
            return result

        setattr(wrapper, "_auth_name", name)
        logger.info(f"Successfully registered auth method '{name}' with function '{func.__name__}'")
        return wrapper
    return decorator

# ------------------------------------------------------------------------------
# Auth Helper Functions
# ------------------------------------------------------------------------------

def validate_credentials(
    auth_method: AUTH_METHOD_NAMES,
    provider_name: ModelProvider,
    config: AUTH_CONFIG_TYPE,
    env: AUTH_ENV_VARS
) -> Dict[str, str]:
    """
    Validates that all required credentials for an auth method are present.
    Uses provider-specific configurations with fallback to defaults.

    Args:
        auth_method: The authentication method name.
        provider_name: The provider name from ModelProvider.
        config: The configuration dictionary.
        env: The environment variables dictionary.

    Returns:
        Dict[str, str]: Dictionary of validated credentials.

    Raises:
        AstralMissingCredentialsError: If any required credentials are missing.
    """
    logger.info(f"Validating credentials for provider '{provider_name}', auth method '{auth_method}'")

    # Normalize provider name to access the config
    provider = provider_name.lower()
    logger.info(f"Normalized provider name: '{provider}'")

    # Try to get provider-specific config for this auth method
    provider_config = None
    if provider_name in AUTH_CONFIG and auth_method in AUTH_CONFIG[provider_name]:
        provider_config = AUTH_CONFIG[provider_name][auth_method]
        logger.info(f"Found provider-specific auth configuration for '{provider_name}.{auth_method}': {provider_config}")
    else:
        # Fall back to default config for this auth method
        provider_config = DEFAULT_AUTH_CONFIG.get(auth_method, {})
        logger.info(f"Using default auth configuration for '{auth_method}': {provider_config}")

    # Get required credentials and environment variable mappings
    required_creds = provider_config.get("required", [])
    env_vars = provider_config.get("env_vars", {})
    logger.info(f"Required credentials: {required_creds}")
    logger.info(f"Environment variable mappings: {env_vars}")

    # If we're using the default config, replace {provider} placeholders
    for cred, env_var in env_vars.items():
        if "{provider}" in env_var:
            original = env_vars[cred]
            env_vars[cred] = env_var.replace("{provider}", provider.upper())
            logger.info(f"Replaced placeholder in env var: '{original}' -> '{env_vars[cred]}'")

    # Check for missing credentials
    missing_creds = []
    credentials = {}

    for cred in required_creds:
        # Try config first, then environment variable
        value = config.get(cred)

        if value:
            logger.info(f"Found credential '{cred}' in configuration")
        elif cred in env_vars:
            env_var_name = env_vars[cred]
            value = env.get(env_var_name)
            if value:
                logger.info(f"Found credential '{cred}' in environment variable '{env_var_name}'")
            else:
                logger.info(f"Environment variable '{env_var_name}' not found or empty")

        if value:
            credentials[cred] = value
            # Mask the credential value for security in logs
            masked_value = "********" if cred in ("api_key", "token", "client_secret") else value
            logger.info(f"Added credential '{cred}' with value: {masked_value}")
        else:
            missing_creds.append(cred)
            logger.info(f"Missing required credential: '{cred}'")

    # If any credentials are missing, raise an error
    if missing_creds:
        logger.info(f"Credential validation failed: missing {missing_creds}")
        
        # Create a detailed error message that explains what's missing and how to fix it
        missing_creds_list = ", ".join(missing_creds)
        env_vars_list = []
        
        # Build a list of expected environment variables for the missing credentials
        for cred in missing_creds:
            if cred in env_vars:
                env_vars_list.append(f"{env_vars[cred]}")
        
        detailed_message = f"Missing required credentials for {provider_name} authentication using '{auth_method}': {missing_creds_list}"
        
        if env_vars_list:
            env_vars_str = ", ".join(env_vars_list)
            detailed_message += f". Set environment variable(s): {env_vars_str}"
            
        raise AstralMissingCredentialsError(
            detailed_message,
            auth_method_name=auth_method,
            provider_name=provider_name,
            required_credentials=required_creds,
            missing_credentials=missing_creds
        ) from None

    logger.info(f"Credential validation successful for '{provider_name}' using '{auth_method}'")
    return credentials

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
    logger.info("Loading environment variables (cached)")
    env_vars = dict(os.environ)

    # Log environment variables related to authentication, masking sensitive values
    auth_related_vars = {k: "********" for k in env_vars if any(
        pattern in k.upper() for pattern in
        ["API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]
    )}

    if auth_related_vars:
        logger.info(f"Found authentication-related environment variables: {list(auth_related_vars.keys())}")
    else:
        logger.info("No authentication-related environment variables found")

    return env_vars
