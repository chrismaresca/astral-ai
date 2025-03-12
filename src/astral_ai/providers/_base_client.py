# -------------------------------------------------------------------------------- #
# Base Provider Client
# -------------------------------------------------------------------------------- #
# This module defines the base provider client that all specific provider
# implementations must inherit from. It handles:
#   - Authentication strategy management
#   - Client caching
#   - Configuration loading
#   - Generic type definitions for requests/responses
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import yaml
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

from abc import ABCMeta

# Provider Types
from astral_ai.providers._generics import (
    ProviderClientT,
    ProviderResponseChatT,
    ProviderResponseStructuredT,
    ProviderResponseStreamingT,
    ProviderRequestChatT,
    ProviderRequestStructuredT,
    ProviderRequestStreamingT,
)


# Authentication
from astral_ai._auth import (
    AuthMethodConfig,
    AuthRegistryMeta,
    AuthCallable,
    AUTH_METHOD_NAMES,
    AUTH_CONFIG_TYPE,
    AUTH_CONFIG_TYPE_WITH_PROVIDER,
    get_env_vars,
)
from astral_ai.constants._models import ModelProvider

# Exceptions
from astral_ai.errors.exceptions import (
    ProviderAuthenticationError,
    UnknownAuthMethodError,
    AuthMethodFailureError,
    AstralAuthError,
    AstralAuthMethodFailureError,
    AstralUnknownAuthMethodError,
)

# Logging
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Helper to Read Config
# -------------------------------------------------------------------------------- #

# TODO: Eventually this should be replaced with a class that reads them all and caches them on init. but is this secure?
def read_config(config_path: Path) -> Optional[AUTH_CONFIG_TYPE_WITH_PROVIDER]:
    """
    Reads a config.yaml file if it exists.
    The config may specify an "auth_method" key (and any other credentials).

    Args:
        config_path: Path to the configuration file
        auth_config_type: Type of authentication configuration

    Returns:
        Optional[AUTH_CONFIG_TYPE]: The loaded configuration or None if file doesn't exist
    """
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config_data: AUTH_CONFIG_TYPE_WITH_PROVIDER = yaml.safe_load(f)
                logger.debug(f"Astral authentication configuration loaded from {config_path}: {config_data}")
                return config_data
        except Exception as e:
            logger.error(f"Failed to load Astral authentication configuration file {config_path}: {e}", exc_info=True)
    logger.debug("No Astral authentication configuration file found; proceeding without a configuration file.")
    return None


# -------------------------------------------------------------------------------- #
# Combined Meta
# -------------------------------------------------------------------------------- #

class CombinedMeta(AuthRegistryMeta, ABCMeta):
    pass

# -------------------------------------------------------------------------------- #
# Base Provider Client
# -------------------------------------------------------------------------------- #


class BaseProviderClient(
    ABC,
    Generic[
        ProviderClientT,
        ProviderRequestChatT,
        ProviderRequestStructuredT,
        ProviderRequestStreamingT,
        ProviderResponseChatT,
        ProviderResponseStructuredT,
        ProviderResponseStreamingT
    ],
    metaclass=CombinedMeta
    # metaclass=ModelProviderMeta
):
    """Base class for all provider clients (OpenAI, Anthropic, etc.).

    This class implements:
    - Multi-strategy authentication with configurable methods
    - Client caching mechanism (can be disabled via config)
    - Configuration loading from YAML
    - Generic type definitions for provider-specific request/response types

    Type Parameters:
        ProviderClientT: The provider's client type (e.g. OpenAI, Anthropic client)
        ProviderRequestChatT: The provider's chat request type
        ProviderRequestStructuredT: The provider's structured request type
        ProviderRequestStreamingT: The provider's streaming request type
        ProviderResponseChatT: The provider's chat response type
        ProviderResponseStructuredT: The provider's structured response type
        ProviderResponseStreamingT: The provider's streaming response type

    Attributes:
        _auth_strategies: Registry of available authentication strategies
        _client_cache: Cache of authenticated provider clients
        client: The authenticated provider client instance
        _config_path: Path to the primary configuration file
    """
    _auth_strategies: Dict[AUTH_METHOD_NAMES, AuthCallable] = {}
    _client_cache: ClassVar[Dict[Any, ProviderClientT]] = {}
    _model_provider: ModelProvider = None
    _config_path: ClassVar[Path] = Path("astral.yaml")

    # --------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------

    def __init__(self, config: Optional[AUTH_CONFIG_TYPE] = None) -> None:
        """Initialize the provider client with optional configuration.

        Args:
            config: Optional configuration dictionary. If not provided, will attempt
                   to load from config.yaml file.
        """

        self._full_config: AUTH_CONFIG_TYPE_WITH_PROVIDER = config or self.load_full_config() or {}
        # Extract the configuration specific to this model provider.
        self._config: AUTH_CONFIG_TYPE = self.get_provider_config()
        logger.debug(f"Provider-specific config for '{self._model_provider}': {self._config}")

        # Handle client caching
        cache_client = self._config.get("cache_client", True)
        cache_key = self.__class__

        if cache_client and cache_key in self._client_cache:
            logger.debug("Using cached provider client.")
            self.client: ProviderClientT = self._client_cache[cache_key]
        else:
            self.client = self._get_or_authenticate_client()
            if cache_client:
                logger.debug("Caching provider client for future use.")
                self._client_cache[cache_key] = self.client

    # --------------------------------------------------------------------------
    # Load Full Config
    # --------------------------------------------------------------------------

    @classmethod
    def load_full_config(cls) -> Optional[AUTH_CONFIG_TYPE_WITH_PROVIDER]:
        """
        Loads the complete configuration from astral.yaml.
        Expected format:

        openai:
          auth_method:
            auth_method: "api_key"
            environment_variables:
              OPENAI_API_KEY: "your_openai_key"
          cache_client: True
          api_base: "https://api.openai.com"

        huggingface:
          auth_method:
            auth_method: "oauth"
            environment_variables:
              HUGGINGFACE_OAUTH_TOKEN: "your_oauth_token_here"
          cache_client: True
          api_base: "https://api-inference.huggingface.co"
        """
        return read_config(cls._config_path)

    # --------------------------------------------------------------------------
    # Get Provider Config
    # --------------------------------------------------------------------------

    def get_provider_config(self) -> Dict[str, Any]:
        """
        Extracts and returns the configuration section for this provider,
        based on its _model_provider identifier.
        """
        return self._full_config.get(self._model_provider, {})

    # --------------------------------------------------------------------------
    # Get or Authenticate Client
    # --------------------------------------------------------------------------

    def _get_or_authenticate_client(self) -> ProviderClientT:
        """
        Attempts to authenticate using a specified auth method (if provided in config)
        or by looping through all available strategies.
        
        Returns:
            ProviderClientT: An authenticated provider client
            
        Raises:
            AstralUnknownAuthMethodError: If the specified auth method is not supported
            AstralAuthMethodFailureError: If the specified auth method fails
            AstralAuthError: If all authentication methods fail
        """
        env = get_env_vars()
        auth_method_config = self._config.get("auth_method")
        supported_methods = list(self._auth_strategies.keys())
        
        # Determine which authentication methods to try
        methods_to_try = []
        if auth_method_config:
            # If specific auth method is configured, only try that one
            auth_method_name = auth_method_config.auth_method
            if auth_method_name not in self._auth_strategies:
                error = AstralUnknownAuthMethodError(auth_method_name, supported_methods)
                logger.error(f"{error}")
                raise error
            methods_to_try = [(auth_method_name, self._auth_strategies[auth_method_name])]
            logger.debug(f"Using configured authentication method: '{auth_method_name}' for '{self._model_provider}'")
        else:
            # Otherwise try all available strategies
            methods_to_try = list(self._auth_strategies.items())
            logger.debug(f"No specific auth method configured for '{self._model_provider}'. Will try all available methods: {supported_methods}")
        
        # Try each authentication method
        errors = []
        for name, strategy in methods_to_try:
            logger.debug(f"Attempting authentication {self._model_provider} using method: '{name}'")
            try:
                client = strategy(self, self._config, env)
                if client:
                    logger.debug(f"Authentication succeeded for '{self._model_provider}' using method: '{name}'")
                    return client
            except Exception as e:
                error_msg = f"Authentication method '{name}' failed for '{self._model_provider}': {e}"
                logger.warning(error_msg, exc_info=True)
                errors.append((name, str(e)))
                
                # If using a specific configured method, fail immediately
                if auth_method_config:
                    raise AstralAuthMethodFailureError(error_msg) from e
        
        # If we get here, all authentication methods failed
        error_details = "\n".join([f"- {name}: {error}" for name, error in errors])
        error_message = f"All authentication methods failed for provider '{self._model_provider}':\n{error_details}"
        logger.error(error_message)
        raise AstralAuthError(error_message)

    # --------------------------------------------------------------------------
    # Create Completion
    # --------------------------------------------------------------------------
    @abstractmethod
    def create_completion_chat(self, request: ProviderRequestChatT) -> ProviderResponseChatT:
        """
        Call the provider API to create a completion.
        """
        pass

    @abstractmethod
    def create_completion_structured(self, request: ProviderRequestStructuredT) -> ProviderResponseStructuredT:
        """
        Call the provider API to create a structured completion.
        """
        pass

    # @abstractmethod
    # def create_completion_streaming(self, request: ProviderRequestStreamingT) -> ProviderResponseStreamingT:
    #     """
    #     Call the provider API to create a streaming completion.
    #     """
    #     pass
