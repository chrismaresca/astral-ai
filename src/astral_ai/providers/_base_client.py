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
    AUTH_METHOD_NAME_TYPES,
    AUTH_CONFIG_TYPE,
    get_env_vars,
)
from astral_ai.constants._models import ModelProvider

# Exceptions
from astral_ai.exceptions import (
    ProviderAuthenticationError,
    UnknownAuthMethodError,
    AuthMethodFailureError,
)

# Logging
from astral_ai.logger import logger

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
    metaclass=AuthRegistryMeta
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
    """
    _auth_strategies: Dict[AUTH_METHOD_NAME_TYPES, AuthCallable] = {}
    _client_cache: ClassVar[Dict[Any, ProviderClientT]] = {}
    _model_provider: ModelProvider = None

    def __init__(self, config: Optional[AUTH_CONFIG_TYPE] = None) -> None:
        """Initialize the provider client with optional configuration.

        Args:
            config: Optional configuration dictionary. If not provided, will attempt
                   to load from config.yaml file.
        """
        # Load configuration from YAML if not provided
        self._config: AUTH_CONFIG_TYPE = config or self.load_config() or {}
        logger.debug(f"Configuration loaded: {self._config}")

        # Handle client caching
        cache_client = self._config.get("cache_client", True)
        cache_key = self.__class__

        if cache_client and cache_key in self._client_cache:
            logger.info("Using cached provider client.")
            self.client: ProviderClientT = self._client_cache[cache_key]
        else:
            self.client = self._get_or_authenticate_client()
            if cache_client:
                logger.info("Caching provider client for future use.")
                self._client_cache[cache_key] = self.client

    @classmethod
    def load_config(cls) -> Optional[AUTH_CONFIG_TYPE]:
        """
        Reads a config.yaml file if it exists.
        The config may specify an "auth_method" key (and any other credentials).
        """
        config_path = Path("config.yaml")
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config_data: AUTH_CONFIG_TYPE = yaml.safe_load(f)
                    logger.debug(f"Config loaded from {config_path}: {config_data}")
                    return config_data
            except Exception as e:
                logger.error(f"Failed to load config file {config_path}: {e}", exc_info=True)
                return None
        logger.debug("No config.yaml found; proceeding without a configuration file.")
        return None

    def _get_or_authenticate_client(self) -> ProviderClientT:
        """
        Attempts to authenticate using a specified auth method (if provided in config)
        or by looping through all available strategies.
        Caches environment variables for efficiency.
        """
        env = get_env_vars()
        auth_method_config = self._config.get("auth_method")
        supported_methods = list(self._auth_strategies.keys())
        logger.debug(f"Supported authentication strategies: {supported_methods}")

        # If a specific auth method is configured, try that one.
        if auth_method_config:
            auth_method_name = auth_method_config.auth_method
            strategy = self._auth_strategies.get(auth_method_name)
            if not strategy:
                error = UnknownAuthMethodError(auth_method_name, supported_methods)
                logger.error(error)
                raise error

            try:
                client = strategy(self, self._config, env)
                if client:
                    logger.info(f"Authentication succeeded using method: '{auth_method_name}'")
                    return client
            except Exception as e:
                logger.error(
                    f"Authentication method '{auth_method_name}' failed: {e}", exc_info=True
                )
                raise AuthMethodFailureError(auth_method_name, e) from e

        # Otherwise, try all available strategies.
        for name, strategy in self._auth_strategies.items():
            logger.info(f"Attempting authentication using strategy: '{name}'")
            try:
                client = strategy(self, self._config, env)
                if client:
                    logger.info(f"Authentication succeeded using strategy: '{name}'")
                    return client
            except Exception as e:
                logger.warning(
                    f"Authentication strategy '{name}' failed: {e}", exc_info=True
                )
                continue

        error_message = (
            f"All authentication methods failed. Strategies attempted: {supported_methods}"
        )
        logger.error(error_message)
        raise ProviderAuthenticationError(error_message)

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