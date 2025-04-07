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
import traceback

# Provider Types
from astral_ai.providers._generics import (
    ProviderResponseChatT,
    ProviderResponseStructuredT,
    ProviderResponseStreamingT,
    ProviderRequestChatT,
    ProviderRequestStructuredT,
    ProviderRequestStreamingT,
    SyncProviderClientT,
    AsyncProviderClientT,
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
    AstralAuthError,
    AstralAuthMethodFailureError,
    AstralUnknownAuthMethodError,
    MultipleAstralAuthenticationErrors,
)

from astral_ai.errors.error_decorators import auth_error_handler
from astral_ai.errors.error_formatter import format_error_message

# Logging
from astral_ai.logger import logger

# -------------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------------- #

CONFIG_LOG_PREFIX = "📋"

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
            logger.error(f"Failed to load Astral authentication configuration file {config_path}: {e}")
    # logger.debug("No Astral authentication configuration file found; proceeding without a configuration file.")
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
        SyncProviderClientT,
        AsyncProviderClientT,
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
        SyncProviderClientT: The provider's synchronous client type (e.g. OpenAI, Anthropic client)
        AsyncProviderClientT: The provider's asynchronous client type (e.g. OpenAI, Anthropic client)
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
    _client_cache: ClassVar[Dict[str, SyncProviderClientT | AsyncProviderClientT]] = {}
    _model_provider: ModelProvider = None
    _config_path: ClassVar[Path] = Path("astral.yaml")

    # --------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------

    def __init__(self, config: Optional[AUTH_CONFIG_TYPE] = None, async_client: bool = False) -> None:
        """Initialize the provider client with optional configuration.

        Args:
            config: Optional configuration dictionary. If not provided, will attempt
                   to load from config.yaml file.
            async_client: Whether to initialize an async client
        """
        self._full_config: AUTH_CONFIG_TYPE_WITH_PROVIDER = config or self.load_full_config() or {}
        self._config: AUTH_CONFIG_TYPE = self.get_provider_config()
        self._async_client_flag = async_client

        if self._config:
            logger.debug(f"{CONFIG_LOG_PREFIX} Provider-specific config for '{self._model_provider}': {self._config}")
        else:
            logger.debug(f"{CONFIG_LOG_PREFIX} No provider-specific config found for '{self._model_provider}'")
        
        # Lazy initialization - clients will be created on first access
        self._sync_client_instance = None
        self._async_client_instance = None

    # --------------------------------------------------------------------------
    # Client Properties (Lazy Initialization)
    # --------------------------------------------------------------------------
    
    @property
    def client(self) -> SyncProviderClientT:
        """Lazily initialize and return the sync client."""
        if self._sync_client_instance is None:
            cache_client = self._config.get("cache_client", True)
            sync_cache_key = f"{self.__class__.__name__}.sync"
            
            if cache_client and sync_cache_key in self._client_cache:
                logger.debug(f"🔄 Using cached sync client for {self._model_provider}")
                self._sync_client_instance = self._client_cache[sync_cache_key]
            else:
                logger.debug(f"🔍 No Sync Client identified for {self._model_provider}. Lazily initializing now.")
                self._sync_client_instance = self._get_or_authenticate_client(async_client=False)
                if cache_client:
                    logger.debug(f"💾 Caching sync client for {self._model_provider}")
                    self._client_cache[sync_cache_key] = self._sync_client_instance
                    
        return self._sync_client_instance
    
    @property
    def async_client(self) -> AsyncProviderClientT:
        """Lazily initialize and return the async client."""
        if self._async_client_instance is None:
            cache_client = self._config.get("cache_client", True)
            async_cache_key = f"{self.__class__.__name__}.async"
            
            if cache_client and async_cache_key in self._client_cache:
                logger.debug(f"🔄 Using cached async client for {self._model_provider}")
                self._async_client_instance = self._client_cache[async_cache_key]
            else:
                logger.debug(f"🔍 No Async Client identified for {self._model_provider}. Lazily initializing now.")
                self._async_client_instance = self._get_or_authenticate_client(async_client=True)
                if cache_client:
                    logger.debug(f"💾 Caching async client for {self._model_provider}")
                    self._client_cache[async_cache_key] = self._async_client_instance
                    
        return self._async_client_instance

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

    @auth_error_handler
    def _get_or_authenticate_client(self, async_client: bool = False) -> SyncProviderClientT | AsyncProviderClientT:
        client_type = "Async" if async_client else "Sync"
        logger.debug(f"🚀 Attempting to initialize {client_type} {self._model_provider} Client with available auth methods")

        env = get_env_vars()
        auth_method_config = self._config.get("auth_method")
        supported_methods = list(self._auth_strategies.keys())

        # Determine which authentication methods to try
        if auth_method_config:
            auth_method_name = auth_method_config.auth_method
            logger.debug(f"⚙️ Using configured method: '{auth_method_name}'")

            if auth_method_name not in self._auth_strategies:
                error = AstralUnknownAuthMethodError(
                    f"Unknown authentication method '{auth_method_name}' for provider '{self._model_provider}'. Supported methods: {supported_methods}",
                    auth_method_name=auth_method_name,
                    provider_name=self._model_provider,
                    supported_methods=supported_methods
                )
                logger.error(f"❌ {error}")
                raise error

            methods_to_try = [(auth_method_name, self._auth_strategies[auth_method_name])]
        else:
            methods_to_try = list(self._auth_strategies.items())
            logger.debug(f"🔄 No specific auth method configured. Trying all methods: {', '.join(supported_methods)}")

        errors = []
        for name, strategy in methods_to_try:
            logger.debug(f"🔑 Trying auth method: '{name}'")
            try:
                client = strategy(self, self._config, env, async_client=async_client)
                if client:
                    logger.debug(f"✅ Authentication succeeded using '{name}'")
                    return client
            except Exception as e:
                logger.warning(f"❌ Auth method '{name}' failed: {str(e)}")
                errors.append((name, e))
                # If there's only one method to try, re-raise the underlying error immediately.
                if len(methods_to_try) == 1:
                    raise e

        # If multiple methods were attempted and all failed, raise a consolidated error.
        raise MultipleAstralAuthenticationErrors(
            "",  # Empty message will be formatted by the decorator.
            provider_name=self._model_provider,
            auth_method_name="multiple_failed",
            error_traceback=traceback.format_exc(),
            errors=errors
        )
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