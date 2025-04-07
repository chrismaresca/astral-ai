from typing import Dict, Optional, overload, Literal
import threading
import json
import uuid

# Astral imports
from astral_ai._types import AstralClientParams
from astral_ai.constants._models import ModelProvider
from astral_ai._auth import AUTH_CONFIG_TYPE
from astral_ai.errors.exceptions import ProviderNotSupportedError

# Provider imports
from astral_ai.providers._base_client import BaseProviderClient
from astral_ai.providers.anthropic import AnthropicProviderClient
from astral_ai.providers.openai import OpenAIProviderClient
from astral_ai.providers.deepseek._client import DeepSeekProviderClient


# Map of provider names to their client classes
_PROVIDER_CLIENT_MAP: Dict[ModelProvider, type[BaseProviderClient]] = {
    "openai": OpenAIProviderClient,
    "anthropic": AnthropicProviderClient,
    "deepseek": DeepSeekProviderClient,
}

class ProviderClientRegistry:
    """
    Thread-safe registry for provider clients.

    Clients are cached by a key derived from:
    1. An explicit unique client_key (if provided in AstralClientParams)
    2. The provider name and its authentication config hash (if config is provided)
    3. Just the provider name (if neither client_key nor config is provided)

    The caching behavior is controlled by the AstralClientParams:
    - If new_client is False (default), a deterministic key is generated from the provider name and config
    - If new_client is True, a unique client_key must be provided to force creation of a new client instance
    """
    _client_registry: Dict[str, BaseProviderClient] = {}
    _lock = threading.RLock()

    @classmethod
    def _generate_registry_key(
        cls,
        provider_name: ModelProvider,
        config: Optional[AUTH_CONFIG_TYPE],
        client_key: Optional[str] = None,
        async_client: bool = False,
    ) -> str:
        """
        Generate a unique registry key for a provider client.

        The key is generated in the following order of precedence:
        1. If client_key is provided, it is used verbatim
        2. If config is provided, the key is generated from the provider name and a hash of the config
        3. Otherwise, just the provider name is used

        The key is suffixed with ".async" if async_client is True.

        Args:
            provider_name: The name of the provider
            config: Optional authentication configuration
            client_key: Optional unique key for the client
            async_client: Whether this is an async client

        Returns:
            str: A unique key for the client registry
        """
        base_key = ""
        if client_key:
            base_key = client_key
        elif config is not None:
            config_str = json.dumps({k: v.model_dump() for k, v in config.items()}, sort_keys=True)
            config_hash = hash(config_str)
            base_key = f"{provider_name}_{config_hash}"
        else:
            base_key = provider_name
            
        # Add suffix for async clients
        return f"{base_key}.async" if async_client else base_key

    @overload
    @classmethod
    def get_client(
        cls,
        provider_name: Literal["openai", "azureOpenAI"],
        astral_client: Optional[AstralClientParams] = None,
        async_client: bool = False,
    ) -> OpenAIProviderClient:
        ...

    @overload
    @classmethod
    def get_client(
        cls,
        provider_name: Literal["anthropic"],
        astral_client: Optional[AstralClientParams] = None,
        async_client: bool = False,
    ) -> AnthropicProviderClient:
        ...

    @overload
    @classmethod
    def get_client(
        cls,
        provider_name: Literal["deepseek"],
        astral_client: Optional[AstralClientParams] = None,
        async_client: bool = False,
    ) -> DeepSeekProviderClient:
        ...
        
    @classmethod
    def get_client(
        cls,
        provider_name: ModelProvider,
        astral_client: Optional[AstralClientParams] = None,
        async_client: bool = False,
    ) -> BaseProviderClient:
        """
        Retrieve or create a provider client instance.

        The client is retrieved from cache if available, otherwise a new instance is created.
        The caching behavior is controlled by the AstralClientParams:

        - If astral_client is None:
            Uses the default (cached) client for that provider
        - If astral_client.new_client is True:
            Forces creation of a new client instance
            - Requires a unique client_key to be provided
            - Raises ValueError if client_key is missing
        - Otherwise:
            Uses a deterministic key based on provider name and client_config
            - Returns cached client if available
            - Creates new client if not in cache

        Args:
            provider_name: The name of the provider to get a client for
            astral_client: Optional client parameters controlling caching behavior
            async_client: Whether to return an async client

        Returns:
            BaseProviderClient: The provider client instance

        Raises:
            ValueError: If new_client is True but no client_key is provided
            ProviderNotSupportedError: If the provider is not supported
        """
        if astral_client is None:
            key = cls._generate_registry_key(provider_name, None, async_client=async_client)
        else:
            if astral_client.new_client:
                if astral_client.client_key:
                    key = cls._generate_registry_key(provider_name, None, astral_client.client_key, async_client)
                else:
                    raise ValueError(
                        "When new_client is True, you must provide a unique client_key."
                    )
            else:
                key = cls._generate_registry_key(
                    provider_name, 
                    astral_client.client_config, 
                    astral_client.client_key,
                    async_client
                )

        with cls._lock:
            if key not in cls._client_registry:
                client_class = _PROVIDER_CLIENT_MAP.get(provider_name)
                if client_class is None:
                    raise ProviderNotSupportedError(provider_name=provider_name)
                cls._client_registry[key] = client_class(
                    astral_client.client_config if astral_client else None,
                    async_client=async_client
                )
            return cls._client_registry[key]

    @classmethod
    def register_client(
        cls,
        provider_name: ModelProvider,
        client: BaseProviderClient,
        client_config: Optional[AUTH_CONFIG_TYPE] = None,
        client_key: Optional[str] = None,
        async_client: bool = False,
    ) -> None:
        """
        Manually register or replace a provider client in the registry.

        The client is registered with a key generated from the provider name,
        client configuration, and optional client key. If a client already exists
        with the same key, it will be replaced.

        Args:
            provider_name: The name of the provider
            client: The client instance to register
            client_config: Optional configuration for the client
            client_key: Optional unique key for the client
            async_client: Whether this is an async client
        """
        key = cls._generate_registry_key(provider_name, client_config, client_key, async_client)
        with cls._lock:
            cls._client_registry[key] = client

    @classmethod
    def unregister_client(
        cls,
        provider_name: ModelProvider,
        client_config: Optional[AUTH_CONFIG_TYPE] = None,
        client_key: Optional[str] = None,
        async_client: bool = False,
    ) -> None:
        """
        Remove a provider client from the registry.

        The client is identified by a key generated from the provider name,
        client configuration, and optional client key. If no client exists
        with the generated key, no action is taken.

        Args:
            provider_name: The name of the provider
            client_config: Optional configuration for the client
            client_key: Optional unique key for the client
            async_client: Whether this is an async client
        """
        key = cls._generate_registry_key(provider_name, client_config, client_key, async_client)
        with cls._lock:
            if key in cls._client_registry:
                del cls._client_registry[key]

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear all registered clients from the registry.

        This removes all cached client instances, forcing new clients
        to be created on subsequent get_client calls.
        """
        with cls._lock:
            cls._client_registry.clear()

    @classmethod
    def get_all_clients(cls) -> Dict[str, BaseProviderClient]:
        """
        Return a shallow copy of all registered clients.

        Returns:
            Dict[str, BaseProviderClient]: A dictionary mapping registry keys to client instances
        """
        with cls._lock:
            return dict(cls._client_registry)

    @classmethod
    def get_client_count(cls) -> int:
        """
        Return the total number of registered clients.

        Returns:
            int: The number of clients currently in the registry
        """
        with cls._lock:
            return len(cls._client_registry)

    @classmethod
    def get_client_by_index(cls, index: int) -> BaseProviderClient:
        """
        Get the client at the given index in the registry.

        Args:
            index: The zero-based index of the client to retrieve

        Returns:
            BaseProviderClient: The client at the given index

        Raises:
            IndexError: If the index is out of range
        """
        with cls._lock:
            values = list(cls._client_registry.values())
            if index < 0 or index >= len(values):
                raise IndexError("Client index out of range")
            return values[index]

    @classmethod
    def get_client_by_name(cls, name: str) -> BaseProviderClient:
        """
        Retrieve a client by its registry key.

        Args:
            name: The registry key of the client to retrieve

        Returns:
            BaseProviderClient: The client associated with the given registry key

        Raises:
            KeyError: If no client exists with the given key
        """
        with cls._lock:
            return cls._client_registry[name]
