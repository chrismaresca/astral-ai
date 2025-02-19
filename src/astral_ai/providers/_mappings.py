# -------------------------------------------------------------------------------- #
# Provider Mappings
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Dict, Optional, overload, Literal

# Astral imports
from astral_ai._types._astral import AstralClientParams
from astral_ai._models import ModelProvider
from astral_ai._auth import AUTH_CONFIG_TYPE
from astral_ai.exceptions import ProviderNotSupportedError

# Provider imports
from astral_ai.providers.base import BaseProviderClient
from astral_ai.providers.anthropic import AnthropicProviderClient
from astral_ai.providers.openai import OpenAIProviderClient

# Adapter imports
from astral_ai.providers._adapters import BaseCompletionAdapter
from astral_ai.providers.openai._adapters import OpenAICompletionAdapter
from astral_ai.providers.anthropic._adapters import AnthropicCompletionAdapter

# -------------------------------------------------------------------------------- #
# Provider Adapter Mapping
# -------------------------------------------------------------------------------- #

_PROVIDER_ADAPTER_MAP: Dict[ModelProvider, BaseCompletionAdapter] = {
    "openai": OpenAICompletionAdapter,
    "anthropic": AnthropicCompletionAdapter,
}

# -------------------------------------------------------------------------------- #
# Get Provider Adapter Overloads
# -------------------------------------------------------------------------------- #

@overload
def get_provider_adapter(model_provider: Literal["openai", "azureOpenAI"]) -> OpenAICompletionAdapter:
    ...


@overload
def get_provider_adapter(model_provider: Literal["anthropic"]) -> AnthropicCompletionAdapter:
    ...


def get_provider_adapter(model_provider: ModelProvider) -> BaseCompletionAdapter:
    """
    Get the provider adapter for the given model provider.
    
    Args:
        model_provider (ModelProvider): The provider to get the adapter for (e.g. "openai", "anthropic")
        
    Returns:
        BaseCompletionAdapter: The appropriate adapter instance for the provider
        
    Raises:
        KeyError: If the provider is not found in the adapter mapping
    """
    return _PROVIDER_ADAPTER_MAP[model_provider]()


# -------------------------------------------------------------------------------- #
# Provider Client Registry
# -------------------------------------------------------------------------------- #

# In-memory cache of provider client instances
_client_registry: Dict[str, BaseProviderClient] = {}

# Map of provider names to their client classes
_PROVIDER_CLIENT_MAP: Dict[ModelProvider, type[BaseProviderClient]] = {
    "openai": OpenAIProviderClient,
    "anthropic": AnthropicProviderClient,
}


def _generate_registry_key(
    provider_name: ModelProvider,
    config: Optional[AUTH_CONFIG_TYPE],
) -> str:
    """
    Generate a unique key for the registry based on provider name and config.
    
    Args:
        provider_name (ModelProvider): The name of the provider
        config (Optional[AUTH_CONFIG_TYPE]): The provider configuration
        
    Returns:
        str: A unique registry key combining the provider name and config hash
    """
    if config and isinstance(config, dict):
        config_key = hash(frozenset(config.items()))
        return f"{provider_name}_{config_key}"
    return provider_name


# -------------------------------------------------------------------------------- #
# Get Provider Client Overloads
# -------------------------------------------------------------------------------- #

@overload
def get_provider_client(
    provider_name: Literal["openai", "azureOpenAI"],
    astral_client: Optional[AstralClientParams] = None,
) -> OpenAIProviderClient:
    ...


@overload
def get_provider_client(
    provider_name: Literal["anthropic"],
    astral_client: Optional[AstralClientParams] = None,
) -> AnthropicProviderClient:
    ...


def get_provider_client(
    provider_name: ModelProvider,
    astral_client: Optional[AstralClientParams] = None,
) -> BaseProviderClient:
    """
    Retrieve an existing provider client from the registry or create a new one if needed.

    Args:
        provider_name (ModelProvider): The provider's name (e.g., "openai", "anthropic")
        astral_client (Optional[AstralClientParams]): Contains parameters such as:
            - new_client (bool): Whether to force the creation of a new client
            - client_config (Optional[AUTH_CONFIG_TYPE]): Configuration for the provider

    Returns:
        BaseProviderClient: An instance of the provider client

    Raises:
        ProviderNotSupportedError: If the provider is not supported
    """
    if astral_client is None:
        new_client = False
        client_config = None
    else:
        new_client = astral_client.new_client
        client_config = astral_client.client_config

    key = _generate_registry_key(provider_name, client_config)

    if key not in _client_registry or new_client:
        client_class = _PROVIDER_CLIENT_MAP.get(provider_name)
        if client_class is None:
            raise ProviderNotSupportedError(provider_name=provider_name)
        _client_registry[key] = client_class(client_config)
    return _client_registry[key]
