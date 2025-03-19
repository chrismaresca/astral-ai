# -------------------------------------------------------------------------------- #
# Provider Adapters
# -------------------------------------------------------------------------------- #
# This module serves as the main entry point for provider adapters.
# It imports provider-specific adapters and provides a factory function for creating 
# the appropriate adapter.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Literal, Any, overload, TypeVar, Dict, Optional, Type, Union

# Base adapter imports
from astral_ai.providers._base_adapters import ProviderAdapter
from astral_ai.constants._models import ModelProvider

# Provider-specific adapter imports
from astral_ai.providers.openai._adapter import OpenAIAdapter
from astral_ai.providers.deepseek._adapter import DeepSeekAdapter
from astral_ai.providers.anthropic._adapter import AnthropicAdapter

# Provider-specific types
from astral_ai.providers.openai._types._request import (
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding,
)

from astral_ai.providers.deepseek._types._request import (
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
    DeepSeekRequestEmbedding,
)

from astral_ai.providers.anthropic._types._request import (
    AnthropicRequestChat,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding,
)

# Type Variables
_ModelProviderT = TypeVar("_ModelProviderT", bound=ModelProvider)

# -------------------------------------------------------------------------------- #
# Factory Function: create_adapter
# -------------------------------------------------------------------------------- #

# Overloads for each recognized provider
@overload
def create_adapter(
    provider: Literal["openai"]
) -> ProviderAdapter[
    Literal["openai"],
    OpenAIRequestChat,
    OpenAIRequestStructured,
    OpenAIRequestEmbedding
]:
    ...


@overload
def create_adapter(
    provider: Literal["deepseek"]
) -> ProviderAdapter[
    Literal["deepseek"],
    DeepSeekRequestChat,
    DeepSeekRequestStructured,
    DeepSeekRequestEmbedding
]:
    ...


@overload
def create_adapter(
    provider: Literal["anthropic"]
) -> ProviderAdapter[
    Literal["anthropic"],
    AnthropicRequestChat,
    AnthropicRequestStructured,
    AnthropicRequestEmbedding
]:
    ...


def create_adapter(
    provider: _ModelProviderT
) -> ProviderAdapter[_ModelProviderT, Any, Any, Any]:
    """
    Creates a typed ProviderAdapter for the given provider string.
    
    The overloads ensure that calling create_adapter with a specific provider
    returns an appropriately typed adapter for that provider.
    
    Args:
        provider: The name of the provider to create an adapter for
        
    Returns:
        A properly typed ProviderAdapter for the specified provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        return ProviderAdapter("openai", OpenAIAdapter())
    elif provider == "anthropic":
        return ProviderAdapter("anthropic", AnthropicAdapter())
    elif provider == "deepseek":
        return ProviderAdapter("deepseek", DeepSeekAdapter())
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# -------------------------------------------------------------------------------- #
# Re-exports
# -------------------------------------------------------------------------------- #

# Export provider-specific adapter classes
__all__ = [
    "create_adapter",
    "OpenAIAdapter",
    "DeepSeekAdapter",
    "AnthropicAdapter",
]


