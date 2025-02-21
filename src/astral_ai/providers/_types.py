# -------------------------------------------------------------------------------- #
# Provider Types
# -------------------------------------------------------------------------------- #
# This module contains type definitions used across the provider implementations
# to avoid circular imports
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
from typing import TypeVar, Protocol

# -------------------------------------------------------------------------------- #
# Provider Client Protocol
# -------------------------------------------------------------------------------- #


class ProviderClientProtocol(Protocol):
    """Protocol defining the interface for provider clients"""
    pass


# Base provider client type
BaseProviderClient = TypeVar("BaseProviderClient", bound=ProviderClientProtocol)
ModelProviderClient = TypeVar("ModelProviderClient", bound=ProviderClientProtocol)
