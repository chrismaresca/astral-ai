# Provider Adapters
# -------------------------------------------------------------------------------- #
# This module contains:
#   - The mapping and functions for converting project messages into
#     provider-specific messages.
#   - Overloads for the get_provider_message_converter function.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in imports
from abc import ABC, abstractmethod
from typing import Generic, Type, Optional, overload, TypeVar, Union

# Pydantic imports
from pydantic import BaseModel

# Astral imports
from astral_ai.providers._generics import (
    ProviderRequestT,
    ProviderResponseT,
    StructuredOutputT,
)

# Astral AI Types
from astral_ai._types._request import AstralCompletionRequest
from astral_ai._types._response import AstralChatResponse, AstralStructuredResponse


# -------------------------------------------------------------------------------- #
# Base Completion Adapter
# -------------------------------------------------------------------------------- #

class BaseCompletionAdapter(ABC, Generic[ProviderRequestT, ProviderResponseT]):
    """
    Abstract base adapter that converts Astral requests/responses to/from
    provider-specific types, bound to a provider client.
    """

    @abstractmethod
    def to_provider_completion_request(
        self, request: AstralCompletionRequest
    ) -> ProviderRequestT:
        """
        Convert an AstralCompletionRequest into a provider-specific request.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------------- #
    # Overloads
    # ---------------------------------------------------------------------------- #

    @overload
    @abstractmethod
    def to_astral_completion_response(self, response: ProviderResponseT) -> AstralChatResponse:
        ...

    @overload
    @abstractmethod
    def to_astral_completion_response(
        self, response: ProviderResponseT, response_model: Type[StructuredOutputT]
    ) -> AstralStructuredResponse:
        ...

    # ---------------------------------------------------------------------------- #
    # Implementation
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def to_astral_completion_response(
        self,
        response: ProviderResponseT,
        response_model: Optional[Type[StructuredOutputT]] = None,
    ) -> Union[AstralChatResponse, AstralStructuredResponse]:
        """
        Convert a provider-specific response into an Astral response.
        If response_model is provided, return a structured response;
        otherwise, return a chat response.
        """
        raise NotImplementedError


# -------------------------------------------------------------------------------- #
# Base Embedding Adapter
# -------------------------------------------------------------------------------- #

