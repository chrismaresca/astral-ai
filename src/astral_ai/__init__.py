# -------------------------------------------------------------------------------- #
# Astral AI Package - Main Entry Point
# -------------------------------------------------------------------------------- #

"""
Astral AI - A Python library for AI development with built-in observability and cost tracking.
Provides a single interface for working with any AI model.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Version Information
# -------------------------------------------------------------------------------- #
__title__ = "astral-ai"
__version__ = "0.1.0"

# -------------------------------------------------------------------------------- #
# Module Imports
# -------------------------------------------------------------------------------- #
# Message handling
from astral_ai.messages import (
    Message,
    MessageList,
    TextMessage,
    LocalImageMessage,
    RemoteImageMessage,
)

# Resources
from astral_ai.resources.completions import (
    Completions,
    complete,
    complete_structured,
    complete_async,
    complete_structured_async,
)

# Error handling
from astral_ai.errors.exceptions import (
    AstralBaseError as AstralError,
    AstralAuthError as AuthenticationError,
    AstralProviderError as ProviderError,
    AstralProviderResponseError as ResponseError,
    ProviderNotFoundForModelError as ModelNotFoundError,
)

# -------------------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------------------- #
__all__ = [
    # Package metadata
    "__title__",
    "__version__",

    # Messages
    "Message",
    "MessageList",
    "TextMessage",
    "LocalImageMessage",
    "RemoteImageMessage",

    # Completions
    "Completions",
    "complete",
    "complete_structured",
    "complete_async",
    "complete_structured_async",
    
    # Errors
    "AstralError",
    "AuthenticationError",
    "ProviderError",
    "ResponseError",
    "ModelNotFoundError",
]


