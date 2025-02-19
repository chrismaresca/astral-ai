# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

"""
This module contains the exceptions for the astral_ai package.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in
from typing import Union

# Astral AI
from astral_ai._models import ModelName

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Provider Authentication Errors
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Base Provider Authentication
# ------------------------------------------------------------------------------


class ProviderAuthenticationError(Exception):
    """General exception raised when provider authentication fails."""
    pass


# ------------------------------------------------------------------------------
# Provider Not Supported Error
# ------------------------------------------------------------------------------


class ProviderNotSupportedError(ProviderAuthenticationError):
    """Exception raised when a provider is not supported."""

    def __init__(self, provider_name: str) -> None:
        message = f"Provider '{provider_name}' is not supported."
        super().__init__(message)
        self.provider_name = provider_name

# ------------------------------------------------------------------------------
# Provider Not Found Error
# ------------------------------------------------------------------------------


class ProviderNotFoundForModelError(ProviderAuthenticationError):
    """Exception raised when a provider is not found for a model."""

    def __init__(self, model_name: Union[ModelName, str]) -> None:
        message = f"No provider registered for model '{model_name}'."
        super().__init__(message)
        self.model_name = model_name


# ------------------------------------------------------------------------------
# Unknown Authentication Method Error
# ------------------------------------------------------------------------------


class UnknownAuthMethodError(ProviderAuthenticationError):
    """Exception raised when a specified authentication method is not registered."""

    def __init__(self, method_name: str, supported_methods: list[str]) -> None:
        message = (
            f"Unknown authentication method specified: '{method_name}'. "
            f"Supported methods for this client: {supported_methods}"
        )
        super().__init__(message)
        self.method_name = method_name
        self.supported_methods = supported_methods


# ------------------------------------------------------------------------------
# Authentication Method Failure Error
# ------------------------------------------------------------------------------


class AuthMethodFailureError(ProviderAuthenticationError):
    """Exception raised when a specific authentication method fails."""

    def __init__(self, method_name: str, original_exception: Exception) -> None:
        message = (
            f"Authentication method '{method_name}' failed with error: {original_exception}"
        )
        super().__init__(message)
        self.method_name = method_name
        self.original_exception = original_exception

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model Name Error
# ------------------------------------------------------------------------------


class ModelError(Exception):
    """General exception raised when a model is not valid."""
    pass

# ------------------------------------------------------------------------------
# Model Name Error
# ------------------------------------------------------------------------------


class ModelNameError(ModelError):
    """Exception raised when a model name is not valid."""

    def __init__(self, model_name: Union[ModelName, str]) -> None:
        message = f"Model name '{model_name}' is not valid."
        super().__init__(message)
        self.model_name = model_name

# ------------------------------------------------------------------------------
# Messages Not Provided Error
# ------------------------------------------------------------------------------


class BaseMessagesError(Exception):
    """Base exception for messages errors."""
    pass


class MessagesNotProvidedError(BaseMessagesError):
    """Exception raised when no messages are provided to the model."""

    def __init__(self, model_name: ModelName):
        self.message = f"No messages provided to the model {model_name}."
        super().__init__(self.message)


# ------------------------------------------------------------------------------
# Invalid Message Error
# ------------------------------------------------------------------------------


class InvalidMessageError(BaseMessagesError):
    """Exception raised when the message is invalid."""

    def __init__(self, message_type: str):
        self.message = f"Invalid message or message list type provided: {message_type}"
        super().__init__(self.message)


class InvalidMessageRoleError(BaseMessagesError):
    """Exception raised when the message role is invalid."""

    def __init__(self, message: str = "Invalid message role provided."):
        self.message = message
        super().__init__(self.message)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Response Model Missing Error
# ------------------------------------------------------------------------------


class ResponseModelMissingError(Exception):
    """Exception raised when a response model is missing."""
    def __init__(self, model_name: ModelName):
        self.message = f"Response model missing for model {model_name}."
        super().__init__(self.message)
