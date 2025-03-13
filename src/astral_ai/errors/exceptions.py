from __future__ import annotations
# -------------------------------------------------------------------------------- #
# Astral AI Exceptions
# -------------------------------------------------------------------------------- #

"""
This module contains all the exceptions for the Astral AI framework.

It includes:
- Provider Authentication Errors
- Provider Not Supported Error
- Provider Not Found For Model Error
- Provider Response Error

"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Union, Any, Optional, Type

# module imports
from astral_ai.constants._models import ModelName, ModelProvider
# -------------------------------------------------------------------------------- #
# Provider Authentication Errors
# -------------------------------------------------------------------------------- #


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

    def __init__(self, model_name: Union['ModelName', str]) -> None:
        message = f"No provider registered for model '{model_name}'."
        super().__init__(message)
        self.model_name = model_name


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

# ------------------------------------------------------------------------------
# Invalid Parameter Error
# ------------------------------------------------------------------------------


class MissingParameterError(Exception):
    """Exception raised when a required parameter is missing from a function call."""

    def __init__(self, parameter_name: str, function_name: str):
        self.message = (
            f"Oops! The function '{function_name}' needs a value for '{parameter_name}' to work properly. "
            f"This parameter is required - could you please provide a value for '{parameter_name}'?"
        )
        super().__init__(self.message)


# ------------------------------------------------------------------------------
# Provider Response Error
# ------------------------------------------------------------------------------

class ProviderResponseError(Exception):
    """Exception raised when we receive an unexpected response from an AI provider."""

    def __init__(self, provider_name: ModelProvider, response_type: str):
        self.message = (
            f"Oops! We got an unexpected response from {provider_name}. "
            f"We received a '{response_type}' response, but that's not what we were expecting. "
            f"This likely means either the API changed or there's a bug in our code."
        )
        super().__init__(self.message)


class ProviderFeatureNotSupportedError(ProviderResponseError):
    """Exception raised when a provider feature is not supported."""

    def __init__(self, provider_name: ModelProvider, feature_name: str):
        self.message = (
            f"Oops! It looks like you're trying to use {feature_name}, but {provider_name} doesn't "
            f"support that feature yet You may want to try a different AI provider that supports "
            f"what you're trying to do, or use a different approach."
        )
        super().__init__(self.message)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# -------------------------------------------------------------------------------- #
# Astral AI Custom Client Exceptions
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Astral AI Error Messages
# -------------------------------------------------------------------------------- #

"""
This module provides exceptions for the Astral AI framework with an emphasis 
on developer experience. Error messages are designed to be immediately useful,
with clear explanations, emojis for quick identification, and actionable suggestions.
"""

# ------------------------------------------------------------------------- #
# Astral AI Base Exceptions
# ------------------------------------------------------------------------- #


class AstralBaseError(Exception):
    """
    Base exception for all Astral AI client errors.

    Provides additional context:
    - status_code
    - request_id
    - error_body
    - error_traceback
    """

    def __init__(self, message: str, *,
                status_code: Optional[int] = None,
                request_id: Optional[str] = None,
                error_body: Optional[Any] = None,
                error_traceback: Optional[str] = None,
                **kwargs: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.error_body = error_body
        self.error_traceback = error_traceback

        # Add any additional keyword arguments to the exception instance
        for key, value in kwargs.items():
            setattr(self, key, value)


# ------------------------------------------------------------------------- #
# Provider Errors
# ------------------------------------------------------------------------- #
"""
Provider-related errors that occur during the interaction with provider APIs.
These errors are raised when there are issues with the provider API, such as
invalid requests, rate limits being exceeded, or connection errors.
"""


class AstralProviderError(AstralBaseError):
    """Base class for all provider-related errors."""
    pass


class AstralProviderAuthenticationError(AstralProviderError):
    """Authentication issue with a provider API (invalid API key, expired token, etc.)"""
    pass


class AstralProviderRateLimitError(AstralProviderError):
    """Rate limit exceeded on a provider API (too many requests)"""
    pass


class AstralProviderConnectionError(AstralProviderError):
    """Connection issue with a provider API (network error, timeout, etc.)"""
    pass


class AstralProviderStatusError(AstralProviderError):
    """Error status received from a provider API (4xx or 5xx responses)"""
    pass


class AstralProviderResponseError(AstralProviderError):
    """Error when the provider returns an unexpected response type"""
    
    def __init__(self, message: str, *,
                provider_name: str,
                expected_response_type: str,
                status_code: Optional[int] = None,
                request_id: Optional[str] = None,
                error_body: Optional[Any] = None,
                error_traceback: Optional[str] = None,
                **kwargs: Any) -> None:
        super().__init__(message, 
                        status_code=status_code,
                        request_id=request_id,
                        error_body=error_body,
                        error_traceback=error_traceback,
                        **kwargs)
        self.provider_name = provider_name
        self.expected_response_type = expected_response_type


# ------------------------------------------------------------------------- #
# Auth Errors
# ------------------------------------------------------------------------- #
"""
Authentication-related errors that occur during the authentication process with providers.
These errors are raised when there are issues with credentials, configuration, or when
authentication methods fail or are not supported by the provider.
"""


class AstralAuthError(AstralBaseError):
    """Base class for all authentication-related errors."""
    
    def __init__(self, message: str = "", *,
                status_code: Optional[int] = None,
                request_id: Optional[str] = None,
                error_body: Optional[Any] = None,
                error_traceback: Optional[str] = None,
                auth_method_name: Optional[str] = None,
                provider_name: Optional[str] = None,
                model_name: Optional[Union[ModelName, str]] = None,
                required_credentials: Optional[list[str]] = None,
                missing_credentials: Optional[list[str]] = None,
                env_variable_name: Optional[str] = None,
                documentation_link: Optional[str] = None,
                **kwargs: Any) -> None:
        super().__init__(message, 
                        status_code=status_code,
                        request_id=request_id,
                        error_body=error_body,
                        error_traceback=error_traceback,
                        **kwargs)
        self.auth_method_name = auth_method_name
        self.provider_name = provider_name
        self.model_name = model_name
        self.required_credentials = required_credentials or []
        self.missing_credentials = missing_credentials or []
        self.env_variable_name = env_variable_name
        self.documentation_link = documentation_link or "https://docs.astralai.com/authentication"


class AstralAuthConfigurationError(AstralAuthError):
    """Raised when there's an issue with the authentication configuration."""
    pass


class AstralMissingCredentialsError(AstralAuthError):
    """Raised when required authentication credentials are missing."""
    pass


class AstralInvalidCredentialsError(AstralAuthError):
    """Raised when authentication credentials exist but are invalid."""
    pass


class AstralEnvironmentVariableError(AstralAuthError):
    """Raised when required environment variables are missing."""
    pass


class AstralAuthMethodFailureError(AstralAuthError):
    """Raised when a specific authentication method fails."""
    pass


class AstralUnknownAuthMethodError(AstralAuthError):
    """Exception raised when an unknown authentication method is specified."""
    
    def __init__(self, message: str, *,
                auth_method_name: Optional[str] = None,
                provider_name: Optional[str] = None,
                supported_methods: Optional[list[str]] = None,
                status_code: Optional[int] = None,
                request_id: Optional[str] = None,
                error_body: Optional[Any] = None,
                error_traceback: Optional[str] = None,
                **kwargs: Any) -> None:
        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            error_body=error_body,
            error_traceback=error_traceback,
            auth_method_name=auth_method_name,
            provider_name=provider_name,
            **kwargs
        )
        self.supported_methods = supported_methods or []


class MultipleAstralAuthenticationErrors(AstralAuthError):
    """Exception raised when multiple authentication methods fail.
    
    This exception stores information about all the individual failed authentication
    attempts and their corresponding error messages.
    """
    
    def __init__(self, message: str = "", *,
                status_code: Optional[int] = None,
                request_id: Optional[str] = None,
                error_body: Optional[Any] = None,
                error_traceback: Optional[str] = None,
                auth_method_name: Optional[str] = None,
                provider_name: Optional[str] = None,
                model_name: Optional[Union[ModelName, str]] = None,
                errors: Optional[list[tuple[str, Exception]]] = None,
                documentation_link: Optional[str] = None,
                **kwargs: Any) -> None:
        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            error_body=error_body,
            error_traceback=error_traceback,
            auth_method_name=auth_method_name,
            provider_name=provider_name,
            model_name=model_name,
            documentation_link=documentation_link,
            **kwargs
        )
        self.errors = errors or []


# ------------------------------------------------------------------------- #
# Unexpected Errors
# ------------------------------------------------------------------------- #


class AstralUnexpectedError(AstralBaseError):
    """Exception for unexpected errors that don't fit other categories."""
    pass
