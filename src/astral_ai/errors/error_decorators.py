# -------------------------------------------------------------------------------- #
# Error Decorators
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import functools
import os
import traceback
from typing import Optional, Dict, Any

# openai imports
from openai import (
    OpenAIError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

# module imports
from astral_ai.logger import logger

# Astral AI Exceptions
from astral_ai.errors.exceptions import (
    # Provider Errors
    AstralProviderError,
    AstralProviderAuthenticationError,
    AstralProviderRateLimitError,
    AstralProviderConnectionError,
    AstralProviderStatusError,
    # Unexpected Errors
    AstralUnexpectedError,

    # Auth Errors
    AstralAuthError,
    AstralAuthConfigurationError,
    AstralMissingCredentialsError,
    AstralInvalidCredentialsError,
    AstralEnvironmentVariableError,
    AstralUnknownAuthMethodError,
    AstralAuthMethodFailureError,
    MultipleAstralAuthenticationErrors,
)

# Error Formatter
from .error_formatter import format_error_message, format_multiple_auth_errors

# Astral AI Logger
from astral_ai.logger import logger

# -------------------------------------------------------------------------------- #
# Auth Errors
# -------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
# Provider Error Handler Decorator
# ------------------------------------------------------------------------- #
def provider_error_handler(func):
    """
    Decorator for provider-level functions.

    This decorator always uses the generalized format_error_message to produce
    a verbose error message and maps known provider exceptions (from OpenAI)
    to their corresponding Astral errors. It extracts context details (status_code,
    request_id, error_body, error_traceback) and re-raises a new error with the
    verbose message.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except OpenAIError as e:
            # Extract additional context conditionally.
            status_code = getattr(e, "status_code", None)
            request_id = kwargs.get("request_id") or getattr(e, "request_id", None)
            error_body = getattr(e, "body", None)
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = getattr(e, "error_traceback", None) or traceback.format_exc()

            # Map the OpenAI error to the correct Astral error type.
            if isinstance(e, AuthenticationError):
                error_type = "authentication"
                astral_error_class = AstralProviderAuthenticationError
            elif isinstance(e, RateLimitError):
                error_type = "rate_limit"
                astral_error_class = AstralProviderRateLimitError
            elif isinstance(e, (APIConnectionError, APITimeoutError)):
                error_type = "connection"
                astral_error_class = AstralProviderConnectionError
            elif isinstance(e, APIStatusError):
                error_type = "status"
                astral_error_class = AstralProviderStatusError
            else:
                error_type = "unexpected"
                astral_error_class = AstralUnexpectedError

            # Retrieve provider name (assumed stored as _model_provider on self).
            provider_name = getattr(self, "_model_provider", "unknown")

            # Format the error message.
            verbose_message = format_error_message(
                error_category="provider",
                error_type=error_type,
                source_name=provider_name,
                additional_message=str(e),
                status_code=status_code,
                request_id=request_id,
                error_body=error_body,
                error_traceback=error_traceback
            )

            raise astral_error_class(verbose_message,
                                     status_code=status_code,
                                     request_id=request_id,
                                     error_body=error_body,
                                     error_traceback=error_traceback) from e
        except Exception as e:
            # Wrap any other exception as an unexpected provider error.
            status_code = None
            request_id = kwargs.get("request_id")
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = traceback.format_exc()
                
            provider_name = getattr(self, "_model_provider", "unknown")
            verbose_message = format_error_message(
                error_category="provider",
                error_type="unexpected",
                source_name=provider_name,
                additional_message=str(e),
                status_code=status_code,
                request_id=request_id,
                error_body=None,
                error_traceback=error_traceback
            )
            raise AstralUnexpectedError(verbose_message,
                                        status_code=status_code,
                                        request_id=request_id,
                                        error_body=None,
                                        error_traceback=error_traceback) from e
    return wrapper


# -------------------------------------------------------------------------------- #
# Auth Error Handler Decorator
# -------------------------------------------------------------------------------- #
def auth_error_handler(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)

        except MultipleAstralAuthenticationErrors as e:
            auth_source = getattr(self, "_model_provider", "unknown")
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = getattr(e, "error_traceback", None) or traceback.format_exc()
                
            # Reuse message if already formatted
            message = str(e) if "=====" in str(e) else format_multiple_auth_errors(
                provider_name=auth_source, 
                errors=e.errors,
                error_traceback=error_traceback  # Include traceback in the message
            )
            # Create a copy of the exception's attributes
            error_attrs = {k: v for k, v in vars(e).items() if k != "args"}
            error_attrs["error_traceback"] = error_traceback
            raise MultipleAstralAuthenticationErrors(message, **error_attrs) from None

        except (
            AstralAuthError,
            AstralUnknownAuthMethodError,
            AstralAuthMethodFailureError,
            AstralAuthConfigurationError,
            AstralMissingCredentialsError,
            AstralInvalidCredentialsError,
            AstralEnvironmentVariableError
        ) as e:
            auth_source = getattr(self, "_model_provider", "unknown")
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = getattr(e, "error_traceback", None) or traceback.format_exc()
                
            # Map exception types to a simpler error type
            error_type_map = {
                AstralUnknownAuthMethodError: "unknown_method",
                AstralAuthMethodFailureError: "method_failure",
                AstralAuthConfigurationError: "configuration",
                AstralMissingCredentialsError: "missing_credentials",
                AstralInvalidCredentialsError: "invalid_credentials",
                AstralEnvironmentVariableError: "environment_variables",
                AstralAuthError: "unexpected",
            }
            error_type = error_type_map.get(type(e), "unexpected")
            additional_message = str(e) or "Authentication error occurred."
            
            message = format_error_message(
                error_category="authentication",
                error_type=error_type,
                source_name=auth_source,
                additional_message=additional_message,
                status_code=getattr(e, "status_code", None),
                request_id=getattr(e, "request_id", None),
                error_body=getattr(e, "error_body", None),
                error_traceback=error_traceback  # Include the traceback in the formatted message
            )
            
            # Create a copy of the exception's attributes
            error_attrs = {k: v for k, v in vars(e).items() if k != "args"}
            error_attrs["error_traceback"] = error_traceback
            
            raise e.__class__(message, **error_attrs) from None

        except Exception as e:
            auth_source = getattr(self, "_model_provider", "unknown")
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = traceback.format_exc()
            
            message = format_error_message(
                error_category="authentication",
                error_type="unexpected",
                source_name=auth_source,
                additional_message=str(e) or "Unexpected error during authentication.",
                status_code=None,
                request_id=None,
                error_body=None,
                error_traceback=error_traceback  # Include the traceback in the formatted message
            )
            
            raise AstralAuthError(
                message,
                status_code=None,
                request_id=None,
                error_body=None,
                error_traceback=error_traceback,
                auth_method_name=getattr(func, "_auth_name", "unknown"),
                provider_name=auth_source
            ) from None

    return wrapper

# ------------------------------------------------------------------------- #
# Resource Error Handler Decorator
# ------------------------------------------------------------------------- #


def resource_error_handler(func):
    """
    Decorator for resource-level functions.

    This decorator always uses the generalized format_error_message (with resource-specific
    details) to produce a verbose error message. It extracts context details and re-raises
    the error with the verbose message.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except AstralProviderError as e:
            raise e
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            request_id = getattr(e, "request_id", None)
            error_body = getattr(e, "error_body", None)
            
            # Only collect traceback if the environment variable is set
            error_traceback = None
            if os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
                error_traceback = getattr(e, "error_traceback", None) or traceback.format_exc()
                
            resource_name = getattr(self, "_resource_name", "unknown")

            verbose_message = format_error_message(
                error_category="resource",
                error_type="default",
                source_name=resource_name,
                additional_message=str(e),
                status_code=status_code,
                request_id=request_id,
                error_body=error_body,
                error_traceback=error_traceback
            )
            raise e.__class__(verbose_message,
                              status_code=status_code,
                              request_id=request_id,
                              error_body=error_body,
                              error_traceback=error_traceback) from e
    return wrapper


# # -------------------------------------------------------------------------------- #
# # Error handler decorator
# # -------------------------------------------------------------------------------- #
# def provider_error_handler(func):
#     """
#     Decorator that transforms provider-specific errors into developer-friendly Astral errors.

#     Features:
#     - Structured, readable error messages with clear sections
#     - Error type and context clearly identified
#     - Potential issues and solutions separated and easy to scan
#     - Consistent format across all providers
#     - Includes traceback for easier debugging
#     """
#     @functools.wraps(func)
#     def _decorator(self, *args, **kwargs):
#         model_provider = getattr(self, "_model_provider", "unknown").upper()

#         try:
#             return func(self, *args, **kwargs)
#         except OpenAIError as e:
#             # Log detailed technical error for debugging
#             logger.error(
#                 f"Provider error in '{func.__name__}' for {model_provider}: {str(e)}",
#                 exc_info=True
#             )

#             # Determine error type and corresponding Astral error class
#             if isinstance(e, AuthenticationError):
#                 error_type = "authentication"
#                 error_emoji = "🔑"
#                 error_title = "Authentication Error"
#                 astral_error_class = AstralAuthenticationError
#             elif isinstance(e, RateLimitError):
#                 error_type = "rate_limit"
#                 error_emoji = "🐢"
#                 error_title = "Rate Limit Error"
#                 astral_error_class = AstralRateLimitError
#             elif isinstance(e, (APIConnectionError, APITimeoutError)):
#                 error_type = "connection"
#                 error_emoji = "🌐"
#                 error_title = "Connection Error"
#                 astral_error_class = AstralConnectionError
#             elif isinstance(e, APIStatusError):
#                 error_type = "status"
#                 error_emoji = "⚠️"
#                 error_title = "API Status Error"
#                 astral_error_class = AstralStatusError
#             else:
#                 error_type = "unexpected"
#                 error_emoji = "❗"
#                 error_title = "Unexpected Error"
#                 astral_error_class = AstralUnexpectedError

#             # Get provider-specific error details
#             provider_errors = PROVIDER_ERROR_MESSAGES.get(model_provider, {})
#             error_data = provider_errors.get(error_type, {})
#             documentation_url = provider_errors.get('documentation_url', 'https://docs.astralai.com/errors/')
#             astral_docs_url = "https://docs.astralai.com/errors/"

#             # Get the basic message and suggestions
#             basic_message = error_data.get(
#                 "message",
#                 f"{error_emoji} [{model_provider}] {error_title}: Please refer to provider documentation here: {documentation_url}"
#             )
#             suggestions = error_data.get("suggestions", [])

#             # Extract additional context when available
#             status_code = getattr(e, "status_code", None)
#             request_id = getattr(e, "request_id", None)
#             error_body = getattr(e, "body", None)

#             # Get traceback information
#             error_traceback = traceback.format_exc()

#             # Format the error message
#             friendly_message = format_error_message(
#                 model_provider=model_provider,
#                 error_type=error_type,
#                 error_emoji=error_emoji,
#                 error_title=error_title,
#                 basic_message=basic_message,
#                 suggestions=suggestions,
#                 documentation_url=documentation_url,
#                 astral_docs_url=astral_docs_url,
#                 status_code=status_code,
#                 request_id=request_id,
#                 error_body=error_body,
#                 error_traceback=error_traceback
#             )

#             # Log the fully formatted message at debug level for reference
#             logger.debug(f"Formatted error message:\n{friendly_message}")

#             # Raise the appropriate Astral error with the friendly message
#             raise astral_error_class(friendly_message) from e

#         except Exception as e:
#             # Catch-all for non-provider errors
#             logger.error(
#                 f"Unexpected error in '{func.__name__}' for provider {model_provider}: {str(e)}",
#                 exc_info=True
#             )
#             raise
#     return _decorator
