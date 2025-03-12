# -------------------------------------------------------------------------------- #
# error_formatting.py
# -------------------------------------------------------------------------------- #

import traceback
from typing import Literal, Optional, Dict, Any

# ------------------------------------------------------------------------- #
# Type Definitions
# ------------------------------------------------------------------------- #
ErrorCategory = Literal["provider", "client", "resource", "authentication"]
ProviderErrorType = Literal["authentication", "rate_limit", "connection", "status", "unexpected"]
AuthErrorType = Literal["unknown_method", "method_failure", "configuration", "missing_credentials", "invalid_credentials", "environment_variables", "unexpected"]

# ------------------------------------------------------------------------- #
# TODO: Move to YAML or DB
# Embedded Error Messages Dictionary (for development)
# ------------------------------------------------------------------------- #
# This dictionary contains configuration for provider-specific errors as well as
# default configurations for client and resource errors.
PROVIDER_ERROR_MESSAGES: Dict[str, Any] = {
    "provider": {
        "OPENAI": {
            "authentication": {
                "base_message": "Please verify your API key or credentials. Check your environment variables or config file.",
                "suggestions": [
                    "Check your API key or credentials.",
                    "Verify your environment variables or config file."
                ],
                "documentation_url": "https://docs.openai.com/api/errors"
            },
            "rate_limit": {
                "base_message": "You're sending requests too fast. Slow down or review your API usage limits.",
                "suggestions": [
                    "Slow down your request rate.",
                    "Review API usage limits."
                ],
                "documentation_url": "https://docs.openai.com/api/errors"
            },
            "connection": {
                "base_message": "A network error occurred. Check your internet connection and consider increasing timeout settings.",
                "suggestions": [
                    "Check your internet connection.",
                    "Consider increasing timeout settings."
                ],
                "documentation_url": "https://docs.openai.com/api/errors"
            },
            "status": {
                "base_message": "The API responded with an error status. Review your request parameters and ensure the service is available.",
                "suggestions": [],
                "documentation_url": "https://docs.openai.com/api/errors"
            },
            "unexpected": {
                "base_message": "Something went wrong. Please review your request and refer to the documentation.",
                "suggestions": [],
                "documentation_url": "https://docs.openai.com/api/errors"
            }
        },
        "DEEPSEEK": {
            "authentication": {
                "base_message": "Please verify that your API key and base URL are correct.",
                "suggestions": [
                    "Verify your API key and base URL."
                ],
                "documentation_url": "https://docs.deepseek.com/api/error-codes"
            },
            "rate_limit": {
                "base_message": "Too many requests. Consider adjusting your request frequency or reviewing your usage limits.",
                "suggestions": [
                    "Slow down your request rate.",
                    "Review usage limits."
                ],
                "documentation_url": "https://docs.deepseek.com/api/error-codes"
            },
            "connection": {
                "base_message": "A network error occurred. Check your internet connection and your timeout settings.",
                "suggestions": [
                    "Check your internet connection.",
                    "Review your timeout settings."
                ],
                "documentation_url": "https://docs.deepseek.com/api/error-codes"
            },
            "status": {
                "base_message": "The API responded with an error status.",
                "suggestions": [
                    "Potential issue: insufficient API credits. Check your balance here: https://platform.deepseek.com/top_up"
                ],
                "documentation_url": "https://docs.deepseek.com/api/error-codes"
            },
            "unexpected": {
                "base_message": "An unknown error occurred. Please refer to the docs for troubleshooting.",
                "suggestions": [],
                "documentation_url": "https://docs.deepseek.com/api/error-codes"
            }
        }
    },
    "authentication": {
        "OPENAI": {
            "unknown_method": {
                "base_message": "The authentication method you specified is not supported for OpenAI.",
                "suggestions": [
                    "Use 'api_key' authentication for OpenAI.",
                    "Check the auth_method parameter in your config."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "method_failure": {
                "base_message": "Authentication with OpenAI failed due to an error in the authentication method.",
                "suggestions": [
                    "Double-check your authentication configuration.",
                    "Ensure you're using 'api_key' authentication for OpenAI."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "configuration": {
                "base_message": "Your OpenAI authentication configuration is incorrect.",
                "suggestions": [
                    "Check your configuration file for errors.",
                    "Make sure all required parameters are properly formatted."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "missing_credentials": {
                "base_message": "Required OpenAI credentials are missing.",
                "suggestions": [
                    "Make sure you've provided an API key.",
                    "Check your environment variables or configuration file."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "invalid_credentials": {
                "base_message": "Your OpenAI credentials are invalid or rejected by the API.",
                "suggestions": [
                    "Verify your API key is correct and active.",
                    "Check if your OpenAI account has billing enabled.",
                    "Make sure your API key has the necessary permissions."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "environment_variables": {
                "base_message": "Required environment variables for OpenAI authentication are missing.",
                "suggestions": [
                    "Set the OPENAI_API_KEY environment variable.",
                    "Check your .env file or system environment variables."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            },
            "unexpected": {
                "base_message": "An unexpected error occurred during OpenAI authentication.",
                "suggestions": [
                    "Check your network connection.",
                    "Verify your API key is valid.",
                    "Make sure your OpenAI account is active and has billing enabled."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/openai"
            }
        },
        "ANTHROPIC": {
            "unknown_method": {
                "base_message": "The authentication method you specified is not supported for Anthropic.",
                "suggestions": [
                    "Use 'api_key' authentication for Anthropic.",
                    "Check the auth_method parameter in your config."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "method_failure": {
                "base_message": "Authentication with Anthropic failed due to an error in the authentication method.",
                "suggestions": [
                    "Double-check your authentication configuration.",
                    "Ensure you're using 'api_key' authentication for Anthropic."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "configuration": {
                "base_message": "Your Anthropic authentication configuration is incorrect.",
                "suggestions": [
                    "Check your configuration file for errors.",
                    "Make sure all required parameters are properly formatted."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "missing_credentials": {
                "base_message": "Required Anthropic credentials are missing.",
                "suggestions": [
                    "Make sure you've provided an API key.",
                    "Check your environment variables or configuration file."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "invalid_credentials": {
                "base_message": "Your Anthropic credentials are invalid or rejected by the API.",
                "suggestions": [
                    "Verify your API key is correct and active.",
                    "Check if your Anthropic account is in good standing."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "environment_variables": {
                "base_message": "Required environment variables for Anthropic authentication are missing.",
                "suggestions": [
                    "Set the ANTHROPIC_API_KEY environment variable.",
                    "Check your .env file or system environment variables."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            },
            "unexpected": {
                "base_message": "An unexpected error occurred during Anthropic authentication.",
                "suggestions": [
                    "Check your network connection.",
                    "Verify your API key is valid.",
                    "Make sure your Anthropic account is active."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/anthropic"
            }
        }
    },
    "client": {
        "default": {
            "message": "Client Error: An error occurred within the client.",
            "emoji": "🚫",
            "suggestions": [
                "Ensure the client is properly configured.",
                "Check for network issues."
            ],
            "documentation_url": "https://docs.astralai.com/client/errors"
        }
    },
    "resource": {
        "default": {
            "message": "Resource Error: An error occurred at the resource level.",
            "emoji": "❗",
            "suggestions": [
                "Ensure the resource is configured correctly.",
                "Review resource documentation."
            ],
            "documentation_url": "https://docs.astralai.com/resources/errors"
        }
    }
}

# ------------------------------------------------------------------------- #
# Common Provider Error Parts
# ------------------------------------------------------------------------- #
# These parts remain the same across providers.
PROVIDER_ERROR_COMMON: Dict[ProviderErrorType, Dict[str, str]] = {
    "authentication": {
        "message_template": "[{provider}] Authentication Error: {base_message}",
        "emoji": "🔑",
    },
    "rate_limit": {
        "message_template": "[{provider}] Rate Limit Error: {base_message}",
        "emoji": "🐢",
    },
    "connection": {
        "message_template": "[{provider}] Connection Error: {base_message}",
        "emoji": "🌐",
    },
    "status": {
        "message_template": "[{provider}] API Status Error: {base_message}",
        "emoji": "⚠️",
    },
    "unexpected": {
        "message_template": "[{provider}] Unexpected Error: {base_message}",
        "emoji": "❗",
    },
}

# ------------------------------------------------------------------------- #
# Common Authentication Error Parts
# ------------------------------------------------------------------------- #
# These parts remain the same across providers for authentication errors.
AUTH_ERROR_COMMON: Dict[AuthErrorType, Dict[str, str]] = {
    "unknown_method": {
        "message_template": "[{provider}] Unknown Auth Method: {base_message}",
        "emoji": "❓",
    },
    "method_failure": {
        "message_template": "[{provider}] Auth Method Failure: {base_message}",
        "emoji": "⚠️",
    },
    "configuration": {
        "message_template": "[{provider}] Auth Configuration Error: {base_message}",
        "emoji": "⚙️",
    },
    "missing_credentials": {
        "message_template": "[{provider}] Missing Credentials: {base_message}",
        "emoji": "🔍",
    },
    "invalid_credentials": {
        "message_template": "[{provider}] Invalid Credentials: {base_message}",
        "emoji": "🔒",
    },
    "environment_variables": {
        "message_template": "[{provider}] Environment Variable Error: {base_message}",
        "emoji": "🌍",
    },
    "unexpected": {
        "message_template": "[{provider}] Unexpected Auth Error: {base_message}",
        "emoji": "❗",
    },
}

# ------------------------------------------------------------------------- #
# Generalized Error Message Formatter
# ------------------------------------------------------------------------- #
def format_error_message(
    error_category: ErrorCategory,
    error_type: str,  # For provider errors, use one of ProviderErrorType; for client/resource, typically "default".
    source_name: str,  # Provider name for provider errors, resource name for resource errors.
    additional_message: Optional[str] = None,
    status_code: Optional[int] = None,
    request_id: Optional[str] = None,
    error_body: Optional[Any] = None,
    error_traceback: Optional[str] = None
) -> str:
    """
    Format a structured, verbose error message.

    For provider errors, this function uses the common provider error template (with the provider name)
    and combines it with provider-specific details (base_message, suggestions, documentation URL)
    from the PROVIDER_ERROR_MESSAGES dictionary.

    For client and resource errors, it uses the default configuration.
    """

    source_name = source_name.upper()

    if error_category == "provider":
        # Lookup provider-specific details.
        provider_msgs = PROVIDER_ERROR_MESSAGES.get("provider", {})
        provider_specific = provider_msgs.get(source_name, {}).get(error_type, {})
        base_message: str = provider_specific.get("base_message", "An error occurred.")
        suggestions = provider_specific.get("suggestions", [])
        documentation_url = provider_specific.get("documentation_url", "https://docs.astralai.com/errors")
        # Use the common provider error template.
        common = PROVIDER_ERROR_COMMON.get(error_type, {"message_template": "{base_message}", "emoji": ""})
        formatted_message = common["message_template"].format(provider=source_name, base_message=base_message)
        emoji = common["emoji"]
    elif error_category == "authentication":
        # Lookup authentication-specific details.
        auth_msgs = PROVIDER_ERROR_MESSAGES.get("authentication", {})
        auth_specific = auth_msgs.get(source_name, {}).get(error_type, {})
        base_message: str = auth_specific.get("base_message", "An authentication error occurred.")
        suggestions = auth_specific.get("suggestions", [])
        documentation_url = auth_specific.get("documentation_url", "https://docs.astralai.com/authentication")
        # Use the common authentication error template.
        common = AUTH_ERROR_COMMON.get(error_type, {"message_template": "{base_message}", "emoji": ""})
        formatted_message = common["message_template"].format(provider=source_name, base_message=base_message)
        emoji = common["emoji"]
    else:
        # For client and resource errors, use the default configuration.
        config = PROVIDER_ERROR_MESSAGES.get(error_category, {}).get("default", {})
        formatted_message = config.get("message", "An error occurred.")
        emoji = config.get("emoji", "")
        suggestions = config.get("suggestions", [])
        documentation_url = config.get("documentation_url", "https://docs.astralai.com/errors")
    
    # Append additional context if provided.
    # TODO: Do we want this?
    # if additional_message:
    #     formatted_message += f" Details: {additional_message}"
    
    # Build the structured error message.
    message_parts = []
    message_parts.append("\n" + "=" * 80)
    message_parts.append(f"  {emoji}  {error_type.upper()}  {emoji}")
    message_parts.append("=" * 80 + "\n")
    message_parts.append(f"📌 ERROR TYPE: {error_type.replace('_', ' ').upper()}")
    
    # Add source information.
    if error_category == "provider":
        message_parts.append(f"🏢 PROVIDER: {source_name}")
    elif error_category == "resource":
        message_parts.append(f"🏢 RESOURCE: {source_name}")
    else:
        message_parts.append(f"🏢 SOURCE: {source_name}")
    
    message_parts.append(f"📝 MESSAGE: {formatted_message}")
    message_parts.append("")  # Spacing
    
    # Append technical details if available.
    tech_details = []
    if status_code is not None:
        tech_details.append(f"Status code: {status_code}")
    if request_id:
        tech_details.append(f"Request ID: {request_id}")
    if error_body:
        body_str = str(error_body)
        if len(body_str) > 200:
            body_str = body_str[:200] + "..."
        tech_details.append(f"Response body: {body_str}")
    
    if tech_details:
        message_parts.append("🛠️  TECHNICAL DETAILS:")
        for detail in tech_details:
            message_parts.append(f"  • {detail}")
        message_parts.append("")
    
    # Append troubleshooting suggestions.
    if suggestions:
        message_parts.append("💡 POTENTIAL SOLUTIONS:")
        for suggestion in suggestions:
            message_parts.append(f"  • {suggestion}")
        message_parts.append("")
    
    # Append documentation links.
    message_parts.append("📚 DOCUMENTATION LINKS:")
    message_parts.append(f"  • Documentation: {documentation_url}")
    message_parts.append("")
    
    # Append error traceback if available.
    if error_traceback:
        message_parts.append("🔍 ERROR TRACEBACK:")
        message_parts.append(f"{error_traceback}")
    
    message_parts.append("=" * 80)
    
    return "\n".join(message_parts)
