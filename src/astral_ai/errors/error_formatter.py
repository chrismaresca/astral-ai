# -------------------------------------------------------------------------------- #
# error_formatting.py
# -------------------------------------------------------------------------------- #

import os
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
        },
        "DEEPSEEK": {
            "unknown_method": {
                "base_message": "The authentication method you specified is not supported for DeepSeek.",
                "suggestions": [
                    "Use 'api_key' or 'api_key_with_base_url' authentication for DeepSeek.",
                    "Check the auth_method parameter in your config."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "method_failure": {
                "base_message": "Authentication with DeepSeek failed due to an error in the authentication method.",
                "suggestions": [
                    "Double-check your authentication configuration.",
                    "Ensure you're using 'api_key' or 'api_key_with_base_url' authentication for DeepSeek."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "configuration": {
                "base_message": "Your DeepSeek authentication configuration is incorrect.",
                "suggestions": [
                    "Check your configuration file for errors.",
                    "Make sure all required parameters are properly formatted."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "missing_credentials": {
                "base_message": "Required DeepSeek credentials are missing.",
                "suggestions": [
                    "For 'api_key' authentication, you must provide an API key.",
                    "For 'api_key_with_base_url' authentication, you must provide both an API key and a base URL.",
                    "Check your environment variables or configuration file."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "invalid_credentials": {
                "base_message": "Your DeepSeek credentials are invalid or rejected by the API.",
                "suggestions": [
                    "Verify your API key is correct and active.",
                    "Check if your DeepSeek account is in good standing.",
                    "Make sure your base URL is correctly formatted if using 'api_key_with_base_url'."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "environment_variables": {
                "base_message": "Required environment variables for DeepSeek authentication are missing.",
                "suggestions": [
                    "For 'api_key' authentication, set the DEEPSEEK_API_KEY environment variable.",
                    "For 'api_key_with_base_url', set both DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL environment variables.",
                    "Check your .env file or system environment variables."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
            },
            "unexpected": {
                "base_message": "An unexpected error occurred during DeepSeek authentication.",
                "suggestions": [
                    "Check your network connection.",
                    "Verify your API key is valid.",
                    "Make sure your DeepSeek account is active.",
                    "Check the base URL format if using 'api_key_with_base_url'."
                ],
                "documentation_url": "https://docs.astralai.com/authentication/deepseek"
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
    
    # Initialize message_parts for building the message
    message_parts = []

    # Process error category-specific data
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
    
    # 1. Build the header
    message_parts.append("\n\n" + "=" * 80)
    message_parts.append(f"  {emoji}  {error_type.upper()}  {emoji}")
    message_parts.append("=" * 80 + "\n")
    
    # 2. Add basic error information
    message_parts.append(f"📌 ERROR TYPE: {error_type.replace('_', ' ').upper()}")
    
    # Add source information.
    if error_category == "provider":
        message_parts.append(f"🏢 PROVIDER: {source_name}")
    elif error_category == "resource":
        message_parts.append(f"🏢 RESOURCE: {source_name}")
    else:
        message_parts.append(f"🏢 SOURCE: {source_name}")
    
    message_parts.append(f"📝 MESSAGE: {formatted_message}")
    
    # 3. Add additional details if provided (after the basic info)
    if additional_message:
        message_parts.append(f"📋 DETAILS: {additional_message}")
    
    message_parts.append("")  # Spacing
    
    # 4. Add technical details if available
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
    
    # 5. Add troubleshooting suggestions
    if suggestions:
        message_parts.append("💡 POTENTIAL SOLUTIONS:")
        for suggestion in suggestions:
            message_parts.append(f"  • {suggestion}")
        message_parts.append("")
    
    # 6. Add documentation links
    message_parts.append("📚 DOCUMENTATION LINKS:")
    message_parts.append(f"  • Documentation: {documentation_url}")
    message_parts.append("")
    
    # 7. Add error traceback if enabled
    if error_traceback and os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
        message_parts.append("🔍 ERROR TRACEBACK:")
        message_parts.append(f"{error_traceback}")
    
    # 8. Add footer
    message_parts.append("=" * 80)
    
    return "\n".join(message_parts)

def format_multiple_auth_errors(
    provider_name: str,
    errors: list[tuple[str, Exception]],
    error_traceback: Optional[str] = None
) -> str:
    """
    Format an error message for multiple authentication failures.
    
    Args:
        provider_name: The name of the provider
        errors: List of tuples containing (auth_method_name, exception)
        error_traceback: Optional traceback string
        
    Returns:
        Formatted error message with details about all failed authentication attempts
    """
    provider_name = provider_name.upper()
    
    message_parts = []
    
    # 1. Header
    message_parts.append("\n\n" + "=" * 80)
    if len(errors) == 1:
        message_parts.append("  🔒  AUTHENTICATION FAILURE  🔒")
    else:
        message_parts.append("  🔒  MULTIPLE AUTHENTICATION FAILURES  🔒")
    message_parts.append("=" * 80 + "\n")
    
    # 2. Basic information
    if len(errors) == 1:
        auth_method, _ = errors[0]
        message_parts.append(f"📌 ERROR TYPE: AUTHENTICATION FAILURE")
        message_parts.append(f"🏢 PROVIDER: {provider_name}")
        message_parts.append(f"🔑 METHOD: {auth_method}")
        message_parts.append(f"📝 MESSAGE: Authentication failed for provider '{provider_name}' using method '{auth_method}'.")
    else:
        message_parts.append(f"📌 ERROR TYPE: MULTIPLE AUTHENTICATION FAILURES")
        message_parts.append(f"🏢 PROVIDER: {provider_name}")
        message_parts.append(f"📝 MESSAGE: All authentication methods failed for provider '{provider_name}'.")
    
    # Extract any additional details from the first error
    if len(errors) == 1:
        _, error = errors[0]
        if hasattr(error, "_message") and error._message:
            message_parts.append(f"📋 DETAILS: {error._message}")
    
    message_parts.append("")  # Spacing
    
    # 3. Individual errors
    if len(errors) == 1:
        message_parts.append("❌ AUTHENTICATION ERROR DETAILS:")
    else:
        message_parts.append("❌ FAILED AUTHENTICATION ATTEMPTS:")
        
    for idx, (auth_method, error) in enumerate(errors, 1):
        if len(errors) > 1:
            message_parts.append(f"\n--- ATTEMPT #{idx}: '{auth_method}' ---")
        
        # Extract error information based on error type
        if hasattr(error, "missing_credentials") and error.missing_credentials:
            missing_creds = ", ".join(error.missing_credentials)
            message_parts.append(f"Missing credentials: {missing_creds}")
            
            # For missing credentials errors, check if we need to set env variables
            if hasattr(error, "provider_name"):
                env_vars = []
                for cred in error.missing_credentials:
                    env_var = f"{error.provider_name.upper()}_{cred.upper()}"
                    env_vars.append(env_var)
                if env_vars:
                    env_vars_str = ", ".join(env_vars)
                    message_parts.append(f"Required environment variables: {env_vars_str}")
        elif isinstance(error, Exception):
            # Get error message without all the formatting
            error_msg = str(error)
            # Try to extract just the basic error message, not the full formatted version
            if "DETAILS:" in error_msg:
                error_msg = error_msg.split("DETAILS:")[1].split("\n")[0].strip()
            elif "\n" in error_msg:
                error_msg = error_msg.split("\n")[0]
                
            message_parts.append(f"Error: {error_msg}")
            
        # Add more specific attributes if available
        for attr_name in dir(error):
            if not attr_name.startswith('_') and not callable(getattr(error, attr_name)) and attr_name not in [
                "args", "with_traceback", "errors", "missing_credentials", "provider_name", "auth_method_name"
            ]:
                value = getattr(error, attr_name)
                if isinstance(value, (str, int, bool, list)) and value:
                    # Skip very long values or empty values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    message_parts.append(f"{attr_name}: {value}")
            
    message_parts.append("\n" + "-" * 80)
    
    # 4. Suggestions
    message_parts.append("\n💡 POTENTIAL SOLUTIONS:")
    
    # Tailored suggestions based on error types
    missing_creds_errors = [e for _, e in errors if hasattr(e, "missing_credentials") and e.missing_credentials]
    if missing_creds_errors:
        message_parts.append("  • Set the required environment variables mentioned above")
        message_parts.append("  • Add the missing credentials to your configuration file")
    
    message_parts.append("  • Verify your API keys and other credentials are correctly set")
    message_parts.append("  • Configure a specific authentication method in your astral.yaml file")
    
    if len(errors) > 1:
        message_parts.append("  • Review the individual error messages above for specific issues")
    message_parts.append("")
    
    # 5. Documentation
    message_parts.append("📚 DOCUMENTATION LINKS:")
    message_parts.append(f"  • Authentication guide: https://docs.astralai.com/authentication")
    message_parts.append(f"  • Provider-specific documentation: https://docs.astralai.com/authentication/{provider_name.lower()}")
    message_parts.append("")
    
    # 6. Traceback - only include if enabled
    if error_traceback and os.environ.get("ASTRAL_TRACEBACK_IN_MESSAGE", "").lower() == "true":
        message_parts.append("🔍 ERROR TRACEBACK:")
        message_parts.append(f"{error_traceback}")
    
    # 7. Footer
    message_parts.append("=" * 80)
    
    return "\n".join(message_parts)
