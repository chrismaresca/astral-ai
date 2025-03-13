from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Model Support Exceptions
# -------------------------------------------------------------------------------- #

"""
This module contains exceptions related to model capabilities and feature support.

These exceptions are raised when a requested feature is not supported by a model.
The module provides specialized exceptions for different types of unsupported features
and a base class for all model capability errors.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import Optional, Any, Union, List

# module imports
from astral_ai.constants._models import ModelName, ModelProvider
from astral_ai.errors.exceptions import AstralBaseError
from astral_ai.constants._supported_models import supports_feature, FEATURE_NAME


# ------------------------------------------------------------------------------
# Model Error
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

# -------------------------------------------------------------------------------- #
# Helper Functions
# -------------------------------------------------------------------------------- #


def get_model_feature_error_message(model_name: Union[ModelName, str], feature_name: str) -> str:
    """
    Get a standardized error message for when a model doesn't support a feature.

    Args:
        model_name: The name of the model that doesn't support the feature
        feature_name: The name of the unsupported feature

    Returns:
        A formatted error message
    """
    return f"The model '{model_name}' does not support the '{feature_name}' feature."


# -------------------------------------------------------------------------------- #
# Base Model Capability Error
# -------------------------------------------------------------------------------- #


class ModelCapabilityError(AstralBaseError):
    """Base exception for model capability errors."""

    def __init__(self, message: str, *,
                 model_name: Optional[Union[ModelName, str]] = None,
                 model_provider: Optional[Union[ModelProvider, str]] = None,
                 feature_name: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.model_provider = model_provider
        self.feature_name = feature_name
        self.documentation_link = documentation_link


# -------------------------------------------------------------------------------- #
# Response Model Missing Error
# -------------------------------------------------------------------------------- #


class ResponseModelMissingError(ModelCapabilityError):
    """Exception raised when a response model is required but not provided."""

    feature_name = "structured_output"

    def __init__(self,
                 message: str = "Response model is required for structured output",
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        super().__init__(
            message,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# Structured Output Not Supported Error
# -------------------------------------------------------------------------------- #


class StructuredOutputNotSupportedError(ModelCapabilityError):
    """
    Exception raised when structured output is not supported by the model.

    Models that support structured output include:
    - OpenAI: gpt-4o, o1, o1-mini, o3-mini
    - Anthropic: claude-3-haiku, claude-3-5-sonnet, claude-3-opus
    """

    feature_name = "structured_output"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# Reasoning Effort Not Supported Error
# -------------------------------------------------------------------------------- #


class ReasoningEffortNotSupportedError(ModelCapabilityError):
    """
    Exception raised when reasoning effort is not supported by the model.

    Models that support reasoning effort include:
    - OpenAI: o1, o3-mini
    """

    feature_name = "reasoning_effort"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# Tools Not Supported Error
# -------------------------------------------------------------------------------- #


class ToolsNotSupportedError(ModelCapabilityError):
    """
    Exception raised when tools/function calling is not supported by the model.

    Models that support function calls include:
    - OpenAI: gpt-4o, o1, o1-mini, o3-mini
    - Anthropic: claude-3-haiku, claude-3-5-sonnet, claude-3-opus
    """

    feature_name = "function_calls"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# Image Ingestion Not Supported Error
# -------------------------------------------------------------------------------- #


class ImageIngestionNotSupportedError(ModelCapabilityError):
    """
    Exception raised when image ingestion is not supported by the model.

    Models that support image ingestion include:
    - OpenAI: gpt-4o, o1
    - Anthropic: claude-3-5-sonnet, claude-3-opus
    """

    feature_name = "image_ingestion"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# System Message Not Supported Error
# -------------------------------------------------------------------------------- #


class SystemMessageNotSupportedError(ModelCapabilityError):
    """
    Exception raised when system messages are not supported by the model.

    Models that support system messages include:
    - OpenAI: gpt-4o
    - Anthropic: claude-3-5-sonnet, claude-3-opus
    """

    feature_name = "system_message"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )


# -------------------------------------------------------------------------------- #
# Developer Message Not Supported Error
# -------------------------------------------------------------------------------- #


class DeveloperMessageNotSupportedError(ModelCapabilityError):
    """
    Exception raised when developer messages are not supported by the model.

    Models that support developer messages include:
    - OpenAI: o1, o3-mini
    """

    feature_name = "developer_message"

    def __init__(self,
                 model_name: Union[ModelName, str],
                 message: Optional[str] = None,
                 documentation_link: Optional[str] = None,
                 **kwargs: Any) -> None:
        if message is None:
            message = get_model_feature_error_message(model_name, self.feature_name)

        super().__init__(
            message,
            model_name=model_name,
            feature_name=self.feature_name,
            documentation_link=documentation_link,
            **kwargs
        )
