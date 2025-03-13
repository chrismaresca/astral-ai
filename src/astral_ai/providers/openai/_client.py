# -------------------------------------------------------------------
# OpenAI Provider Client
# -------------------------------------------------------------------

# Typing
from typing import Optional, Dict, Any, Union

# OpenAI
from openai import OpenAI, AsyncOpenAI, APIError


# Astral AI Models and Types (assumed to be defined elsewhere)
from astral_ai.constants._models import ModelProvider, OpenAIModels
from astral_ai.providers._base_client import BaseProviderClient

# OpenAI Types
from ._types import (
    OpenAIRequestChatType,
    OpenAIRequestStreamingType,
    OpenAIRequestStructuredType,
    OpenAIStructuredResponseType,
    OpenAIChatResponseType,
    OpenAIStreamingResponseType
)
# Exceptions
from astral_ai.errors.exceptions import (
    AstralProviderResponseError,
    AstralAuthMethodFailureError
)
from astral_ai.errors.error_decorators import provider_error_handler

# Astral Auth
from astral_ai._auth import AUTH_CONFIG_TYPE, auth_method, AUTH_ENV_VARS, validate_credentials

# Provider Types
from astral_ai.providers.openai._types import OpenAIClientsType


class OpenAIProviderClient(BaseProviderClient[
        OpenAIClientsType,
        OpenAIRequestChatType,
        OpenAIRequestStructuredType,
        OpenAIRequestStreamingType,
        OpenAIChatResponseType,
        OpenAIStructuredResponseType,
        OpenAIStreamingResponseType]):
    """
    Client for OpenAI.
    """

    # --------------------------------------------------------------------------
    # Model Provider
    # --------------------------------------------------------------------------

    _model_provider: ModelProvider = "openai"

    # --------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------
    def __init__(self, config: Optional[AUTH_CONFIG_TYPE] = None):
        # Initialize the base class (which performs authentication)
        super().__init__(config)

    @auth_method("api_key")
    def auth_via_api_key(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> OpenAI:
        """
        Authenticate using an API key from config or environment variables.

        Args:
            config: Configuration dictionary
            env: Environment variables dictionary
            credentials: Pre-validated credentials from the auth_method decorator
        """
        credentials = validate_credentials(
            auth_method="api_key",  # Use api_key for testing with base_url requirement
            provider_name=self._model_provider,
            config=config,
            env=env
        )

        return OpenAI(api_key=credentials["api_key"])

    # # --------------------------------------------------------------------------
    # # Create Completion Stream
    # # --------------------------------------------------------------------------
    # def create_completion_stream(
    #     self,
    #     request: OpenAIRequest,
    # ) -> Stream[ChatCompletionChunk]:
    #     """
    #     Create a completion stream using the OpenAI API.
    #     """

    #     response = self.client.chat.completions.create(**request)

    #     return response

    # --------------------------------------------------------------------------
    # Create Completion
    # --------------------------------------------------------------------------

    @provider_error_handler
    def create_completion_chat(self, request: OpenAIRequestChatType) -> OpenAIChatResponseType:
        """
        Create a completion using the OpenAI API.

        Args:
            request: The request to create a completion.

        Returns:
            The completion.
        """

        openai_response = self.client.chat.completions.create(**request)

        if isinstance(openai_response, OpenAIChatResponseType):
            return openai_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="OpenAIChatResponse"
            )

    # --------------------------------------------------------------------------
    # Create Structured Completion
    # --------------------------------------------------------------------------

    @provider_error_handler
    def create_completion_structured(self, request: OpenAIRequestStructuredType) -> OpenAIStructuredResponseType:
        """
        Create a structured completion using the OpenAI API.

        Args:
            request: The request to create a structured completion.

        Returns:
            The structured completion.
        """
        openai_response = self.client.beta.chat.completions.parse(**request)

        if isinstance(openai_response, OpenAIStructuredResponseType):
            return openai_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="OpenAIStructuredResponse"
            )
