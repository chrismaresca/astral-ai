# -------------------------------------------------------------------
# OpenAI Provider Client
# -------------------------------------------------------------------

# Typing
from typing import Optional, Dict, Any, Union

# OpenAI
from openai import OpenAI, AsyncOpenAI


# Astral AI Models and Types (assumed to be defined elsewhere)
from astral_ai.constants._models import ModelProvider, DeepSeekModels
from astral_ai.providers._base_client import BaseProviderClient

# DeepSeek Types
from ._types import (
    DeepSeekRequestChatType,
    DeepSeekRequestStreamingType,
    DeepSeekRequestStructuredType,
    DeepSeekStructuredResponseType,
    DeepSeekChatResponseType,
    DeepSeekStreamingResponseType
)

# DeepSeek Constants
from ._constants import DEEPSEEK_BASE_URL

# Exceptions
from astral_ai.errors.exceptions import (
    AstralProviderResponseError,
    AstralAuthMethodFailureError,
    AstralMissingCredentialsError
)
from astral_ai.errors.error_decorators import provider_error_handler


# Astral Auth
from astral_ai._auth import AUTH_CONFIG_TYPE, auth_method, AUTH_ENV_VARS, AUTH_METHOD_NAMES

# Provider Types
from astral_ai.providers.deepseek._types import DeepSeekClientsType


class DeepSeekProviderClient(BaseProviderClient[
        DeepSeekClientsType,
        DeepSeekRequestChatType,
        DeepSeekRequestStructuredType,
        DeepSeekRequestStreamingType,
        DeepSeekChatResponseType,
        DeepSeekStructuredResponseType,
        DeepSeekStreamingResponseType]):
    """
    Client for OpenAI.
    """

    # --------------------------------------------------------------------------
    # Model Provider
    # --------------------------------------------------------------------------

    _model_provider: ModelProvider = "deepseek"

    # --------------------------------------------------------------------------
    # Initialize
    # --------------------------------------------------------------------------
    def __init__(self, config: Optional[AUTH_CONFIG_TYPE] = None):
        # Initialize the base class (which performs authentication)
        super().__init__(config)

    # --------------------------------------------------------------------------
    # Validate Credentials
    # --------------------------------------------------------------------------

    def _validate_credentials(self, auth_method_name: AUTH_METHOD_NAMES, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> Dict[AUTH_METHOD_NAMES, str]:
        """
        Validate the credentials for the DeepSeek provider.
        """
        credentials = {}

        if auth_method_name == "api_key_with_base_url":
            credentials["api_key"] = config.get(self._model_provider, {}).get(auth_method_name, None) or env.get("DEEPSEEK_API_KEY")
            if not credentials["api_key"]:
                raise AstralAuthMethodFailureError("API key is required")

            return credentials
        else:
            raise AstralMissingCredentialsError(
                f"Invalid authentication method: {auth_method_name}",
                auth_method_name=auth_method_name,
                provider_name=self._model_provider,
                required_credentials=["api_key"],
                missing_credentials=[auth_method_name]
            )

    # --------------------------------------------------------------------------
    # Authenticate
    # --------------------------------------------------------------------------

    @auth_method("api_key_with_base_url")
    def auth_via_api_key_with_base_url(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> OpenAI:
        """
        Authenticate using an API key from config or environment variables.

        Args:
            config: Configuration dictionary
            env: Environment variables dictionary

        Returns:
            OpenAI: Initialized OpenAI client for DeepSeek

        Raises:
            AstralMissingCredentialsError: If any required credentials are missing
            AstralAuthMethodFailureError: If client initialization fails
        """
        credentials = self._validate_credentials(
            auth_method_name="api_key_with_base_url",
            config=config,
            env=env
        )

        # Extract the credentials
        api_key = credentials["api_key"]

        # IMPORTANT: This is hard-coded for DeepSeek locally
        base_url = DEEPSEEK_BASE_URL

        # Initialize the client with the credentials and hard-coded base URL
        # Any exceptions will be caught by the auth_method decorator and wrapped appropriately
        # IMPORTANT: We use the OpenAI client for DeepSeek
        return OpenAI(api_key=api_key, base_url=base_url)

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
    def create_completion_chat(self, request: DeepSeekRequestChatType) -> DeepSeekChatResponseType:
        """
        Create a completion using the OpenAI SDK to communicate with the DeepSeek API.

        Args:
            request: The request to create a completion.

        Returns:
            The completion.
        """

        # IMPORTANT: We use the OpenAI client for DeepSeek
        openai_response = self.client.chat.completions.create(**request)

        if isinstance(openai_response, DeepSeekChatResponseType):
            return openai_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="DeepSeekChatResponse"
            )

    # --------------------------------------------------------------------------
    # Create Completion Async
    # --------------------------------------------------------------------------

    @provider_error_handler
    async def create_completion_chat_async(self, request: DeepSeekRequestChatType) -> DeepSeekChatResponseType:
        """
        Create a completion asynchronously using the OpenAI SDK to communicate with the DeepSeek API.

        Args:
            request: The request to create a completion.

        Returns:
            The completion.
        """
        # Initialize AsyncOpenAI with the same credentials and base URL
        async_client = AsyncOpenAI(api_key=self.client.api_key, base_url=self.client.base_url)

        # IMPORTANT: We use the OpenAI client for DeepSeek
        openai_response = await async_client.chat.completions.create(**request)

        if isinstance(openai_response, DeepSeekChatResponseType):
            return openai_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="DeepSeekChatResponse"
            )

    # --------------------------------------------------------------------------
    # Create Structured Completion
    # --------------------------------------------------------------------------

    @provider_error_handler
    def create_completion_structured(self, request: DeepSeekRequestStructuredType) -> DeepSeekStructuredResponseType:
        """
        Create a structured completion using the OpenAI SDK to communicate with the DeepSeek API.

        Args:
            request: The request to create a structured completion.

        Returns:
            The structured completion.
        """
        deepseek_response = self.client.chat.completions.create(**request)

        if isinstance(deepseek_response, DeepSeekStructuredResponseType):
            return deepseek_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="DeepSeekStructuredResponse"
            )

    # --------------------------------------------------------------------------
    # Create Structured Completion Async
    # --------------------------------------------------------------------------

    @provider_error_handler
    async def create_completion_structured_async(self, request: DeepSeekRequestStructuredType) -> DeepSeekStructuredResponseType:
        """
        Create a structured completion asynchronously using the OpenAI SDK to communicate with the DeepSeek API.

        Args:
            request: The request to create a structured completion.

        Returns:
            The structured completion.
        """
        # Initialize AsyncOpenAI with the same credentials and base URL
        async_client = AsyncOpenAI(api_key=self.client.api_key, base_url=self.client.base_url)

        deepseek_response = await async_client.chat.completions.create(**request)

        if isinstance(deepseek_response, DeepSeekStructuredResponseType):
            return deepseek_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="DeepSeekStructuredResponse"
            )
