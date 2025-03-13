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
    AstralAuthMethodFailureError,
    AstralMissingCredentialsError
)
from astral_ai.errors.error_decorators import provider_error_handler

# Astral Auth
from astral_ai._auth import AUTH_CONFIG_TYPE, auth_method, AUTH_ENV_VARS, AUTH_METHOD_NAMES

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


    # --------------------------------------------------------------------------
    # Validate Credentials
    # --------------------------------------------------------------------------

    def _validate_credentials(self, auth_method_name: AUTH_METHOD_NAMES, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS) -> Dict[str, str]:
        """
        Validate the credentials for the OpenAI provider.
        """
        credentials = {}

        if auth_method_name == "api_key":

            credentials["api_key"] = config.get(self._model_provider, {}).get(auth_method_name, None) or env.get("OPENAI_API_KEY")
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

    @auth_method("api_key")
    def auth_via_api_key(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS, async_client: bool = False) -> OpenAI:
        """
        Authenticate using an API key from config or environment variables.

        Args:
            config: Configuration dictionary
            env: Environment variables dictionary
            credentials: Pre-validated credentials from the auth_method decorator
        """

        credentials = self._validate_credentials(
            auth_method_name="api_key",
            config=config,
            env=env
        )

        # Extract the credentials
        api_key = credentials["api_key"]

        if async_client:
            return AsyncOpenAI(api_key=api_key)
        else:
            return OpenAI(api_key=api_key)

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
    # Create Completion Async
    # --------------------------------------------------------------------------

    @provider_error_handler
    async def create_completion_chat_async(self, request: OpenAIRequestChatType) -> OpenAIChatResponseType:
        """
        Create a completion asynchronously using the OpenAI API.

        Args:
            request: The request to create a completion.

        Returns:
            The completion.
        """
        # Initialize AsyncOpenAI with the same credentials
        async_client = AsyncOpenAI(api_key=self.client.api_key)
        
        openai_response = await self.client.chat.completions.create(**request)

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

    # --------------------------------------------------------------------------
    # Create Structured Completion Async
    # --------------------------------------------------------------------------

    @provider_error_handler
    async def create_completion_structured_async(self, request: OpenAIRequestStructuredType) -> OpenAIStructuredResponseType:
        """
        Create a structured completion asynchronously using the OpenAI API.

        Args:
            request: The request to create a structured completion.

        Returns:
            The structured completion.
        """
        # Initialize AsyncOpenAI with the same credentials
        async_client = AsyncOpenAI(api_key=self.client.api_key)
        
        openai_response = await async_client.beta.chat.completions.parse(**request)

        if isinstance(openai_response, OpenAIStructuredResponseType):
            return openai_response
        else:
            raise AstralProviderResponseError(
                f"Unexpected response type from {self._model_provider}",
                provider_name=self._model_provider,
                expected_response_type="OpenAIStructuredResponse"
            )
