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
from astral_ai.errors.exceptions import ProviderResponseError, AstralAuthMethodFailureError
from astral_ai.errors.error_decorators import provider_error_handler


# Astral Auth
from astral_ai._auth import AUTH_CONFIG_TYPE, auth_method, AUTH_ENV_VARS, validate_credentials

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

    @auth_method("api_key_with_base_url")
    def auth_via_api_key_with_base_url(self, config: AUTH_CONFIG_TYPE, env: AUTH_ENV_VARS, credentials: Dict[str, str] = None) -> OpenAI:
        """
        Authenticate using an API key from config or environment variables.
        
        Args:
            config: Configuration dictionary
            env: Environment variables dictionary
            credentials: Pre-validated credentials from the auth_method decorator
        
        Returns:
            OpenAI: Initialized OpenAI client for DeepSeek
            
        Raises:
            AstralAuthMethodFailureError: If client initialization fails
        """
        # If credentials weren't passed (shouldn't happen with our decorator), validate them
        if credentials is None:
            credentials = validate_credentials(
                auth_method="api_key_with_base_url",
                provider_name=self._model_provider,
                config=config,
                env=env
            )

        # Initialize the client with the credentials
        try:
            # IMPORTANT: We use the OpenAI client for DeepSeek
            return OpenAI(api_key=credentials["api_key"], base_url=DEEPSEEK_BASE_URL)
        except Exception as e:
            raise AstralAuthMethodFailureError(
                f"Failed to initialize DeepSeek client: {str(e)}",
                auth_method_name="api_key_with_base_url",
                provider_name=self._model_provider
            ) from e

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
            raise ProviderResponseError(provider_name=self._model_provider, response_type="DeepSeekChatResponse")

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
            raise ProviderResponseError(provider_name=self._model_provider, response_type="DeepSeekStructuredResponse")
