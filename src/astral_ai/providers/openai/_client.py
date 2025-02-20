# -------------------------------------------------------------------
# OpenAI Provider Client
# -------------------------------------------------------------------

# Typing
from typing import Optional, Dict, Any, Union

# OpenAI
from openai import OpenAI, AsyncOpenAI


# Astral AI Models and Types (assumed to be defined elsewhere)
from astral_ai.constants._models import ModelProvider, OpenAIModels
from astral_ai.providers._base_client import BaseProviderClient

# OpenAI Types
from astral_ai.providers.openai._types import OpenAIRequestChatT, OpenAIRequestStreamingT, OpenAIRequestStructuredT, OpenAIStructuredResponse, OpenAIChatResponse, OpenAIStreamingResponse

# Exceptions
from astral_ai.exceptions import ProviderAuthenticationError, ProviderResponseError

# Astral Auth
from astral_ai._auth import AUTH_METHOD_NAME_TYPES, AUTH_CONFIG_TYPE, auth_method, AUTH_ENV_VARS

# OpenAI Provider Client Type
from astral_ai.providers._generics import OpenAIClients


class OpenAIProviderClient(BaseProviderClient[
        OpenAIClients,
        OpenAIRequestChatT,
        OpenAIRequestStructuredT,
        OpenAIRequestStreamingT,
        OpenAIChatResponse,
        OpenAIStructuredResponse,
        OpenAIStreamingResponse]):
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
        """
        api_key = config.get("api_key") or env.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderAuthenticationError("No API key provided for OpenAI authentication.")
        # Assume the OpenAI client accepts an 'api_key' parameter.
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

    @calculate_cost
    def create_completion_chat(self, request: OpenAIRequestChatT) -> OpenAIChatResponse:
        """
        Create a completion using the OpenAI API.
        """

        openai_response = self.client.chat.completions.create(**request)

        if isinstance(openai_response, OpenAIChatResponse):
            return openai_response
        else:
            raise ProviderResponseError(provider_name=self._model_provider, response_type="OpenAIChatResponse")

    # --------------------------------------------------------------------------
    # Create Structured Completion
    # --------------------------------------------------------------------------

    @calculate_cost
    def create_completion_structured(self, request: OpenAIRequestStructuredT) -> OpenAIStructuredResponse:
        """
        Create a structured completion using the OpenAI API.
        """
        openai_response = self.client.beta.chat.completions.parse(**request)

        if isinstance(openai_response, OpenAIStructuredResponse):
            return openai_response
        else:
            raise ProviderResponseError(provider_name=self._model_provider, response_type="OpenAIStructuredResponse")
