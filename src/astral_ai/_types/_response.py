# ------------------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------------------

"""
Response Models for Astral AI
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Standard Library
import time
import uuid

# Types
from typing import Optional, Iterable, TypeVar

# Typed Extensions
from typing_extensions import Self

# Pydantic
from pydantic import BaseModel, PrivateAttr, Field, model_validator

# Astral AI
from astral_ai._models import ModelName, ModelProvider, get_provider_from_model_name
from astral_ai._types._request import ToolAlias, Metadata

# ------------------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Provider Response
# ------------------------------------------------------------------------------


class ProviderResponseObject(BaseModel):
    """
    Provider Response Model for Astral AI
    """
    provider_object: str = Field(description="The object type of the provider response")
    provider_response_id: str = Field(description="The ID of the provider response")
    provider_model_id: str = Field(description="The ID of the provider model")
    provider_request_id: str = Field(description="The ID of the provider request")
    provider_created: int = Field(description="The created time (UNIX timestamp) for the provider response")


class BaseResponse(BaseModel):
    """
    Base Response Model for Astral AI
    """
    _response_id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    _time_created: float = PrivateAttr(default_factory=lambda: time.time())
    _provider_name: ModelProvider = PrivateAttr()
    _provider_response: ProviderResponseObject = PrivateAttr()

    model: ModelName = Field(description="The model used to generate the response")

    @property
    def response_id(self) -> str:
        """
        Get the response ID
        """
        return self._response_id

    @property
    def time_created(self) -> float:
        """
        Get the time created
        """
        return self._time_created

    @property
    def provider_name(self) -> ModelProvider:
        """
        Get the provider name
        """
        return self._provider_name

    @property
    def provider_response(self) -> ProviderResponseObject:
        """
        Get the provider response
        """
        return self._provider_response

    @model_validator(mode="after")
    def set_provider_name(self) -> Self:
        """
        Set the provider name
        """
        self._provider_name = get_provider_from_model_name(self.model)
        return self


# ------------------------------------------------------------------------------
# Chat Response
# ------------------------------------------------------------------------------


# class RequestInputObject(BaseModel):
#     """
#     Request Input Object for Astral AI
#     """
#     seed: Optional[str] = Field(description="The seed for the request")
#     temperature: Optional[float] = Field(description="The temperature for the request")
#     max_tokens: Optional[int] = Field(description="The maximum number of tokens for the request")
#     top_p: Optional[float] = Field(description="The top p for the request")
#     frequency_penalty: Optional[float] = Field(description="The frequency penalty for the request")
#     presence_penalty: Optional[float] = Field(description="The presence penalty for the request")
#     service_tier: Optional[str] = Field(description="The service tier for the request")
#     tools: Optional[Iterable[ToolAlias]] = Field(description="The tools for the request")
#     metadata: Optional[Metadata] = Field(description="The metadata for the request")
#     system_fingerprint: Optional[str] = Field(description="The system fingerprint for the request")


# ------------------------------------------------------------------------------
# Usage Details
# ------------------------------------------------------------------------------


class ChatUsageDetails(BaseModel):
    """
    Chat Completion Usage Details Model for Astral AI
    """
    accepted_prediction_tokens: Optional[int] = Field(description="The accepted prediction tokens for the request")
    audio_tokens: Optional[int] = Field(description="The audio tokens for the request")
    reasoning_tokens: Optional[int] = Field(description="The reasoning tokens for the request")
    rejected_prediction_tokens: Optional[int] = Field(description="The rejected prediction tokens for the request")


class PromptUsageDetails(BaseModel):
    """
    Prompt Usage Details Model for Astral AI
    """
    audio_tokens: Optional[int] = Field(description="The audio tokens for the request")
    cached_tokens: Optional[int] = Field(description="The cached tokens for the request")


# TODO: This or decorator???
class ChatCostDetails(BaseModel):
    """
    Chat Cost Details Model for Astral AI
    """
    completion_cost: float = Field(description="The completion cost for the request")
    prompt_cost: float = Field(description="The prompt cost for the request")


class ChatUsage(BaseModel):
    """
    Chat Usage Model for Astral AI
    """
    completion_tokens: int = Field(description="The completion tokens for the request")
    prompt_tokens: int = Field(description="The prompt tokens for the request")
    total_tokens: int = Field(description="The total tokens for the request")

    # Details
    completion_tokens_details: Optional[ChatUsageDetails] = Field(description="The completion tokens details for the request")
    prompt_tokens_details: Optional[PromptUsageDetails] = Field(description="The prompt tokens details for the request")


# ------------------------------------------------------------------------------
# AI Response Message Objects
# ------------------------------------------------------------------------------

class AIResponseMessage(BaseModel):
    """
    Message Model for Astral AI
    """
    role: str = Field(description="The role of the message")
    content: str = Field(description="The content of the message")


class AIResponseMessageObject(BaseModel):
    """
    Message Object Model for Astral AI
    """
    index: int = Field(description="The index of the message")
    message: AIResponseMessage = Field(description="The message for the response")
    finish_reason: str = Field(description="The finish reason for the message")


# ------------------------------------------------------------------------------
# Chat Response
# ------------------------------------------------------------------------------


class AstralChatResponse(BaseResponse):
    """
    Chat Response Model for Astral AI
    """
    response: Iterable[AIResponseMessageObject] = Field(description="The messages for the response")
    usage: ChatUsage = Field(description="The usage for the response")


# ------------------------------------------------------------------------------
# AI Structured Response Objects
# ------------------------------------------------------------------------------


StructuredOutputResponse = TypeVar('StructuredOutputResponse', bound=BaseModel)

# ------------------------------------------------------------------------------
# Structured Response
# ------------------------------------------------------------------------------


class AstralStructuredResponse(BaseResponse):
    """
    Structured Response Model for Astral AI
    """
    response: StructuredOutputResponse = Field(description="The response for the structured response")
    usage: ChatUsage = Field(description="The usage for the response")
