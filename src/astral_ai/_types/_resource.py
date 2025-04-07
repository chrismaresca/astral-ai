# -------------------------------------------------------------------------------- #
# Astral Base Resource Model
# -------------------------------------------------------------------------------- #

"""
Base Resource Model for Astral AI - shared common base for requests and responses
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import uuid
from abc import ABC
from typing import Optional, Dict, Any
from datetime import datetime
# Typing Extensions
from typing_extensions import Self

# Pydantic
from pydantic import BaseModel, PrivateAttr, Field, model_validator

# Astral AI Constants
from astral_ai.constants._models import ModelName, ModelProvider, ResourceType

# Astral AI Utils
from astral_ai.utilities import (
    get_provider_from_model_name,
    get_resource_type_from_model_name,
)

# -------------------------------------------------------------------------------- #
# Base Resource
# -------------------------------------------------------------------------------- #


class AstralBaseResource(BaseModel, ABC):
    """
    Base Resource Model for Astral AI

    This is a common base class that unifies shared functionality between
    AstralBaseRequest and AstralBaseResponse classes.
    """
    # Resource Identifiers
    _resource_id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    _time_created: datetime = PrivateAttr(default_factory=lambda: datetime.now())

    # Identifiers
    _organization_id: Optional[str] = PrivateAttr(default=None)
    _project_id: Optional[str] = PrivateAttr(default=None)
    _user_id: Optional[str] = PrivateAttr(default=None)

    # Private Attributes that are set after validation automatically
    _provider_name: ModelProvider = PrivateAttr()
    _resource_type: ResourceType = PrivateAttr()

    # Model
    model: ModelName = Field(description="The model used for the resource.")

    def __init__(self, **data: Dict[str, Any]) -> None:
        """Override init to prevent direct instantiation of abstract class."""
        if self.__class__ == AstralBaseResource:
            raise TypeError("Cannot instantiate abstract class AstralBaseResource directly")
        super().__init__(**data)

    # --------------------------------------------------------------------------
    # Resource ID
    # --------------------------------------------------------------------------

    @property
    def resource_id(self) -> str:
        """The resource ID."""
        return self._resource_id

    # --------------------------------------------------------------------------
    # Time Created
    # --------------------------------------------------------------------------

    @property
    def time_created(self) -> datetime:
        """Get the time created."""
        return self._time_created

    # --------------------------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Organization ID
    # --------------------------------------------------------------------------

    @property
    def organization_id(self) -> Optional[str]:
        """Get the organization ID"""
        return self._organization_id

    @organization_id.setter
    def organization_id(self, value: Optional[str]) -> None:
        """Set the organization ID"""
        self._organization_id = value

    # --------------------------------------------------------------------------
    # Project ID
    # --------------------------------------------------------------------------

    @property
    def project_id(self) -> Optional[str]:
        """Get the project ID"""
        return self._project_id

    @project_id.setter
    def project_id(self, value: Optional[str]) -> None:
        """Set the project ID"""
        self._project_id = value

    # --------------------------------------------------------------------------
    # User ID
    # --------------------------------------------------------------------------

    @property
    def user_id(self) -> Optional[str]:
        """Get the user ID"""
        return self._user_id

    @user_id.setter
    def user_id(self, value: Optional[str]) -> None:
        """Set the user ID"""
        self._user_id = value

    # --------------------------------------------------------------------------
    # Provider Name
    # --------------------------------------------------------------------------

    @property
    def provider_name(self) -> ModelProvider:
        """The provider name for the resource."""
        return self._provider_name

    @model_validator(mode="after")
    def set_provider_name(self) -> Self:
        """Set the provider name for the resource."""
        self._provider_name = get_provider_from_model_name(model_name=self.model)
        return self

    # --------------------------------------------------------------------------
    # Resource Type
    # --------------------------------------------------------------------------

    @property
    def resource_type(self) -> ResourceType:
        """The resource type for the resource."""
        return self._resource_type

    @model_validator(mode="after")
    def set_resource_type(self) -> Self:
        """Set the resource type for the resource."""
        self._resource_type = get_resource_type_from_model_name(model_name=self.model)
        return self
