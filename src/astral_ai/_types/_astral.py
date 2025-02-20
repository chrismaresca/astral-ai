# ------------------------------------------------------------------------------
# Astral AI Specific Types and Models
# ------------------------------------------------------------------------------

"""
Astral AI Specific Types and Models.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in
from typing import Optional, TypedDict

# Pydantic
from pydantic import BaseModel, Field

# Astral AI
from astral_ai._auth import AUTH_CONFIG_TYPE


# ------------------------------------------------------------------------------
# Astral Client Parameters
# ------------------------------------------------------------------------------


class AstralClientParams(BaseModel):
    """
    Astral AI Client Parameters.
    """
    new_client: bool = Field(default=False, description="If True, force the creation of a new client even if one exists.")
    client_config: Optional[AUTH_CONFIG_TYPE] = Field(default=None, description="Config used to instantiate a new client if needed.")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


class AstralUsageParams(BaseModel):
    """
    Astral Usage Parameters.
    """
    usage: bool = Field(default=True, description="If True, include usage information in the response.")
    cost: bool = Field(default=True, description="If True, include cost information in the response.")
    latency: bool = Field(default=True, description="If True, include latency information in the response.")


# ------------------------------------------------------------------------------
# Astral Usage
# ------------------------------------------------------------------------------

class AstralParams(BaseModel):
    """
    Astral Parameters.
    """
    astral_client: AstralClientParams = Field(default_factory=AstralClientParams, description="Astral client parameters.")
    usage: AstralUsageParams = Field(default_factory=AstralUsageParams, description="Usage parameters.")


# ------------------------------------------------------------------------------
# Astral Params Dict
# ------------------------------------------------------------------------------

AstralParamsDict = TypedDict("AstralParamsDict", {"astral_client": AstralClientParams, "usage": AstralUsageParams})
