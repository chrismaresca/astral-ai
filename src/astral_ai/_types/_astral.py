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
from typing import Optional

# Pydantic
from pydantic import BaseModel, Field, ConfigDict

# Astral AI
from astral_ai._auth import AUTH_CONFIG_TYPE
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy

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


# class AstralUsageParams(BaseModel):
#     """
#     Astral Usage Parameters.
#     """
#     usage: bool = Field(default=True, description="If True, include usage information in the response.")
#     cost: bool = Field(default=True, description="If True, include cost information in the response.")
#     latency: bool = Field(default=True, description="If True, include latency information in the response.")


# ------------------------------------------------------------------------------
# Astral Usage
# ------------------------------------------------------------------------------

class AstralParams(BaseModel):
    """
    Astral Parameters.
    """
    
    astral_client: AstralClientParams = Field(default_factory=AstralClientParams, description="Astral client parameters.")
    # usage: AstralUsageParams = Field(default_factory=AstralUsageParams, description="Usage parameters.")
    cost_strategy: BaseCostStrategy = Field(default_factory=ReturnCostStrategy, description="Cost strategy.")


    # --------------------------------------------------------------------------
    # Config
    # --------------------------------------------------------------------------

    model_config = ConfigDict(arbitrary_types_allowed=True)
