from __future__ import annotations
"""
Astral AI Specific Types and Models.
"""

from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from astral_ai._auth import AUTH_CONFIG_TYPE
from astral_ai.tracing._cost_strategies import BaseCostStrategy, ReturnCostStrategy

if TYPE_CHECKING:
    from astral_ai.providers._base_client import BaseProviderClient

# ------------------------------------------------------------------------------
# Astral Client Parameters
# ------------------------------------------------------------------------------

class AstralClientParams(BaseModel):
    """
    Astral AI Client Parameters.
    """
    new_client: bool = Field(
        default=False,
        description="If True, force creation of a new client. A unique client_key or new config must be provided."
    )
    client_config: Optional[AUTH_CONFIG_TYPE] = Field(
        default=None,
        description="Authentication config used to instantiate a new client."
    )
    client_key: Optional[str] = Field(
        default=None,
        description="Optional unique key for the client. If provided, it overrides key generation based on config."
    )

# ------------------------------------------------------------------------------
# Astral Usage / Parameters
# ------------------------------------------------------------------------------

class AstralParams(BaseModel):
    """
    Astral Parameters.
    """
    astral_client: AstralClientParams = Field(
        default_factory=AstralClientParams, description="Astral client parameters."
    )
    cost_strategy: BaseCostStrategy = Field(
        default_factory=ReturnCostStrategy, description="Cost strategy."
    )
    # Optionally, the user can supply a specific provider client instance.
    provider_client: Optional[BaseProviderClient] = Field(
        default=None, description="Optionally override the default provider client with a specific instance."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
