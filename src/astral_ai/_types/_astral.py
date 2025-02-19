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
