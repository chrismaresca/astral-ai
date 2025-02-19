# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

"""
Config for Astral AI
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in
from typing import Tuple

# Pydantic
from pydantic import Field
from pydantic_settings import BaseSettings


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

class AstralConfig(BaseSettings):
    """
    Config for Astral AI
    """
    SUPPORTED_MODEL_TYPES: set[str] = Field(
        default={"llm", "embedding"},
        description="Supported model types",
    )

# Initialize Config
config = AstralConfig()

print(config.SUPPORTED_MODEL_TYPES)
