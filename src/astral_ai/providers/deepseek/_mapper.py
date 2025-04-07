# -------------------------------------------------------------------------------- #
    # DeepSeek Mappers
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import List, Optional, Union, Dict, Any
import logging

# module imports
from astral_ai._types import NOT_GIVEN

from astral_ai.constants._models import DeepSeekModels

# Model Capabilities
from astral_ai.constants._model_capabilities import get_model_max_tokens

# Types
from astral_ai._types._request._request_params import ReasoningEffort

# Get logger
from astral_ai.logger import logger

# ------------------------------------------------------------------------------
# DeepSeek Message Mapping
# ------------------------------------------------------------------------------


from astral_ai.constants._model_capabilities import ModelCapabilities

def to_deepseek_max_tokens(model: DeepSeekModels, max_tokens: Optional[int] = None) -> int:
    """
    Convert the max tokens to the DeepSeek max tokens.
    """
    logger.debug(f"DeepSeek does not need a specific max tokens value but we will use the model's max tokens if not provided.")

    if max_tokens is NOT_GIVEN or max_tokens is None:
        return NOT_GIVEN
    else:
        model_spec_max_tokens = get_model_max_tokens(model, with_reasoning_effort=False)
        if model_spec_max_tokens is None:
            logger.warning(f"Something went wrong when attempting to get the max tokens for model {model}.",
                           "Setting to default.")
            return NOT_GIVEN
        else:
            if max_tokens > model_spec_max_tokens:
                logger.warning(f"The max tokens you provided ({max_tokens}) is greater than the model's max tokens ({model_spec_max_tokens}).",
                               "Setting to the model's max tokens.")
                return model_spec_max_tokens
            else:
                return max_tokens
