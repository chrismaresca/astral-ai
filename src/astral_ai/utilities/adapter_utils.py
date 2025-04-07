# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Built-in imports
from typing import Any, Dict

# Astral AI Types
from astral_ai._types import NOT_GIVEN

# ------------------------------------------------------------------------------
# Apply Key Mapping
# ------------------------------------------------------------------------------


def apply_key_mapping(
    raw: Dict[str, Any],
    key_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Rename & filter a raw Astral→provider payload dict.

    - Drops any field whose value is NOT_GIVEN
    - Drops any Astral field not present in key_map. This means that the
      provider will not accept the field, so therefore passing it will result
      in potential runtime errors.
    - Renames keys via key_map
    - Preserves explicit None values

    Returns a provider‑ready dict.
    """
    return {
        key_map[astral_key]: value
        for astral_key, value in raw.items()
        if value is not NOT_GIVEN and astral_key in key_map
    }

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
