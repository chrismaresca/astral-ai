# -------------------------------------------------------------------------------- #
# Completions Resource Module
# -------------------------------------------------------------------------------- #

"""
Exposes the Completions resource and convenience functions.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from __future__ import annotations

# module imports
from .completions import (
    Completions,
    complete,
    complete_structured,
    complete_async,
    complete_structured_async,
)

# -------------------------------------------------------------------------------- #
# Exports
# -------------------------------------------------------------------------------- #
__all__ = [
    "Completions",
    "complete",
    "complete_structured",
    "complete_async",
    "complete_structured_async",
]
