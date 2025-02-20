# ------------------------------------------------------------
# Base Model
# ------------------------------------------------------------

"""
Base Model for Astral AI
"""


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# Built-in
from typing import Union, Literal, override, TypeVar, TypeAlias, Final


# ------------------------------------------------------------
# Not Given (Originally Copied from OpenAI in openai/_types.py)
# ------------------------------------------------------------


class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response: ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"

# ------------------------------------------------------------------------------
# Not Given or TypeVar
# ------------------------------------------------------------------------------


# TypeVar for NotGivenOr
_T = TypeVar("_T")

# Not Given or TypeVar
NotGivenOr = Union[_T, NotGiven]

# Not Given Final
NOT_GIVEN: Final[NotGiven] = NotGiven()
