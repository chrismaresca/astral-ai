# -------------------------------------------------------------------------------- #
# Pydantic Abstract Base Class Example
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from abc import ABC, abstractmethod
from typing import Any

# Pydantic imports
from pydantic import BaseModel


# -------------------------------------------------------------------------------- #
# Abstract Base Class Implementation
# -------------------------------------------------------------------------------- #
class AbstractBase(BaseModel):
    """Abstract base class for Pydantic models."""
    field1: str
    field2: int
    
    def __init__(self, **data: Any) -> None:
        """Override init to prevent direct instantiation of abstract class."""
        if self.__class__ == AbstractBase:
            raise TypeError("Cannot instantiate abstract class AbstractBase directly")
        super().__init__(**data)


class ConcreteSubclass(AbstractBase):
    """Concrete subclass of AbstractBase."""

    field1: str
    field2: int


try:
    instance = AbstractBase(field1="example", field2=123)
    print(instance)
except TypeError as e:
    print(f"Error: {e}")
# Expected output: Error: Cannot instantiate abstract class AbstractBase directly

instance = ConcreteSubclass(field1="example", field2=123)
print(instance)
# Expected output: field1='example' field2=123