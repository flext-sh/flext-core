"""FLEXT Core types and type aliases.

Backward compatibility module that re-exports types from typings.py.
This module exists to maintain compatibility with existing tests.
"""

# Re-export everything from typings for backward compatibility
# Legacy aliases for backward compatibility - use Protocol instead
from flext_core.protocols import FlextValidator as TPredicate  # Legacy alias
from flext_core.typings import (
    E,
    F,
    FlextEntityId,
    FlextTypes,
    R,
    T,
    TAnyDict,
    TAnyObject,
    TComparable,
    TEntityId,
    TErrorMessage,
    TSerializable,
    TUserData,
    TValidatable,
    U,
    V,
)

__all__ = [
    "E",
    "F",
    "FlextEntityId",
    "FlextTypes",
    "R",
    "T",
    "TAnyDict",
    "TAnyObject",
    "TComparable",
    "TEntityId",
    "TErrorMessage",
    "TPredicate",  # Legacy alias
    "TSerializable",
    "TUserData",
    "TValidatable",
    "U",
    "V",
]
