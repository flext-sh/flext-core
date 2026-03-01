"""Rule 2 violation: TypeVar defined outside typings.py."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")  # VIOLATION — TypeVars must be in typings.py


class FlextTestModels:
    """Models namespace."""

    pass


m = FlextTestModels
__all__ = ["FlextTestModels", "m"]
