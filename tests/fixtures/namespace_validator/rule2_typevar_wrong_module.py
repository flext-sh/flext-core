"""Rule 2 violation: TypeVar defined outside typings.py."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class FlextTestModels:
    """Models namespace."""


m = FlextTestModels
__all__ = []
