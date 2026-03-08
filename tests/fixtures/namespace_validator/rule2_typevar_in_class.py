"""Rule 2 violation: TypeVar defined inside class body."""

from __future__ import annotations

from typing import TypeVar


class FlextTestTypes:
    """Types with TypeVar inside class — WRONG."""

    T = TypeVar("T")  # VIOLATION — TypeVar must be at module level
    E = TypeVar("E")  # VIOLATION


t = FlextTestTypes
__all__ = ["FlextTestTypes", "t"]
