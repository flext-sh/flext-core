"""Rule 2 violation: Protocol defined inside Types class."""

from __future__ import annotations

from typing import Protocol

from flext_core import t as _t


class FlextTestTypes(_t):
    """Types with Protocol inside — WRONG."""

    class ProtocolInsideTypes(Protocol):
        """Protocol inside Types — VIOLATION."""

        def to_dict(self) -> _t.StrMapping:
            """Convert to dict."""
            ...


t = FlextTestTypes
__all__ = ["FlextTestTypes", "t"]
