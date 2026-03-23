"""Rule 2 violation: Protocol defined inside Types class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class FlextTestTypes:
    """Types with Protocol inside — WRONG."""

    class ProtocolInsideTypes(Protocol):
        """Protocol inside Types — VIOLATION."""

        def to_dict(self) -> Mapping[str, str]:
            """Convert to dict."""
            ...


t = FlextTestTypes
__all__ = ["FlextTestTypes", "t"]
