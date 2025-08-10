"""FLEXT Singer base patterns.

This module provides Singer-specific base patterns and utilities for data integration.
Currently provides stubs for backward compatibility during refactoring.
"""

from __future__ import annotations

from flext_core.models import FlextModel
from flext_core.result import FlextResult


class FlextSingerBase(FlextModel):
    """Base class for Singer-related functionality."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize Singer base."""
        super().__init__(**kwargs)

    def extract(self) -> FlextResult[dict[str, object]]:
        """Extract data using Singer patterns."""
        return FlextResult.ok({})

    def load(self, data: dict[str, object]) -> FlextResult[bool]:  # noqa: ARG002
        """Load data using Singer patterns."""
        return FlextResult.ok(True)  # noqa: FBT003


class FlextSingerTap(FlextSingerBase):
    """Base class for Singer taps (extractors)."""


class FlextSingerTarget(FlextSingerBase):
    """Base class for Singer targets (loaders)."""


__all__: list[str] = [
    "FlextSingerBase",
    "FlextSingerTap",
    "FlextSingerTarget",
]
