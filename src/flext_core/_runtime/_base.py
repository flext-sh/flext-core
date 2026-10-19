"""Base runtime helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from flext_core import c, p, t
from flext_core._utilities import FlextUtilitiesBase


class FlextRuntimeBase(FlextUtilitiesBase):
    """Foundational runtime helpers shared by higher runtime namespaces."""

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None.

        Returns:
            The resulting ``datetime | None``.

        """
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def resolve_effective_log_level(
        *,
        trace: bool,
        debug: bool,
        log_level: t.LogLevel,
        log_level_explicit: bool = True,
    ) -> t.LogLevel:
        """Resolve severity once; trace controls detail, never the threshold.

        Returns:
            The resulting ``t.LogLevel``.

        """
        _ = trace
        if debug and not log_level_explicit:
            return c.LogLevel.DEBUG
        return log_level

    @staticmethod
    def normalize_alnum(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase the result.

        Returns:
            The resulting ``str``.

        """
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def to_scalar(
        item: p.GuardInput | None,
    ) -> t.Scalar:
        """Coerce any runtime value to ``t.Scalar``.

        Returns:
            The resulting ``tb.Scalar``.

        """
        if item is None:
            return ""
        return item if isinstance(item, c.SCALAR_TYPES) else str(item)


__all__: list[str] = ["FlextRuntimeBase"]
