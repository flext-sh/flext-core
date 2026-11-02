# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core import p


class FlextProtocolsErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ResultFailureSpec(Protocol):
        """Structural data contract projected from the owning model."""

        error: str | None
        error_code: str | None
        error_data: t.JsonMapping | p.ConfigModelInput | None
        exception: BaseException | None

    @runtime_checkable
    class LoggedCallSpec(Protocol):
        """Structural data contract projected from the owning model."""

        func_name: str
        func_module: str
        op_name: str
        correlation_id: str | None
        track_perf: bool
        start_time: float

    @runtime_checkable
    class ExceptionMetricsSnapshot(Protocol):
        """Structural data contract projected from the owning model."""

        total_exceptions: t.NonNegativeInt
        exception_counts: t.IntMapping
        exception_counts_summary: str
        unique_exception_types: t.NonNegativeInt

    @runtime_checkable
    class ExceptionMetricsState(Protocol):
        """Structural data contract projected from the owning model."""

        exception_counts: t.IntMapping


__all__: list[str] = [
    "FlextProtocolsErrors",
]
