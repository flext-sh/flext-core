# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsResult
    from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextProtocolsErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ResultFailureSpec(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def error(self) -> str | None:
            """Human-readable failure message."""
            ...

        @property
        def error_code(self) -> str | None:
            """Canonical failure error code."""
            ...

        @property
        def error_data(
            self,
        ) -> FlextTypingBase.JsonMapping | FlextProtocolsResult.ConfigModelInput | None:
            """Structured error payload attached to the failure."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Underlying exception captured by the failure."""
            ...

    @runtime_checkable
    class LoggedCallSpec(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def func_name(self) -> str:
            """Wrapped function name."""
            ...

        @property
        def func_module(self) -> str:
            """Wrapped function module qualified name."""
            ...

        @property
        def op_name(self) -> str:
            """Logical operation name used in log events."""
            ...

        @property
        def correlation_id(self) -> str | None:
            """Active correlation identifier."""
            ...

        @property
        def track_perf(self) -> bool:
            """Track and log the call duration."""
            ...

        @property
        def start_time(self) -> float:
            """Monotonic start time of the call."""
            ...

    @runtime_checkable
    class ExceptionMetricsSnapshot(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def total_exceptions(self) -> FlextTypesPydantic.NonNegativeInt:
            """Total recorded exception occurrences."""
            ...

        @property
        def exception_counts(self) -> FlextTypingBase.IntMapping:
            """Per-exception occurrence totals keyed by type name."""
            ...

        @property
        def exception_counts_summary(self) -> str:
            """Human-readable summary for logs and diagnostics."""
            ...

        @property
        def unique_exception_types(self) -> FlextTypesPydantic.NonNegativeInt:
            """Number of unique exception types recorded."""
            ...

        def to_config_map(self) -> FlextTypingBase.JsonMapping:
            """Expose the snapshot through the canonical flat config contract."""

    @runtime_checkable
    class ExceptionMetricsState(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def exception_counts(self) -> FlextTypingBase.IntMapping:
            """Recorded counts keyed by exception type name."""
            ...

        def record_exception(self, exception_type: type[BaseException]) -> Self:
            """Return a new state with one additional recorded exception."""

        def clear(self) -> Self:
            """Return a cleared metrics state."""

        def snapshot(self) -> FlextProtocolsErrors.ExceptionMetricsSnapshot:
            """Build the validated public metrics snapshot."""


__all__: list[str] = [
    "FlextProtocolsErrors",
]
