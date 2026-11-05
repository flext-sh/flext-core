# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import flext_core
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsResult


class FlextProtocolsErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ResultFailureSpec(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        error: str | None
        error_code: str | None

        @property
        def error_data(
            self,
        ) -> flext_core.t.JsonMapping | FlextProtocolsResult.ConfigModelInput | None:
            """Structured error payload attached to the failure."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Underlying exception captured by the failure."""
            ...

    @runtime_checkable
    class LoggedCallSpec(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        func_name: str
        func_module: str
        op_name: str
        correlation_id: str | None
        track_perf: bool
        start_time: float

    @runtime_checkable
    class ExceptionMetricsSnapshot(FlextProtocolsBase.StrictModel, Protocol):
        """Structural data contract projected from the owning model."""

        total_exceptions: flext_core.t.NonNegativeInt
        exception_counts: flext_core.t.IntMapping
        exception_counts_summary: str
        unique_exception_types: flext_core.t.NonNegativeInt

        @property
        def has_exceptions(self) -> bool:
            """Whether the metrics snapshot contains recorded exceptions."""
            ...

        def to_config_map(self) -> flext_core.t.JsonMapping:
            """Expose the snapshot through the canonical flat config contract."""
            ...

    @runtime_checkable
    class ExceptionMetricsState(FlextProtocolsBase.StrictModel, Protocol):
        """Structural data contract projected from the owning model."""

        exception_counts: flext_core.t.IntMapping

        @property
        def total_exceptions(self) -> int:
            """Total recorded exception occurrences."""
            ...

        @property
        def unique_exception_types(self) -> int:
            """Number of unique exception types recorded."""
            ...

        @property
        def exception_counts_summary(self) -> str:
            """Human-readable summary for logs and diagnostics."""
            ...

        def record_exception(self, exception_type: type[BaseException]) -> Self:
            """Return a new state with one additional recorded exception."""
            ...

        def clear(self) -> Self:
            """Return a cleared metrics state."""
            ...

        def snapshot(self) -> FlextProtocolsErrors.ExceptionMetricsSnapshot:
            """Build the validated public metrics snapshot."""
            ...


__all__: list[str] = [
    "FlextProtocolsErrors",
]
