"""FlextModelsErrors namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Self

from flext_core import (
    FlextModelsPydantic as mp,
    FlextTypes as t,
    FlextUtilitiesPydantic as up,
)
from flext_core._models.base import FlextModelsBase as m


class FlextModelsErrors:
    """Canonical Pydantic models for structured errors and error metrics."""

    class ExceptionMetricsSnapshot(m.StrictModel):
        """Validated public snapshot for exception metric exports."""

        total_exceptions: Annotated[
            t.NonNegativeInt,
            mp.Field(description="Total recorded exception occurrences."),
        ] = 0
        exception_counts: Annotated[
            t.MutableIntMapping,
            mp.Field(
                default_factory=dict,
                description="Per-exception occurrence totals keyed by type name.",
            ),
        ]
        exception_counts_summary: Annotated[
            str,
            mp.Field(description="Human-readable summary for logs and diagnostics."),
        ] = ""
        unique_exception_types: Annotated[
            t.NonNegativeInt,
            mp.Field(description="Number of unique exception types recorded."),
        ] = 0

        @up.computed_field()
        @property
        def has_exceptions(self) -> bool:
            """Whether the metrics snapshot contains recorded exceptions."""
            return self.total_exceptions > 0

        def to_config_map(self) -> t.JsonMapping:
            """Expose the snapshot through the canonical flat config contract."""
            payload: dict[str, t.JsonValue] = {
                "total_exceptions": self.total_exceptions,
                "exception_counts_summary": self.exception_counts_summary,
                "unique_exception_types": self.unique_exception_types,
            }
            for key, value in self.exception_counts.items():
                payload[f"exception_counts.{key}"] = value
            return payload

    class ExceptionMetricsState(m.StrictModel):
        """Mutable-through-copy runtime state for exception counters."""

        exception_counts: Annotated[
            t.MutableIntMapping,
            mp.Field(
                default_factory=dict,
                description="Recorded counts keyed by exception type name.",
            ),
        ]

        @up.computed_field()
        @property
        def total_exceptions(self) -> int:
            """Total recorded exception occurrences."""
            return sum(self.exception_counts.values(), 0)

        @up.computed_field()
        @property
        def unique_exception_types(self) -> int:
            """Number of unique exception types recorded."""
            return len(self.exception_counts)

        @up.computed_field()
        @property
        def exception_counts_summary(self) -> str:
            """Human-readable summary for logs and diagnostics."""
            return ";".join(
                f"{exception_name}:{count}"
                for exception_name, count in self.exception_counts.items()
            )

        def record_exception(self, exception_type: type[BaseException]) -> Self:
            """Return a new state with one additional recorded exception."""
            name = exception_type.__qualname__
            counts = dict(self.exception_counts)
            counts[name] = counts.get(name, 0) + 1
            return self.model_copy(update={"exception_counts": counts})

        def clear(self) -> Self:
            """Return a cleared metrics state."""
            return self.model_copy(update={"exception_counts": {}})

        def snapshot(self) -> FlextModelsErrors.ExceptionMetricsSnapshot:
            """Build the validated public metrics snapshot."""
            return FlextModelsErrors.ExceptionMetricsSnapshot.model_validate({
                "total_exceptions": self.total_exceptions,
                "exception_counts": dict(self.exception_counts),
                "exception_counts_summary": self.exception_counts_summary,
                "unique_exception_types": self.unique_exception_types,
            })
