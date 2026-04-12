"""FlextModelsErrors namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, computed_field

from flext_core import FlextModelsBase, t


class FlextModelsErrors:
    """Canonical Pydantic models for structured errors and error metrics."""

    class StructuredErrorSnapshot(FlextModelsBase.StrictModel):
        """Validated public snapshot for structured error serialization."""

        error_type: Annotated[str, Field(description="Concrete exception type name.")]
        message: Annotated[str, Field(description="Human-readable error message.")]
        error_code: Annotated[
            str, Field(description="Canonical structured error code.")
        ]
        error_domain: Annotated[
            str | None,
            Field(description="Canonical routing domain for the error."),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(description="Correlation identifier propagated with the error."),
        ] = None
        timestamp: Annotated[
            t.Numeric,
            Field(description="Unix timestamp when the error instance was created."),
        ]
        attributes: Annotated[
            t.ConfigMap,
            Field(description="Flattenable metadata attributes exposed publicly."),
        ] = Field(default_factory=lambda: t.ConfigMap(root={}))

        @computed_field
        @property
        def error_message(self) -> str:
            """Public alias expected by structured error consumers."""
            return self.message

        def to_payload(self) -> t.ConfigMap:
            """Flatten the snapshot into the public payload shape."""
            payload = t.ConfigMap.model_validate(
                self.model_dump(exclude={"attributes"})
            )
            for key, value in self.attributes.items():
                if key not in payload:
                    payload[key] = value
            return payload

    class ExceptionMetricsSnapshot(FlextModelsBase.StrictModel):
        """Validated public snapshot for exception metric exports."""

        total_exceptions: Annotated[
            t.NonNegativeInt,
            Field(description="Total recorded exception occurrences."),
        ] = 0
        exception_counts: Annotated[
            t.ConfigMap,
            Field(description="Per-exception occurrence totals keyed by type name."),
        ] = Field(default_factory=lambda: t.ConfigMap(root={}))
        exception_counts_summary: Annotated[
            str,
            Field(description="Human-readable summary for logs and diagnostics."),
        ] = ""
        unique_exception_types: Annotated[
            t.NonNegativeInt,
            Field(description="Number of unique exception types recorded."),
        ] = 0

        @computed_field
        @property
        def has_exceptions(self) -> bool:
            """Whether the metrics snapshot contains recorded exceptions."""
            return self.total_exceptions > 0

        def to_config_map(self) -> t.ConfigMap:
            """Expose the snapshot through the canonical config container."""
            return t.ConfigMap(
                root={
                    "total_exceptions": self.total_exceptions,
                    "exception_counts": self.exception_counts,
                    "exception_counts_summary": self.exception_counts_summary,
                    "unique_exception_types": self.unique_exception_types,
                }
            )

    class ExceptionMetricsState(FlextModelsBase.StrictModel):
        """Mutable-through-copy runtime state for exception counters."""

        exception_counts: Annotated[
            t.IntMapping,
            Field(description="Recorded counts keyed by exception type name."),
        ] = Field(default_factory=dict)

        @computed_field
        @property
        def total_exceptions(self) -> int:
            """Total recorded exception occurrences."""
            return sum(self.exception_counts.values(), 0)

        @computed_field
        @property
        def unique_exception_types(self) -> int:
            """Number of unique exception types recorded."""
            return len(self.exception_counts)

        @computed_field
        @property
        def exception_counts_summary(self) -> str:
            """Human-readable summary for logs and diagnostics."""
            return ";".join(
                f"{exception_name}:{count}"
                for exception_name, count in self.exception_counts.items()
            )

        @staticmethod
        def resolve_exception_name(exception_type: type[BaseException]) -> str:
            """Resolve the canonical registry key for an exception type."""
            return exception_type.__qualname__

        def record_exception(self, exception_type: type[BaseException]) -> Self:
            """Return a new state with one additional recorded exception."""
            exception_name = self.resolve_exception_name(exception_type)
            counts = dict(self.exception_counts)
            counts[exception_name] = counts.get(exception_name, 0) + 1
            return self.model_copy(update={"exception_counts": counts})

        def clear(self) -> Self:
            """Return a cleared metrics state."""
            return self.model_copy(update={"exception_counts": {}})

        def snapshot(self) -> FlextModelsErrors.ExceptionMetricsSnapshot:
            """Build the validated public metrics snapshot."""
            return FlextModelsErrors.ExceptionMetricsSnapshot.model_validate({
                "total_exceptions": self.total_exceptions,
                "exception_counts": t.ConfigMap.model_validate(
                    dict(self.exception_counts),
                ),
                "exception_counts_summary": self.exception_counts_summary,
                "unique_exception_types": self.unique_exception_types,
            })
