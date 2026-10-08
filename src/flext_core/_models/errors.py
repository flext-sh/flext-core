"""FlextModelsErrors namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar, Self

from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up


class FlextModelsErrors:
    """Canonical Pydantic models for structured errors and error metrics."""

    class ResultFailureSpec(m.ArbitraryTypesModel):
        """Failure-side payload spec for Result construction.

        Bundles the failure constructor knobs so ``r[T].fail`` builds a result
        through a single typed payload while the success path stays
        allocation-free.
        """

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="forbid",
            strict=True,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        error: Annotated[
            str | None,
            mp.Field(default=None, description="Human-readable failure message."),
        ] = None
        error_code: Annotated[
            str | None,
            mp.Field(default=None, description="Canonical failure error code."),
        ] = None
        error_data: Annotated[
            t.JsonMapping | t.ConfigModelInput | None,
            mp.Field(
                default=None,
                description="Structured error payload attached to the failure.",
            ),
        ] = None
        exception: Annotated[
            BaseException | None,
            mp.Field(
                default=None,
                description="Underlying exception captured by the failure.",
            ),
        ] = None

    class LoggedCallSpec(m.ArbitraryTypesModel):
        """Static call context for decorator-driven logging execution."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="forbid",
            strict=True,
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        func_name: Annotated[
            str,
            mp.Field(description="Wrapped function name."),
        ] = ""
        func_module: Annotated[
            str,
            mp.Field(description="Wrapped function module qualified name."),
        ] = ""
        op_name: Annotated[
            str,
            mp.Field(description="Logical operation name used in log events."),
        ] = ""
        correlation_id: Annotated[
            str | None,
            mp.Field(default=None, description="Active correlation identifier."),
        ] = None
        track_perf: Annotated[
            bool,
            mp.Field(default=False, description="Track and log the call duration."),
        ] = False
        start_time: Annotated[
            float,
            mp.Field(default=0.0, description="Monotonic start time of the call."),
        ] = 0.0

    class ExceptionMetricsSnapshot(m.StrictModel):
        """Validated public snapshot for exception metric exports."""

        total_exceptions: Annotated[
            t.NonNegativeInt,
            mp.Field(description="Total recorded exception occurrences."),
        ] = 0
        exception_counts: Annotated[
            t.IntMapping,
            mp.Field(description="Per-exception occurrence totals keyed by type name."),
        ] = mp.Field(default_factory=lambda: MappingProxyType[str, int]({}))
        exception_counts_summary: Annotated[
            str,
            mp.Field(description="Human-readable summary for logs and diagnostics."),
        ] = ""
        unique_exception_types: Annotated[
            t.NonNegativeInt,
            mp.Field(description="Number of unique exception types recorded."),
        ] = 0

        @up.computed_field
        @property
        def has_exceptions(self) -> bool:
            """Whether the metrics snapshot contains recorded exceptions."""
            return self.total_exceptions > 0

        def to_config_map(self) -> t.JsonMapping:
            """Expose the snapshot through the canonical flat config contract.

            Returns:
                The resulting ``t.JsonMapping``.

            """
            payload: t.JsonDict = {
                "total_exceptions": self.total_exceptions,
                "exception_counts_summary": self.exception_counts_summary,
                "unique_exception_types": self.unique_exception_types,
            }
            for key, value in self.exception_counts.items():
                payload[f"exception_counts.{key}"] = value
            return payload

    class ExceptionMetricsState(m.StrictModel):
        """Copy-updated runtime state for exception counters."""

        exception_counts: Annotated[
            t.IntMapping,
            mp.Field(description="Recorded counts keyed by exception type name."),
        ] = mp.Field(default_factory=lambda: MappingProxyType[str, int]({}))

        @up.computed_field
        @property
        def total_exceptions(self) -> int:
            """Total recorded exception occurrences."""
            return sum(self.exception_counts.values(), 0)

        @up.computed_field
        @property
        def unique_exception_types(self) -> int:
            """Number of unique exception types recorded."""
            return len(self.exception_counts)

        @up.computed_field
        @property
        def exception_counts_summary(self) -> str:
            """Human-readable summary for logs and diagnostics."""
            return ";".join(
                f"{exception_name}:{count}"
                for exception_name, count in self.exception_counts.items()
            )

        def record_exception(self, exception_type: type[BaseException]) -> Self:
            """Return a new state with one additional recorded exception.

            Returns:
                A new state with one additional recorded exception.

            """
            name = exception_type.__qualname__
            counts = dict(self.exception_counts)
            counts[name] = counts.get(name, 0) + 1
            updated: Self = self.model_copy(update={"exception_counts": counts})
            return updated

        def clear(self) -> Self:
            """Return a cleared metrics state.

            Returns:
                A cleared metrics state.

            """
            cleared: Self = self.model_copy(
                update={"exception_counts": MappingProxyType({})},
            )
            return cleared

        def snapshot(self) -> FlextModelsErrors.ExceptionMetricsSnapshot:
            """Build the validated public metrics snapshot.

            Returns:
                The resulting ``FlextModelsErrors.ExceptionMetricsSnapshot``.

            """
            snapshot: FlextModelsErrors.ExceptionMetricsSnapshot = (
                FlextModelsErrors.ExceptionMetricsSnapshot.model_validate({
                    "total_exceptions": self.total_exceptions,
                    "exception_counts": dict(self.exception_counts),
                    "exception_counts_summary": self.exception_counts_summary,
                    "unique_exception_types": self.unique_exception_types,
                })
            )
            return snapshot
