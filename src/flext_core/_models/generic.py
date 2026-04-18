"""Generic reusable models for the FLEXT ecosystem.

TIER 1: Uses Tier 0 modules (constants, typings) and base models.

This module provides generic models for reusable operation progress tracking.

All downstream projects (flext-cli, flext-ldif, flext-ldap, flext-oud-mig)
should use these models instead of creating their own dict patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from datetime import datetime
from typing import Annotated, ClassVar

from flext_core import (
    FlextModelsBase as m,
    FlextUtilitiesDomain,
    FlextUtilitiesGenerators as u,
    FlextUtilitiesPydantic as up,
    t,
)


class FlextGenericModels:
    """Generic models for mutable progress accumulation during operations."""

    class Operation(m.ArbitraryTypesModel):
        """Progress tracking for ongoing operations.

        Used by: batch operations, migrations, sync, data processing.
        """

        success_count: Annotated[
            t.NonNegativeInt,
            up.Field(default=0, description="Successes"),
        ] = 0
        failure_count: Annotated[
            t.NonNegativeInt,
            up.Field(default=0, description="Failures"),
        ] = 0
        skipped_count: Annotated[
            t.NonNegativeInt,
            up.Field(default=0, description="Skipped"),
        ] = 0
        warning_count: Annotated[
            t.NonNegativeInt,
            up.Field(default=0, description="Warnings"),
        ] = 0
        retry_count: Annotated[
            t.NonNegativeInt,
            up.Field(default=0, description="Retries"),
        ] = 0
        start_time: Annotated[
            datetime | None,
            up.Field(default=None, description="Start time"),
        ] = None
        last_update: Annotated[
            datetime | None,
            up.Field(default=None, description="Last update"),
        ] = None
        estimated_total: Annotated[
            t.NonNegativeInt | None,
            up.Field(default=None, description="Estimated total"),
        ] = None
        current_item: Annotated[
            str | None,
            up.Field(default=None, description="Current item"),
        ] = None
        operation_name: Annotated[
            str | None,
            up.Field(default=None, description="Operation name"),
        ] = None
        metadata: Annotated[
            t.Dict,
            up.Field(
                description="Operation metadata",
            ),
        ] = up.Field(default_factory=t.Dict)

        def record_failure(self) -> None:
            """Record a failed operation."""
            self.failure_count += 1
            self.last_update = u.generate_datetime_utc()

        def record_retry(self) -> None:
            """Record a retry attempt."""
            self.retry_count += 1
            self.last_update = u.generate_datetime_utc()

        def record_skip(self) -> None:
            """Record a skipped operation."""
            self.skipped_count += 1
            self.last_update = u.generate_datetime_utc()

        def record_success(self) -> None:
            """Record a successful operation."""
            self.success_count += 1
            self.last_update = u.generate_datetime_utc()

        def record_warning(self) -> None:
            """Record an operation with warnings."""
            self.warning_count += 1
            self.last_update = u.generate_datetime_utc()

        def start_operation(
            self,
            name: str | None = None,
            estimated_total: int | None = None,
        ) -> None:
            """Start the operation tracking."""
            self.operation_name = name
            self.estimated_total = estimated_total
            self.start_time = u.generate_datetime_utc()
            self.last_update = self.start_time

    class Conversion(m.ArbitraryTypesModel):
        """Conversion progress tracking with error reporting.

        Used by: flext-ldif conversion, data transformations, ETL.

        Enforcement exemption: progress tracker with intentionally mutable
        ``converted``/``errors``/``warnings``/``skipped`` collectors — items
        are appended as conversion proceeds.
        """

        _flext_enforcement_exempt: ClassVar[bool] = True

        converted: Annotated[
            MutableSequence[t.ValueOrModel],
            up.Field(description="Converted items"),
        ] = up.Field(default_factory=list[t.ValueOrModel])
        errors: Annotated[
            MutableSequence[str],
            up.Field(description="Error messages"),
        ] = up.Field(default_factory=list)
        warnings: Annotated[
            MutableSequence[str],
            up.Field(description="Warning messages"),
        ] = up.Field(default_factory=list)
        skipped: Annotated[
            MutableSequence[t.ValueOrModel],
            up.Field(description="Skipped items"),
        ] = up.Field(default_factory=list[t.ValueOrModel])
        start_time: Annotated[
            datetime | None,
            up.Field(default=None, description="Start time"),
        ] = None
        end_time: Annotated[
            datetime | None,
            up.Field(default=None, description="End time"),
        ] = None
        source_format: Annotated[
            str | None,
            up.Field(default=None, description="Source format"),
        ] = None
        target_format: Annotated[
            str | None,
            up.Field(default=None, description="Target format"),
        ] = None
        total_input_count: Annotated[
            t.NonNegativeInt | None,
            up.Field(default=None, description="Total input count"),
        ] = None
        metadata: Annotated[
            t.Dict,
            up.Field(
                description="Conversion metadata",
            ),
        ] = up.Field(default_factory=t.Dict)

        def add_converted(self, item: t.ValueOrModel) -> None:
            """Add a successfully converted item."""
            self.converted.append(item)

        def add_error(self, error: str, item: t.ValueOrModel | None = None) -> None:
            """Add an error with optional failed item."""
            self.errors.append(error)
            if item is not None:
                FlextUtilitiesDomain.append_metadata_sequence_item(
                    self.metadata,
                    "failed_items",
                    item,
                )

        def add_skipped(self, item: t.ValueOrModel, reason: str | None = None) -> None:
            """Add a skipped item with optional reason."""
            self.skipped.append(item)
            if reason:
                FlextUtilitiesDomain.upsert_skip_reason(self.metadata, item, reason)

        def add_warning(self, warning: str, item: t.ValueOrModel | None = None) -> None:
            """Add a warning with optional item."""
            self.warnings.append(warning)
            if item is not None:
                FlextUtilitiesDomain.append_metadata_sequence_item(
                    self.metadata,
                    "warning_items",
                    item,
                )

        def complete_conversion(self) -> None:
            """Mark conversion as completed."""
            self.end_time = u.generate_datetime_utc()

        def start_conversion(
            self,
            source_format: str | None = None,
            target_format: str | None = None,
            total_input_count: int | None = None,
        ) -> None:
            """Start conversion tracking."""
            self.source_format = source_format
            self.target_format = target_format
            self.total_input_count = total_input_count
            self.start_time = u.generate_datetime_utc()


__all__: list[str] = ["FlextGenericModels"]
