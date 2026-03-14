"""Generic reusable models for the FLEXT ecosystem.

TIER 1: Uses Tier 0 modules (constants, typings) and base models.

This module provides generic models organized by business function:
- Value Objects: Immutable data compared by value (FrozenValueModel)
- Snapshots: State captured at a specific moment (FrozenStrictModel)
- Progress Trackers: Mutable accumulators during operations (ArbitraryTypesModel)

All downstream projects (flext-cli, flext-ldif, flext-ldap, flext-oud-mig)
should use these models instead of creating their own dict patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from flext_core import t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.containers import FlextModelsContainers


class FlextGenericModels:
    """Generic models organized by business function.

    Categories:
    - Value: Immutable data objects compared by value
    - Snapshot: State captured at a specific moment
    - Progress: Mutable accumulators during operations
    """

    class OperationContext(FlextModelFoundation.FrozenValueModel):
        """Immutable context of an operation.

        Used by: all projects
        Function: Metadata of operation that doesn't change during execution.
        """

        correlation_id: Annotated[
            str,
            Field(
                default_factory=lambda: str(uuid.uuid4()),
                description="Unique correlation ID for tracing",
            ),
        ]
        operation_id: Annotated[
            str,
            Field(
                default_factory=lambda: str(uuid.uuid4()),
                description="Unique operation ID",
            ),
        ]
        timestamp: Annotated[
            datetime,
            Field(
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when created",
            ),
        ]
        source: Annotated[
            str | None, Field(default=None, description="Source system")
        ] = None
        user_id: Annotated[str | None, Field(default=None, description="User ID")] = (
            None
        )
        tenant_id: Annotated[
            str | None, Field(default=None, description="Tenant ID")
        ] = None
        environment: Annotated[
            str | None, Field(default=None, description="Environment")
        ] = None
        version: Annotated[
            str, Field(default="1.0.0", description="Schema version")
        ] = "1.0.0"
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Additional metadata",
            ),
        ]
        message: Annotated[
            t.NormalizedValue | BaseModel,
            Field(default=None, description="Message payload"),
        ] = None
        message_type: Annotated[str, Field(default="", description="Message type")] = ""
        dispatch_type: Annotated[
            str, Field(default="", description="Dispatch type")
        ] = ""
        timeout_override: Annotated[
            int | None, Field(default=None, description="Timeout override seconds")
        ] = None

    class Service(FlextModelFoundation.FrozenStrictModel):
        """Snapshot of service state.

        Used by: FlextService.get_info(), monitoring, health checks.
        """

        name: Annotated[str, Field(description="Service name")]
        version: Annotated[
            str | None, Field(default=None, description="Service version")
        ] = None
        status: Annotated[
            str, Field(default="active", description="Service status")
        ] = "active"
        uptime_seconds: Annotated[
            float | None, Field(default=None, description="Uptime in seconds")
        ] = None
        start_time: Annotated[
            datetime | None, Field(default=None, description="Start time")
        ] = None
        last_health_check: Annotated[
            datetime | None, Field(default=None, description="Last health check")
        ] = None
        health_status: Annotated[
            str, Field(default="unknown", description="Health status")
        ] = "unknown"
        port: Annotated[int | None, Field(default=None, description="Port")] = None
        host: Annotated[str | None, Field(default=None, description="Host")] = None
        pid: Annotated[int | None, Field(default=None, description="Process ID")] = None
        memory_usage_mb: Annotated[
            float | None, Field(default=None, description="Memory MB")
        ] = None
        cpu_usage_percent: Annotated[
            float | None, Field(default=None, description="CPU %")
        ] = None
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Service metadata",
            ),
        ]

    class Configuration(FlextModelFoundation.FrozenStrictModel):
        """Configuration snapshot.

        Used by: CLI config info, debug, auditing.
        """

        config: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Config key-value pairs",
            ),
        ]
        captured_at: Annotated[
            datetime,
            Field(
                default_factory=lambda: datetime.now(UTC),
                description="Capture timestamp",
            ),
        ]
        source: Annotated[
            str | None, Field(default=None, description="Config source")
        ] = None
        environment: Annotated[
            str | None, Field(default=None, description="Target environment")
        ] = None
        version: Annotated[
            str, Field(default="1.0.0", description="Schema version")
        ] = "1.0.0"
        checksum: Annotated[str | None, Field(default=None, description="Checksum")] = (
            None
        )
        validation_errors: Annotated[
            list[str],
            Field(default_factory=list, description="Validation errors"),
        ]
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Config metadata",
            ),
        ]

    class Health(FlextModelFoundation.FrozenStrictModel):
        """Health check result.

        Used by: /health endpoints, monitoring, alerting.
        """

        healthy: Annotated[bool, Field(default=True, description="Overall health")] = (
            True
        )
        checks: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Check results",
            ),
        ]
        details: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Check details",
            ),
        ]
        checked_at: Annotated[
            datetime,
            Field(
                default_factory=lambda: datetime.now(UTC),
                description="Check timestamp",
            ),
        ]
        service_name: Annotated[
            str | None, Field(default=None, description="Service name")
        ] = None
        service_version: Annotated[
            str | None, Field(default=None, description="Service version")
        ] = None
        duration_ms: Annotated[
            float | None, Field(default=None, description="Check duration ms")
        ] = None
        environment: Annotated[
            str | None, Field(default=None, description="Environment")
        ] = None
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Health metadata",
            ),
        ]

    """Progress trackers - mutable, accumulate during operation."""

    @staticmethod
    def safe_percentage(processed: int, total: int | None) -> float:
        """Percentage with zero-safe fallback, capped at 100."""
        if not total or total == 0:
            return 0.0
        return min(processed / total * 100.0, 100.0)

    @staticmethod
    def safe_rate(numerator: int, denominator: int) -> float:
        """Division with zero-safe fallback."""
        if denominator == 0:
            return 0.0
        return numerator / denominator

    class Operation(FlextModelFoundation.ArbitraryTypesModel):
        """Progress tracking for ongoing operations.

        Used by: batch operations, migrations, sync, data processing.
        """

        success_count: Annotated[int, Field(default=0, description="Successes")] = 0
        failure_count: Annotated[int, Field(default=0, description="Failures")] = 0
        skipped_count: Annotated[int, Field(default=0, description="Skipped")] = 0
        warning_count: Annotated[int, Field(default=0, description="Warnings")] = 0
        retry_count: Annotated[int, Field(default=0, description="Retries")] = 0
        start_time: Annotated[
            datetime | None, Field(default=None, description="Start time")
        ] = None
        last_update: Annotated[
            datetime | None, Field(default=None, description="Last update")
        ] = None
        estimated_total: Annotated[
            int | None, Field(default=None, description="Estimated total")
        ] = None
        current_item: Annotated[
            str | None, Field(default=None, description="Current item")
        ] = None
        operation_name: Annotated[
            str | None, Field(default=None, description="Operation name")
        ] = None
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Operation metadata",
            ),
        ]

        def record_failure(self) -> None:
            """Record a failed operation."""
            self.failure_count += 1
            self._update_timestamp()

        def record_retry(self) -> None:
            """Record a retry attempt."""
            self.retry_count += 1
            self._update_timestamp()

        def record_skip(self) -> None:
            """Record a skipped operation."""
            self.skipped_count += 1
            self._update_timestamp()

        def record_success(self) -> None:
            """Record a successful operation."""
            self.success_count += 1
            self._update_timestamp()

        def record_warning(self) -> None:
            """Record an operation with warnings."""
            self.warning_count += 1
            self._update_timestamp()

        def start_operation(
            self, name: str | None = None, estimated_total: int | None = None
        ) -> None:
            """Start the operation tracking."""
            self.operation_name = name
            self.estimated_total = estimated_total
            self.start_time = datetime.now(UTC)
            self.last_update = self.start_time

        def _update_timestamp(self) -> None:
            """Update the last update timestamp."""
            self.last_update = datetime.now(UTC)

    class Conversion(FlextModelFoundation.ArbitraryTypesModel):
        """Conversion progress tracking with error reporting.

        Used by: flext-ldif conversion, data transformations, ETL.
        """

        converted: Annotated[
            list[t.NormalizedValue | BaseModel],
            Field(default_factory=list, description="Converted items"),
        ]
        errors: Annotated[
            list[str],
            Field(default_factory=list, description="Error messages"),
        ]
        warnings: Annotated[
            list[str],
            Field(default_factory=list, description="Warning messages"),
        ]
        skipped: Annotated[
            list[t.NormalizedValue | BaseModel],
            Field(default_factory=list, description="Skipped items"),
        ]
        start_time: Annotated[
            datetime | None, Field(default=None, description="Start time")
        ] = None
        end_time: Annotated[
            datetime | None, Field(default=None, description="End time")
        ] = None
        source_format: Annotated[
            str | None, Field(default=None, description="Source format")
        ] = None
        target_format: Annotated[
            str | None, Field(default=None, description="Target format")
        ] = None
        total_input_count: Annotated[
            int | None, Field(default=None, description="Total input count")
        ] = None
        metadata: Annotated[
            FlextModelsContainers.Dict,
            Field(
                default_factory=FlextModelsContainers.Dict,
                description="Conversion metadata",
            ),
        ]

        def add_converted(self, item: t.NormalizedValue | BaseModel) -> None:
            """Add a successfully converted item."""
            self.converted.append(item)

        def add_error(
            self, error: str, item: t.NormalizedValue | BaseModel | None = None
        ) -> None:
            """Add an error with optional failed item."""
            self.errors.append(error)
            if item is not None:
                self._append_metadata_item("failed_items", item)

        def add_skipped(
            self, item: t.NormalizedValue | BaseModel, reason: str | None = None
        ) -> None:
            """Add a skipped item with optional reason."""
            self.skipped.append(item)
            if reason:
                self._upsert_skip_reason(item, reason)

        def add_warning(
            self, warning: str, item: t.NormalizedValue | BaseModel | None = None
        ) -> None:
            """Add a warning with optional item."""
            self.warnings.append(warning)
            if item is not None:
                self._append_metadata_item("warning_items", item)

        def complete_conversion(self) -> None:
            """Mark conversion as completed."""
            self.end_time = datetime.now(UTC)

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
            self.start_time = datetime.now(UTC)

        def _append_metadata_item(
            self,
            key: Literal["failed_items", "warning_items"],
            item: t.NormalizedValue | BaseModel,
        ) -> None:
            if key not in self.metadata.root:
                self.metadata.root[key] = []
            raw_items = self.metadata.root.get(key, [])
            items = raw_items if isinstance(raw_items, list) else []
            items.append(item)
            self.metadata.root[key] = items

        def _upsert_skip_reason(
            self, item: t.NormalizedValue | BaseModel, reason: str
        ) -> None:
            raw_reasons = self.metadata.root.get("skip_reasons", {})
            reasons: dict[str, str] = {}
            if isinstance(raw_reasons, Mapping):
                reasons = {str(k): str(v) for k, v in raw_reasons.items()}
            reasons[str(item)] = reason
            self.metadata.root["skip_reasons"] = reasons

    BatchResultDict = FlextModelsContainers.BatchResultDict


__all__ = ["FlextGenericModels"]
