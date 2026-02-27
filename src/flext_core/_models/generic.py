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
from typing import Literal

from pydantic import Field

from flext_core._models.base import FlextModelFoundation
from flext_core.typings import t


class FlextGenericModels:
    """Generic models organized by business function.

    Categories:
    - Value: Immutable data objects compared by value
    - Snapshot: State captured at a specific moment
    - Progress: Mutable accumulators during operations
    """

    class Value:
        """Value objects - immutable, compared by value."""

        class OperationContext(FlextModelFoundation.FrozenValueModel):
            """Immutable context of an operation.

            Used by: all projects
            Function: Metadata of operation that doesn't change during execution.
            """

            correlation_id: str = Field(
                default_factory=lambda: str(uuid.uuid4()),
                description="Unique correlation ID for tracing",
            )
            operation_id: str = Field(
                default_factory=lambda: str(uuid.uuid4()),
                description="Unique operation ID",
            )
            timestamp: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when created",
            )
            source: str | None = Field(default=None, description="Source system")
            user_id: str | None = Field(default=None, description="User ID")
            tenant_id: str | None = Field(default=None, description="Tenant ID")
            environment: str | None = Field(default=None, description="Environment")
            version: str = Field(default="1.0.0", description="Schema version")
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Additional metadata",
            )
            message: t.ConfigMapValue = Field(
                default=None,
                description="Message payload",
            )
            message_type: str = Field(default="", description="Message type")
            dispatch_type: str = Field(default="", description="Dispatch type")
            timeout_override: int | None = Field(
                default=None,
                description="Timeout override seconds",
            )

    class Snapshot:
        """Snapshots - state captured at a specific moment."""

        class Service(FlextModelFoundation.FrozenStrictModel):
            """Snapshot of service state.

            Used by: FlextService.get_info(), monitoring, health checks.
            """

            name: str = Field(description="Service name")
            version: str | None = Field(default=None, description="Service version")
            status: str = Field(default="active", description="Service status")
            uptime_seconds: float | None = Field(
                default=None,
                description="Uptime in seconds",
            )
            start_time: datetime | None = Field(default=None, description="Start time")
            last_health_check: datetime | None = Field(
                default=None,
                description="Last health check",
            )
            health_status: str = Field(default="unknown", description="Health status")
            port: int | None = Field(default=None, description="Port")
            host: str | None = Field(default=None, description="Host")
            pid: int | None = Field(default=None, description="Process ID")
            memory_usage_mb: float | None = Field(default=None, description="Memory MB")
            cpu_usage_percent: float | None = Field(default=None, description="CPU %")
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Service metadata",
            )

        class Configuration(FlextModelFoundation.FrozenStrictModel):
            """Configuration snapshot.

            Used by: CLI config info, debug, auditing.
            """

            config: t.Dict = Field(
                default_factory=t.Dict,
                description="Config key-value pairs",
            )
            captured_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="Capture timestamp",
            )
            source: str | None = Field(default=None, description="Config source")
            environment: str | None = Field(
                default=None,
                description="Target environment",
            )
            version: str = Field(default="1.0.0", description="Schema version")
            checksum: str | None = Field(default=None, description="Checksum")
            validation_errors: list[str] = Field(
                default_factory=list,
                description="Validation errors",
            )
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Config metadata",
            )

        class Health(FlextModelFoundation.FrozenStrictModel):
            """Health check result.

            Used by: /health endpoints, monitoring, alerting.
            """

            healthy: bool = Field(default=True, description="Overall health")
            checks: t.Dict = Field(default_factory=t.Dict, description="Check results")
            details: t.Dict = Field(default_factory=t.Dict, description="Check details")
            checked_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="Check timestamp",
            )
            service_name: str | None = Field(default=None, description="Service name")
            service_version: str | None = Field(
                default=None,
                description="Service version",
            )
            duration_ms: float | None = Field(
                default=None,
                description="Check duration ms",
            )
            environment: str | None = Field(default=None, description="Environment")
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Health metadata",
            )

    class Progress:
        """Progress trackers - mutable, accumulate during operation."""

        @staticmethod
        def safe_rate(numerator: int, denominator: int) -> float:
            """Division with zero-safe fallback."""
            if denominator == 0:
                return 0.0
            return numerator / denominator

        @staticmethod
        def safe_percentage(processed: int, total: int | None) -> float:
            """Percentage with zero-safe fallback, capped at 100."""
            if not total or total == 0:
                return 0.0
            return min((processed / total) * 100.0, 100.0)

        class Operation(FlextModelFoundation.ArbitraryTypesModel):
            """Progress tracking for ongoing operations.

            Used by: batch operations, migrations, sync, data processing.
            """

            success_count: int = Field(default=0, description="Successes")
            failure_count: int = Field(default=0, description="Failures")
            skipped_count: int = Field(default=0, description="Skipped")
            warning_count: int = Field(default=0, description="Warnings")
            retry_count: int = Field(default=0, description="Retries")
            start_time: datetime | None = Field(default=None, description="Start time")
            last_update: datetime | None = Field(
                default=None,
                description="Last update",
            )
            estimated_total: int | None = Field(
                default=None,
                description="Estimated total",
            )
            current_item: str | None = Field(default=None, description="Current item")
            operation_name: str | None = Field(
                default=None,
                description="Operation name",
            )
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Operation metadata",
            )

            def record_success(self) -> None:
                """Record a successful operation."""
                self.success_count += 1
                self._update_timestamp()

            def record_failure(self) -> None:
                """Record a failed operation."""
                self.failure_count += 1
                self._update_timestamp()

            def record_skip(self) -> None:
                """Record a skipped operation."""
                self.skipped_count += 1
                self._update_timestamp()

            def record_warning(self) -> None:
                """Record an operation with warnings."""
                self.warning_count += 1
                self._update_timestamp()

            def record_retry(self) -> None:
                """Record a retry attempt."""
                self.retry_count += 1
                self._update_timestamp()

            def start_operation(
                self,
                name: str | None = None,
                estimated_total: int | None = None,
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

            converted: list[t.GuardInputValue] = Field(
                default_factory=list,
                description="Converted items",
            )
            errors: list[str] = Field(
                default_factory=list,
                description="Error messages",
            )
            warnings: list[str] = Field(
                default_factory=list,
                description="Warning messages",
            )
            skipped: list[t.GuardInputValue] = Field(
                default_factory=list,
                description="Skipped items",
            )
            start_time: datetime | None = Field(default=None, description="Start time")
            end_time: datetime | None = Field(default=None, description="End time")
            source_format: str | None = Field(default=None, description="Source format")
            target_format: str | None = Field(default=None, description="Target format")
            total_input_count: int | None = Field(
                default=None,
                description="Total input count",
            )
            metadata: t.Dict = Field(
                default_factory=t.Dict,
                description="Conversion metadata",
            )

            def _append_metadata_item(
                self,
                key: Literal["failed_items", "warning_items"],
                item: t.ConfigMapValue,
            ) -> None:
                if key not in self.metadata.root:
                    self.metadata.root[key] = []
                raw_items = self.metadata.root.get(key, [])
                items = raw_items if isinstance(raw_items, list) else []
                items.append(item)
                self.metadata.root[key] = items

            def _upsert_skip_reason(self, item: t.ConfigMapValue, reason: str) -> None:
                raw_reasons = self.metadata.root.get("skip_reasons", {})
                reasons: dict[str, str] = {}
                if isinstance(raw_reasons, Mapping):
                    reasons = {str(k): str(v) for k, v in raw_reasons.items()}
                reasons[str(item)] = reason
                self.metadata.root["skip_reasons"] = reasons

            def add_converted(self, item: t.GuardInputValue) -> None:
                """Add a successfully converted item."""
                self.converted.append(item)

            def add_error(
                self,
                error: str,
                item: t.ConfigMapValue | None = None,
            ) -> None:
                """Add an error with optional failed item."""
                self.errors.append(error)
                if item is not None:
                    self._append_metadata_item("failed_items", item)

            def add_warning(
                self,
                warning: str,
                item: t.ConfigMapValue | None = None,
            ) -> None:
                """Add a warning with optional item."""
                self.warnings.append(warning)
                if item is not None:
                    self._append_metadata_item("warning_items", item)

            def add_skipped(
                self,
                item: t.GuardInputValue,
                reason: str | None = None,
            ) -> None:
                """Add a skipped item with optional reason."""
                self.skipped.append(item)
                if reason:
                    self._upsert_skip_reason(item, reason)

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

            def complete_conversion(self) -> None:
                """Mark conversion as completed."""
                self.end_time = datetime.now(UTC)

    BatchResultDict = t.BatchResultDict


__all__ = ["FlextGenericModels"]
