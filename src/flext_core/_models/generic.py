"""Generic reusable models for the FLEXT ecosystem.

TIER 1: Uses Tier 0 modules (constants, typings) and base models.

This module provides generic models organized by business function:
- Value Objects: Immutable data compared by value (FrozenValueModel)
- Snapshots: State captured at a specific moment (FrozenStrictModel)
- Progress Trackers: Mutable accumulators during operations (ArbitraryTypesModel)

All downstream projects (flext-cli, flext-ldif, flext-ldap, algar-oud-mig)
should use these models instead of creating their own dict patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from flext_core._models.base import FlextModelFoundation
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextGenericModels:
    """Generic models organized by business function.

    Categories:
    - Value: Immutable data objects compared by value
    - Snapshot: State captured at a specific moment
    - Progress: Mutable accumulators during operations
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # VALUE OBJECTS (Immutable - compared by value)
    # Function: Represent data that doesn't change after creation
    # ═══════════════════════════════════════════════════════════════════════════

    class Value:
        """Value objects - immutable, compared by value."""

        # NOTE: LdapEntryAttributes REMOVED - LDAP-specific, belongs in flext-ldap
        # Use: from flext_ldap.models import m; m.Ldap.EntryAttributes

        class OperationContext(FlextModelFoundation.FrozenValueModel):
            """Immutable context of an operation with advanced tracking capabilities.

            Used by: all projects
            Function: Metadata of operation that doesn't change during execution
            Provides comprehensive operation tracking with correlation, timing, and metadata
            """

            correlation_id: str = Field(
                default_factory=FlextRuntime.generate_id,
                description="Unique correlation ID for tracing operations across services",
            )
            operation_id: str = Field(
                default_factory=FlextRuntime.generate_id,
                description="Unique operation ID for this specific operation instance",
            )
            timestamp: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="UTC timestamp when operation context was created",
            )
            source: str | None = Field(
                default=None,
                description="Source system or component that initiated the operation",
            )
            user_id: str | None = Field(
                default=None,
                description="ID of user who initiated the operation (if applicable)",
            )
            tenant_id: str | None = Field(
                default=None,
                description="Multi-tenant identifier for operation isolation",
            )
            environment: str | None = Field(
                default=None,
                description="Environment context (dev, staging, prod, etc.)",
            )
            version: str = Field(
                default="1.0.0",
                description="Version of the operation context schema",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Additional context-specific metadata",
            )

            @property
            def age_seconds(self) -> float:
                """Calculate age of operation context in seconds.

                Returns:
                    float: Age in seconds since context creation.

                """
                now = datetime.now(UTC)
                return (now - self.timestamp).total_seconds()

            @property
            def age_minutes(self) -> float:
                """Calculate age of operation context in minutes.

                Returns:
                    float: Age in minutes since context creation.

                """
                return self.age_seconds / 60.0

            @property
            def is_recent(self) -> bool:
                """Check if operation context is recent (created within last 5 minutes).

                Returns:
                    bool: True if context was created within last 5 minutes.

                """
                return self.age_minutes <= c.Performance.VERY_RECENT_THRESHOLD_MINUTES

            @property
            def formatted_timestamp(self) -> str:
                """Get ISO 8601 formatted timestamp string.

                Returns:
                    str: ISO 8601 formatted timestamp.

                """
                return self.timestamp.isoformat()

            @property
            def has_user_context(self) -> bool:
                """Check if operation has user context information.

                Returns:
                    bool: True if user_id is set.

                """
                return self.user_id is not None

            @property
            def has_tenant_context(self) -> bool:
                """Check if operation has tenant context information.

                Returns:
                    bool: True if tenant_id is set.

                """
                return self.tenant_id is not None

            @property
            def context_summary(self) -> str:
                """Generate a summary string of the operation context.

                Returns:
                    str: Formatted summary including key identifiers.

                """
                parts = [
                    f"op:{self.operation_id[:8]}",
                    f"corr:{self.correlation_id[:8]}",
                ]
                if self.source:
                    parts.append(f"src:{self.source}")
                if self.user_id:
                    parts.append(f"user:{self.user_id}")
                if self.tenant_id:
                    parts.append(f"tenant:{self.tenant_id}")

                return " | ".join(parts)

            def with_metadata(
                self, **kwargs: t.GeneralValueType
            ) -> FlextGenericModels.Value.OperationContext:
                """Create new context with additional metadata.

                Args:
                    **kwargs: Metadata key-value pairs to add.

                Returns:
                    OperationContext: New context with merged metadata.

                """
                new_metadata = {**self.metadata, **kwargs}
                return self.__class__(
                    correlation_id=self.correlation_id,
                    operation_id=self.operation_id,
                    timestamp=self.timestamp,
                    source=self.source,
                    user_id=self.user_id,
                    tenant_id=self.tenant_id,
                    environment=self.environment,
                    version=self.version,
                    metadata=new_metadata,
                )

            def for_child_operation(
                self, child_operation_id: str | None = None
            ) -> FlextGenericModels.Value.OperationContext:
                """Create context for child operation with same correlation.

                Args:
                    child_operation_id: Optional child operation ID, generated if None.

                Returns:
                    OperationContext: Child context with same correlation but new operation ID.

                """
                return self.__class__(
                    correlation_id=self.correlation_id,
                    operation_id=child_operation_id or FlextRuntime.generate_id(),
                    timestamp=datetime.now(UTC),
                    source=self.source,
                    user_id=self.user_id,
                    tenant_id=self.tenant_id,
                    environment=self.environment,
                    version=self.version,
                    metadata=self.metadata,
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # SNAPSHOTS (Immutable - captured state)
    # Function: Capture state at a specific moment
    # ═══════════════════════════════════════════════════════════════════════════

    class Snapshot:
        """Snapshots - state captured at a specific moment."""

        class Service(FlextModelFoundation.FrozenStrictModel):
            """Comprehensive snapshot of service state with health and metrics.

            Used by: FlextService.get_info(), monitoring, health checks
            Function: Capture current service state for logging/debug/metrics
            """

            name: str = Field(description="Service name identifier")
            version: str | None = Field(
                default=None,
                description="Service version (semantic versioning)",
            )
            status: str = Field(
                default="active",
                description="Current service status (active, inactive, degraded, etc.)",
            )
            uptime_seconds: float | None = Field(
                default=None,
                description="Service uptime in seconds",
            )
            start_time: datetime | None = Field(
                default=None,
                description="When service was started",
            )
            last_health_check: datetime | None = Field(
                default=None,
                description="Timestamp of last health check",
            )
            health_status: str = Field(
                default="unknown",
                description="Health status (healthy, unhealthy, degraded)",
            )
            port: int | None = Field(
                default=None,
                description="Service port number",
            )
            host: str | None = Field(
                default=None,
                description="Service host address",
            )
            pid: int | None = Field(
                default=None,
                description="Process ID of service",
            )
            memory_usage_mb: float | None = Field(
                default=None,
                description="Current memory usage in MB",
            )
            cpu_usage_percent: float | None = Field(
                default=None,
                description="Current CPU usage percentage",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Additional service-specific metadata",
            )

            @property
            def is_active(self) -> bool:
                """Check if service is in active status.

                Returns:
                    bool: True if status is 'active'.

                """
                return self.status == "active"

            @property
            def is_healthy(self) -> bool:
                """Check if service health status is healthy.

                Returns:
                    bool: True if health_status is 'healthy'.

                """
                return self.health_status == "healthy"

            @property
            def uptime_hours(self) -> float | None:
                """Calculate uptime in hours.

                Returns:
                    float | None: Uptime in hours, or None if uptime_seconds not available.

                """
                return self.uptime_seconds / 3600.0 if self.uptime_seconds else None

            @property
            def uptime_days(self) -> float | None:
                """Calculate uptime in days.

                Returns:
                    float | None: Uptime in days, or None if uptime_seconds not available.

                """
                return self.uptime_hours / 24.0 if self.uptime_hours else None

            @property
            def formatted_uptime(self) -> str:
                """Format uptime as human-readable string.

                Returns:
                    str: Formatted uptime string (e.g., "2d 3h 45m").

                """
                if not self.uptime_seconds:
                    return "unknown"

                days = int(self.uptime_seconds // 86400)
                hours = int((self.uptime_seconds % 86400) // 3600)
                minutes = int((self.uptime_seconds % 3600) // 60)

                parts = []
                if days > 0:
                    parts.append(f"{days}d")
                if hours > 0 or days > 0:
                    parts.append(f"{hours}h")
                parts.append(f"{minutes}m")

                return " ".join(parts)

            @property
            def endpoint_url(self) -> str | None:
                """Construct service endpoint URL if host and port are available.

                Returns:
                    str | None: Full endpoint URL or None if incomplete.

                """
                if self.host and self.port:
                    return f"http://{self.host}:{self.port}"
                return None

            @property
            def health_check_age_minutes(self) -> float | None:
                """Calculate minutes since last health check.

                Returns:
                    float | None: Minutes since last health check, or None if no check performed.

                """
                if not self.last_health_check:
                    return None

                now = datetime.now(UTC)
                return (now - self.last_health_check).total_seconds() / 60.0

            @property
            def needs_health_check(self) -> bool:
                """Check if service needs a health check (last check > 5 minutes ago).

                Returns:
                    bool: True if health check is needed.

                """
                age = self.health_check_age_minutes
                return age is None or age > c.Performance.HEALTH_CHECK_STALE_MINUTES

            @property
            def resource_summary(self) -> str:
                """Generate resource usage summary.

                Returns:
                    str: Formatted resource summary.

                """
                parts = []
                if self.memory_usage_mb is not None:
                    parts.append(f"RAM: {self.memory_usage_mb:.1f}MB")
                if self.cpu_usage_percent is not None:
                    parts.append(f"CPU: {self.cpu_usage_percent:.1f}%")

                return " | ".join(parts) if parts else "no metrics"

            def to_health_check_format(self) -> dict[str, t.GeneralValueType]:
                """Convert to standard health check format.

                Returns:
                    dict: Health check compatible dictionary.

                """
                return {
                    "name": self.name,
                    "version": self.version,
                    "status": self.status,
                    "health": self.health_status,
                    "uptime": self.formatted_uptime,
                    "timestamp": self.last_health_check.isoformat()
                    if self.last_health_check
                    else None,
                    **self.metadata,
                }

        class Configuration(FlextModelFoundation.FrozenStrictModel):
            """Comprehensive configuration snapshot with validation and metadata.

            Used by: CLI config info, debug, auditing, configuration management
            Function: Capture current configuration for comparison/logging/validation
            """

            config: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Configuration key-value pairs",
            )
            captured_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="When configuration was captured",
            )
            source: str | None = Field(
                default=None,
                description="Configuration source (file, env, database, etc.)",
            )
            environment: str | None = Field(
                default=None,
                description="Target environment (dev, staging, prod, etc.)",
            )
            version: str = Field(
                default="1.0.0",
                description="Configuration schema version",
            )
            checksum: str | None = Field(
                default=None,
                description="Configuration checksum for integrity validation",
            )
            validation_errors: list[str] = Field(
                default_factory=list,
                description="Configuration validation errors if any",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Configuration metadata",
            )

            @property
            def is_valid(self) -> bool:
                """Check if configuration has no validation errors.

                Returns:
                    bool: True if no validation errors exist.

                """
                return len(self.validation_errors) == 0

            @property
            def has_validation_errors(self) -> bool:
                """Check if configuration has validation errors.

                Returns:
                    bool: True if validation errors exist.

                """
                return len(self.validation_errors) > 0

            @property
            def validation_error_count(self) -> int:
                """Get count of validation errors.

                Returns:
                    int: Number of validation errors.

                """
                return len(self.validation_errors)

            @property
            def config_keys(self) -> list[str]:
                """Get sorted list of configuration keys.

                Returns:
                    list[str]: Sorted configuration keys.

                """
                return sorted(self.config.keys())

            @property
            def config_size(self) -> int:
                """Get number of configuration entries.

                Returns:
                    int: Number of configuration key-value pairs.

                """
                return len(self.config)

            @property
            def age_minutes(self) -> float:
                """Calculate age of configuration snapshot in minutes.

                Returns:
                    float: Age in minutes since capture.

                """
                now = datetime.now(UTC)
                return (now - self.captured_at).total_seconds() / 60.0

            @property
            def is_recent(self) -> bool:
                """Check if configuration is recent (captured within last hour).

                Returns:
                    bool: True if captured within last hour.

                """
                return self.age_minutes <= c.Performance.RECENT_THRESHOLD_MINUTES

            @property
            def formatted_captured_at(self) -> str:
                """Get ISO 8601 formatted capture timestamp.

                Returns:
                    str: ISO 8601 formatted timestamp.

                """
                return self.captured_at.isoformat()

            def get(
                self, key: str, default: t.GeneralValueType = None
            ) -> t.GeneralValueType:
                """Get configuration value with optional default.

                Args:
                    key: Configuration key to retrieve.
                    default: Default value if key not found.

                Returns:
                    Configuration value or default.

                """
                return self.config.get(key, default)

            def has_key(self, key: str) -> bool:
                """Check if configuration contains a key.

                Args:
                    key: Key to check for existence.

                Returns:
                    bool: True if key exists in configuration.

                """
                return key in self.config

            def to_environment_variables(self, prefix: str = "") -> dict[str, str]:
                """Convert configuration to environment variable format.

                Args:
                    prefix: Optional prefix for environment variable names.

                Returns:
                    dict: Environment variable name-value pairs.

                """
                env_vars = {}
                for key, value in self.config.items():
                    env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                    env_vars[env_key] = str(value)
                return env_vars

            def validate_required_keys(self, required_keys: list[str]) -> list[str]:
                """Validate that required configuration keys are present.

                Args:
                    required_keys: List of keys that must be present.

                Returns:
                    list[str]: List of missing required keys.

                """
                return [key for key in required_keys if key not in self.config]

            def with_validation_errors(
                self, errors: list[str]
            ) -> FlextGenericModels.Snapshot.Configuration:
                """Create new configuration with validation errors.

                Args:
                    errors: List of validation errors to add.

                Returns:
                    Configuration: New configuration with errors.

                """
                return self.__class__(
                    config=self.config,
                    captured_at=self.captured_at,
                    source=self.source,
                    environment=self.environment,
                    version=self.version,
                    checksum=self.checksum,
                    validation_errors=errors,
                    metadata=self.metadata,
                )

        class Health(FlextModelFoundation.FrozenStrictModel):
            """Comprehensive health check result with detailed diagnostics.

            Used by: endpoints /health, monitoring, alerting systems
            Function: Result of a health check with detailed component analysis
            """

            healthy: bool = Field(
                default=True,
                description="Overall health status",
            )
            checks: dict[str, bool] = Field(
                default_factory=dict,
                description="Individual health check results (component -> status)",
            )
            details: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Detailed information for each health check",
            )
            checked_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="When health check was performed",
            )
            service_name: str | None = Field(
                default=None,
                description="Name of service being checked",
            )
            service_version: str | None = Field(
                default=None,
                description="Version of service being checked",
            )
            duration_ms: float | None = Field(
                default=None,
                description="Time taken to perform health check in milliseconds",
            )
            environment: str | None = Field(
                default=None,
                description="Environment where health check was performed",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Additional health check metadata",
            )

            @property
            def total_checks(self) -> int:
                """Get total number of health checks performed.

                Returns:
                    int: Number of health checks.

                """
                return len(self.checks)

            @property
            def healthy_checks_count(self) -> int:
                """Get count of healthy checks.

                Returns:
                    int: Number of checks that passed.

                """
                return sum(1 for status in self.checks.values() if status)

            @property
            def unhealthy_checks_count(self) -> int:
                """Get count of unhealthy checks.

                Returns:
                    int: Number of checks that failed.

                """
                return sum(1 for status in self.checks.values() if not status)

            @property
            def health_percentage(self) -> float:
                """Calculate health percentage (0.0 to 100.0).

                Returns:
                    float: Percentage of healthy checks.

                """
                if self.total_checks == 0:
                    return 100.0
                return (self.healthy_checks_count / self.total_checks) * 100.0

            @property
            def unhealthy_checks(self) -> list[str]:
                """Get list of unhealthy check names.

                Returns:
                    list[str]: Names of failed health checks.

                """
                return [name for name, status in self.checks.items() if not status]

            @property
            def healthy_checks(self) -> list[str]:
                """Get list of healthy check names.

                Returns:
                    list[str]: Names of passed health checks.

                """
                return [name for name, status in self.checks.items() if status]

            @property
            def age_seconds(self) -> float:
                """Calculate age of health check result in seconds.

                Returns:
                    float: Age in seconds since check was performed.

                """
                now = datetime.now(UTC)
                return (now - self.checked_at).total_seconds()

            @property
            def is_recent(self) -> bool:
                """Check if health check is recent (performed within last 2 minutes).

                Returns:
                    bool: True if check was performed within last 2 minutes.

                """
                return self.age_seconds <= c.Performance.RECENT_THRESHOLD_SECONDS

            @property
            def formatted_checked_at(self) -> str:
                """Get ISO 8601 formatted check timestamp.

                Returns:
                    str: ISO 8601 formatted timestamp.

                """
                return self.checked_at.isoformat()

            @property
            def status_summary(self) -> str:
                """Generate health status summary string.

                Returns:
                    str: Formatted status summary (e.g., "3/5 checks passed").

                """
                return f"{self.healthy_checks_count}/{self.total_checks} checks passed"

            @property
            def severity_level(self) -> str:
                """Determine health severity level based on failure rate.

                Returns:
                    str: Severity level (healthy, warning, critical, unknown).

                """
                if self.total_checks == 0:
                    return "unknown"

                failure_rate = self.unhealthy_checks_count / self.total_checks

                if failure_rate == 0.0:
                    return "healthy"
                if failure_rate <= c.Performance.FAILURE_RATE_WARNING_THRESHOLD:
                    return "warning"
                return "critical"

            def get_check_detail(self, check_name: str) -> t.GeneralValueType:
                """Get detailed information for a specific health check.

                Args:
                    check_name: Name of the health check.

                Returns:
                    Detailed information for the check, or None if not found.

                """
                return self.details.get(check_name)

            def to_monitoring_format(self) -> dict[str, t.GeneralValueType]:
                """Convert to monitoring system compatible format.

                Returns:
                    dict: Monitoring compatible dictionary.

                """
                return {
                    "healthy": self.healthy,
                    "status": "up" if self.healthy else "down",
                    "checks": self.checks,
                    "total_checks": self.total_checks,
                    "healthy_count": self.healthy_checks_count,
                    "unhealthy_count": self.unhealthy_checks_count,
                    "health_percentage": self.health_percentage,
                    "severity": self.severity_level,
                    "checked_at": self.formatted_checked_at,
                    "service": self.service_name,
                    "version": self.service_version,
                    "duration_ms": self.duration_ms,
                    **self.metadata,
                }

            def with_additional_check(
                self, name: str, status: bool, detail: t.GeneralValueType = None
            ) -> FlextGenericModels.Snapshot.Health:
                """Create new health result with additional check.

                Args:
                    name: Check name.
                    status: Check status.
                    detail: Optional check detail.

                Returns:
                    Health: New health result with additional check.

                """
                new_checks = {**self.checks, name: status}
                new_details = {**self.details}
                if detail is not None:
                    new_details[name] = detail

                new_healthy = all(new_checks.values())

                return self.__class__(
                    healthy=new_healthy,
                    checks=new_checks,
                    details=new_details,
                    checked_at=self.checked_at,
                    service_name=self.service_name,
                    service_version=self.service_version,
                    duration_ms=self.duration_ms,
                    environment=self.environment,
                    metadata=self.metadata,
                )

        # NOTE: ObjectClassGroups REMOVED - LDAP-specific, belongs in flext-ldap
        # Use: from flext_ldap.models import m; m.Ldap.ObjectClassGroups

    # ═══════════════════════════════════════════════════════════════════════════
    # PROGRESS TRACKERS (Mutable - accumulate during operation)
    # Function: Track progress of operations that evolve
    # ═══════════════════════════════════════════════════════════════════════════

    class Progress:
        """Progress trackers - mutable, accumulate during operation."""

        class Operation(FlextModelFoundation.ArbitraryTypesModel):
            """Comprehensive progress tracking for ongoing operations.

            Used by: batch operations, migrations, sync, data processing
            Function: Accumulate counters, track progress, and provide metrics during processing
            """

            success_count: int = Field(
                default=0,
                description="Number of successfully processed items",
            )
            failure_count: int = Field(
                default=0,
                description="Number of failed items",
            )
            skipped_count: int = Field(
                default=0,
                description="Number of skipped items",
            )
            warning_count: int = Field(
                default=0,
                description="Number of items with warnings",
            )
            retry_count: int = Field(
                default=0,
                description="Number of retry attempts",
            )
            start_time: datetime | None = Field(
                default=None,
                description="When operation started",
            )
            last_update: datetime | None = Field(
                default=None,
                description="When progress was last updated",
            )
            estimated_total: int | None = Field(
                default=None,
                description="Estimated total items to process",
            )
            current_item: str | None = Field(
                default=None,
                description="Currently processing item identifier",
            )
            operation_name: str | None = Field(
                default=None,
                description="Name/description of the operation",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Additional operation metadata",
            )

            @property
            def total_count(self) -> int:
                """Calculate total processed items.

                Returns:
                    int: Total of success + failure + skipped + warning counts.

                """
                return (
                    self.success_count
                    + self.failure_count
                    + self.skipped_count
                    + self.warning_count
                )

            @property
            def success_rate(self) -> float:
                """Calculate success rate as ratio.

                Returns:
                    float: Success rate between 0.0 and 1.0.

                """
                if self.total_count == 0:
                    return 0.0
                return self.success_count / self.total_count

            @property
            def failure_rate(self) -> float:
                """Calculate failure rate as ratio.

                Returns:
                    float: Failure rate between 0.0 and 1.0.

                """
                if self.total_count == 0:
                    return 0.0
                return self.failure_count / self.total_count

            @property
            def completion_percentage(self) -> float:
                """Calculate completion percentage based on estimated total.

                Returns:
                    float: Completion percentage (0.0 to 100.0), or 0.0 if no estimate.

                """
                if not self.estimated_total or self.estimated_total == 0:
                    return 0.0
                return min((self.total_count / self.estimated_total) * 100.0, 100.0)

            @property
            def remaining_count(self) -> int | None:
                """Calculate estimated remaining items.

                Returns:
                    int | None: Estimated remaining items, or None if no estimate.

                """
                if self.estimated_total is None:
                    return None
                return max(0, self.estimated_total - self.total_count)

            @property
            def duration_seconds(self) -> float | None:
                """Calculate operation duration in seconds.

                Returns:
                    float | None: Duration in seconds since start, or None if not started.

                """
                if not self.start_time:
                    return None

                end_time = self.last_update or datetime.now(UTC)
                return (end_time - self.start_time).total_seconds()

            @property
            def items_per_second(self) -> float | None:
                """Calculate processing rate in items per second.

                Returns:
                    float | None: Processing rate, or None if duration not available.

                """
                duration = self.duration_seconds
                if not duration or duration == 0:
                    return None
                return self.total_count / duration

            @property
            def estimated_time_remaining_seconds(self) -> float | None:
                """Estimate time remaining based on current processing rate.

                Returns:
                    float | None: Estimated seconds remaining, or None if cannot calculate.

                """
                if not self.items_per_second or not self.remaining_count:
                    return None
                return self.remaining_count / self.items_per_second

            @property
            def is_complete(self) -> bool:
                """Check if operation is complete (all estimated items processed).

                Returns:
                    bool: True if operation appears complete.

                """
                if self.estimated_total is None:
                    return False
                return self.total_count >= self.estimated_total

            @property
            def has_errors(self) -> bool:
                """Check if operation has any failures.

                Returns:
                    bool: True if there are any failures.

                """
                return self.failure_count > 0

            @property
            def has_warnings(self) -> bool:
                """Check if operation has any warnings.

                Returns:
                    bool: True if there are any warnings.

                """
                return self.warning_count > 0

            @property
            def status_summary(self) -> str:
                """Generate operation status summary string.

                Returns:
                    str: Formatted status summary.

                """
                parts = [
                    f"{self.success_count} success",
                    f"{self.failure_count} failed",
                    f"{self.skipped_count} skipped",
                ]
                if self.warning_count > 0:
                    parts.append(f"{self.warning_count} warnings")

                if self.estimated_total:
                    parts.append(".1f")

                return ", ".join(parts)

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

            def set_current_item(self, item: str) -> None:
                """Set the currently processing item.

                Args:
                    item: Identifier of current item being processed.

                """
                self.current_item = item
                self._update_timestamp()

            def start_operation(
                self, name: str | None = None, estimated_total: int | None = None
            ) -> None:
                """Start the operation tracking.

                Args:
                    name: Optional operation name.
                    estimated_total: Optional estimated total items.

                """
                self.operation_name = name
                self.estimated_total = estimated_total
                self.start_time = datetime.now(UTC)
                self.last_update = self.start_time

            def _update_timestamp(self) -> None:
                """Update the last update timestamp."""
                self.last_update = datetime.now(UTC)

            def to_progress_report(self) -> dict[str, t.GeneralValueType]:
                """Convert to progress report format for monitoring.

                Returns:
                    dict: Progress report dictionary.

                """
                return {
                    "operation": self.operation_name,
                    "total_processed": self.total_count,
                    "success": self.success_count,
                    "failed": self.failure_count,
                    "skipped": self.skipped_count,
                    "warnings": self.warning_count,
                    "success_rate": ".3f",
                    "completion_percentage": ".1f",
                    "estimated_remaining": self.remaining_count,
                    "current_item": self.current_item,
                    "duration_seconds": self.duration_seconds,
                    "items_per_second": ".2f" if self.items_per_second else None,
                    "estimated_time_remaining": ".0f"
                    if self.estimated_time_remaining_seconds
                    else None,
                    "is_complete": self.is_complete,
                    "has_errors": self.has_errors,
                    "has_warnings": self.has_warnings,
                }

        class Conversion(FlextModelFoundation.ArbitraryTypesModel):
            """Comprehensive conversion progress tracking with detailed error reporting.

            Used by: flext-ldif conversion, data transformations, ETL processes
            Function: Accumulate results, track conversion metrics, and provide detailed error reporting
            """

            converted: list[t.GeneralValueType] = Field(
                default_factory=list,
                description="Successfully converted items",
            )
            errors: list[str] = Field(
                default_factory=list,
                description="Conversion error messages",
            )
            warnings: list[str] = Field(
                default_factory=list,
                description="Conversion warning messages",
            )
            skipped: list[t.GeneralValueType] = Field(
                default_factory=list,
                description="Items that were skipped during conversion",
            )
            start_time: datetime | None = Field(
                default=None,
                description="When conversion started",
            )
            end_time: datetime | None = Field(
                default=None,
                description="When conversion completed",
            )
            source_format: str | None = Field(
                default=None,
                description="Source data format",
            )
            target_format: str | None = Field(
                default=None,
                description="Target data format",
            )
            total_input_count: int | None = Field(
                default=None,
                description="Total number of input items",
            )
            metadata: dict[str, t.GeneralValueType] = Field(
                default_factory=dict,
                description="Conversion metadata",
            )

            @property
            def has_errors(self) -> bool:
                """Check if any errors occurred.

                Returns:
                    bool: True if errors list is not empty.

                """
                return len(self.errors) > 0

            @property
            def has_warnings(self) -> bool:
                """Check if any warnings occurred.

                Returns:
                    bool: True if warnings list is not empty.

                """
                return len(self.warnings) > 0

            @property
            def converted_count(self) -> int:
                """Get count of converted items.

                Returns:
                    int: Number of items in converted list.

                """
                return len(self.converted)

            @property
            def skipped_count(self) -> int:
                """Get count of skipped items.

                Returns:
                    int: Number of items in skipped list.

                """
                return len(self.skipped)

            @property
            def error_count(self) -> int:
                """Get count of errors.

                Returns:
                    int: Number of error messages.

                """
                return len(self.errors)

            @property
            def warning_count(self) -> int:
                """Get count of warnings.

                Returns:
                    int: Number of warning messages.

                """
                return len(self.warnings)

            @property
            def total_processed_count(self) -> int:
                """Get total count of processed items (converted + skipped).

                Returns:
                    int: Total processed items.

                """
                return self.converted_count + self.skipped_count

            @property
            def success_rate(self) -> float:
                """Calculate conversion success rate.

                Returns:
                    float: Success rate between 0.0 and 1.0.

                """
                if self.total_processed_count == 0:
                    return 0.0
                return self.converted_count / self.total_processed_count

            @property
            def duration_seconds(self) -> float | None:
                """Calculate conversion duration in seconds.

                Returns:
                    float | None: Duration in seconds, or None if not completed.

                """
                if not self.start_time or not self.end_time:
                    return None
                return (self.end_time - self.start_time).total_seconds()

            @property
            def items_per_second(self) -> float | None:
                """Calculate conversion rate in items per second.

                Returns:
                    float | None: Conversion rate, or None if duration not available.

                """
                duration = self.duration_seconds
                if not duration or duration == 0:
                    return None
                return self.total_processed_count / duration

            @property
            def is_complete(self) -> bool:
                """Check if conversion is complete.

                Returns:
                    bool: True if end_time is set.

                """
                return self.end_time is not None

            @property
            def completion_percentage(self) -> float:
                """Calculate completion percentage based on total input count.

                Returns:
                    float: Completion percentage (0.0 to 100.0), or 0.0 if no total known.

                """
                if not self.total_input_count or self.total_input_count == 0:
                    return 0.0
                return min(
                    (self.total_processed_count / self.total_input_count) * 100.0, 100.0
                )

            @property
            def status_summary(self) -> str:
                """Generate conversion status summary string.

                Returns:
                    str: Formatted status summary.

                """
                parts = [
                    f"{self.converted_count} converted",
                    f"{self.error_count} errors",
                    f"{self.warning_count} warnings",
                    f"{self.skipped_count} skipped",
                ]

                if self.completion_percentage > 0:
                    parts.append(".1f")

                return ", ".join(parts)

            @property
            def conversion_summary(self) -> str:
                """Generate detailed conversion summary.

                Returns:
                    str: Multi-line conversion summary.

                """
                lines = [
                    f"Conversion: {self.source_format or 'unknown'} → {self.target_format or 'unknown'}",
                    f"Status: {'Complete' if self.is_complete else 'In Progress'}",
                    f"Items: {self.converted_count} converted, {self.skipped_count} skipped",
                    f"Issues: {self.error_count} errors, {self.warning_count} warnings",
                ]

                if self.success_rate > 0:
                    lines.append(".1%")

                if self.duration_seconds:
                    lines.append(".2f")

                if self.items_per_second:
                    lines.append(".1f")

                return "\n".join(lines)

            def add_converted(self, item: t.GeneralValueType) -> None:
                """Add a successfully converted item.

                Args:
                    item: The converted item to add.

                """
                self.converted.append(item)

            def add_error(
                self, error: str, item: t.GeneralValueType | None = None
            ) -> None:
                """Add an error message with optional failed item.

                Args:
                    error: Error message to add.
                    item: Optional item that caused the error.

                """
                self.errors.append(error)
                if item is not None:
                    self.metadata.setdefault("failed_items", []).append(item)

            def add_warning(
                self, warning: str, item: t.GeneralValueType | None = None
            ) -> None:
                """Add a warning message with optional item.

                Args:
                    warning: Warning message to add.
                    item: Optional item that generated the warning.

                """
                self.warnings.append(warning)
                if item is not None:
                    self.metadata.setdefault("warning_items", []).append(item)

            def add_skipped(
                self, item: t.GeneralValueType, reason: str | None = None
            ) -> None:
                """Add a skipped item with optional reason.

                Args:
                    item: The item that was skipped.
                    reason: Optional reason for skipping.

                """
                self.skipped.append(item)
                if reason:
                    self.metadata.setdefault("skip_reasons", {})[str(item)] = reason

            def start_conversion(
                self,
                source_format: str | None = None,
                target_format: str | None = None,
                total_input_count: int | None = None,
            ) -> None:
                """Start the conversion tracking.

                Args:
                    source_format: Source data format.
                    target_format: Target data format.
                    total_input_count: Total number of input items.

                """
                self.source_format = source_format
                self.target_format = target_format
                self.total_input_count = total_input_count
                self.start_time = datetime.now(UTC)

            def complete_conversion(self) -> None:
                """Mark conversion as completed."""
                self.end_time = datetime.now(UTC)

            def to_conversion_report(self) -> dict[str, t.GeneralValueType]:
                """Convert to conversion report format for monitoring.

                Returns:
                    dict: Conversion report dictionary.

                """
                return {
                    "source_format": self.source_format,
                    "target_format": self.target_format,
                    "converted_count": self.converted_count,
                    "error_count": self.error_count,
                    "warning_count": self.warning_count,
                    "skipped_count": self.skipped_count,
                    "success_rate": ".3f",
                    "completion_percentage": ".1f",
                    "duration_seconds": self.duration_seconds,
                    "items_per_second": ".1f" if self.items_per_second else None,
                    "is_complete": self.is_complete,
                    "has_errors": self.has_errors,
                    "has_warnings": self.has_warnings,
                    "status_summary": self.status_summary,
                    "conversion_summary": self.conversion_summary,
                }


# Short alias for internal use
gm = FlextGenericModels
