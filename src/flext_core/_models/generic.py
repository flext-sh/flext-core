"""Generic reusable models for FLEXT ecosystem (Value, Snapshot, Progress).

These models are used across all FLEXT consumer projects (flext-ldif, flext-ldap,
flext-cli, client-a-oud-mig) for common data patterns: immutable value objects,
captured snapshots, and mutable progress trackers.

TIER 1: Uses Tier 0 (constants, typings) and Tier 1 (base models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import KeysView
from datetime import UTC, datetime

from pydantic import Field, computed_field

from flext_core._models.base import FlextModelFoundation as m_base
from flext_core.typings import t

# ═══════════════════════════════════════════════════════════════════════════
# VALUE OBJECTS (Immutable - compared by value)
# Function: Represent data that doesn't change after creation
# ═══════════════════════════════════════════════════════════════════════════


class LdapEntryAttributes(m_base.FrozenValueModel):
    """Normalized LDAP attributes - immutable after server read.

    Represents attributes from an LDAP entry with dict-like access.
    Immutable for use in sets, as dict keys, and in value comparisons.

    Used by: flext-ldap, flext-ldif, client-a-oud-mig
    Function: Encapsulate dict of attributes with type-safe access
    """

    attributes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="LDAP attributes mapping",
    )

    def get(self, name: str) -> list[str] | None:
        """Get attribute value by name.

        Args:
            name: Attribute name (case-sensitive)

        Returns:
            List of attribute values or None if not found

        """
        return self.attributes.get(name)

    def __getitem__(self, name: str) -> list[str]:
        """Get attribute value by name (raises KeyError if not found).

        Args:
            name: Attribute name (case-sensitive)

        Returns:
            List of attribute values

        Raises:
            KeyError: If attribute not found

        """
        return self.attributes[name]

    def keys(self) -> KeysView[str]:
        """Get all attribute names.

        Returns:
            Keys view of attribute names

        """
        return self.attributes.keys()


class OperationContext(m_base.FrozenValueModel):
    """Immutable context for an operation.

    Represents metadata about an operation that doesn't change during execution.
    Provides correlation and operation tracking across the system.

    Used by: all projects
    Function: Metadata for operation that is immutable during execution
    """

    correlation_id: str = Field(
        default_factory=lambda: getattr(
            __import__("flext_core.runtime", fromlist=["FlextRuntime"]).FlextRuntime,
            "generate_id",
            lambda: __import__("uuid").uuid4().hex,
        )(),
        description="Correlation ID for request tracing",
    )
    operation_id: str = Field(
        default_factory=lambda: getattr(
            __import__("flext_core.runtime", fromlist=["FlextRuntime"]).FlextRuntime,
            "generate_id",
            lambda: __import__("uuid").uuid4().hex,
        )(),
        description="Unique operation ID",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Operation timestamp",
    )
    source: str | None = Field(
        default=None,
        description="Operation source (cli, api, worker, etc.)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOTS (Immutable - captured state)
# Function: Capture state at a specific moment
# ═══════════════════════════════════════════════════════════════════════════


class ServiceSnapshot(m_base.FrozenStrictModel):
    """Snapshot of service state - immutable after check.

    Captures the current state of a service for logging/debug/monitoring.

    Used by: FlextService.get_info(), monitoring
    Function: Capture current service state for logging/debug
    """

    name: str = Field(description="Service name")
    version: str | None = Field(default=None, description="Service version")
    status: str = Field(default="active", description="Service status")
    uptime_seconds: float | None = Field(
        default=None,
        description="Seconds service has been running",
    )
    metadata: dict[str, t.GeneralValueType] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class ConfigurationSnapshot(m_base.FrozenStrictModel):
    """Snapshot of configuration at a moment - immutable after capture.

    Captures the current configuration state for comparison/logging/audit.

    Used by: CLI config info, debug, audit
    Function: Capture current configuration for comparison/logging
    """

    config: dict[str, t.GeneralValueType] = Field(
        default_factory=dict,
        description="Configuration snapshot",
    )
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When configuration was captured",
    )
    source: str | None = Field(
        default=None,
        description="Source of configuration (env, file, code)",
    )
    environment: str | None = Field(
        default=None,
        description="Environment name (dev, staging, prod)",
    )


class HealthStatus(m_base.FrozenStrictModel):
    """Result of health check - immutable after verification.

    Represents the results of system health checks.

    Used by: endpoints /health, monitoring
    Function: Result of a health check
    """

    healthy: bool = Field(
        default=True,
        description="Overall health status",
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual component health status",
    )
    details: dict[str, t.GeneralValueType] = Field(
        default_factory=dict,
        description="Detailed health information",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When health check was performed",
    )


class ObjectClassGroups(m_base.FrozenStrictModel):
    """Entries grouped by objectClass - result of grouping.

    Groups LDAP entries by their object class.

    Used by: flext-ldap, flext-ldif (schema analysis)
    Function: Result of grouping by object type
    """

    groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="objectClass -> list[dn]",
    )
    total_entries: int = Field(
        default=0,
        description="Total number of entries",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKERS (Mutable - accumulate during operation)
# Function: Track progress of operations that evolve
# ═══════════════════════════════════════════════════════════════════════════


class OperationProgress(m_base.ArbitraryTypesModel):
    """Progress of operation in progress - mutable.

    Tracks success/failure/skip counters during batch operations.

    Used by: batch operations, migrations, sync
    Function: Accumulate counters during processing
    """

    success_count: int = Field(
        default=0,
        description="Number of successful operations",
    )
    failure_count: int = Field(
        default=0,
        description="Number of failed operations",
    )
    skipped_count: int = Field(
        default=0,
        description="Number of skipped operations",
    )

    @computed_field
    def total_count(self) -> int:
        """Total number of operations.

        Returns:
            Sum of success, failure, and skip counts

        """
        return self.success_count + self.failure_count + self.skipped_count

    @computed_field
    def success_rate(self) -> float:
        """Rate of successful operations.

        Returns:
            Percentage of successful operations (0.0 to 1.0)

        """
        total = self.success_count + self.failure_count + self.skipped_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def record_success(self) -> None:
        """Record a successful operation."""
        self.success_count += 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1

    def record_skip(self) -> None:
        """Record a skipped operation."""
        self.skipped_count += 1


class ConversionProgress(m_base.ArbitraryTypesModel):
    """Progress of conversion with error/warning accumulation - mutable.

    Tracks converted items and accumulated errors/warnings during conversion.

    Used by: flext-ldif conversion, data transformations
    Function: Accumulate results and problems during conversion
    """

    converted: list[t.GeneralValueType] = Field(
        default_factory=list,
        description="Successfully converted items",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages encountered",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages encountered",
    )

    @computed_field
    def has_errors(self) -> bool:
        """Whether any errors have been recorded.

        Returns:
            True if errors list is not empty

        """
        return len(self.errors) > 0

    @computed_field
    def has_warnings(self) -> bool:
        """Whether any warnings have been recorded.

        Returns:
            True if warnings list is not empty

        """
        return len(self.warnings) > 0

    def add_converted(self, item: t.GeneralValueType) -> None:
        """Add a successfully converted item.

        Args:
            item: The converted item to add

        """
        self.converted.append(item)

    def add_error(self, error: str) -> None:
        """Add an error message.

        Args:
            error: Error message to add

        """
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message.

        Args:
            warning: Warning message to add

        """
        self.warnings.append(warning)


__all__ = [
    "ConfigurationSnapshot",
    "ConversionProgress",
    "HealthStatus",
    "LdapEntryAttributes",
    "ObjectClassGroups",
    "OperationContext",
    "OperationProgress",
    "ServiceSnapshot",
]
