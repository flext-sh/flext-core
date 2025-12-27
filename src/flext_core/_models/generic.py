"""Generic reusable models for the FLEXT ecosystem.

TIER 1: Uses Tier 0 modules (constants, typings) and base models.

This module provides generic models organized by business function:
- Value Objects: Immutable data compared by value (FrozenValueModel)
- Snapshots: State captured at a specific moment (FrozenStrictModel)
- Progress Trackers: Mutable accumulators during operations (ArbitraryTypesModel)

All downstream projects (flext-cli, flext-ldif, flext-ldap, client-a-oud-mig)
should use these models instead of creating their own dict patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import KeysView
from datetime import UTC, datetime

from pydantic import Field

from flext_core._models.base import FlextModelFoundation
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

        class LdapEntryAttributes(FlextModelFoundation.FrozenValueModel):
            """Normalized LDAP attributes - immutable after reading from server.

            Used by: flext-ldap, flext-ldif, client-a-oud-mig
            Function: Encapsulate dict of attributes with type-safe access
            """

            attributes: dict[str, list[str]] = Field(default_factory=dict)

            def get(self, name: str) -> list[str] | None:
                """Get attribute value by name.

                Args:
                    name: Attribute name to look up.

                Returns:
                    List of attribute values, or None if not found.

                """
                return self.attributes.get(name)

            def __getitem__(self, name: str) -> list[str]:
                """Get attribute value by name using subscript notation.

                Args:
                    name: Attribute name to look up.

                Returns:
                    List of attribute values.

                Raises:
                    KeyError: If attribute not found.

                """
                return self.attributes[name]

            def keys(self) -> KeysView[str]:
                """Get all attribute names.

                Returns:
                    View of all attribute names.

                """
                return self.attributes.keys()

            def __contains__(self, name: str) -> bool:
                """Check if attribute exists.

                Args:
                    name: Attribute name to check.

                Returns:
                    True if attribute exists, False otherwise.

                """
                return name in self.attributes

            def __len__(self) -> int:
                """Get number of attributes.

                Returns:
                    Number of attributes.

                """
                return len(self.attributes)

        class OperationContext(FlextModelFoundation.FrozenValueModel):
            """Immutable context of an operation.

            Used by: all projects
            Function: Metadata of operation that doesn't change during execution
            """

            correlation_id: str = Field(
                default_factory=FlextRuntime.generate_id,
            )
            operation_id: str = Field(
                default_factory=FlextRuntime.generate_id,
            )
            timestamp: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
            )
            source: str | None = None

    # ═══════════════════════════════════════════════════════════════════════════
    # SNAPSHOTS (Immutable - captured state)
    # Function: Capture state at a specific moment
    # ═══════════════════════════════════════════════════════════════════════════

    class Snapshot:
        """Snapshots - state captured at a specific moment."""

        class Service(FlextModelFoundation.FrozenStrictModel):
            """Snapshot of service state.

            Used by: FlextService.get_info(), monitoring
            Function: Capture current service state for logging/debug
            """

            name: str
            version: str | None = None
            status: str = "active"
            uptime_seconds: float | None = None
            metadata: dict[str, t.GeneralValueType] = Field(default_factory=dict)

        class Configuration(FlextModelFoundation.FrozenStrictModel):
            """Snapshot of configuration at a moment.

            Used by: CLI config info, debug, auditing
            Function: Capture current configuration for comparison/logging
            """

            config: dict[str, t.GeneralValueType] = Field(default_factory=dict)
            captured_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
            )
            source: str | None = None
            environment: str | None = None

        class Health(FlextModelFoundation.FrozenStrictModel):
            """Result of health check - immutable after verification.

            Used by: endpoints /health, monitoring
            Function: Result of a health check
            """

            healthy: bool = True
            checks: dict[str, bool] = Field(default_factory=dict)
            details: dict[str, t.GeneralValueType] = Field(default_factory=dict)
            checked_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
            )

        class ObjectClassGroups(FlextModelFoundation.FrozenStrictModel):
            """Entries grouped by objectClass - result of grouping.

            Used by: flext-ldap, flext-ldif (schema analysis)
            Function: Result of grouping by object type
            """

            groups: dict[str, list[str]] = Field(
                default_factory=dict,
            )  # objectClass -> list[dn]
            total_entries: int = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # PROGRESS TRACKERS (Mutable - accumulate during operation)
    # Function: Track progress of operations that evolve
    # ═══════════════════════════════════════════════════════════════════════════

    class Progress:
        """Progress trackers - mutable, accumulate during operation."""

        class Operation(FlextModelFoundation.ArbitraryTypesModel):
            """Progress of ongoing operation - mutable.

            Used by: batch operations, migrations, sync
            Function: Accumulate counters during processing
            """

            success_count: int = 0
            failure_count: int = 0
            skipped_count: int = 0

            @property
            def total_count(self) -> int:
                """Calculate total processed items.

                Returns:
                    Total of success + failure + skipped counts.

                """
                return self.success_count + self.failure_count + self.skipped_count

            @property
            def success_rate(self) -> float:
                """Calculate success rate as ratio.

                Returns:
                    Success rate between 0.0 and 1.0.

                """
                if self.total_count == 0:
                    return 0.0
                return self.success_count / self.total_count

            def record_success(self) -> None:
                """Record a successful operation."""
                self.success_count += 1

            def record_failure(self) -> None:
                """Record a failed operation."""
                self.failure_count += 1

            def record_skip(self) -> None:
                """Record a skipped operation."""
                self.skipped_count += 1

        class Conversion(FlextModelFoundation.ArbitraryTypesModel):
            """Progress of conversion with accumulation of errors/warnings.

            Used by: flext-ldif conversion, data transformations
            Function: Accumulate results and problems during conversion
            """

            converted: list[t.GeneralValueType] = Field(default_factory=list)
            errors: list[str] = Field(default_factory=list)
            warnings: list[str] = Field(default_factory=list)

            @property
            def has_errors(self) -> bool:
                """Check if any errors occurred.

                Returns:
                    True if errors list is not empty.

                """
                return len(self.errors) > 0

            @property
            def has_warnings(self) -> bool:
                """Check if any warnings occurred.

                Returns:
                    True if warnings list is not empty.

                """
                return len(self.warnings) > 0

            @property
            def converted_count(self) -> int:
                """Get count of converted items.

                Returns:
                    Number of items in converted list.

                """
                return len(self.converted)

            def add_converted(self, item: t.GeneralValueType) -> None:
                """Add a successfully converted item.

                Args:
                    item: The converted item to add.

                """
                self.converted.append(item)

            def add_error(self, error: str) -> None:
                """Add an error message.

                Args:
                    error: Error message to add.

                """
                self.errors.append(error)

            def add_warning(self, warning: str) -> None:
                """Add a warning message.

                Args:
                    warning: Warning message to add.

                """
                self.warnings.append(warning)


# Short alias for internal use
gm = FlextGenericModels
