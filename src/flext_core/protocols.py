"""Protocol definitions for interface contracts and type safety in FLEXT ecosystem.

Provides FlextProtocols, a hierarchical collection of runtime-checkable protocol
definitions that establish interface contracts and enable type-safe structural
typing throughout the FLEXT ecosystem. All protocols use @runtime_checkable for
runtime validation and follow a layered architecture pattern.

Scope: Foundation protocols organized in architectural layers (Layer 0: Foundation,
Layer 0.5: Circular Import Prevention, Layer 1: Domain, Layer 2: Application,
Layer 3: Infrastructure, Layer 4: Extensions). Protocols enable type-safe
implementations across 32+ dependent projects without requiring explicit
inheritance, using structural typing and protocol composition. All protocols
are nested within the FlextProtocols class following the single-class pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Generic, Protocol, runtime_checkable

from flext_core.typings import T, T_co


class FlextProtocols:
    """Hierarchical protocol definitions for enterprise FLEXT ecosystem.

    Architecture: Layer 0 (Pure Constants - no implementation)
    Provides foundational protocol definitions for the entire FLEXT ecosystem,
    establishing interface contracts and enabling type-safe, structural typing
    compliance across all 32+ dependent projects.

    Key Distinction: These are PROTOCOL DEFINITIONS, not implementations.
    Actual implementations live in their respective layers.
    """

    # Layer 0: Foundation Protocols
    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for objects that can dump model data."""

        def model_dump(self) -> dict[str, object]:
            """Dump model data."""
            ...

    @runtime_checkable
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for objects with model fields."""

        @property
        def model_fields(self) -> dict[str, object]:
            """Model fields."""
            ...

    @runtime_checkable
    class HasResultValue(Protocol[T_co]):
        """Protocol for result-like objects with value."""

        @property
        def value(self) -> T_co:
            """Result value."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

    @runtime_checkable
    class HasValidateCommand(Protocol):
        """Protocol for command validation."""

        def validate_command(
            self, command: object
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate command."""
            ...

    @runtime_checkable
    class HasInvariants(Protocol):
        """Protocol for DDD aggregate invariant checking."""

        def check_invariants(self) -> FlextProtocols.ResultProtocol[bool]:
            """Check invariants."""
            ...

    @runtime_checkable
    class HasTimestamps(Protocol):
        """Protocol for audit timestamp tracking."""

        @property
        def created_at(self) -> datetime:
            """Creation timestamp."""
            ...

        @property
        def updated_at(self) -> datetime:
            """Update timestamp."""
            ...

    @runtime_checkable
    class HasHandlerType(Protocol):
        """Protocol for handler type identification."""

        @property
        def handler_type(self) -> str:
            """Handler type."""
            ...

    @runtime_checkable
    class Configurable(Protocol):
        """Protocol for component configuration."""

        def configure(self, config: dict[str, object]) -> None:
            """Configure component."""
            ...

    # Layer 0.5: Circular Import Prevention Protocols
    @runtime_checkable
    class ResultProtocol(Protocol[T]):
        """Result type interface (prevents circular imports)."""

        @property
        def value(self) -> T:
            """Result value."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

        def ok(self, value: T) -> FlextProtocols.ResultProtocol[T]:
            """Create success result."""
            ...

        def fail(
            self,
            error: str,
            error_code: str | None = None,
            metadata: dict[str, object] | None = None,
        ) -> FlextProtocols.ResultProtocol[T]:
            """Create failure result."""
            ...

        def map(
            self, func: Callable[[T], object]
        ) -> FlextProtocols.ResultProtocol[object]:
            """Map success value."""
            ...

        def flat_map(
            self, func: Callable[[T], FlextProtocols.ResultProtocol[object]]
        ) -> FlextProtocols.ResultProtocol[object]:
            """Flat map success value."""
            ...

        def unwrap(self) -> T:
            """Unwrap success value."""
            ...

    @runtime_checkable
    class ResultLike(Protocol[T_co]):
        """Result-like protocol for compatibility with FlextResult operations."""

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

        @property
        def value(self) -> T_co:
            """Result value."""
            ...

        @property
        def error(self) -> str | None:
            """Error message."""
            ...

        def unwrap(self) -> T_co:
            """Unwrap value."""
            ...

    @runtime_checkable
    class ConfigProtocol(Protocol):
        """Configuration interface (prevents circular imports)."""

        def get(self, key: str, default: object = None) -> object:
            """Get configuration value."""
            ...

        def set(self, key: str, value: object) -> None:
            """Set configuration value."""
            ...

    @runtime_checkable
    class ModelProtocol(HasModelDump, Protocol):
        """Model type interface (prevents circular imports)."""

        def validate(self) -> FlextProtocols.ResultProtocol[bool]:
            """Validate model."""
            ...

    # Layer 1: Domain Protocols
    @runtime_checkable
    class Service(Protocol, Generic[T]):
        """Base domain service interface."""

        def execute(self, command: object) -> FlextProtocols.ResultProtocol[T]:
            """Execute command."""
            ...

        def validate_business_rules(
            self, command: object
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate business rules."""
            ...

        def is_valid(self) -> bool:
            """Check validity."""
            ...

        def get_service_info(self) -> dict[str, object]:
            """Get service info."""
            ...

    @runtime_checkable
    class Repository(Protocol, Generic[T]):
        """Data access interface."""

        def get_by_id(self, entity_id: str) -> FlextProtocols.ResultProtocol[T]:
            """Get entity by ID."""
            ...

        def save(self, entity: T) -> FlextProtocols.ResultProtocol[T]:
            """Save entity."""
            ...

        def delete(self, entity_id: str) -> FlextProtocols.ResultProtocol[bool]:
            """Delete entity."""
            ...

        def find_all(self) -> FlextProtocols.ResultProtocol[list[T]]:
            """Find all entities."""
            ...

    # Layer 2: Application Protocols
    @runtime_checkable
    class Handler(Protocol):
        """Command/Query handler interface."""

        def handle(self, message: object) -> FlextProtocols.ResultProtocol[object]:
            """Handle message."""
            ...

        def validate_command(
            self, command: object
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate command."""
            ...

        def validate_query(self, query: object) -> FlextProtocols.ResultProtocol[bool]:
            """Validate query."""
            ...

        def can_handle(self, message_type: str) -> bool:
            """Check if can handle message type."""
            ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Command routing and execution."""

        def register_handler(
            self, handler_type: str, handler: FlextProtocols.Handler
        ) -> None:
            """Register handler."""
            ...

        def execute(self, command: object) -> FlextProtocols.ResultProtocol[object]:
            """Execute command."""
            ...

    @runtime_checkable
    class Middleware(Protocol):
        """Processing pipeline."""

        def process(
            self,
            command: object,
            next_handler: Callable[[object], FlextProtocols.ResultProtocol[object]],
        ) -> FlextProtocols.ResultProtocol[object]:
            """Process command."""
            ...

    # Layer 3: Infrastructure Protocols
    @runtime_checkable
    class LoggerProtocol(Protocol):
        """Logging interface."""

        def log(
            self, level: str, message: str, _context: dict[str, object] | None = None
        ) -> None:
            """Log message."""
            ...

        def debug(
            self, message: str, _context: dict[str, object] | None = None
        ) -> None:
            """Debug log."""
            ...

        def info(self, message: str, _context: dict[str, object] | None = None) -> None:
            """Info log."""
            ...

        def warning(
            self, message: str, _context: dict[str, object] | None = None
        ) -> None:
            """Warning log."""
            ...

        def error(
            self, message: str, _context: dict[str, object] | None = None
        ) -> None:
            """Error log."""
            ...

    @runtime_checkable
    class Connection(Protocol):
        """External system connection."""

        def test_connection(self) -> FlextProtocols.ResultProtocol[bool]:
            """Test connection."""
            ...

        def get_connection_string(self) -> str:
            """Get connection string."""
            ...

        def close_connection(self) -> None:
            """Close connection."""
            ...

    # Layer 4: Extensions
    @runtime_checkable
    class PluginContext(Protocol):
        """Plugin execution context."""

        @property
        def config(self) -> dict[str, object]:
            """Plugin config."""
            ...

        @property
        def runtime_id(self) -> str:
            """Runtime ID."""
            ...

    @runtime_checkable
    class Observability(Protocol):
        """Metrics and monitoring."""

        def record_metric(
            self, name: str, value: object, tags: dict[str, str] | None = None
        ) -> None:
            """Record metric."""
            ...

        def log_event(
            self, level: str, message: str, _context: dict[str, object] | None = None
        ) -> None:
            """Log event."""
            ...

    class ValidationInfo(Protocol):
        """Protocol for Pydantic ValidationInfo to avoid explicit Any types.

        Used in field validators where Pydantic's ValidationInfo is needed
        but we want to avoid importing pydantic directly in protocols.
        """

        @property
        def field_name(self) -> str | None:
            """Field name being validated."""
            ...

        @property
        def data(self) -> dict[str, object] | None:
            """Validation data dictionary."""
            ...

        @property
        def mode(self) -> str:
            """Validation mode."""
            ...


__all__ = ["FlextProtocols"]
