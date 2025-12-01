"""Runtime-checkable protocols that describe FLEXT public contracts.

Define ``FlextProtocols`` as the structural typing surface for dispatcher
handlers, services, results, and utilities. All protocols are
``@runtime_checkable`` so integration points can be verified without
inheritance, keeping layers decoupled while remaining type-safe.

TIER 0.5: Uses FlextTypes for type definitions (no duplication)
FlextProtocols imports from FlextTypes to maintain single source of truth.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from structlog.typing import BindableLogger

from flext_core.typings import FlextTypes, T_co


class FlextProtocols:
    """Hierarchical protocol definitions for dispatcher-aligned components.

    Architecture: Uses FlextTypes for type definitions (single source of truth)
    ===========================================================================
    Describes the minimal interfaces for results, handlers, services, logging,
    and validation so downstream code can rely on structural typing instead of
    inheritance. Implementations live in their respective modules but conform
    to these contracts for interoperability.

    ESSENTIAL PROTOCOLS (actively used - 15 core):
    - ResultProtocol: Railway-oriented result type (114 usages)
    - HasModelDump: Pydantic model compatibility (25 usages)
    - ContextProtocol: Context management (25 usages)
    - ContainerProtocol: Dependency injection (11 usages)
    - ConfigProtocol: Configuration access (9 usages)
    - VariadicCallable: Callable patterns (8 usages)
    - Handler: Handler pattern (8 usages)
    - Processor: Pipeline processing (7 usages)
    - Service: Service pattern (6 usages)
    - CommandBus: CQRS command bus (6 usages)
    - Configurable: Configuration base (4 usages)
    - ResultLike: Result compatibility (3 usages)
    - HasInvariants: Domain validation (3 usages)
    - Entry: LDIF entry pattern (3 usages)
    - Repository: Repository pattern (2 usages)

    EXTENSION PROTOCOLS (for specialized use cases):
    - HasModelFields, HasResultValue, HasValidateCommand, HasTimestamps
    - HasHandlerType, ContextServiceProtocol, ContextCorrelationProtocol
    - ContextUtilitiesProtocol, ModelProtocol, Middleware, Connection
    - PluginContext, Observability, ValidationInfo, LoggerProtocol

    Type Definitions from FlextTypes:
    - FlextTypes.ScalarValue: str | int | float | bool | datetime | None
    - FlextTypes.FlexibleValue: scalar + single-level collections
    - FlextTypes.GeneralValueType: recursive nested structures
    """

    # =========================================================================
    # LAYER 0: Foundation Protocols
    # =========================================================================

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for objects that can dump model data."""

        def model_dump(self) -> Mapping[str, FlextTypes.FlexibleValue]:
            """Dump model data."""
            ...

    @runtime_checkable
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for objects with model fields."""

        @property
        def model_fields(self) -> Mapping[str, FlextTypes.FlexibleValue]:
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
            self,
            command: FlextTypes.FlexibleValue,
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

        def configure(self, config: Mapping[str, FlextTypes.FlexibleValue]) -> None:
            """Configure component."""
            ...

    # =========================================================================
    # Context, Config, and Container Protocols for circular-import safety
    # =========================================================================

    @runtime_checkable
    class ConfigProtocol(Protocol):
        """Protocol for configuration objects with cloning support."""

        @property
        def app_name(self) -> str:
            """Application name bound to the configuration."""
            ...

        @property
        def version(self) -> str:
            """Semantic version of the running application."""
            ...

        def get(
            self,
            key: str,
            default: FlextTypes.FlexibleValue | None = None,
        ) -> FlextTypes.FlexibleValue | None:
            """Get configuration value by key."""
            ...

        def set(
            self,
            key: str,
            value: FlextTypes.FlexibleValue,
        ) -> None:
            """Set configuration value."""
            ...

        def model_copy(
            self,
            *,
            update: Mapping[str, FlextTypes.FlexibleValue] | None = None,
            deep: bool = False,
        ) -> FlextProtocols.ConfigProtocol:
            """Clone configuration with optional updates."""
            ...

    @runtime_checkable
    class ContextServiceProtocol(Protocol):
        """Protocol for service-scoped context management (static methods)."""

        @staticmethod
        def service_context(
            service_name: str, version: str | None = None
        ) -> AbstractContextManager[None]:
            """Context manager for entering a service scope."""
            ...

    @runtime_checkable
    class ContextCorrelationProtocol(Protocol):
        """Protocol for correlation-aware contexts (static methods)."""

        @staticmethod
        def get_correlation_id() -> str | None:
            """Get current correlation identifier."""
            ...

        @staticmethod
        def set_correlation_id(correlation_id: str | None) -> None:
            """Set or reset correlation identifier."""
            ...

        @staticmethod
        def reset_correlation_id() -> None:
            """Clear correlation identifier from context."""
            ...

    @runtime_checkable
    class ContextUtilitiesProtocol(Protocol):
        """Protocol for utility helpers bound to a context (static methods)."""

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure a correlation identifier exists and return it."""
            ...

    @runtime_checkable
    class ContextProtocol(Protocol):
        """Protocol for execution contexts with cloning semantics."""

        def clone(self) -> FlextProtocols.ContextProtocol:
            """Clone context for isolated execution."""
            ...

        def set(
            self,
            key: str,
            value: FlextTypes.GeneralValueType,
            scope: str = ...,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Set a context value."""
            ...

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
            """Get a context value."""
            ...

    @runtime_checkable
    class ContainerProtocol(Configurable, Protocol):
        """Protocol for dependency injection containers."""

        @property
        def config(self) -> FlextProtocols.ConfigProtocol:
            """Configuration bound to the container."""
            ...

        @property
        def context(self) -> FlextProtocols.ContextProtocol:
            """Execution context bound to the container."""
            ...

        def scoped(
            self,
            *,
            config: FlextProtocols.ConfigProtocol | None = None,
            context: FlextProtocols.ContextProtocol | None = None,
            subproject: str | None = None,
            services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
            factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]]
            | None = None,
        ) -> FlextProtocols.ContainerProtocol:
            """Create an isolated container scope with optional overrides."""
            ...

        def has_service(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def register(
            self,
            name: str,
            service: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Register a service instance."""
            ...

        def register_factory(
            self,
            name: str,
            factory: Callable[[], FlextTypes.GeneralValueType],
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Register a service factory."""
            ...

        def with_service(
            self,
            name: str,
            service: FlextTypes.GeneralValueType,
        ) -> FlextProtocols.ContainerProtocol:
            """Fluent interface for service registration."""
            ...

        def with_factory(
            self,
            name: str,
            factory: Callable[[], FlextTypes.GeneralValueType],
        ) -> FlextProtocols.ContainerProtocol:
            """Fluent interface for factory registration."""
            ...

        def get(self, name: str) -> FlextProtocols.ResultProtocol[T_co]:
            """Get service by name."""
            ...

        def get_typed[T](
            self, name: str, type_cls: type[T]
        ) -> FlextProtocols.ResultProtocol[T]:
            """Get service with type safety."""
            ...

        def list_services(self) -> Sequence[str]:
            """List all registered services."""
            ...

        def clear_all(self) -> None:
            """Clear all services and factories."""
            ...

    # =========================================================================
    # LAYER 0.5: Circular Import Prevention Protocols
    # =========================================================================

    @runtime_checkable
    class ResultProtocol[T](Protocol):
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

        @property
        def error(self) -> str | None:
            """Error message."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code for categorization."""
            ...

        # NOTE: Factory methods (ok, fail) and transformation methods (map, flat_map)
        # are omitted from this protocol for simplicity:
        # - ok/fail are classmethods in FlextResult (Protocols can't define classmethods)
        # - map/flat_map are implementation-specific (FlextResult returns FlextResult[U])
        # Use FlextResult directly when you need factory or transformation methods.

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

    class ModelProtocol(HasModelDump, Protocol):
        """Model type interface (prevents circular imports)."""

        def validate(self) -> FlextProtocols.ResultProtocol[bool]:
            """Validate model."""
            ...

    # =========================================================================
    # LAYER 1: Domain Protocols
    # =========================================================================

    @runtime_checkable
    class Service[T](Protocol):
        """Base domain service interface."""

        def execute(
            self,
            command: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[T]:
            """Execute command."""
            ...

        def validate_business_rules(
            self,
            command: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate business rules."""
            ...

        def is_valid(self) -> bool:
            """Check validity."""
            ...

        def get_service_info(self) -> Mapping[str, FlextTypes.FlexibleValue]:
            """Get service info."""
            ...

    @runtime_checkable
    class Repository[T](Protocol):
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

        def find_all(self) -> FlextProtocols.ResultProtocol[Sequence[T]]:
            """Find all entities."""
            ...

    # =========================================================================
    # LAYER 2: Application Protocols
    # =========================================================================

    @runtime_checkable
    class VariadicCallable(Protocol[T_co]):
        """Protocol for variadic callables returning T_co."""

        def __call__(
            self,
            *args: FlextTypes.FlexibleValue,
            **kwargs: FlextTypes.FlexibleValue,
        ) -> T_co:
            """Call the function with any arguments, returning T_co."""
            ...

    @runtime_checkable
    class Handler(Protocol):
        """Command/Query handler interface."""

        def handle(
            self,
            message: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[T_co]:
            """Handle message."""
            ...

        def validate_command(
            self,
            command: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate command."""
            ...

        def validate_query(
            self,
            query: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool]:
            """Validate query."""
            ...

        def can_handle(self, message_type: str) -> bool:
            """Check if can handle message type."""
            ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Command routing and execution."""

        def register_handler(
            self,
            request: FlextTypes.GeneralValueType,
            handler: FlextTypes.GeneralValueType | None = None,
        ) -> FlextProtocols.ResultProtocol[Mapping[str, FlextTypes.GeneralValueType]]:
            """Register handler."""
            ...

        def register_command[TCommand, TResult](
            self,
            command_type: type[TCommand],
            handler: FlextTypes.GeneralValueType,
            *,
            handler_config: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
            """Register command handler."""
            ...

        @staticmethod
        def create_handler_from_function(
            handler_func: Callable[..., FlextTypes.GeneralValueType],
            handler_config: Mapping[str, FlextTypes.FlexibleValue] | None = None,
            mode: str = ...,
        ) -> FlextProtocols.ResultProtocol[FlextProtocols.Handler]:
            """Create handler from function (static method)."""
            ...

        def execute(
            self,
            command: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[T_co]:
            """Execute command."""
            ...

    @runtime_checkable
    class Middleware(Protocol):
        """Processing pipeline."""

        def process[T](
            self,
            command: FlextTypes.FlexibleValue,
            next_handler: Callable[
                [FlextTypes.FlexibleValue],
                FlextProtocols.ResultProtocol[T],
            ],
        ) -> FlextProtocols.ResultProtocol[T]:
            """Process command."""
            ...

    @runtime_checkable
    class Processor(Protocol):
        """Processor interface for data transformation pipelines.

        Processors can be objects with a process() method that takes data
        and returns a result (which will be normalized to ResultProtocol).
        Accepts GeneralValueType, BaseModel, or ResultProtocol for processing.

        The return type is flexible to support:
        - Direct values (FlextTypes.GeneralValueType)
        - BaseModel instances (Pydantic models)
        - ResultProtocol instances (structural typing)
        - Objects with is_success/is_failure properties (FlextResult compatibility)
        """

        def process(
            self,
            data: (
                FlextTypes.GeneralValueType
                | BaseModel
                | FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]
            ),
        ) -> (
            FlextTypes.GeneralValueType
            | BaseModel
            | FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]
        ):
            """Process data and return result.

            Returns can be:
            - Direct value (FlextTypes.GeneralValueType)
            - BaseModel instance (Pydantic model)
            - ResultProtocol (structural typing compatible)
            """
            ...

    # =========================================================================
    # LAYER 3: Infrastructure Protocols
    # =========================================================================

    @runtime_checkable
    class LoggerProtocol(Protocol):
        """Logging interface."""

        def log(
            self,
            level: str,
            message: str,
            _context: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        ) -> None:
            """Log message."""
            ...

        def debug(
            self,
            message: str,
            *args: FlextTypes.FlexibleValue,
            return_result: bool = False,
            **context: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool] | None:
            """Debug log."""
            ...

        def info(
            self,
            message: str,
            *args: FlextTypes.FlexibleValue,
            return_result: bool = False,
            **context: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool] | None:
            """Info log."""
            ...

        def warning(
            self,
            message: str,
            *args: FlextTypes.FlexibleValue,
            return_result: bool = False,
            **context: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool] | None:
            """Warning log."""
            ...

        def error(
            self,
            message: str,
            *args: FlextTypes.FlexibleValue,
            return_result: bool = False,
            **context: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool] | None:
            """Error log."""
            ...

        def exception(
            self,
            message: str,
            *,
            exception: BaseException | None = None,
            exc_info: bool = True,
            return_result: bool = False,
            **kwargs: FlextTypes.FlexibleValue,
        ) -> FlextProtocols.ResultProtocol[bool] | None:
            """Exception log."""
            ...

    @runtime_checkable
    class StructlogLogger(BindableLogger, Protocol):
        """Protocol for structlog logger with all logging methods.

        Extends BindableLogger to add explicit method signatures for logging methods
        (debug, info, warning, error, etc.) that are available via __getattr__ at runtime.

        Structlog loggers implement this protocol through dynamic method dispatch.
        """

        def debug(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType | Exception,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log debug message."""
            ...

        def info(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log info message."""
            ...

        def warning(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log warning message."""
            ...

        def warn(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType,
        ) -> None:
            """Log warning message (alias)."""
            ...

        def error(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log error message."""
            ...

        def critical(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log critical message."""
            ...

        def exception(
            self,
            msg: str | FlextTypes.GeneralValueType,
            *args: FlextTypes.GeneralValueType,
            **kw: FlextTypes.GeneralValueType | Exception,
        ) -> None:
            """Log exception with traceback."""
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

    # =========================================================================
    # LAYER 4: Extensions
    # =========================================================================

    @runtime_checkable
    class PluginContext(Protocol):
        """Plugin execution context."""

        @property
        def config(self) -> Mapping[str, FlextTypes.FlexibleValue]:
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
            self,
            name: str,
            value: FlextTypes.FlexibleValue,
            tags: Mapping[str, str] | None = None,
        ) -> None:
            """Record metric."""
            ...

        def log_event(
            self,
            level: str,
            message: str,
            _context: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        ) -> None:
            """Log event."""
            ...

    @runtime_checkable
    class ValidationInfo(Protocol):
        """Protocol for Pydantic ValidationInfo to avoid explicit Any types."""

        @property
        def field_name(self) -> str | None:
            """Field name being validated."""
            ...

        @property
        def data(self) -> Mapping[str, FlextTypes.FlexibleValue] | None:
            """Validation data dictionary."""
            ...

        @property
        def mode(self) -> str:
            """Validation mode."""
            ...

    # =========================================================================
    # DOMAIN PROTOCOLS (Layer 1)
    # =========================================================================

    class Entry:
        """Entry-related protocols."""

        @runtime_checkable
        class EntryProtocol(Protocol):
            """Protocol for entry objects - read-only."""

            @property
            def dn(self) -> str:
                """Distinguished name."""
                ...

            @property
            def attributes(self) -> Mapping[str, Sequence[str]]:
                """Entry attributes as immutable mapping."""
                ...

            def to_dict(self) -> Mapping[str, FlextTypes.FlexibleValue]:
                """Convert to dictionary representation."""
                ...

            def to_ldif(self) -> str:
                """Convert to LDIF format."""
                ...

        @runtime_checkable
        class MutableEntryProtocol(EntryProtocol, Protocol):
            """Protocol for mutable entry objects."""

            def set_attribute(
                self,
                name: str,
                values: Sequence[str],
            ) -> FlextProtocols.Entry.EntryProtocol:
                """Set attribute values, returning self for chaining."""
                ...

            def add_attribute(
                self,
                name: str,
                values: Sequence[str],
            ) -> FlextProtocols.Entry.EntryProtocol:
                """Add attribute values, returning self for chaining."""
                ...

            def remove_attribute(self, name: str) -> FlextProtocols.Entry.EntryProtocol:
                """Remove attribute, returning self for chaining."""
                ...


__all__ = ["FlextProtocols"]
