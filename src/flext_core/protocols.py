"""Runtime-checkable structural typing protocols for FLEXT framework.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import ModuleType, TracebackType
from typing import ParamSpec, Protocol, Self, runtime_checkable

from pydantic import BaseModel
from structlog.typing import BindableLogger

from flext_core.typings import T_co, t

P_HandlerFunc = ParamSpec("P_HandlerFunc")


class FlextProtocols:
    """Hierarchical protocol namespace organized by Interface Segregation Principle."""

    # =========================================================================
    # CORE RESULT PROTOCOL (Root Level for Self-Reference)
    # =========================================================================

    @runtime_checkable
    class Result[T](Protocol):
        """Result type interface for railway-oriented programming.

        Used extensively for all operations that can fail. Provides
        structural typing interface for FlextResult without circular imports.
        Fully compatible with FlextResult and FlextRuntime usage patterns.

        Defined at root level to allow self-referencing in method signatures
        (e.g., `def map[U](...) -> FlextProtocols.Result[U]`).
        """

        @property
        def value(self) -> T:
            """Result value (available on success, never None)."""
            ...

        @property
        def data(self) -> T:
            """Alias for value (backward compatibility)."""
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
            """Error message (available on failure, None on success)."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code for categorization."""
            ...

        @property
        def error_data(
            self,
        ) -> t.ConfigurationMapping | None:
            """Error metadata (optional)."""
            ...

        @property
        def result(self) -> object:
            """Access internal Result for advanced operations."""
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Unwrap success value or return default on failure."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocols.Result[U]:
            """Transform success value using function."""
            ...

        def flat_map[U](
            self,
            func: Callable[[T], FlextProtocols.Result[U]],
        ) -> FlextProtocols.Result[U]:
            """Chain operations returning Result."""
            ...

        def map_error(self, func: Callable[[str], str]) -> Self:
            """Transform error message on failure.

            Returns self on success, new Result with transformed error on failure.
            """
            ...

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> Self:
            """Filter success value using predicate.

            Returns self if predicate passes or result is failure,
            new failed Result if predicate fails.
            """
            ...

        def flow_through[U](
            self,
            *funcs: Callable[[T | U], FlextProtocols.Result[U]],
        ) -> FlextProtocols.Result[U]:
            """Chain multiple operations in a pipeline."""
            ...

        def alt(
            self,
            func: Callable[[str], str],
        ) -> Self:
            """Apply alternative function on failure."""
            ...

        def lash(
            self,
            func: Callable[[str], FlextProtocols.Result[T]],
        ) -> Self:
            """Apply recovery function on failure."""
            ...

        def to_maybe(self) -> object:
            """Convert to returns.maybe.Maybe."""
            ...

        def to_io(self) -> object:
            """Convert to returns.io.IO."""
            ...

        def to_io_result(self) -> object:
            """Convert to returns.io.IOResult.

            Returns IOFlextProtocols.Result[T, str] - success wraps value, failure wraps error.
            """
            ...

        @classmethod
        def from_io_result(
            cls,
            io_result: object,
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Create Result from returns.io.IOResult."""
            ...

        @classmethod
        def from_maybe(
            cls,
            maybe: object,
            error: str = "Value is Nothing",
        ) -> FlextProtocols.Result[T]:
            """Create Result from returns.maybe.Maybe."""
            ...

        @classmethod
        def create_from_callable(
            cls,
            func: Callable[[], T],
            error_code: str | None = None,
        ) -> FlextProtocols.Result[T]:
            """Create result from callable, catching exceptions."""
            ...

        @classmethod
        def traverse[TItem, UResult](
            cls,
            items: Sequence[TItem],
            func: Callable[[TItem], FlextProtocols.Result[UResult]],
            *,
            fail_fast: bool = True,
        ) -> FlextProtocols.Result[list[UResult]]:
            """Map over sequence with configurable failure handling."""
            ...

        @classmethod
        def accumulate_errors(
            cls,
            *results: FlextProtocols.Result[t.GeneralValueType],
        ) -> FlextProtocols.Result[list[t.GeneralValueType]]:
            """Collect all successes, fail if any failure."""
            ...

        @classmethod
        def with_resource[TResource](
            cls,
            factory: Callable[[], TResource],
            op: Callable[[TResource], FlextProtocols.Result[T]],
            cleanup: Callable[[TResource], None] | None = None,
        ) -> FlextProtocols.Result[T]:
            """Resource management with automatic cleanup."""
            ...

        def __enter__(self) -> Self:
            """Context manager entry."""
            ...

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Context manager exit."""
            ...

        def __repr__(self) -> str:
            """String representation."""
            ...

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
            ...

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            ...

    @runtime_checkable
    class ResultLike(Protocol[T_co]):
        """Result-like protocol for compatibility with FlextResult operations.

        Used for type compatibility when working with result-like objects.
        """

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
    class HasModelDump(Protocol):
        """Protocol for objects that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> Mapping[str, t.FlexibleValue]:
            """Dump model data to dictionary."""
            ...

    @runtime_checkable
    class HasModelFields(Protocol):
        """Protocol for objects with model fields.

        Extends HasModelDump with model fields access.
        Used for Pydantic model introspection.
        """

        def model_dump(self) -> Mapping[str, t.FlexibleValue]:
            """Dump model data to dictionary."""
            ...

        @property
        def model_fields(self) -> Mapping[str, t.FlexibleValue]:
            """Model fields mapping."""
            ...

    @runtime_checkable
    class Model(Protocol):
        """Model type interface for validation.

        Used for model validation without circular imports.
        """

        def model_dump(self) -> Mapping[str, t.FlexibleValue]:
            """Dump model data to dictionary."""
            ...

        def validate(self) -> FlextProtocols.Result[bool]:
            """Validate model."""
            ...

        # =========================================================================
        # CONFIGURATION: Configuration Protocols
        # =========================================================================

    @runtime_checkable
    class Configurable(Protocol):
        """Protocol for component configuration."""

        def configure(self, config: Mapping[str, t.FlexibleValue]) -> None:
            """Configure component with settings."""
            ...

    @runtime_checkable
    class Config(Protocol):
        """Configuration object protocol based on Pydantic BaseSettings pattern.

        Reflects real implementations like FlextSettings which uses Pydantic BaseSettings.
        Configuration objects use direct field access (Pydantic standard) rather than
        explicit get/set methods. Supports cloning via model_copy() and optional
        override methods.
        """

        # Required fields (access as attributes, not necessarily properties)
        app_name: str
        """Application name bound to the configuration."""

        version: str
        """Semantic version of the running application."""

        def model_copy(
            self,
            *,
            update: Mapping[str, t.FlexibleValue] | None = None,
            deep: bool = False,
        ) -> Self:
            """Clone configuration with optional updates (Pydantic standard method)."""
            ...

    # =========================================================================
    # CONTEXT: Context Management Protocols
    # =========================================================================

    @runtime_checkable
    class Ctx(Protocol):
        """Execution context protocol with cloning semantics.

        Uses ResultLike (covariant) for return types to allow implementations
        to return FlextResult while protocol defines the interface.
        """

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def set(
            self,
            key: str,
            value: t.GeneralValueType,
            scope: str = ...,
        ) -> FlextProtocols.ResultLike[bool]:
            """Set a context value."""
            ...

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> FlextProtocols.ResultLike[t.GeneralValueType]:
            """Get a context value."""
            ...

    # =========================================================================
    # CONTAINER: Dependency Injection Protocols
    # =========================================================================

    @runtime_checkable
    class DI(Protocol):
        """Dependency injection container protocol.

        Extends Configurable to allow container configuration.
        Implements configure() method from Configurable protocol.
        """

        def configure(self, config: Mapping[str, t.FlexibleValue]) -> None:
            """Configure component with settings (from Configurable protocol)."""
            ...

        @property
        def config(self) -> FlextProtocols.Config:
            """Configuration bound to the container."""
            ...

        @property
        def context(self) -> FlextProtocols.Ctx:
            """Execution context bound to the container."""
            ...

        def scoped(
            self,
            *,
            config: FlextProtocols.Config | None = None,
            context: FlextProtocols.Ctx | None = None,
            subproject: str | None = None,
            services: Mapping[str, t.GeneralValueType] | None = None,
            factories: Mapping[str, t.FactoryCallable] | None = None,
            resources: Mapping[str, t.ResourceCallable] | None = None,
        ) -> Self:
            """Create an isolated container scope with optional overrides."""
            ...

        def wire_modules(
            self,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: Sequence[str] | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules/packages to the DI bridge for @inject/Provide usage."""
            ...

        def get_config(self) -> t.ConfigurationMapping:
            """Return the merged configuration exposed by this container."""
            ...

        def has_service(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def register(
            self,
            name: str,
            service: t.GeneralValueType,
        ) -> FlextProtocols.ResultLike[bool]:
            """Register a service instance."""
            ...

        def register_factory(
            self,
            name: str,
            factory: Callable[[], t.GeneralValueType],
        ) -> FlextProtocols.ResultLike[bool]:
            """Register a service factory."""
            ...

        def with_service(
            self,
            name: str,
            service: t.GeneralValueType,
        ) -> Self:
            """Fluent interface for service registration."""
            ...

        def with_factory(
            self,
            name: str,
            factory: Callable[[], t.GeneralValueType],
        ) -> Self:
            """Fluent interface for factory registration."""
            ...

        def get(self, name: str) -> FlextProtocols.ResultLike[t.GeneralValueType]:
            """Get service by name.

            Returns the resolved service as GeneralValueType. For type-safe
            resolution with runtime validation, use get_typed[T](name, type_cls).
            """
            ...

        def get_typed[T](
            self,
            name: str,
            type_cls: type[T],
        ) -> FlextProtocols.ResultLike[T]:
            """Get service with type safety and runtime validation.

            Reflects real implementations like FlextContainer.get_typed()
            which validates runtime type after resolution.
            """
            ...

        def list_services(self) -> Sequence[str]:
            """List all registered services."""
            ...

        def clear_all(self) -> None:
            """Clear all services and factories."""
            ...

    # =========================================================================
    # DOMAIN: Domain-Specific Protocols
    # =========================================================================

    @runtime_checkable
    class Service[T](Protocol):
        """Base domain service interface.

        Reflects real implementations like FlextService which executes
        domain logic without requiring command parameters (services are
        self-contained with their own configuration).
        """

        def execute(self) -> FlextProtocols.Result[T]:
            """Execute domain service logic.

            Reflects real implementations like FlextService which don't
            require command parameters - services are self-contained with
            their own configuration and context.
            """
            ...

        def validate_business_rules(self) -> FlextProtocols.Result[bool]:
            """Validate business rules with extensible validation pipeline.

            Reflects real implementations like FlextService which perform
            business rule validation without external command parameters.
            """
            ...

        def is_valid(self) -> bool:
            """Check if service is in valid state for execution.

            Reflects real implementations like FlextService which check
            validity based on internal state and business rules.
            """
            ...

        def get_service_info(self) -> Mapping[str, t.FlexibleValue]:
            """Get service metadata and configuration information.

            Reflects real implementations like FlextService which provide
            service metadata for observability and debugging.
            """
            ...

    @runtime_checkable
    class Repository[T](Protocol):
        """Data access interface."""

        def get_by_id(
            self,
            entity_id: str,
        ) -> FlextProtocols.Result[T]:
            """Get entity by ID."""
            ...

        def save(
            self,
            entity: T,
        ) -> FlextProtocols.Result[T]:
            """Save entity."""
            ...

        def delete(
            self,
            entity_id: str,
        ) -> FlextProtocols.Result[bool]:
            """Delete entity."""
            ...

        def find_all(
            self,
        ) -> FlextProtocols.Result[Sequence[T]]:
            """Find all entities."""
            ...

    class Validation:
        """Validation protocols for domain rules."""

        @runtime_checkable
        class HasInvariants(Protocol):
            """Protocol for DDD aggregate invariant checking.

            Reflects real implementations like FlextModelsEntity.AggregateRoot
            which checks invariants and raises exceptions on violation rather
            than returning FlextProtocols.Result[bool].
            """

            def check_invariants(self) -> None:
                """Check invariants, raising exception on violation.

                Reflects real implementations like FlextModelsEntity.AggregateRoot
                which raises ValidationError when invariants are violated.
                """
                ...

    # =========================================================================
    # APPLICATION: Application Layer Protocols
    # =========================================================================

    @runtime_checkable
    class Handler(Protocol):
        """Command/Query handler interface.

        Reflects real implementations like FlextHandlers which provide
        comprehensive validation and execution pipelines for CQRS handlers.
        """

        def handle(
            self,
            message: t.FlexibleValue,
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Handle message - core business logic method.

            Reflects real implementations like FlextHandlers.handle() which
            executes handler business logic for commands, queries, or events.
            """
            ...

        def validate(
            self,
            data: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool]:
            """Validate input data using extensible validation pipeline.

            Reflects real implementations like FlextHandlers.validate() which
            performs base validation that can be overridden by subclasses.
            """
            ...

        def validate_command(
            self,
            command: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool]:
            """Validate command message with command-specific rules.

            Reflects real implementations like FlextHandlers.validate_command()
            which delegates to validate() by default but can be overridden.
            """
            ...

        def validate_query(
            self,
            query: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool]:
            """Validate query message with query-specific rules.

            Reflects real implementations like FlextHandlers.validate_query()
            which delegates to validate() by default but can be overridden.
            """
            ...

        def can_handle(self, message_type: type[object]) -> bool:
            """Check if handler can handle the specified message type.

            Reflects real implementations like FlextHandlers.can_handle() which
            checks message type compatibility using duck typing and class hierarchy.
            """
            ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Command routing and execution protocol.

        Reflects real implementations like FlextDispatcher which provides
        CQRS routing, handler registration, and execution with context
        propagation and reliability controls.
        """

        def register_handler(
            self,
            request: t.HandlerType | t.GeneralValueType | BaseModel | object,
            handler: t.GeneralValueType | None = None,
        ) -> FlextProtocols.Result[t.ConfigurationMapping]:
            """Register handler dynamically.

            Reflects real implementations like FlextDispatcher that accept
            dict, Pydantic model, handler callable, handler object (FlextHandlers),
            or any object for registration. Uses object to allow handler instances
            that can't be referenced directly from protocol definitions.
            Returns ConfigurationMapping with registration details.
            """
            ...

        def register_command[TCommand, TResult](
            self,
            command_type: type[TCommand],
            handler: t.GeneralValueType,
            *,
            handler_config: Mapping[str, t.FlexibleValue] | None = None,
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Register command handler."""
            ...

        @staticmethod
        def create_handler_from_function(
            handler_func: Callable[P_HandlerFunc, t.GeneralValueType],
            handler_config: Mapping[str, t.FlexibleValue] | None = None,
            mode: str = ...,
        ) -> FlextProtocols.Result[FlextProtocols.Handler]:
            """Create handler from function (static method)."""
            ...

        def execute(
            self,
            command: t.FlexibleValue,
        ) -> FlextProtocols.Result[
            t.GeneralValueType
        ]:  # Use t.GeneralValueType instead of T_co
            """Execute command."""
            ...

        def dispatch(
            self,
            message_or_type: t.GeneralValueType,
            data: t.GeneralValueType | None = None,
            *,
            config: t.GeneralValueType | None = None,
            metadata: t.GeneralValueType | None = None,
            correlation_id: str | None = None,
            timeout_override: int | None = None,
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Dispatch message (primary method for real implementations).

            Reflects real implementations like FlextDispatcher which provides
            flexible dispatch accepting message objects or (type, data) tuples.
            """
            ...

    @runtime_checkable
    class Registry(Protocol):
        """Handler registry protocol for CQRS handler registration.

        Reflects real implementations like FlextRegistry which provides
        handler registration, batch operations, and idempotent tracking
        for CQRS handlers.
        """

        def register_handler(
            self,
            handler: t.GeneralValueType | None,
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Register a handler instance.

            Reflects real implementations like FlextRegistry.register_handler()
            which registers handlers with idempotent tracking.
            """
            ...

        def register_handlers(
            self,
            handlers: Sequence[t.GeneralValueType],
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Register multiple handlers in batch.

            Reflects real implementations like FlextRegistry.register_handlers()
            which provides batch registration with summary reporting.
            """
            ...

        def register_bindings(
            self,
            bindings: Mapping[type[t.GeneralValueType], t.GeneralValueType],
        ) -> FlextProtocols.Result[t.GeneralValueType]:
            """Register message-to-handler bindings.

            Reflects real implementations like FlextRegistry.register_bindings()
            which maps message types to handlers.
            """
            ...

        @classmethod
        def create(
            cls,
            dispatcher: FlextProtocols.CommandBus | None = None,
            *,
            auto_discover_handlers: bool = False,
        ) -> Self:
            """Factory method to create a registry instance.

            Reflects real implementations like FlextRegistry.create()
            which provides zero-config handler registration with auto-discovery.
            """
            ...

    @runtime_checkable
    class Middleware(Protocol):
        """Processing pipeline middleware."""

        def process[TResult](
            self,
            command: t.FlexibleValue,
            next_handler: Callable[
                [t.FlexibleValue],
                FlextProtocols.Result[TResult],
            ],
        ) -> FlextProtocols.Result[TResult]:
            """Process command."""
            ...

    @runtime_checkable
    class Processor(Protocol):
        """Processor interface for data transformation pipelines.

        Processors can be objects with a process() method that takes data
        and returns a result (which will be normalized to Result).
        Accepts t.GeneralValueType, BaseModel, or Result for processing.

        The return type is flexible to support:
        - Direct values (t.GeneralValueType)
        - BaseModel instances (Pydantic models)
        - Result instances (structural typing)
        - Objects with is_success/is_failure properties (FlextResult compatibility)
        """

        def process(
            self,
            data: (
                t.GeneralValueType
                | BaseModel
                | FlextProtocols.Result[t.GeneralValueType]
            ),
        ) -> t.GeneralValueType | BaseModel | FlextProtocols.Result[t.GeneralValueType]:
            """Process data and return result.

            Returns can be:
            - Direct value (t.GeneralValueType)
            - BaseModel instance (Pydantic model)
            - Result (structural typing compatible)
            """
            ...

    # =========================================================================
    # INFRASTRUCTURE: Infrastructure Protocols
    # =========================================================================

    @runtime_checkable
    class Log(Protocol):
        """Logging interface protocol."""

        def log(
            self,
            level: str,
            message: str,
            _context: Mapping[str, t.FlexibleValue] | None = None,
        ) -> None:
            """Log message."""
            ...

        def debug(
            self,
            message: str,
            *args: t.FlexibleValue,
            return_result: bool = False,
            **context: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool] | None:
            """Debug log."""
            ...

        def info(
            self,
            message: str,
            *args: t.FlexibleValue,
            return_result: bool = False,
            **context: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool] | None:
            """Info log."""
            ...

        def warning(
            self,
            message: str,
            *args: t.FlexibleValue,
            return_result: bool = False,
            **context: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool] | None:
            """Warning log."""
            ...

        def error(
            self,
            message: str,
            *args: t.FlexibleValue,
            return_result: bool = False,
            **context: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool] | None:
            """Error log."""
            ...

        def exception(
            self,
            message: str,
            *,
            exception: BaseException | None = None,
            exc_info: bool = True,
            return_result: bool = False,
            **kwargs: t.FlexibleValue,
        ) -> FlextProtocols.Result[bool] | None:
            """Exception log."""
            ...

        @runtime_checkable
        class StructlogLogger(BindableLogger, Protocol):
            """Protocol for structlog logger with all logging methods.

            Extends BindableLogger to add explicit method signatures for
            logging methods (debug, info, warning, error, etc.) that are
            available via __getattr__ at runtime.

            Structlog loggers implement this protocol through dynamic
            method dispatch.
            """

            def debug(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType | Exception,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log debug message."""
                ...

            def info(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log info message."""
                ...

            def warning(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log warning message."""
                ...

            def warn(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType,
            ) -> None:
                """Log warning message (alias)."""
                ...

            def error(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log error message."""
                ...

            def critical(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log critical message."""
                ...

            def exception(
                self,
                msg: str | t.GeneralValueType,
                *args: t.GeneralValueType,
                **kw: t.GeneralValueType | Exception,
            ) -> None:
                """Log exception with traceback."""
                ...

        @runtime_checkable
        class Connection(Protocol):
            """External system connection protocol."""

            def test_connection(
                self,
            ) -> FlextProtocols.Result[bool]:
                """Test connection."""
                ...

            def get_connection_string(self) -> str:
                """Get connection string."""
                ...

            def close_connection(self) -> None:
                """Close connection."""
                ...

        @runtime_checkable
        class Metadata(Protocol):
            """Metadata object protocol."""

            @property
            def created_at(self) -> datetime:
                """Creation timestamp."""
                ...

            @property
            def updated_at(self) -> datetime:
                """Update timestamp."""
                ...

            @property
            def version(self) -> str:
                """Version string."""
                ...

            @property
            def attributes(self) -> t.ConfigurationMapping:
                """Metadata attributes."""
                ...

    # =========================================================================
    # UTILITY: Utility Protocols
    # =========================================================================

    @runtime_checkable
    class VariadicCallable(Protocol[T_co]):
        """Protocol for variadic callables returning T_co.

        Used for flexible function signatures that accept any arguments.
        Accepts *args and **kwargs, making it suitable for services, handlers,
        factories, and callbacks.
        """

        def __call__(
            self,
            *args: t.GeneralValueType,
            **kwargs: t.GeneralValueType,
        ) -> T_co:
            """Call the function with any arguments, returning T_co."""
            ...

    @runtime_checkable
    class ValidatorSpec(Protocol):
        """Protocol for validator specifications with operator composition.

        Validators implement __call__ to validate values and support composition
        via __and__ (both must pass), __or__ (either passes), and __invert__ (negation).

        Example:
            validator = V.string.non_empty & V.string.max_length(100)
            is_valid = validator("hello")  # True

        """

        def __call__(self, value: object) -> bool:
            """Validate value, return True if valid."""
            ...

        def __and__(
            self,
            other: FlextProtocols.ValidatorSpec,
        ) -> FlextProtocols.ValidatorSpec:
            """Compose with AND - both validators must pass."""
            ...

        def __or__(
            self,
            other: FlextProtocols.ValidatorSpec,
        ) -> FlextProtocols.ValidatorSpec:
            """Compose with OR - at least one validator must pass."""
            ...

        def __invert__(self) -> FlextProtocols.ValidatorSpec:
            """Negate validator - passes when original fails."""
            ...

    # =========================================================================
    # SPECIALIZED: Specialized Domain Protocols
    # =========================================================================

    class Entry:
        """Entry-related protocols for LDIF processing."""

        @runtime_checkable
        class Base(Protocol):
            """Entry object protocol (read-only)."""

            @property
            def dn(self) -> str:
                """Distinguished name."""
                ...

            @property
            def attributes(self) -> Mapping[str, Sequence[str]]:
                """Entry attributes as immutable mapping."""
                ...

            def to_dict(self) -> Mapping[str, t.FlexibleValue]:
                """Convert to dictionary representation."""
                ...

            def to_ldif(self) -> str:
                """Convert to LDIF format."""
                ...

        @runtime_checkable
        class Mutable(Base, Protocol):
            """Mutable entry object protocol.

            Extends Base with mutation methods.
            """

            def set_attribute(
                self,
                name: str,
                values: Sequence[str],
            ) -> FlextProtocols.Entry.Mutable:
                """Set attribute values, returning self for chaining."""
                ...

            def add_attribute(
                self,
                name: str,
                values: Sequence[str],
            ) -> FlextProtocols.Entry.Mutable:
                """Add attribute values, returning self for chaining."""
                ...

            def remove_attribute(
                self,
                name: str,
            ) -> FlextProtocols.Entry.Mutable:
                """Remove attribute, returning self for chaining."""
                ...

    # =========================================================================
    # MAPPER PROTOCOLS (For Collection Operations)
    # =========================================================================

    @runtime_checkable
    class SingleValueMapper[T, R](Protocol):
        """Protocol for mappers that transform single values."""

        def __call__(self, value: T) -> R:
            """Map a single value to a result."""
            ...

    @runtime_checkable
    class KeyValueMapper[T, R](Protocol):
        """Protocol for mappers that transform key-value pairs."""

        def __call__(self, key: str, value: T) -> R:
            """Map a key-value pair to a result."""
            ...

    # =========================================================================
    # UTILITIES PROTOCOLS
    # =========================================================================

    class Utilities:
        """Protocols for utility operations."""

        @runtime_checkable
        class CallableWithHints(Protocol):
            """Protocol for callables that support type hints introspection."""

            __annotations__: t.ConfigurationDict

    # =========================================================================


p = FlextProtocols
fc = FlextProtocols

# Export nested protocols for direct import compatibility
VariadicCallable = FlextProtocols.VariadicCallable
ValidatorSpec = FlextProtocols.ValidatorSpec

__all__ = ["FlextProtocols", "ValidatorSpec", "VariadicCallable", "fc", "p"]
