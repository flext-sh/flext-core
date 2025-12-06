"""Runtime-checkable protocols that describe FLEXT public contracts.

Define ``p`` as the structural typing surface for dispatcher
handlers, services, results, and utilities. All protocols are
``@runtime_checkable`` so integration points can be verified without
inheritance, keeping layers decoupled while remaining type-safe.

Organized in hierarchical namespaces following SOLID principles:
- Foundation: Core protocols (Result, Model)
- Configuration: Config protocols
- Context: Context management protocols
- Container: Dependency injection protocols
- Domain: Domain-specific protocols (Service, Repository, Validation)
- Application: Application layer protocols (Handler, CommandBus, Processor)
- Infrastructure: Infrastructure protocols (Logger, Connection, Metadata)
- Utility: Utility protocols (Callable)
- Specialized: Specialized domain protocols (Entry for LDIF)

All protocols use types from t (flext_core.typings) to maintain
single source of truth.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import ModuleType
from typing import ParamSpec, Protocol, runtime_checkable

from pydantic import BaseModel
from structlog.typing import BindableLogger

from flext_core.typings import T_co, t

P_HandlerFunc = ParamSpec("P_HandlerFunc")


class FlextProtocols:
    """Hierarchical protocol definitions organized by SOLID principles.

    All protocols are organized in namespaces following the Interface
    Segregation Principle, allowing clients to depend only on the protocols
    they actually need.

    **Namespace Structure:**

    - **Foundation**: Core protocols used throughout the system
      (Result, Model)

    - **Configuration**: Configuration and component setup protocols
      (Config, Configurable)

    - **Context**: Context management and execution context protocols
      (Context)

    - **Container**: Dependency injection container protocols
      (Container)

    - **Domain**: Domain-specific business logic protocols
      (Service, Repository, Validation)

    - **Application**: Application layer protocols for handlers and processing
      (Handler, CommandBus, Middleware, Processor)

    - **Infrastructure**: Infrastructure protocols for external systems
      (Logger, Connection, Metadata)

    - **Utility**: Utility protocols for common patterns
      (Callable)

    - **Specialized**: Specialized domain protocols
      (Entry for LDIF processing)

    **Usage:**

    ```python
    from flext_core.protocols import p

    # Foundation protocols
    result: p.Foundation.Result[str]
    model: p.Foundation.HasModelDump

    # Domain protocols
    service: p.Domain.Service[str]
    repository: p.Domain.Repository[Entity]

    # Application protocols
    handler: p.Application.Handler
    command_bus: p.Application.CommandBus
    ```

    **Composition:**

    Protocols are designed to be composed:
    - Container.DI extends Configurable
    - Foundation.Model extends HasModelDump
    - Infrastructure.Logger.StructlogLogger extends BindableLogger

    This follows the Interface Segregation Principle, keeping protocols
    focused and composable.
    """

    # =========================================================================
    # FOUNDATION: Core Protocols
    # =========================================================================

    class Foundation:
        """Foundation protocols used throughout the system."""

        @runtime_checkable
        class Result[T](Protocol):
            """Result type interface for railway-oriented programming.

            Used extensively for all operations that can fail. Provides
            structural typing interface for FlextResult without circular imports.
            """

            @property
            def value(self) -> T:
                """Result value (available on success)."""
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
                """Error message (available on failure)."""
                ...

            @property
            def error_code(self) -> str | None:
                """Error code for categorization."""
                ...

            def unwrap(self) -> T:
                """Unwrap success value (raises on failure)."""
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

            def validate(self) -> FlextProtocols.Foundation.Result[bool]:
                """Validate model."""
                ...

    # =========================================================================
    # CONFIGURATION: Configuration Protocols
    # =========================================================================

    class Configuration:
        """Configuration and component setup protocols."""

        @runtime_checkable
        class Configurable(Protocol):
            """Protocol for component configuration."""

            def configure(self, config: Mapping[str, t.FlexibleValue]) -> None:
                """Configure component with settings."""
                ...

        @runtime_checkable
        class Config(Protocol):
            """Configuration object protocol based on Pydantic BaseSettings pattern.

            Reflects real implementations like FlextConfig which uses Pydantic BaseSettings.
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
            ) -> FlextProtocols.Configuration.Config:
                """Clone configuration with optional updates (Pydantic standard method)."""
                ...

    # =========================================================================
    # CONTEXT: Context Management Protocols
    # =========================================================================

    class Context:
        """Context management and execution context protocols."""

        @runtime_checkable
        class Ctx(Protocol):
            """Execution context protocol with cloning semantics."""

            def clone(self) -> FlextProtocols.Context.Ctx:
                """Clone context for isolated execution."""
                ...

            def set(
                self,
                key: str,
                value: t.GeneralValueType,
                scope: str = ...,
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Set a context value."""
                ...

            def get(
                self,
                key: str,
                scope: str = ...,
            ) -> FlextProtocols.Foundation.Result[t.GeneralValueType]:
                """Get a context value."""
                ...

    # =========================================================================
    # CONTAINER: Dependency Injection Protocols
    # =========================================================================

    class Container:
        """Dependency injection container protocols."""

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
            def config(self) -> FlextProtocols.Configuration.Config:
                """Configuration bound to the container."""
                ...

            @property
            def context(self) -> FlextProtocols.Context.Ctx:
                """Execution context bound to the container."""
                ...

            def scoped(
                self,
                *,
                config: FlextProtocols.Configuration.Config | None = None,
                context: FlextProtocols.Context.Ctx | None = None,
                subproject: str | None = None,
                services: Mapping[
                    str,
                    t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType],
                ]
                | None = None,
                factories: Mapping[
                    str,
                    Callable[
                        [],
                        (
                            t.ScalarValue
                            | Sequence[t.ScalarValue]
                            | Mapping[str, t.ScalarValue]
                        ),
                    ],
                ]
                | None = None,
                resources: Mapping[str, Callable[[], t.GeneralValueType]] | None = None,
            ) -> FlextProtocols.Container.DI:
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

            def get_config(self) -> t.Types.ConfigurationMapping:
                """Return the merged configuration exposed by this container."""
                ...

            def has_service(self, name: str) -> bool:
                """Check if a service is registered."""
                ...

            def register(
                self,
                name: str,
                service: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Register a service instance."""
                ...

            def register_factory(
                self,
                name: str,
                factory: Callable[[], t.GeneralValueType],
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Register a service factory."""
                ...

            def with_service(
                self,
                name: str,
                service: t.GeneralValueType,
            ) -> FlextProtocols.Container.DI:
                """Fluent interface for service registration."""
                ...

            def with_factory(
                self,
                name: str,
                factory: Callable[[], t.GeneralValueType],
            ) -> FlextProtocols.Container.DI:
                """Fluent interface for factory registration."""
                ...

            def get[T](self, name: str) -> FlextProtocols.Foundation.Result[T]:
                """Get service by name with type safety.

                Reflects real implementations like FlextContainer which uses
                generic type parameter for type-safe resolution.
                Returns r[T] directly (not wrapped in another Result).
                """
                ...

            def get_typed[T](
                self,
                name: str,
                type_cls: type[T],
            ) -> FlextProtocols.Foundation.Result[T]:
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

    class Domain:
        """Domain-specific business logic protocols."""

        @runtime_checkable
        class Service[T](Protocol):
            """Base domain service interface.

            Reflects real implementations like FlextService which executes
            domain logic without requiring command parameters (services are
            self-contained with their own configuration).
            """

            def execute(self) -> FlextProtocols.Foundation.Result[T]:
                """Execute domain service logic.

                Reflects real implementations like FlextService which don't
                require command parameters - services are self-contained with
                their own configuration and context.
                """
                ...

            def validate_business_rules(self) -> FlextProtocols.Foundation.Result[bool]:
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
            ) -> FlextProtocols.Foundation.Result[T]:
                """Get entity by ID."""
                ...

            def save(
                self,
                entity: T,
            ) -> FlextProtocols.Foundation.Result[T]:
                """Save entity."""
                ...

            def delete(
                self,
                entity_id: str,
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Delete entity."""
                ...

            def find_all(
                self,
            ) -> FlextProtocols.Foundation.Result[Sequence[T]]:
                """Find all entities."""
                ...

        class Validation:
            """Validation protocols for domain rules."""

            @runtime_checkable
            class HasInvariants(Protocol):
                """Protocol for DDD aggregate invariant checking.

                Reflects real implementations like FlextModelsEntity.AggregateRoot
                which checks invariants and raises exceptions on violation rather
                than returning Result[bool].
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

    class Application:
        """Application layer protocols for handlers and processing."""

        @runtime_checkable
        class Handler(Protocol):
            """Command/Query handler interface.

            Reflects real implementations like FlextHandlers which provide
            comprehensive validation and execution pipelines for CQRS handlers.
            """

            def handle(
                self,
                message: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[t.GeneralValueType]:
                """Handle message - core business logic method.

                Reflects real implementations like FlextHandlers.handle() which
                executes handler business logic for commands, queries, or events.
                """
                ...

            def validate(
                self,
                data: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Validate input data using extensible validation pipeline.

                Reflects real implementations like FlextHandlers.validate() which
                performs base validation that can be overridden by subclasses.
                """
                ...

            def validate_command(
                self,
                command: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[bool]:
                """Validate command message with command-specific rules.

                Reflects real implementations like FlextHandlers.validate_command()
                which delegates to validate() by default but can be overridden.
                """
                ...

            def validate_query(
                self,
                query: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[bool]:
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
                request: t.GeneralValueType | BaseModel,
                handler: t.GeneralValueType | None = None,
            ) -> FlextProtocols.Foundation.Result[t.Types.ConfigurationMapping]:
                """Register handler dynamically.

                Reflects real implementations like FlextDispatcher that accept
                dict, Pydantic model, or handler object for registration.
                Returns ConfigurationMapping with registration details.
                """
                ...

            def register_command[TCommand, TResult](
                self,
                command_type: type[TCommand],
                handler: t.GeneralValueType,
                *,
                handler_config: Mapping[str, t.FlexibleValue] | None = None,
            ) -> FlextProtocols.Foundation.Result[t.GeneralValueType]:
                """Register command handler."""
                ...

            @staticmethod
            def create_handler_from_function(
                handler_func: Callable[P_HandlerFunc, t.GeneralValueType],
                handler_config: Mapping[str, t.FlexibleValue] | None = None,
                mode: str = ...,
            ) -> FlextProtocols.Foundation.Result[FlextProtocols.Application.Handler]:
                """Create handler from function (static method)."""
                ...

            def execute(
                self,
                command: t.FlexibleValue,
            ) -> FlextProtocols.Foundation.Result[T_co]:
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
            ) -> FlextProtocols.Foundation.Result[t.GeneralValueType]:
                """Dispatch message (primary method for real implementations).

                Reflects real implementations like FlextDispatcher which provides
                flexible dispatch accepting message objects or (type, data) tuples.
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
                    FlextProtocols.Foundation.Result[TResult],
                ],
            ) -> FlextProtocols.Foundation.Result[TResult]:
                """Process command."""
                ...

        @runtime_checkable
        class Processor(Protocol):
            """Processor interface for data transformation pipelines.

            Processors can be objects with a process() method that takes data
            and returns a result (which will be normalized to Foundation.Result).
            Accepts GeneralValueType, BaseModel, or Foundation.Result for processing.

            The return type is flexible to support:
            - Direct values (t.GeneralValueType)
            - BaseModel instances (Pydantic models)
            - Foundation.Result instances (structural typing)
            - Objects with is_success/is_failure properties (FlextResult compatibility)
            """

            def process(
                self,
                data: (
                    t.GeneralValueType
                    | BaseModel
                    | FlextProtocols.Foundation.Result[t.GeneralValueType]
                ),
            ) -> (
                t.GeneralValueType
                | BaseModel
                | FlextProtocols.Foundation.Result[t.GeneralValueType]
            ):
                """Process data and return result.

                Returns can be:
                - Direct value (t.GeneralValueType)
                - BaseModel instance (Pydantic model)
                - Foundation.Result (structural typing compatible)
                """
                ...

    # =========================================================================
    # INFRASTRUCTURE: Infrastructure Protocols
    # =========================================================================

    class Infrastructure:
        """Infrastructure protocols for external systems."""

        class Logger:
            """Logging protocols."""

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
                ) -> FlextProtocols.Foundation.Result[bool] | None:
                    """Debug log."""
                    ...

                def info(
                    self,
                    message: str,
                    *args: t.FlexibleValue,
                    return_result: bool = False,
                    **context: t.FlexibleValue,
                ) -> FlextProtocols.Foundation.Result[bool] | None:
                    """Info log."""
                    ...

                def warning(
                    self,
                    message: str,
                    *args: t.FlexibleValue,
                    return_result: bool = False,
                    **context: t.FlexibleValue,
                ) -> FlextProtocols.Foundation.Result[bool] | None:
                    """Warning log."""
                    ...

                def error(
                    self,
                    message: str,
                    *args: t.FlexibleValue,
                    return_result: bool = False,
                    **context: t.FlexibleValue,
                ) -> FlextProtocols.Foundation.Result[bool] | None:
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
                ) -> FlextProtocols.Foundation.Result[bool] | None:
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
            ) -> FlextProtocols.Foundation.Result[bool]:
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
            def attributes(self) -> Mapping[str, t.GeneralValueType]:
                """Metadata attributes."""
                ...

    # =========================================================================
    # UTILITY: Utility Protocols
    # =========================================================================

    class Utility:
        """Utility protocols for common patterns."""

        @runtime_checkable
        class Callable(Protocol[T_co]):
            """Protocol for variadic callables returning T_co.

            Used for flexible function signatures that accept any arguments.
            """

            def __call__(
                self,
                *args: t.GeneralValueType,
                **kwargs: t.GeneralValueType,
            ) -> T_co:
                """Call the function with any arguments, returning T_co."""
                ...

    # =========================================================================
    # SPECIALIZED: Specialized Domain Protocols
    # =========================================================================

    class Specialized:
        """Specialized domain protocols for specific use cases."""

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
                ) -> FlextProtocols.Specialized.Entry.Mutable:
                    """Set attribute values, returning self for chaining."""
                    ...

                def add_attribute(
                    self,
                    name: str,
                    values: Sequence[str],
                ) -> FlextProtocols.Specialized.Entry.Mutable:
                    """Add attribute values, returning self for chaining."""
                    ...

                def remove_attribute(
                    self,
                    name: str,
                ) -> FlextProtocols.Specialized.Entry.Mutable:
                    """Remove attribute, returning self for chaining."""
                    ...

    # =========================================================================
    # ROOT-LEVEL ALIASES (Minimize nesting for common protocols)
    # =========================================================================
    # These aliases provide direct access to commonly used protocols without
    # requiring nested namespace traversal. Both access patterns work:
    #   - p.Result[T]  (new, concise)
    #   - p.Foundation.Result[T]  (old, still works for backward compatibility)

    # Foundation protocols (most commonly used)
    Result = Foundation.Result
    ResultLike = Foundation.ResultLike
    HasModelDump = Foundation.HasModelDump
    HasModelFields = Foundation.HasModelFields
    Model = Foundation.Model

    # Configuration protocols
    Configurable = Configuration.Configurable
    Config = Configuration.Config

    # Context protocols
    Ctx = Context.Ctx

    # Container protocols
    DI = Container.DI

    # Domain protocols
    Service = Domain.Service
    Repository = Domain.Repository
    HasInvariants = Domain.Validation.HasInvariants

    # Application protocols
    Handler = Application.Handler
    CommandBus = Application.CommandBus
    Middleware = Application.Middleware
    Processor = Application.Processor

    # Infrastructure protocols
    Log = Infrastructure.Logger.Log
    StructlogLogger = Infrastructure.Logger.StructlogLogger
    Connection = Infrastructure.Connection
    Metadata = Infrastructure.Metadata
    MetadataProtocol = Infrastructure.Metadata  # Backward compat alias

    # Utility protocols
    VariadicCallable = Utility.Callable

    # Specialized protocols
    Entry = Specialized.Entry.Base
    MutableEntry = Specialized.Entry.Mutable


# Alias for simplified usage
p = FlextProtocols

__all__ = [
    "FlextProtocols",
    "p",
]
