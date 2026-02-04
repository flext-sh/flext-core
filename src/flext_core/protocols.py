"""Runtime-checkable structural typing protocols for FLEXT framework.

Copyright (t) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import ModuleType, TracebackType
from typing import (
    Protocol,
    Self,
    TypedDict,
    _ProtocolMeta,  # noqa: PLC2701 - Required for metaclass resolution
    runtime_checkable,
)

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass  # noqa: PLC2701
from pydantic_settings import BaseSettings
from structlog.typing import BindableLogger

from flext_core.typings import P, T, t

# =============================================================================
# PROTOCOL DETECTION AND VALIDATION HELPERS (Module-level)
# =============================================================================


def _is_protocol(cls: type) -> bool:
    """Check if a class is a typing.Protocol.

    This function detects Protocol classes by checking for the _is_protocol
    attribute set by Python's typing module on Protocol classes.

    Args:
        cls: The class to check.

    Returns:
        True if cls is a Protocol, False otherwise.

    """
    # Check if cls has the Protocol marker attribute
    if not hasattr(cls, "_is_protocol"):
        return False
    # Get the attribute value safely
    is_proto = getattr(cls, "_is_protocol", False)
    # Handle both bool and MethodType (some Python versions)
    # Wrap callable result in bool() to ensure return type is always bool
    return bool(is_proto) if not callable(is_proto) else bool(is_proto())


def _validate_protocol_compliance(
    cls: type,
    protocol: type,
    class_name: str,
) -> None:
    """Validate that a class implements all required protocol members.

    This function checks structural typing compliance at class definition time,
    providing clear error messages if the protocol contract is not satisfied.

    For Pydantic models, fields are declared as annotations and may not be
    accessible via hasattr during metaclass processing. This function checks
    both hasattr AND class annotations (including inherited) for compliance.

    Args:
        cls: The class to validate.
        protocol: The protocol the class should implement.
        class_name: Name of the class (for error messages).

    Raises:
        TypeError: If the class doesn't implement required protocol members.

    """
    # Get protocol annotations (required members)
    protocol_annotations = getattr(protocol, "__annotations__", {})
    raw_attrs: object = getattr(protocol, "__protocol_attrs__", set())
    protocol_methods: set[str] = {
        x
        for x in (raw_attrs if isinstance(raw_attrs, set) else set())
        if isinstance(x, str)
    }

    # Build set of required members
    required_members: set[str] = set(protocol_annotations.keys())
    if protocol_methods:
        required_members.update(protocol_methods)

    # Filter out private attributes and Protocol internals
    required_members = {
        m for m in required_members if not m.startswith("_") or m.startswith("__")
    }

    # Collect all annotations from class and its bases (for Pydantic fields)
    all_annotations: set[str] = set()
    for base in cls.__mro__:
        base_annotations = getattr(base, "__annotations__", {})
        all_annotations.update(base_annotations.keys())

    # Check each required member (check hasattr OR annotations)
    def _has_member(member: str) -> bool:
        """Check if class has member via attribute or annotation."""
        return hasattr(cls, member) or member in all_annotations

    missing = [member for member in required_members if not _has_member(member)]

    if missing:
        protocol_name = getattr(protocol, "__name__", str(protocol))
        missing_str = ", ".join(sorted(missing))
        msg = (
            f"Class '{class_name}' does not implement required members "
            f"of protocol '{protocol_name}': {missing_str}"
        )
        raise TypeError(msg)


def _partition_protocol_bases(
    bases: tuple[type, ...],
) -> tuple[list[type], list[type]]:
    """Separate Protocol bases from regular class bases.

    This function partitions a tuple of base classes into two lists:
    - protocols: Classes that are typing.Protocol subclasses
    - model_bases: Regular classes (including Pydantic bases)

    Args:
        bases: Tuple of base classes from class definition.

    Returns:
        Tuple of (protocols, model_bases) lists.

    """
    protocols: list[type] = []
    model_bases: list[type] = []

    for base in bases:
        if _is_protocol(base):
            protocols.append(base)
        else:
            model_bases.append(base)

    return protocols, model_bases


def _get_class_protocols(cls: type) -> tuple[type, ...]:
    """Get the protocols a class implements.

    Args:
        cls: The class to check.

    Returns:
        Tuple of protocol types the class implements.

    """
    return getattr(cls, "__protocols__", ())


def _check_implements_protocol(instance: object, protocol: type) -> bool:
    """Check if an instance's class implements a protocol.

    This function checks both:
    1. Explicit protocol registration via __protocols__
    2. Structural typing compatibility via isinstance

    Args:
        instance: The object to check.
        protocol: The protocol to check against.

    Returns:
        True if the instance implements the protocol.

    """
    # Check explicit registration
    cls = type(instance)
    registered_protocols = _get_class_protocols(cls)
    if protocol in registered_protocols:
        return True

    # Check structural typing (for @runtime_checkable protocols)
    if hasattr(protocol, "__protocol_attrs__"):
        # Protocol has __protocol_attrs__ if @runtime_checkable
        return isinstance(instance, protocol)

    return False


# Define combined metaclasses inheriting from both Pydantic's ModelMetaclass and
# typing's _ProtocolMeta. This resolves the metaclass conflict when classes
# inherit from both BaseModel/BaseSettings and Protocol subclasses.
# Note: BaseSettings uses the same ModelMetaclass as BaseModel.


class _CombinedModelMeta(ModelMetaclass, _ProtocolMeta):
    """Combined metaclass for Pydantic BaseModel + Protocol inheritance."""


class _CombinedSettingsMeta(ModelMetaclass, _ProtocolMeta):
    """Combined metaclass for Pydantic BaseSettings + Protocol inheritance."""


class FlextProtocols:
    """Hierarchical protocol namespace organized by Interface Segregation Principle.

    Hierarchy follows architectural layers:
    - Base: Fundamental interfaces
    - Core: Result handling and model protocols
    - Configuration: Config and context management
    - Infrastructure: DI and container protocols
    - Domain: Business domain protocols
    - Application: CQRS and application layer protocols
    - Utility: Supporting utility protocols
    """

    # =========================================================================
    # BASE PROTOCOLS (Fundamental Interfaces)
    # =========================================================================

    @runtime_checkable
    class BaseProtocol(Protocol):
        """Base protocol that all FLEXT protocols inherit from implicitly.

        Ensures all protocols follow structural typing principles and
        maintain consistency across the framework.
        """

        def _protocol_name(self) -> str:
            """Return the protocol name for introspection."""
            ...

    # =========================================================================
    # CONTEXT PROTOCOLS (Context Management)
    # =========================================================================

    @runtime_checkable
    class ContextLike(Protocol):
        """Context protocol for type safety without circular imports.

        Defined in protocols.py to keep all protocol definitions together.
        Full context protocol p.Context extends this minimal interface.

        Methods use generic return types (object) for structural compatibility
        with p.Context which uses ResultLike[T] (also covariant with object).
        """

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def set(
            self,
            key: str,
            value: t.GeneralValueType,
            scope: str = ...,
        ) -> object:
            """Set a context value. Returns Result-like object."""
            ...

        def get(
            self,
            key: str,
            scope: str = ...,
        ) -> object:
            """Get a context value. Returns Result-like object."""
            ...

    # RuntimeBootstrapOptions moved to Pydantic model in _models/service.py
    # Use type alias for backward compatibility
    # from flext_core._models.service import RuntimeBootstrapOptions

    # =========================================================================
    # CORE PROTOCOLS (Result Handling and Models)
    # =========================================================================

    @runtime_checkable
    class Result[T](BaseProtocol, Protocol):
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
        def error_data(self) -> t.ConfigurationMapping | None:
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
            factory: FlextProtocols.ResourceFactory[TResource],
            op: FlextProtocols.ResourceOperation[TResource, T],
            cleanup: FlextProtocols.ResourceCleanup[TResource] | None = None,
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
    class ResultLike[T_co](BaseProtocol, Protocol):
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
    class HasModelDump(BaseProtocol, Protocol):
        """Protocol for objects that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> Mapping[str, t.FlexibleValue]:
            """Dump model data to dictionary."""
            ...

    @runtime_checkable
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for objects with model fields.

        Extends HasModelDump with model fields access.
        Used for Pydantic model introspection.
        """

        @property
        def model_fields(self) -> Mapping[str, t.FlexibleValue]:
            """Model fields mapping."""
            ...

    @runtime_checkable
    class Model(HasModelDump, Protocol):
        """Model type interface for validation.

        Used for model validation without circular imports.
        """

        def validate(self) -> FlextProtocols.Result[bool]:
            """Validate model."""
            ...

    # =========================================================================
    # CONFIGURATION PROTOCOLS (Config and Context Management)
    # =========================================================================

    @runtime_checkable
    class Configurable(BaseProtocol, Protocol):
        """Protocol for component configuration."""

        def configure(self, config: Mapping[str, t.FlexibleValue]) -> None:
            """Configure component with settings."""
            ...

    @runtime_checkable
    class Config(BaseProtocol, Protocol):
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

        enable_caching: bool
        """Enable caching for query operations."""

        timeout_seconds: float
        """Default timeout in seconds for operations."""

        dispatcher_auto_context: bool
        """Enable automatic context management in dispatcher."""

        dispatcher_enable_logging: bool
        """Enable logging in dispatcher operations."""

        def model_copy(
            self,
            *,
            update: Mapping[str, t.FlexibleValue] | None = None,
            deep: bool = False,
        ) -> Self:
            """Clone configuration with optional updates (Pydantic standard method)."""
            ...

        def model_dump(
            self,
            *,
            mode: str = "python",
            include: t.FlexibleValue = None,
            exclude: t.FlexibleValue = None,
            by_alias: bool = False,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            round_trip: bool = False,
            warnings: bool = True,
        ) -> dict[str, t.FlexibleValue]:
            """Serialize configuration to dictionary (Pydantic standard method)."""
            ...

    # =========================================================================
    # CONTEXT: Context Management Protocols
    # =========================================================================

    @runtime_checkable
    class Context(ContextLike, Protocol):
        """Execution context protocol with cloning semantics.

        Extends FlextProtocols.ContextLike (minimal protocol) for full
        context operations. Uses ResultLike (covariant) for return types
        to allow implementations to return FlextResult.
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
    # INFRASTRUCTURE PROTOCOLS (DI and Container Management)
    # =========================================================================

    @runtime_checkable
    class DI(Configurable, Protocol):
        """Dependency injection container protocol.

        Extends Configurable to allow container configuration.
        Implements configure() method from Configurable protocol.
        """

        @property
        def config(self) -> FlextProtocols.Config:
            """Configuration bound to the container."""
            ...

        @property
        def context(self) -> FlextProtocols.Context:
            """Execution context bound to the container."""
            ...

        def scoped(
            self,
            *,
            config: FlextProtocols.Config | None = None,
            context: FlextProtocols.Context | None = None,
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
            service: t.RegisterableService,
        ) -> FlextProtocols.ResultLike[bool]:
            """Register a service instance."""
            ...

        def register_factory(
            self,
            name: str,
            factory: FlextProtocols.ResourceFactory[t.RegisterableService],
        ) -> FlextProtocols.ResultLike[bool]:
            """Register a service factory returning RegisterableService."""
            ...

        def with_service(
            self,
            name: str,
            service: t.RegisterableService,
        ) -> Self:
            """Fluent interface for service registration."""
            ...

        def with_factory(
            self,
            name: str,
            factory: FlextProtocols.ResourceFactory[t.RegisterableService],
        ) -> Self:
            """Fluent interface for factory registration."""
            ...

        def get(
            self,
            name: str,
        ) -> FlextProtocols.ResultLike[t.RegisterableService]:
            """Get service by name.

            Returns the resolved service as RegisterableService. For type-safe
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
    # DOMAIN PROTOCOLS (Business Logic Interfaces)
    # =========================================================================

    @runtime_checkable
    class Service[T](BaseProtocol, Protocol):
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
    class Repository[T](BaseProtocol, Protocol):
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
        class Predicate(Protocol):
            """Protocol for callable predicates that accept a value and return bool.

            Used in validation utilities for filtering and conditional logic.
            Supports any callable that accepts a value and returns bool.
            """

            def __call__(self, value: t.GeneralValueType) -> bool:  # INTERFACE
                """Evaluate predicate on value."""
                ...

        @runtime_checkable
        class HasInvariants(Protocol):
            """Protocol for DDD aggregate invariant checking.

            Reflects real implementations like FlextModelsEntity.AggregateRoot
            which checks invariants and raises exceptions on violation rather
            than returning FlextProtocols.Result[bool].
            """

            def check_invariants(self) -> None:  # INTERFACE
                """Check invariants, raising exception on violation.

                Reflects real implementations like FlextModelsEntity.AggregateRoot
                which raises ValidationError when invariants are violated.
                """
                ...

    # =========================================================================
    # APPLICATION: Application Layer Protocols
    # =========================================================================

    @runtime_checkable
    class Handler(BaseProtocol, Protocol):
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
    class CommandBus(BaseProtocol, Protocol):
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
            handler_func: Callable[P, t.GeneralValueType],
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
    class Registry(BaseProtocol, Protocol):
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
    class Middleware(BaseProtocol, Protocol):
        """Processing pipeline middleware."""

        def process[TResult](
            self,
            command: t.FlexibleValue,
            next_handler: FlextProtocols.ResourceOperation[
                t.FlexibleValue,
                TResult,
            ],
        ) -> FlextProtocols.Result[TResult]:
            """Process command."""
            ...

    @runtime_checkable
    class Processor(BaseProtocol, Protocol):
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

    class Log:
        """Logging namespace with StructlogLogger and Metadata protocols.

        Access patterns:
        - p.Log.StructlogLogger - structlog logger protocol
        - p.Log.Metadata - metadata protocol
        """

        @runtime_checkable
        class StructlogLogger(BindableLogger, Protocol):
            """Protocol for structlog logger with all logging methods.

            Extends BindableLogger to add explicit method signatures for
            logging methods (debug, info, warning, error, etc.) that are
            available via __getattr__ at runtime.
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

    @runtime_checkable
    class Connection(BaseProtocol, Protocol):
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

    # =========================================================================
    # UTILITY: Utility Protocols
    # =========================================================================

    @runtime_checkable
    class VariadicCallable[T_co](Protocol):
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
    class ResourceFactory[TResource](Protocol):
        """Protocol for resource factory callables.

        Used in with_resource pattern to create resources.
        Replaces Callable[[], TResource] for type safety.
        """

        def __call__(self) -> TResource:
            """Create and return a resource instance."""
            ...

    @runtime_checkable
    class ResourceOperation[TResource, T](Protocol):
        """Protocol for resource operation callables.

        Used in with_resource pattern to operate on resources.
        Replaces Callable[[TResource], Result[T]] for type safety.
        """

        def __call__(self, resource: TResource) -> FlextProtocols.Result[T]:
            """Execute operation on resource, returning Result."""
            ...

    class ResourceCleanup[TResource](Protocol):
        """Protocol for resource cleanup callables.

        Used in with_resource pattern for optional cleanup.
        Replaces Callable[[TResource], None] for type safety.
        """

        def __call__(self, resource: TResource) -> None:
            """Clean up the resource."""
            ...

    @runtime_checkable
    class ValidatorSpec(BaseProtocol, Protocol):
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

    class Entry(
        BaseProtocol, Protocol
    ):  # Cannot inherit BaseProtocol due to Python nested class limitations
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

        def set_attribute(
            self,
            name: str,
            values: Sequence[str],
        ) -> Self:
            """Set attribute values, returning self for chaining."""
            ...

        def add_attribute(
            self,
            name: str,
            values: Sequence[str],
        ) -> Self:
            """Add attribute values, returning self for chaining."""
            ...

        def remove_attribute(
            self,
            name: str,
        ) -> Self:
            """Remove attribute, returning self for chaining."""
            ...

    # =========================================================================
    # MAPPER PROTOCOLS (For Collection Operations)
    # =========================================================================

    class SingleValueMapper[T, R](BaseProtocol, Protocol):
        """Protocol for mappers that transform single values."""

        def __call__(self, value: T) -> R:
            """Map a single value to a result."""
            ...

    class KeyValueMapper[T, R](BaseProtocol, Protocol):
        """Protocol for mappers that transform key-value pairs."""

        def __call__(self, key: str, value: T) -> R:
            """Map a key-value pair to a result."""
            ...

    # =========================================================================
    # UTILITIES PROTOCOLS
    # =========================================================================
    class CallableWithHints(
        BaseProtocol, Protocol
    ):  # Cannot inherit BaseProtocol due to Python nested class limitations
        """Protocol for callables that support type hints introspection."""

        __annotations__: dict[str, t.GeneralValueType]

    # =========================================================================
    # TYPE ALIASES FOR UTILITIES
    # =========================================================================

    # Type for data sources that get() can access
    # Supports: ConfigurationDict (GeneralValueType), JsonValue dicts, and object dicts
    # NOTE: Explicit dict types needed because pyright treats dict as invariant
    type AccessibleData = (
        dict[str, t.GeneralValueType]
        | dict[str, t.JsonValue]
        | t.ConfigurationMapping
        | Mapping[str, t.JsonValue]
        | Mapping[str, t.GeneralValueType]
        | BaseModel
        | "FlextProtocols.HasModelDump"
        | "FlextProtocols.ValidatorSpec"
    )

    # =========================================================================

    # =========================================================================
    # CONTAINER SERVICE TYPE (Union of all registerable service protocols)
    # =========================================================================

    # RegisterableService: Type alias for all services that can be registered
    # in FlextContainer. Includes protocols, models, and callables.
    # This replaces object/dict usage in DI container methods.
    # Type for services registerable in FlextContainer.
    # Union of all protocol types that can be registered as DI services:
    # - GeneralValueType: Primitives, BaseModel, sequences, mappings
    # - BindableLogger: Logger protocol
    # - Callable: Factories that return GeneralValueType
    RegisterableService = t.GeneralValueType | BindableLogger

    # ServiceFactory: Factory callable that returns RegisterableService
    type ServiceFactory = FlextProtocols.ResourceFactory[t.GeneralValueType]
    """Factory callable returning any registerable service type.

    Broader than t.FactoryCallable (which returns GeneralValueType).
    Supports factories that create protocols like Log,             ..., Config, etc.

    Usage:
        def create_logger() -> FlextLogger:
            return FlextLogger.create_module_logger("app")

        container.register_factory("logger", create_logger)  # OK
    """

    # =========================================================================
    # PROTOCOL + PYDANTIC INTEGRATION (Metaclass, Base Classes, Decorator)
    # =========================================================================

    class ProtocolModelMeta(_CombinedModelMeta):
        """Metaclass combining Pydantic with Protocol structural typing.

        This metaclass inherits from a dynamically-created combined metaclass
        that includes both Pydantic's ModelMetaclass AND Protocol's _ProtocolMeta.
        This allows classes using this metaclass to inherit from Protocol
        subclasses without metaclass conflicts.

        The key insight is to separate Protocol types from real bases,
        create the class with only model bases (avoiding metaclass conflict),
        then validate and store protocol information for runtime checking.

        Usage:
            class MyModel(p.ProtocolModel, p.Domain.Entity):
                name: str
                value: int

                def _protocol_name(self) -> str:
                    return "MyModel"
        """

        def __new__(
            mcs: type[Self],
            name: str,
            bases: tuple[type, ...],
            namespace: dict[str, object],
            **_kwargs: object,
        ) -> type:
            """Create a new class with protocol validation.

            Args:
                name: The class name.
                bases: Tuple of base classes (may include protocols).
                namespace: The class namespace dictionary.
                **_kwargs: Additional keyword arguments for metaclass.

            Returns:
                The newly created class with protocols validated.

            """
            # Separate protocols from model bases
            protocols, model_bases = _partition_protocol_bases(bases)

            # Ensure we have at least one real base
            if not model_bases:
                model_bases = [BaseModel]

            # Create class with only model bases (no metaclass conflict)
            cls: type = super().__new__(
                mcs,
                name,
                tuple(model_bases),
                namespace,
            )

            # Store protocols using setattr (avoids type: ignore)
            setattr(cls, "__protocols__", tuple(protocols))

            # Validate protocol compliance at class definition time
            for protocol in protocols:
                _validate_protocol_compliance(cls, protocol, name)

            return cls

    class ProtocolModel(BaseModel, metaclass=ProtocolModelMeta):
        """Base class for Pydantic models that implement protocols.

        Enables natural multi-inheritance with protocols without metaclass
        conflicts. Protocol compliance is validated at class definition time.

        Usage:
            class MyEntity(p.ProtocolModel, p.Domain.Entity):
                name: str
                value: int

                def _protocol_name(self) -> str:
                    return "MyEntity"

            # Check protocols at runtime
            entity = MyEntity(name="test", value=42)
            assert entity.implements_protocol(p.Domain.Entity)
            assert entity.get_protocols() == (p.Domain.Entity,)
        """

        def implements_protocol(self, protocol: type) -> bool:
            """Check if this instance implements a protocol.

            Args:
                protocol: The protocol type to check.

            Returns:
                True if this instance implements the protocol.

            """
            return _check_implements_protocol(self, protocol)

        @classmethod
        def get_protocols(cls) -> tuple[type, ...]:
            """Return all protocols this class implements.

            Returns:
                Tuple of protocol types.

            """
            return _get_class_protocols(cls)

        def _protocol_name(self) -> str:
            """Return the protocol name for introspection.

            Returns:
                The class name as protocol name.

            """
            return type(self).__name__

    class ProtocolSettings(BaseSettings, metaclass=ProtocolModelMeta):
        """Base class for Pydantic Settings that implement protocols.

        Extends the ProtocolModel pattern to BaseSettings, enabling
        environment variable loading alongside protocol compliance.

        Usage:
            class MySettings(p.ProtocolSettings, p.Configuration.Config):
                app_name: str = Field(default="myapp")
                debug: bool = Field(default=False)

                model_config = SettingsConfigDict(env_prefix="MY_")

                def _protocol_name(self) -> str:
                    return "MySettings"
        """

        def implements_protocol(self, protocol: type) -> bool:
            """Check if this instance implements a protocol.

            Args:
                protocol: The protocol type to check.

            Returns:
                True if this instance implements the protocol.

            """
            return _check_implements_protocol(self, protocol)

        @classmethod
        def get_protocols(cls) -> tuple[type, ...]:
            """Return all protocols this class implements.

            Returns:
                Tuple of protocol types.

            """
            return _get_class_protocols(cls)

        def _protocol_name(self) -> str:
            """Return the protocol name for introspection.

            Returns:
                The class name as protocol name.

            """
            return type(self).__name__

    @staticmethod
    def implements(*protocols: type) -> Callable[[type[T]], type[T]]:
        """Decorator to mark non-Pydantic classes as implementing protocols.

        Validates protocol compliance at class definition time and adds
        protocol introspection capabilities to the decorated class.

        This decorator is for classes that don't inherit from Pydantic
        BaseModel or BaseSettings. For Pydantic classes, use ProtocolModel
        or ProtocolSettings base classes instead.

        Usage:
            @p.implements(p.Handler, p.Domain.Repository)
            class MyHandler(FlextHandlers[Command, Result]):
                def handle(self, message: Command) -> Result:
                    ...

                def _protocol_name(self) -> str:
                    return "MyHandler"

            # Check protocols at runtime
            handler = MyHandler()
            assert handler.implements_protocol(p.Handler)
            assert MyHandler.get_protocols() == (p.Handler, p.Domain.Repository)

        Args:
            *protocols: Protocol types that the class implements.

        Returns:
            A decorator that validates and registers protocols on the class.

        """

        def decorator(cls: type[T]) -> type[T]:
            # Validate each protocol at decoration time
            # Use getattr for type-safe access to __name__
            class_name = getattr(cls, "__name__", str(cls))
            for protocol in protocols:
                _validate_protocol_compliance(cls, protocol, class_name)

            # Store protocols using setattr (avoids type: ignore)
            setattr(cls, "__protocols__", tuple(protocols))

            # Add helper method for instance protocol checking
            def _instance_implements_protocol(
                self: object,
                protocol: type,
            ) -> bool:
                return _check_implements_protocol(self, protocol)

            setattr(cls, "implements_protocol", _instance_implements_protocol)

            # Add classmethod for getting protocols
            def _class_get_protocols(kls: type) -> tuple[type, ...]:
                return _get_class_protocols(kls)

            setattr(cls, "get_protocols", classmethod(_class_get_protocols))

            return cls

        return decorator

    # Expose utility functions as static methods for external use
    @staticmethod
    def is_protocol(target_cls: type) -> bool:
        """Check if a class is a typing.Protocol.

        Args:
            target_cls: The class to check.

        Returns:
            True if target_cls is a Protocol, False otherwise.

        """
        return _is_protocol(target_cls)

    @staticmethod
    def check_implements_protocol(instance: object, protocol: type) -> bool:
        """Check if an instance's class implements a protocol.

        Args:
            instance: The object to check.
            protocol: The protocol to check against.

        Returns:
            True if the instance implements the protocol.

        """
        return _check_implements_protocol(instance, protocol)

    # Alias for convenience (matches instance method name)
    implements_protocol = check_implements_protocol


p = FlextProtocols

__all__ = [
    "FlextProtocols",
    "p",
]
