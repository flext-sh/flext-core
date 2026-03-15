"""Runtime-checkable structural typing protocols for FLEXT framework.

Copyright (t) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import ModuleType, TracebackType
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    Self,
    overload,
    override,
    runtime_checkable,
)

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from structlog.typing import BindableLogger

from flext_core import T, t
from flext_core._models.containers import FlextModelsContainers

if TYPE_CHECKING:
    from pydantic._internal._model_construction import (
        ModelMetaclass as _TypeCheckModelMeta,
    )

    from flext_core import r


class _ProtocolIntrospection:
    """Internal helpers for protocol detection and compliance checks."""

    @staticmethod
    def _get_protocol_attrs(protocol: type) -> tuple[str, ...]:
        try:
            raw_attrs_candidate = protocol.__protocol_attrs__
        except AttributeError:
            return ()
        try:
            return tuple(raw_attrs_candidate)
        except TypeError:
            return ()

    @classmethod
    def check_implements_protocol(
        cls,
        instance: FlextProtocols.Base | t.Container,
        protocol: type,
    ) -> bool:
        """Check if an instance implements a protocol."""
        registered_protocols = cls.get_class_protocols(instance.__class__)
        if protocol in registered_protocols:
            return True
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        raw_attrs = set(cls._get_protocol_attrs(protocol))
        protocol_methods: set[str] = set()
        protocol_methods.update(raw_attrs)
        required_members: set[str] = set(protocol_annotations.keys())
        required_members.update(protocol_methods)
        required_members = {
            m
            for m in required_members
            if not m.startswith("_")
            or m.startswith("__")
            or (m in {"metadata_extra", "sealed"})
        }
        if not required_members:
            return False
        return all(hasattr(instance, member) for member in required_members)

    @classmethod
    def partition_protocol_bases(
        cls, bases: tuple[type, ...]
    ) -> tuple[list[type], list[type]]:
        """Separate Protocol bases from regular class bases."""
        protocols: list[type] = []
        model_bases: list[type] = []
        for base in bases:
            if cls.is_protocol(base):
                protocols.append(base)
            else:
                model_bases.append(base)
        return (protocols, model_bases)

    @staticmethod
    def get_class_protocols(target_cls: type) -> tuple[type, ...]:
        """Get the protocols a class implements."""
        iterable_protocols: Sequence[type]
        try:
            iterable_protocols = tuple(target_cls.__protocols__)
        except (AttributeError, TypeError):
            return ()
        try:
            typed_protocols = list(iterable_protocols)
            return tuple(typed_protocols)
        except TypeError:
            return ()

    @staticmethod
    def is_protocol(target_cls: type) -> bool:
        """Check if a class is a typing.Protocol."""
        is_proto = getattr(target_cls, "_is_protocol", False)
        if callable(is_proto):
            return bool(is_proto())
        return bool(is_proto)

    @staticmethod
    def validate_protocol_compliance(
        target_cls: type, protocol: type, class_name: str
    ) -> None:
        """Validate that a class implements all required protocol members."""
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        raw_attrs = set(_ProtocolIntrospection._get_protocol_attrs(protocol))
        protocol_methods: set[str] = set()
        protocol_methods.update(raw_attrs)
        required_members: set[str] = set(protocol_annotations.keys())
        if protocol_methods:
            required_members.update(protocol_methods)
        required_members = {
            m
            for m in required_members
            if not m.startswith("_")
            or m.startswith("__")
            or (m in {"metadata_extra", "sealed"})
        }
        all_annotations: set[str] = set()
        for base in target_cls.mro():
            base_annotations: Mapping[str, type | str] = (
                base.__annotations__ if hasattr(base, "__annotations__") else {}
            )
            all_annotations.update(base_annotations.keys())
        missing = [
            member
            for member in required_members
            if not (hasattr(target_cls, member) or member in all_annotations)
        ]
        if missing:
            protocol_name = (
                protocol.__name__ if hasattr(protocol, "__name__") else str(protocol)
            )
            missing_str = ", ".join(sorted(missing))
            msg = f"Class '{class_name}' does not implement required members of protocol '{protocol_name}': {missing_str}"
            raise TypeError(msg)


if TYPE_CHECKING:

    class _CombinedModelMeta(_TypeCheckModelMeta):
        """TYPE_CHECKING stub: metaclass chain for mypy resolution."""

else:

    def _build_combined_model_meta() -> type:
        return type("_CombinedModelMeta", (type(BaseModel), type(Protocol)), {})

    _CombinedModelMeta: type = _build_combined_model_meta()


_METACLASS_STRICT: bool = os.environ.get("FLEXT_METACLASS_STRICT", "1") == "1"


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

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

        pass

    @runtime_checkable
    class Model(Base, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        model_config: ClassVar[Mapping[str, t.Container]]
        model_fields: ClassVar[Mapping[str, type | str]]

        def model_dump(
            self, **kwargs: t.Container
        ) -> Mapping[str, t.NormalizedValue | BaseModel]:
            """Dump model to dictionary."""
            ...

        @classmethod
        def model_validate(
            cls,
            obj: t.NormalizedValue | BaseModel,
            **kwargs: t.Container,
        ) -> Self:
            """Validate object against model."""
            ...

        def validate(self) -> FlextProtocols.Result[bool]:
            """Validate model."""
            ...

    @runtime_checkable
    class Routable(Protocol):
        """Protocol for messages that carry explicit route information."""

        @property
        def command_type(self) -> str | None:
            """Command type identifier."""
            ...

        @property
        def event_type(self) -> str | None:
            """Event type identifier."""
            ...

        @property
        def query_type(self) -> str | None:
            """Query type identifier."""
            ...

    _protocol_members_cache: ClassVar[dict[type, frozenset[str]]] = {}
    _class_annotations_cache: ClassVar[dict[type, frozenset[str]]] = {}
    _compliance_results_cache: ClassVar[dict[tuple[type, type], bool]] = {}

    @classmethod
    def _get_protocol_members(cls, protocol: type) -> frozenset[str]:
        if protocol not in cls._protocol_members_cache:
            cls._protocol_members_cache[protocol] = frozenset(
                _ProtocolIntrospection._get_protocol_attrs(protocol)
            )
        return cls._protocol_members_cache[protocol]

    @classmethod
    def _get_class_annotation_members(cls, target_cls: type) -> frozenset[str]:
        if target_cls not in cls._class_annotations_cache:
            all_annotations: set[str] = set()
            for base in target_cls.mro():
                base_annotations: Mapping[str, type | str] = (
                    base.__annotations__ if hasattr(base, "__annotations__") else {}
                )
                all_annotations.update(base_annotations.keys())
            cls._class_annotations_cache[target_cls] = frozenset(all_annotations)
        return cls._class_annotations_cache[target_cls]

    @classmethod
    def _get_protocol_required_members(cls, protocol: type) -> frozenset[str]:
        protocol_annotations: Mapping[str, type | str] = (
            protocol.__annotations__ if hasattr(protocol, "__annotations__") else {}
        )
        required_members: set[str] = set(protocol_annotations.keys())
        required_members.update(cls._get_protocol_members(protocol))
        filtered_members = {
            member
            for member in required_members
            if not member.startswith("_")
            or member.startswith("__")
            or (member in {"metadata_extra", "sealed"})
        }
        return frozenset(filtered_members)

    @classmethod
    def _get_compliance_cache_key(
        cls, target_cls: type, protocol: type
    ) -> tuple[type, type]:
        return (target_cls, protocol)

    @classmethod
    def _check_protocol_compliance(
        cls,
        instance: FlextProtocols.Base | t.Container,
        protocol: type,
    ) -> bool:
        target_cls = instance.__class__
        cache_key = cls._get_compliance_cache_key(target_cls, protocol)
        cached_result = cls._compliance_results_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        try:
            runtime_compliant = isinstance(instance, protocol)
        except TypeError:
            runtime_compliant = False
        if runtime_compliant:
            cls._compliance_results_cache[cache_key] = True
            return True
        registered_protocols = _ProtocolIntrospection.get_class_protocols(target_cls)
        if protocol in registered_protocols:
            cls._compliance_results_cache[cache_key] = True
            return True
        required_members = cls._get_protocol_required_members(protocol)
        if not required_members:
            cls._compliance_results_cache[cache_key] = False
            return False
        class_annotations = cls._get_class_annotation_members(target_cls)
        is_compliant = all(
            hasattr(instance, member) or member in class_annotations
            for member in required_members
        )
        cls._compliance_results_cache[cache_key] = is_compliant
        return is_compliant

    @classmethod
    def _validate_protocol_compliance(
        cls, target_cls: type, protocol: type, class_name: str
    ) -> None:
        cache_key = cls._get_compliance_cache_key(target_cls, protocol)
        cached_result = cls._compliance_results_cache.get(cache_key)
        if cached_result is True:
            return
        required_members = cls._get_protocol_required_members(protocol)
        class_annotations = cls._get_class_annotation_members(target_cls)
        missing = [
            member
            for member in required_members
            if not (hasattr(target_cls, member) or member in class_annotations)
        ]
        if missing:
            cls._compliance_results_cache[cache_key] = False
            protocol_name = (
                protocol.__name__ if hasattr(protocol, "__name__") else str(protocol)
            )
            missing_str = ", ".join(sorted(missing))
            msg = f"Class '{class_name}' does not implement required members of protocol '{protocol_name}': {missing_str}"
            raise TypeError(msg)
        cls._compliance_results_cache[cache_key] = True

    @runtime_checkable
    class Context(Protocol):
        """Context protocol for type safety without circular imports.

        Defined in protocols.py to keep all protocol definitions together.
        Full context protocol p.Context extends this minimal interface.

        Methods use generic return types (Any) for structural compatibility
        with p.Context which uses ResultLike[T] (also covariant with Any).
        """

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def get(self, key: str, scope: str = ...) -> r[t.Container | BaseModel]:
            """Get a context value. Returns Result-like object."""
            ...

        @overload
        def set(
            self, key_or_data: str, value: t.Container | BaseModel, *, scope: str = ...
        ) -> r[bool]: ...

        @overload
        def set(
            self,
            key_or_data: FlextModelsContainers.ConfigMap,
            value: None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]: ...

        def set(
            self,
            key_or_data: str | FlextModelsContainers.ConfigMap,
            value: t.Container | BaseModel | None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]:
            """Set a context value. Returns Result-like object."""
            ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        config_type: type[BaseSettings] | None
        config_overrides: Mapping[str, t.Scalar] | None
        context: FlextProtocols.Context | None
        subproject: str | None
        services: Mapping[str, t.RegisterableService] | None
        factories: Mapping[str, t.FactoryCallable] | None
        resources: Mapping[str, t.ResourceCallable] | None
        container_overrides: Mapping[str, t.Scalar] | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: Sequence[str] | None
        wire_classes: Sequence[type] | None

    @runtime_checkable
    class Result[T](Base, Protocol):
        """Result type interface for railway-oriented programming.

        Used extensively for all operations that can fail. Provides
        structural typing interface for r without circular imports.
        Fully compatible with r and FlextRuntime usage patterns.

        Defined at root level to allow self-referencing in method signatures
        (e.g., `def map[U](...) -> FlextProtocols.Result[U]`).
        """

        @classmethod
        @override
        def __subclasshook__(cls, cls_: type) -> bool:
            """Enable isinstance() for Pydantic-backed implementations.

            Python 3.12+ Protocol isinstance checks use class __dict__ lookup,
            which misses Pydantic v2 model fields (stored in __pydantic_fields__,
            not __dict__). This hook uses class-level attrs that ARE in __dict__.
            """
            if cls is FlextProtocols.Result:
                # Check only attrs that exist in class __dict__ (not Pydantic fields)
                required = frozenset({"is_failure", "value", "flat_map", "lash"})
                if all(any(a in B.__dict__ for B in cls_.__mro__) for a in required):
                    return True
            return NotImplemented

        @override
        def __repr__(self) -> str:
            """String representation."""
            ...

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
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

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
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
        def error_data(self) -> FlextModelsContainers.ConfigMap | None:
            """Error metadata (optional)."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Exception captured during operation (if any)."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def result(self) -> Self:
            """Access internal Result for advanced operations."""
            ...

        @property
        def value(self) -> T:
            """Result value (available on success, never None)."""
            ...

        @classmethod
        def accumulate_errors[TItem](
            cls, *results: FlextProtocols.Result[TItem]
        ) -> FlextProtocols.Result[Sequence[TItem]]:
            """Collect all successes, fail if any failure."""
            ...

        @classmethod
        def create_from_callable(
            cls, func: Callable[[], T], error_code: str | None = None
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
        ) -> FlextProtocols.Result[Sequence[UResult]]:
            """Map over sequence with configurable failure handling."""
            ...

        def filter(self, predicate: Callable[[T], bool]) -> Self:
            """Filter success value using predicate.

            Returns self if predicate passes or result is failure,
            new failed Result if predicate fails.
            """
            ...

        def flat_map[U](
            self, func: Callable[[T], FlextProtocols.Result[U]]
        ) -> FlextProtocols.Result[U]:
            """Chain operations returning Result."""
            ...

        def flow_through[U](
            self, *funcs: Callable[[T | U], FlextProtocols.Result[U]]
        ) -> FlextProtocols.Result[U]:
            """Chain multiple operations in a pipeline."""
            ...

        def lash(self, func: Callable[[str], FlextProtocols.Result[T]]) -> Self:
            """Apply recovery function on failure."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocols.Result[U]:
            """Transform success value using function."""
            ...

        def map_error(self, func: Callable[[str], str]) -> Self:
            """Transform error message on failure.

            Returns self on success, new Result with transformed error on failure.
            """
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Unwrap success value or return default on failure."""
            ...

    @runtime_checkable
    class ResultLike[T_co](Base, Protocol):
        """Result-like protocol for compatibility with r operations.

        Used for type compatibility when working with result-like items.
        """

        @property
        def error(self) -> str | None:
            """Error message."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code."""
            ...

        @property
        def error_data(self) -> FlextModelsContainers.ConfigMap | None:
            """Error data."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Exception captured during operation (if any)."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def value(self) -> T_co:
            """Result value."""
            ...

        def unwrap(self) -> T_co:
            """Unwrap value."""
            ...

    @runtime_checkable
    class HasModelDump(Base, Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> Mapping[str, t.Scalar]:
            """Dump model data to dictionary."""
            ...

    @runtime_checkable
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for items with model fields.

        Extends HasModelDump with model fields access.
        Used for Pydantic model introspection.
        """

        @property
        def model_fields(self) -> Mapping[str, t.Scalar]:
            """Model fields mapping."""
            ...

    @runtime_checkable
    class Configurable(Base, Protocol):
        """Protocol for component configuration."""

        def configure(self, config: Mapping[str, t.Container] | None = None) -> Self:
            """Configure component with settings."""
            ...

    @runtime_checkable
    class Config(HasModelDump, Base, Protocol):
        """Configuration protocol based on Pydantic BaseSettings pattern.

        Reflects real implementations like FlextSettings which uses Pydantic BaseSettings.
        Configuration items use direct field access (Pydantic standard) rather than
        explicit get/set methods. Supports cloning via model_copy() and optional
        override methods.
        """

        app_name: str
        "Application name bound to the configuration."
        version: str
        "Semantic version of the running application."
        enable_caching: bool
        "Enable caching for query operations."
        timeout_seconds: float
        "Default timeout in seconds for operations."
        dispatcher_auto_context: bool
        "Enable automatic context management in dispatcher."
        dispatcher_enable_logging: bool
        "Enable logging in dispatcher operations."

        def model_copy(
            self,
            *,
            update: Mapping[str, t.Container] | None = None,
            deep: bool = False,
        ) -> Self:
            """Create a copy of the model, optionally updating fields or deep copying.

            Args:
                update: Dictionary of values to update in the copied model.
                deep: If True, perform a deep copy of the model fields.

            Returns:
                A new instance of the model.

            """
            ...

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

        def clear_all(self) -> None:
            """Clear all services and factories."""
            ...

        @overload
        def get[T: t.RegisterableService](
            self, name: str, *, type_cls: type[T]
        ) -> r[T]: ...

        @overload
        def get(
            self, name: str, *, type_cls: None = None
        ) -> r[t.RegisterableService]: ...

        def get_config(self) -> FlextModelsContainers.ConfigMap:
            """Return the merged configuration exposed by this container."""
            ...

        def has_service(self, name: str) -> bool:
            """Check if a service is registered."""
            ...

        def list_services(self) -> Sequence[str]:
            """List all registered services."""
            ...

        def register(
            self,
            name: str,
            impl: t.RegisterableService,
            *,
            kind: Literal["service", "factory", "resource"] = "service",
        ) -> Self:
            """Register an implementation by kind."""
            ...

        def scoped(
            self,
            *,
            config: FlextProtocols.Config | None = None,
            context: FlextProtocols.Context | None = None,
            subproject: str | None = None,
            services: Mapping[str, t.RegisterableService] | None = None,
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

    @runtime_checkable
    class Service[T](Base, Protocol):
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

        def get_service_info(self) -> Mapping[str, t.Scalar]:
            """Get service metadata and configuration information.

            Reflects real implementations like FlextService which provide
            service metadata for observability and debugging.
            """
            ...

        def is_valid(self) -> bool:
            """Check if service is in valid state for execution.

            Reflects real implementations like FlextService which check
            validity based on internal state and business rules.
            """
            ...

        def validate_business_rules(self) -> FlextProtocols.Result[bool]:
            """Validate business rules with extensible validation pipeline.

            Reflects real implementations like FlextService which perform
            business rule validation without external command parameters.
            """
            ...

    @runtime_checkable
    class Repository[T](Base, Protocol):
        """Data access interface."""

        def delete(self, entity_id: str) -> FlextProtocols.Result[bool]:
            """Delete entity."""
            ...

        def find_all(self) -> FlextProtocols.Result[Sequence[T]]:
            """Find all entities."""
            ...

        def get_by_id(self, entity_id: str) -> FlextProtocols.Result[T]:
            """Get entity by ID."""
            ...

        def save(self, entity: T) -> FlextProtocols.Result[T]:
            """Save entity."""
            ...

    @runtime_checkable
    class Predicate[T](Protocol):
        """Protocol for callable predicates that accept a value and return bool.

        Used in validation utilities for filtering and conditional logic.
        Supports any callable that accepts a value and returns bool.

        Type Parameters:
            T: The type of value the predicate evaluates.
        """

        def __call__(self, value: T) -> bool:
            """Evaluate predicate on value."""
            ...

    @runtime_checkable
    class Handler[MessageT: Model, ResultT](Base, Protocol):
        """Command/Query handler interface (generic).

        Reflects real implementations like FlextHandlers which provide
        comprehensive validation and execution pipelines for CQRS handlers.

        Type Parameters:
        - MessageT: Type of message handled (command, query, or event)
        - ResultT: Type of result returned by handler
        """

        def can_handle(self, message_type: type) -> bool:
            """Check if handler can handle the specified message type.

            Reflects real implementations like FlextHandlers.can_handle() which
            checks message type compatibility using duck typing and class hierarchy.
            """
            ...

        def handle(self, message: MessageT) -> r[ResultT]:
            """Handle message - core business logic method.

            Reflects real implementations like FlextHandlers.handle() which
            executes handler business logic for commands, queries, or events.
            """
            ...

    @runtime_checkable
    class CommandBus(Base, Protocol):
        """Command routing and execution protocol.

        Matches FlextDispatcher: strict handler registration and message dispatch.
        """

        def dispatch(
            self, message: FlextProtocols.Routable
        ) -> FlextProtocols.Result[FlextProtocols.Model]:
            """Dispatch a CQRS message to its registered handler."""
            ...

        def publish(
            self, event: FlextProtocols.Routable | Sequence[FlextProtocols.Routable]
        ) -> FlextProtocols.Result[bool]:
            """Publish events to registered subscribers."""
            ...

        def register_handler(
            self, handler: t.HandlerLike, *, is_event: bool = False
        ) -> FlextProtocols.Result[bool]:
            """Register a handler with route auto-discovery.

            Handler must expose message_type, event_type, or can_handle
            for route resolution.
            """
            ...

    @runtime_checkable
    class Registry[MessageT: Model, ResultT](Base, Protocol):
        """Handler registry protocol for CQRS handler registration.

        Reflects real implementations like FlextRegistry which provides
        handler registration, batch operations, and idempotent tracking
        for CQRS handlers.
        """

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

        def register_bindings(
            self,
            bindings: Mapping[
                t.MessageTypeSpecifier,
                FlextProtocols.Handler[MessageT, ResultT],
            ],
        ) -> FlextProtocols.Result[FlextProtocols.Model]:
            """Register message-to-handler bindings.

            Reflects real implementations like FlextRegistry.register_bindings()
            which maps message types to handlers.
            """
            ...

        def register_handler(
            self, handler: FlextProtocols.Handler[MessageT, ResultT]
        ) -> FlextProtocols.Result[FlextProtocols.Model]:
            """Register a handler instance.

            Reflects real implementations like FlextRegistry.register_handler()
            which registers handlers with idempotent tracking.
            """
            ...

        def register_handlers(
            self,
            handlers: Sequence[FlextProtocols.Handler[MessageT, ResultT]],
        ) -> FlextProtocols.Result[FlextProtocols.Model]:
            """Register multiple handlers in batch.

            Reflects real implementations like FlextRegistry.register_handlers()
            which provides batch registration with summary reporting.
            """
            ...

    @runtime_checkable
    class Middleware(Base, Protocol):
        """Processing pipeline middleware."""

        def process[TResult](
            self,
            command: FlextProtocols.Model,
            next_handler: Callable[
                [FlextProtocols.Model], FlextProtocols.Result[TResult]
            ],
        ) -> FlextProtocols.Result[TResult]:
            """Process command."""
            ...

    @runtime_checkable
    class Processor(Base, Protocol):
        """Processor interface for data transformation pipelines.

        Processors can be items with a process() method that takes data
        and returns a result (which will be normalized to Result).
        Accepts t.Container, BaseModel, or Result for processing.

        The return type is flexible to support:
        - Direct values (t.Container)
        - BaseModel instances (Pydantic models)
        - Result instances (structural typing)
        - Objects with is_success/is_failure properties (r compatibility)
        """

        def process(
            self,
            data: FlextProtocols.Model | FlextProtocols.Result[FlextProtocols.Model],
        ) -> FlextProtocols.Model | FlextProtocols.Result[FlextProtocols.Model]:
            """Process data and return result.

            Returns can be:
            - BaseModel instance (Pydantic model)
            - Result (structural typing compatible)
            """
            ...

    @runtime_checkable
    class MetricsTracker(Base, Protocol):
        """Metrics tracking protocol for handler execution metrics.

        Reflects real implementations like FlextMixins.CQRS.MetricsTracker which
        tracks handler execution metrics (latency, success/failure counts, etc.).
        """

        def get_metrics(self) -> FlextProtocols.Result[FlextModelsContainers.ConfigMap]:
            """Get current metrics dictionary.

            Returns:
                Result[ConfigMap]: Success result with metrics collection

            """
            ...

        def record_metric(
            self, name: str, value: t.Container
        ) -> FlextProtocols.Result[bool]:
            """Record a metric value.

            Args:
                name: Metric name
                value: Metric value to record

            Returns:
                Result[bool]: Success result

            """
            ...

    @runtime_checkable
    class ContextStack(Base, Protocol):
        """Execution context stack protocol for CQRS operations.

        Reflects real implementations like FlextMixins.CQRS.ContextStack which
        manages a stack of execution contexts for nested handler invocations.
        """

        def current_context(self) -> FlextProtocols.Model | None:
            """Get current execution context without popping.

            Returns:
                ExecutionContext | None: Current context or None if stack is empty

            """
            ...

        def pop_context(self) -> FlextProtocols.Result[FlextProtocols.Model]:
            """Pop execution context from the stack.

            Returns:
                Result[Model]: Success result with popped context

            """
            ...

        def push_context(
            self, ctx: FlextProtocols.Model
        ) -> FlextProtocols.Result[bool]:
            """Push execution context onto the stack.

            Args:
                ctx: Execution context to push

            Returns:
                Result[bool]: Success result

            """
            ...

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

            def critical(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log critical message."""
                ...

            def debug(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log debug message."""
                ...

            def error(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log error message."""
                ...

            def exception(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log exception with traceback."""
                ...

            def info(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log info message."""
                ...

            def warning(
                self,
                msg: str,
                *args: t.Container,
                **kw: t.Container | Exception,
            ) -> r[bool] | None:
                """Log warning message."""
                ...

        @runtime_checkable
        class Metadata(Protocol):
            """Metadata protocol."""

            @property
            def attributes(self) -> FlextModelsContainers.ConfigMap:
                """Metadata attributes."""
                ...

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

    @runtime_checkable
    class Connection(Base, Protocol):
        """External system connection protocol."""

        def close_connection(self) -> None:
            """Close connection."""
            ...

        def get_connection_string(self) -> str:
            """Get connection string."""
            ...

        def test_connection(self) -> FlextProtocols.Result[bool]:
            """Test connection."""
            ...

    @runtime_checkable
    class VariadicCallable[T_co](Protocol):
        """Protocol for variadic callables returning T_co.

        Used for flexible function signatures that accept any arguments.
        Accepts *args and **kwargs, making it suitable for services, handlers,
        factories, and callbacks.
        """

        def __call__(self, *args: t.Container, **kwargs: t.Container) -> T_co:
            """Call the function with any arguments, returning T_co."""
            ...

    @runtime_checkable
    class ValidatorSpec(Base, Protocol):
        """Protocol for validator specifications with operator composition.

        Validators implement __call__ to validate values and support composition
        via __and__ (both must pass), __or__ (either passes), and __invert__ (negation).

        Example:
            validator = V.string.non_empty & V.string.max_length(100)
            is_valid = validator("hello")  # True

        """

        def __call__(self, value: t.Container) -> bool:
            """Validate value, return True if valid."""
            ...

        def __and__(
            self, other: FlextProtocols.ValidatorSpec
        ) -> FlextProtocols.ValidatorSpec:
            """Compose with AND - both validators must pass."""
            ...

        def __invert__(self) -> FlextProtocols.ValidatorSpec:
            """Negate validator - passes when original fails."""
            ...

        def __or__(
            self, other: FlextProtocols.ValidatorSpec
        ) -> FlextProtocols.ValidatorSpec:
            """Compose with OR - at least one validator must pass."""
            ...

    @runtime_checkable
    class Decorator[P, R](Base, Protocol):
        """Protocol for decorator factory pattern.

        Captures the factory pattern used by all FLEXT decorators:
        1. Configuration phase: Accept config parameters
        2. Decorator phase: Return a decorator function
        3. Wrapper phase: Return a wrapper that executes with added behavior

        All FLEXT decorators (@inject, @log_operation,
        @railway, @retry, @timeout, @with_correlation, @combined) follow
        this structural pattern.

        Type Parameters:
        - P: ParamSpec for function parameters
        - R: Return type of wrapped function

        Example:
            @FlextDecorators.log_operation("my_op")
            def my_function(x: int) -> str:
                return str(x)

            # Expands to:
            # my_function = FlextDecorators.log_operation("my_op")(my_function)

        """

        def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
            """Apply decorator to function, returning wrapped callable.

            Args:
                func: The function to decorate

            Returns:
                Wrapped function with same signature and return type

            """
            ...

    @runtime_checkable
    class Entry(Base, Protocol):
        """Entry protocol (read-only)."""

        @property
        def attributes(self) -> Mapping[str, Sequence[str]]:
            """Entry attributes as immutable mapping."""
            ...

        @property
        def dn(self) -> str:
            """Distinguished name."""
            ...

        def add_attribute(self, name: str, values: Sequence[str]) -> Self:
            """Add attribute values, returning self for chaining."""
            ...

        def remove_attribute(self, name: str) -> Self:
            """Remove attribute, returning self for chaining."""
            ...

        def set_attribute(self, name: str, values: Sequence[str]) -> Self:
            """Set attribute values, returning self for chaining."""
            ...

        def to_dict(self) -> Mapping[str, t.Scalar]:
            """Convert to dictionary representation."""
            ...

        def to_ldif(self) -> str:
            """Convert to LDIF format."""
            ...

    @runtime_checkable
    class CallableWithHints(Base, Protocol):
        """Protocol for callables that support type hints introspection."""

    type AccessibleData = (
        FlextModelsContainers.ConfigMap
        | Mapping[str, t.NormalizedValue | BaseModel]
        | t.NormalizedValue
        | BaseModel
        | "FlextProtocols.HasModelDump"
        | "FlextProtocols.ValidatorSpec"
    )
    type RegisterableService = (
        t.Container
        | BindableLogger
        | Callable[..., t.Container]
        | Config
        | Context
        | DI
        | Service[t.Container]
        | CommandBus
    )
    type ServiceFactory = t.FactoryCallable
    'Factory callable returning any registerable service type.\n\n    Broader than t.FactoryCallable (which returns RegisterableService).\n    Supports factories that create protocols like Log,             ..., Config, etc.\n\n    Usage:\n        def create_logger() -> FlextLogger:\n            return FlextLogger.create_module_logger("app")\n\n        container.register_factory("logger", create_logger)  # OK\n    '

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
        """

        def __new__(
            cls,
            name: str,
            bases: tuple[type, ...],
            namespace: Mapping[
                str,
                t.Container | BaseModel | type | Callable[..., t.Container | BaseModel],
            ],
            **_kwargs: t.Scalar,
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
            protocols, model_bases = _ProtocolIntrospection.partition_protocol_bases(
                bases
            )
            if not model_bases:
                model_bases = [BaseModel]
            built_cls: type = super().__new__(
                cls, name, tuple(model_bases), dict(namespace)
            )
            setattr(built_cls, "__protocols__", tuple(protocols))
            if _METACLASS_STRICT:
                for protocol in protocols:
                    FlextProtocols._validate_protocol_compliance(
                        built_cls, protocol, name
                    )
            return built_cls

    class ProtocolModel(metaclass=ProtocolModelMeta):
        """Base class for Pydantic models that implement protocols.

        Enables natural multi-inheritance with protocols without metaclass
        conflicts. Protocol compliance is validated at class definition time.

        Usage:
            class MyEntity(p.ProtocolModel, p.Domain.Entity):
                name: str
                value: int

            # Check protocols at runtime
            entity = MyEntity(name="test", value=42)
            assert entity.implements_protocol(p.Domain.Entity)
            assert entity.get_protocols() == (p.Domain.Entity,)
        """

        @classmethod
        def get_protocols(cls) -> tuple[type, ...]:
            """Return all protocols this class implements.

            Returns:
                Tuple of protocol types.

            """
            return _ProtocolIntrospection.get_class_protocols(cls)

        def implements_protocol(self, protocol: type) -> bool:
            """Check if this instance implements a protocol.

            Args:
                protocol: The protocol type to check.

            Returns:
                True if this instance implements the protocol.

            """
            return _ProtocolIntrospection.check_implements_protocol(self, protocol)

    class ProtocolSettings(BaseSettings, metaclass=ProtocolModelMeta):
        """Base class for Pydantic Settings that implement protocols.

        Extends the ProtocolModel pattern to BaseSettings, enabling
        environment variable loading alongside protocol compliance.

        Usage:
            class MySettings(p.ProtocolSettings, p.Configuration.Config):
                app_name: str = Field(default="myapp")
                debug: bool = Field(default=False)

                model_config = SettingsConfigDict(env_prefix="MY_")
        """

        @classmethod
        def get_protocols(cls) -> tuple[type, ...]:
            """Return all protocols this class implements.

            Returns:
                Tuple of protocol types.

            """
            return _ProtocolIntrospection.get_class_protocols(cls)

        def implements_protocol(self, protocol: type) -> bool:
            """Check if this instance implements a protocol.

            Args:
                protocol: The protocol type to check.

            Returns:
                True if this instance implements the protocol.

            """
            return _ProtocolIntrospection.check_implements_protocol(self, protocol)

    @staticmethod
    def check_implements_protocol(
        instance: FlextProtocols.Base | t.Container,
        protocol: type,
    ) -> bool:
        """Check if an instance's class implements a protocol.

        Args:
            instance: The item to check.
            protocol: The protocol to check against.

        Returns:
            True if the instance implements the protocol.

        """
        return FlextProtocols._check_protocol_compliance(instance, protocol)

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
            class_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
            for protocol in protocols:
                FlextProtocols._validate_protocol_compliance(cls, protocol, class_name)
            setattr(cls, "__protocols__", tuple(protocols))

            def _instance_implements_protocol(
                self: FlextProtocols.Base | t.Container,
                protocol: type,
            ) -> bool:
                return FlextProtocols._check_protocol_compliance(self, protocol)

            setattr(cls, "implements_protocol", _instance_implements_protocol)

            def _class_get_protocols(kls: type) -> tuple[type, ...]:
                return _ProtocolIntrospection.get_class_protocols(kls)

            setattr(cls, "get_protocols", classmethod(_class_get_protocols))
            return cls

        return decorator

    @staticmethod
    def is_protocol(target_cls: type) -> bool:
        """Check if a class is a typing.Protocol."""
        return _ProtocolIntrospection.is_protocol(target_cls)

    implements_protocol = check_implements_protocol


p = FlextProtocols
__all__ = ["FlextProtocols", "p"]
