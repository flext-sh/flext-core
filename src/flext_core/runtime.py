"""Runtime bridge exposing external libraries with dispatcher-safe boundaries.

**ARCHITECTURE LAYER 0.5** - Integration Bridge (Minimal Dependencies)

This module provides runtime utilities that consume patterns from c and
expose external library APIs to higher-level modules, maintaining proper dependency
hierarchy while eliminating code duplication. Implements structural typing via
p (duck typing - no inheritance required).

**Protocol Compliance** (Structural Typing):
FlextRuntime provides utility methods without requiring protocol compliance.
It serves as a bridge to external libraries (structlog, dependency-injector)
and provides type guards and serialization utilities.

**Core Components** (8 functional categories):
1. **Type Guard Utilities** - Pattern-based type validation (email, URL, phone, UUID, path, JSON)
2. **Serialization Utilities** - Safe object-to-dict conversion without circular imports
3. **Type Introspection** - Optional type checking, generic arg extraction
4. **Sequence Type Checking** - Sequence type validation via typing module
5. **External Library Access** - Direct access to structlog, dependency-injector
6. **Structured Logging Configuration** - FLEXT-configured structlog setup
7. **Application Integration** - Optional integration helpers for service layer
8. **Context Correlation** - Service resolution and domain event tracking

**External Library Integration** (Zero Circular Dependency Risk):
- structlog: Advanced structured logging configuration
- dependency-injector: Containers and providers for DI integration
- NO imports from higher layers (result.py, container.py, etc.)
- Pure Layer 0.5 implementation - safe from circular imports

**Usage** (simple runtime aliases only; no alias registry):
- Package __init__: c = FlextConstants, m = FlextModels, etc. (direct assignment only). Never use FlextRuntime.Aliases.
- Facades (e.g. FlextUtilities) expose staticmethod aliases from external subclasses so call sites get one flat namespace (u.foo, u.bar), no subdivision (no u.Mapper.foo).
- At call sites use project namespace only: c, m, r, t, u, p, d, e, h, s, x from project __init__. Subprojects: access only via that project's namespace; no cross-project alias subdivision. MRO protocol only; direct methods.
- Runtime helpers via x (e.g. x.create_instance, x.is_dict_like).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import atexit
import contextlib
import inspect
import json
import logging
import queue
import secrets
import string
import sys
import threading
import typing
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, ClassVar, Self, TypeGuard, cast, override

import structlog
from dependency_injector import containers, providers, wiring
from pydantic import BaseModel, TypeAdapter
from structlog.processors import (
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
)
from structlog.stdlib import add_log_level

from flext_core._runtime_metadata import Metadata
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.typings import T, t

_module_logger = logging.getLogger(__name__)


class _LazyMetadata:
    """Descriptor for lazy-loading Metadata class."""

    def __get__(
        self,
        obj: object,
        objtype: type | None = None,
    ) -> type[_runtime_metadata.Metadata]:

        # Cache the loaded class on the class itself
        setattr(objtype or FlextRuntime, "Metadata", Metadata)
        return Metadata


class FlextRuntime:
    """Expose structlog, DI providers, and validation helpers to higher layers.

    **ARCHITECTURE LAYER 0.5** - Integration Bridge with minimal dependencies

    Provides runtime utilities that consume patterns from c and expose
    external library APIs to higher-level modules, maintaining proper dependency
    hierarchy while eliminating code duplication. Implements structural typing via
    p (duck typing through method signatures, no inheritance required).

    **Architecture** (Layer 0.5 - Integration Bridge):
    FlextRuntime provides utility methods without requiring protocol compliance.
    It serves as a bridge to external libraries (structlog, dependency-injector)
    and provides type guards and serialization utilities following stdlib patterns.

    **Type Guard Utilities** (5+ pattern-based validators):
    1. **is_valid_phone()** - International phone number validation
    2. **is_valid_json()** - JSON string validation via json.loads()
    3. **is_valid_identifier()** - Python identifier validation
    4. **is_dict_like()** / **is_list_like()** - Collection type checking

    **Serialization Utilities** (Safe multi-strategy conversion):
       - Strategy 1: Pydantic v2 model_dump()
       - Strategy 2: Legacy Pydantic dict()
       - Strategy 3: Object __dict__ attribute
       - Strategy 4: Direct dict detection
    2. **safe_get_attribute()** - Safe attribute access without AttributeError
    3. All strategies fail gracefully with logging, never raise exceptions

    **Type Introspection** (Typing module utilities):
    2. **extract_generic_args()** - Extract type arguments from generics
    3. **is_sequence_type()** - Detect sequence types via collections.abc

    **External Library Access** (Direct module access):
    1. **structlog()** - Return imported structlog module
    2. **dependency_providers()** - Return dependency-injector providers
    3. **dependency_containers()** - Return dependency-injector containers

    **Structured Logging Configuration**:
    - **configure_structlog()** - One-time configuration with FLEXT defaults
    - **level_based_context_filter()** - Processor for log-level-specific context
    - Supports console and JSON rendering modes
    - Custom processor chain support

    **Application Integration** (Nested class):
    FlextRuntime.Integration provides optional helpers for service layer:
    1. **track_service_resolution()** - Service resolution tracking
    2. **track_domain_event()** - Domain event emission with correlation

    **Core Features** (10 runtime capabilities):
    1. **Type Safety** - TypeGuard utilities for pattern validation
    2. **Serialization** - Multi-strategy safe object conversion
    3. **Type Introspection** - Generic type analysis
    4. **External Libraries** - structlog and dependency-injector adapters
    5. **Structured Logging** - Production-ready logging configuration
    6. **Context Correlation** - UUID4-based correlation ID generation
    7. **Level-Based Filtering** - Log-level-specific context management
    8. **Service Integration** - Optional application-layer helpers
    9. **Domain Events** - Event tracking with correlation
    10. **Zero Circular Imports** - Foundation + bridge layers only


    **Usage** (simple runtime aliases only; no alias registry):
    - Package __init__: c = FlextConstants, m = FlextModels, etc. (direct assignment only). Never use FlextRuntime.Aliases or any registry.
    - Facades use staticmethod aliases from external subclasses so one flat namespace (no u.Mapper.foo); subprojects use project namespace only (from flext_cli import m, x; m.Foo, m.Bar).
    - At call sites use runtime aliases from project __init__: c, m, r, t, u, p, d, e, h, s, x. Access via project runtime alias only; no subdivision. MRO only; direct methods. Examples: x.create_instance(MyClass), r[T].ok(value).

    **Design Principles**:
    - Circular import prevention through foundation + bridge layers only
    - No imports from higher layers (result.py, container.py, context.py, loggings.py)
    - Direct structlog usage as single source of truth for context
    - Safe strategies for all risky operations (serialization)
    - Opt-in integration helpers (not forced on all modules)
    - Pattern-based validation using c (single source of truth)
    """

    _structlog_configured: ClassVar[bool] = False

    Metadata: ClassVar[type[_runtime_metadata.Metadata]] = cast(
        "type[_runtime_metadata.Metadata]",
        _LazyMetadata(),
    )  # Lazy-loaded from _runtime_metadata

    class _AsyncLogWriter:
        """Background log writer using a queue and a separate thread.

        Provides non-blocking logging by buffering log messages to a queue
        and writing them to the destination stream in a background thread.
        """

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self.queue: queue.Queue[str | None] = queue.Queue(
                maxsize=c.Logging.ASYNC_QUEUE_SIZE,
            )
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="flext-async-log-writer",
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)

        def write(self, message: str) -> None:
            """Write message to queue (non-blocking)."""
            with contextlib.suppress(queue.Full):
                self.queue.put(
                    message,
                    block=c.Logging.ASYNC_BLOCK_ON_FULL,
                )

        def flush(self) -> None:
            """Flush stream (best effort)."""
            if hasattr(self.stream, "flush"):
                with contextlib.suppress(OSError, ValueError):
                    self.stream.flush()

        def shutdown(self) -> None:
            """Stop worker thread and flush remaining messages."""
            if self.stop_event.is_set():
                return
            self.stop_event.set()
            with contextlib.suppress(queue.Full):
                self.queue.put_nowait(None)  # Best-effort sentinel
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
            self.flush()

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get(timeout=0.1)
                    if msg is None:
                        break
                    _ = self.stream.write(msg)
                    if hasattr(self.stream, "flush"):
                        _ = self.stream.flush()
                    self.queue.task_done()
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                except (OSError, ValueError, TypeError) as exc:
                    _module_logger.warning(
                        "Async log writer stream operation failed",
                        exc_info=exc,
                    )
                    with contextlib.suppress(OSError, ValueError, TypeError):
                        _ = self.stream.write("Error in async log writer\n")

    _async_writer: ClassVar[_AsyncLogWriter | None] = None

    @classmethod
    def is_structlog_configured(cls) -> bool:
        """Check if structlog has been configured.

        Returns:
            bool: True if structlog is configured, False otherwise

        """
        return cls._structlog_configured

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use).

        This method provides lazy configuration of structlog. It checks if structlog
        has already been configured, and if not, configures it with default settings.
        This ensures structlog is ready for use without requiring explicit configuration
        at application startup.

        Example:
            >>> # Automatic configuration on first logger creation
            >>> logger = FlextLogger.create_module_logger(__name__)
            >>> # structlog is now configured automatically

        """
        if not cls._structlog_configured:
            cls.configure_structlog()
            cls._structlog_configured = True

    # NOTE: Use c.Settings.LogLevel directly - no aliases per FLEXT standards

    # =========================================================================
    # TYPE-SAFE FACTORY UTILITIES
    # =========================================================================

    @staticmethod
    def create_instance[T](class_type: type[T]) -> T:
        """Type-safe factory for creating instances via object.__new__.

        Business Rule: Creates instances using object.__new__() for type-safe
        instantiation without calling __init__. Validates instance type after creation
        to ensure type safety. This pattern eliminates the need for type: ignore comments
        when using object.__new__() directly. Used by factory patterns throughout FLEXT
        for creating instances without side effects from __init__.

        Audit Implication: Instance creation is validated at runtime, ensuring type
        safety for audit trails. All instances are verified to be of expected type
        before being returned. Used by dependency injection and factory patterns.

        This helper function properly types object.__new__() calls, eliminating
        the need for type: ignore comments. Use this instead of direct object.__new__()
        calls in factory patterns.

        Args:
            class_type: The class to instantiate

        Returns:
            An instance of type T

        Raises:
            TypeError: If object.__new__() does not return instance of expected type

        Example:
            >>> instance = FlextRuntime.create_instance(MyClass)
            >>> # instance is properly typed as MyClass

        """
        instance = object.__new__(class_type)
        if not isinstance(instance, class_type):
            msg = f"object.__new__ did not return instance of {class_type.__name__}"
            raise TypeError(msg)
        return instance

    class Bootstrap:
        """Bootstrap helpers for instantiation without calling ``__init__``."""

        @staticmethod
        def create_instance[T](class_type: type[T]) -> T:
            """Create instance using the runtime low-level constructor path."""
            return FlextRuntime.create_instance(class_type)

    # =========================================================================
    # TYPE GUARD UTILITIES
    # =========================================================================

    @staticmethod
    def is_dict_like(
        value: t.ConfigMapValue,
    ) -> TypeGuard[t.ConfigMap]:
        """Type guard to check if value is dict-like.

        Note:
            ``value`` remains broad because this guard is a boundary utility used
            by normalization paths that accept full ``t.ConfigMapValue``.

        Args:
            value: Value to check

        Returns:
            True if value is a ConfigMap or dict-like object, False otherwise

        """
        match value:
            case t.ConfigMap():
                return True
            case Mapping():
                return True
            case _:
                if value is None:
                    return False
                keys = value.keys if hasattr(value, "keys") else None
                items = value.items if hasattr(value, "items") else None
                get = value.get if hasattr(value, "get") else None
                if not (callable(keys) and callable(items) and callable(get)):
                    return False
                try:
                    keys_values = keys()
                    item_values = items()
                    tuple_entry_size = 2
                    if not isinstance(keys_values, Sequence):
                        return False
                    if not isinstance(item_values, Sequence):
                        return False
                    if not all(isinstance(key, str) for key in keys_values):
                        return False
                    for entry in item_values:
                        if not (
                            isinstance(entry, tuple) and len(entry) == tuple_entry_size
                        ):
                            return False
                    return True
                except (AttributeError, TypeError):
                    return False

    @staticmethod
    def is_list_like(
        value: t.ConfigMapValue,
    ) -> TypeGuard[Sequence[t.ConfigMapValue]]:
        """Type guard to check if value is list-like."""
        return isinstance(value, list)

    @staticmethod
    def _is_scalar(
        value: t.ConfigMapValue,
    ) -> TypeGuard[t.ScalarValue]:
        """Check if value is a scalar type accepted by t.ScalarValue."""
        match value:
            case datetime() | None:
                return True
            case _:
                return False

    @staticmethod
    def normalize_to_general_value(
        val: t.ConfigMapValue,
    ) -> t.ConfigMapValue:
        """Normalize any value to t.ConfigMapValue recursively.

        Converts arbitrary objects, t.ConfigMap, list[t.ConfigMapValue], and other types
        to m.ConfigMap, Sequence[t.ConfigMapValue], etc.
        This is the central conversion function for type safety.

        Args:
            val: Value to normalize (accepts object for flexibility with generics)

        Returns:
            Normalized value compatible with t.ConfigMapValue

        Examples:
            >>> FlextRuntime.normalize_to_general_value({"key": "value"})
            {'key': 'value'}
            >>> FlextRuntime.normalize_to_general_value({"nested": {"inner": 123}})
            {'nested': {'inner': 123}}
            >>> FlextRuntime.normalize_to_general_value([1, 2, {"a": "b"}])
            [1, 2, {'a': 'b'}]

        """
        if FlextRuntime._is_scalar(val):
            return val

        if isinstance(val, Path):
            return str(val)

        if isinstance(val, BaseModel):
            dumped_value: t.ConfigMapValue = TypeAdapter(
                t.ConfigMapValue,
            ).validate_python(val.model_dump())
            return FlextRuntime.normalize_to_general_value(dumped_value)

        if FlextRuntime.is_dict_like(val):
            dict_v = val.root if hasattr(val, "root") else val
            result: dict[str, t.ConfigMapValue] = {}
            for k, v in dict_v.items():
                result[str(k)] = FlextRuntime.normalize_to_general_value(v)
            return result

        if FlextRuntime.is_list_like(val):
            return [FlextRuntime.normalize_to_general_value(item) for item in val]
        return val

    @staticmethod
    def normalize_to_metadata_value(
        val: t.ConfigMapValue,
    ) -> t.MetadataAttributeValue:
        """Normalize any value to t.MetadataAttributeValue.

        t.MetadataAttributeValue is more restrictive than t.ConfigMapValue,
        so we need to normalize nested structures to flat types.
        This method is in FlextRuntime (Tier 0.5) to avoid circular dependencies.

        Args:
            val: Value to normalize

        Returns:
            t.MetadataAttributeValue: Normalized value compatible with Metadata attributes

        Example:
            >>> FlextRuntime.normalize_to_metadata_value("test")
            'test'
            >>> FlextRuntime.normalize_to_metadata_value({"key": "value"})
            {'key': 'value'}
            >>> FlextRuntime.normalize_to_metadata_value([1, 2, 3])
            [1, 2, 3]

        """
        if FlextRuntime._is_scalar(val):
            result_scalar: t.MetadataAttributeValue = val
            return result_scalar

        if isinstance(val, BaseModel):
            dumped_value: t.ConfigMapValue = TypeAdapter(
                t.ConfigMapValue,
            ).validate_python(val.model_dump())
            return FlextRuntime.normalize_to_metadata_value(dumped_value)

        if FlextRuntime.is_dict_like(val):
            raw_mapping = val.root if hasattr(val, "root") else val
            normalized_mapping: dict[str, t.ConfigMapValue] = {}
            for key, value in raw_mapping.items():
                normalized_mapping[str(key)] = FlextRuntime.normalize_to_general_value(
                    value,
                )
            return json.dumps(normalized_mapping)

        if FlextRuntime.is_list_like(val):
            # Convert to list of MetadataAttributeValue scalars (including datetime)
            result_list: list[str | int | float | bool | datetime | None] = []
            for item in val:
                if FlextRuntime._is_scalar(item):
                    result_list.append(item)
                else:
                    result_list.append(str(item))
            # Explicit annotation to ensure MetadataAttributeValue return type
            result_list_typed: t.MetadataAttributeValue = result_list
            return result_list_typed
        # Return type is t.MetadataAttributeValue (str type)
        result_str: t.MetadataAttributeValue = str(val)
        return result_str

    @staticmethod
    def is_valid_json(
        value: t.GeneralValueType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is valid JSON string.

        Business Rule: Validates JSON strings using json.loads() for parsing.
        Returns TypeGuard[str] for type narrowing in conditional blocks.
        Catches JSONDecodeError and ValueError for safe validation. Used for
        validating JSON payloads before deserialization.

        Audit Implication: JSON validation ensures audit trail completeness by
        validating JSON payloads before storage. All JSON strings are validated
        before being used in audit systems.

        Args:
            value: Value to check

        Returns:
            True if value is a valid JSON string, False otherwise

        """
        if not isinstance(value, str):
            return False
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    @staticmethod
    def is_valid_identifier(
        value: t.GeneralValueType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is a valid Python identifier."""
        return isinstance(value, str) and value.isidentifier()

    @staticmethod
    def is_base_model(obj: t.ConfigMapValue) -> TypeGuard[object]:
        """Type guard to narrow object to BaseModel (part of PayloadValue).

        This allows isinstance checks to narrow types for FlextRuntime methods
        that accept PayloadValue (which includes BaseModel).
        """
        match obj:
            case BaseModel():
                return True
            case _:
                return False

    # =========================================================================
    # SERIALIZATION UTILITIES (No flext_core imports)
    # =========================================================================

    @staticmethod
    def safe_get_attribute(
        obj: t.ConfigMapValue,
        attr: str,
        default: t.ConfigMapValue = None,
    ) -> t.ConfigMapValue:
        """Safe attribute access without raising AttributeError.

        Business Rule: Accesses object attributes safely using getattr() with
        default value. Never raises AttributeError, always returns default if
        attribute doesn't exist. Used for safe introspection of arbitrary objects
        without type checking.

        Audit Implication: Safe attribute access ensures audit trail completeness
        by preventing AttributeError exceptions during object introspection. All
        attribute access is safe and logged appropriately.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute does not exist

        Returns:
            Attribute value or default

        """
        return getattr(obj, attr) if hasattr(obj, attr) else default

    @staticmethod
    def extract_generic_args(
        type_hint: t.TypeHintSpecifier,
    ) -> tuple[t.GenericTypeArgument | type[Mapping[str, t.ConfigMapValue]], ...]:
        """Extract generic type arguments from a type hint.

        Business Rule: Extracts generic type arguments from type hints using
        typing.get_args() for standard generics,         and mapping for type
        aliases. Returns empty tuple if no arguments found or on error. Used for
        type introspection and generic type analysis.

        Audit Implication: Type argument extraction enables audit trail completeness
        by providing type information for generic types. All type introspection is
        safe and never raises exceptions.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            # First try the standard typing.get_args
            args = typing.get_args(type_hint)
            if args:
                return args

            # Check if it's a known type alias
            if hasattr(type_hint, "__name__"):
                type_name = type_hint.__name__ if hasattr(type_hint, "__name__") else ""
                # Handle common type aliases - use actual type objects
                # GenericTypeArgument = str | type[t.ConfigMapValue]
                # Type objects (str, int, float, bool) are valid GenericTypeArgument
                # since they represent type[T] where T is a scalar t.ConfigMapValue
                if type_name in {"StringList", "List"}:
                    return (str,)
                if type_name == "IntList":
                    return (int,)
                if type_name == "FloatList":
                    return (float,)
                if type_name == "BoolList":
                    return (bool,)
                if type_name in {"Dict", "StringDict"}:
                    return (str, str)
                if type_name == "NestedDict":
                    return (str, dict)
                if type_name == "IntDict":
                    return (str, int)
                if type_name == "FloatDict":
                    return (str, float)
                if type_name == "BoolDict":
                    return (str, bool)

            return ()
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):
            # Defensive: typing module failures are extremely rare
            return ()

    @staticmethod
    def is_sequence_type(type_hint: t.TypeHintSpecifier) -> bool:
        """Check if type hint represents a sequence type (list, tuple, etc.).

        Business Rule: Checks if type hint represents a sequence type using
        typing.get_origin() and issubclass() checks. Supports both standard
        generics and type aliases. Returns False on error for safe type checking.

        Audit Implication: Sequence type checking enables audit trail completeness
        by providing type information for sequence types. All type checking is
        safe and never raises exceptions.

        Args:
            type_hint: Type hint to check

        Returns:
            True if type hint is a sequence type, False otherwise

        """
        try:
            origin = typing.get_origin(type_hint)
            if origin is not None and hasattr(origin, "__mro__"):
                if origin in {list, tuple}:
                    return True
                return Sequence in origin.__mro__

            if type_hint in {list, tuple, str}:
                return True

            # Check if the type itself is a sequence subclass (for type aliases)
            hint_mro = type_hint.__mro__ if hasattr(type_hint, "__mro__") else None
            if hint_mro is not None and Sequence in hint_mro:
                return True

            # Check __name__ for type aliases like StringList
            type_name = type_hint.__name__ if hasattr(type_hint, "__name__") else None
            return bool(
                type_name is not None
                and type_name
                in {"StringList", "IntList", "FloatList", "BoolList", "List"},
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):
            # Defensive: typing/issubclass failures are extremely rare
            return False

    @staticmethod
    def structlog() -> ModuleType:
        """Return the imported structlog module."""
        return structlog

    @staticmethod
    def get_logger(
        name: str | None = None,
    ) -> p.Log.StructlogLogger:
        """Get structlog logger instance - same structure/config used by FlextLogger.

        Returns the exact same structlog logger instance that FlextLogger uses internally.
        This ensures consistent logging structure across the entire FLEXT ecosystem.

        Args:
            name: Logger name (module name). Defaults to __name__ of caller.

        Returns:
            Logger: Typed structlog logger instance (same as FlextLogger.logger).

        Note:
            FlextLogger internally uses: FlextRuntime.get_logger(name).bind(**context)
            This method returns the base logger before context binding.

        """
        if name is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                name = __name__
        # structlog.get_logger returns BoundLoggerLazyProxy which implements p.Log.StructlogLogger protocol
        # All methods (debug, info, warning, error, etc.) are available directly from structlog logger
        # p.Log.StructlogLogger protocol is compatible with structlog's return type via structural typing
        logger: p.Log.StructlogLogger = structlog.get_logger(name)
        return logger

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    class DependencyIntegration:
        """Centralize dependency-injector wiring with provider helpers.

        This bridge keeps dependency-injector usage confined to L1 while
        exposing a narrow API for higher layers. Factories and configuration
        providers are materialized here to avoid duplicate dictionaries or
        direct imports of ``dependency_injector`` outside the runtime module.
        A small DeclarativeContainer captures config/resources so L3 callers
        never import dependency-injector directly.
        """

        class BridgeContainer(containers.DeclarativeContainer):
            """Declarative container grouping config and resource modules."""

            config = providers.Configuration()
            services = providers.DependenciesContainer()
            resources = providers.DependenciesContainer()

        Provide = wiring.Provide
        inject = staticmethod(wiring.inject)

        @classmethod
        def create_layered_bridge(
            cls,
            config: t.ConfigMap | None = None,
        ) -> tuple[
            containers.DeclarativeContainer,
            containers.DynamicContainer,
            containers.DynamicContainer,
        ]:
            """Create a DeclarativeContainer bridged to dynamic modules.

            Returns a tuple with the declarative bridge plus dynamic containers
            for services and resources. The bridge isolates dependency-injector
            usage to L1/L2 while allowing higher layers to work only with
            FlextContainer's APIs. The declarative bridge exposes ``config``,
            ``services``, and ``resources`` providers, plus ``Provide``/``inject``
            helpers re-exported by the runtime. See
            :class:`DependencyIntegration.BridgeContainer` for attributes
            available to wire modules, classes, and functions.
            """
            bridge = cls.BridgeContainer()
            service_module = containers.DynamicContainer()
            resource_module = containers.DynamicContainer()
            # override() returns None or provider instance - we don't need the return value
            # Type narrowing: bridge.services and bridge.resources are DependenciesContainer instances
            # override() accepts DynamicContainer - dependency-injector types are incomplete
            # Use dynamic call to avoid type checker issues with incomplete stubs
            # Both override() calls return context managers that we don't use
            services_provider = bridge.services
            resources_provider = bridge.resources
            # Call override via getattr to work around incomplete type stubs
            _ = services_provider.override(service_module)
            _ = resources_provider.override(resource_module)
            _ = cls.bind_configuration_provider(bridge.config, config)
            return bridge, service_module, resource_module

        @classmethod
        def create_container(
            cls,
            *,
            config: t.ConfigMap | None = None,
            services: Mapping[str, t.RegisterableService] | None = None,
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
            resources: Mapping[str, Callable[[], t.ConfigMapValue]] | None = None,
            wire_modules: Sequence[ModuleType] | None = None,
            wire_packages: Sequence[str] | None = None,
            wire_classes: Sequence[type] | None = None,
            factory_cache: bool = True,
        ) -> containers.DynamicContainer:
            """Create a DynamicContainer with optional pre-registration and wiring.

            Args:
                config: Optional configuration mapping bound to ``config`` provider.
                services: Optional mapping of provider names to concrete objects
                    registered via ``providers.Object``.
                factories: Optional mapping of provider names to factory callables
                    registered via ``providers.Singleton`` (default) or
                    ``providers.Factory`` when ``factory_cache=False``.
                resources: Optional mapping of provider names to resource factories
                    registered via ``providers.Resource``.
                wire_modules: Optional modules to wire immediately for ``@inject``/
                    ``Provide`` usage.
                wire_packages: Optional packages to wire immediately.
                wire_classes: Optional classes to wire immediately.
                factory_cache: Whether factories use singleton semantics. When
                    ``False``, factories are registered with ``providers.Factory``
                    to produce new instances per resolution.

            Returns:
                A dynamic container ready for immediate ``@inject`` consumption
                without manual follow-up registration calls.

            """
            di_container = containers.DynamicContainer()

            if config is not None:
                _ = cls.bind_configuration(di_container, config)

            if services:
                for name, instance in services.items():
                    _ = cls.register_object(di_container, name, instance)

            if factories:
                for name, factory in factories.items():
                    _ = cls.register_factory(
                        di_container,
                        name,
                        factory,
                        cache=factory_cache,
                    )

            if resources:
                for name, resource_factory in resources.items():
                    # register_resource[T] accepts Callable[[], T] and infers T
                    # resource_factory is Callable[[], t.ConfigMapValue] from resources
                    # T is inferred as t.ConfigMapValue, which is valid
                    _ = cls.register_resource(
                        di_container,
                        name,
                        resource_factory,
                    )

            if wire_modules or wire_packages or wire_classes:
                cls.wire(
                    di_container,
                    modules=wire_modules,
                    packages=wire_packages,
                    classes=wire_classes,
                )

            return di_container

        @staticmethod
        def bind_configuration(
            di_container: containers.DynamicContainer,
            config: t.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration mapping to the DI container.

            Uses ``providers.Configuration`` to expose values to downstream
            providers without higher layers interacting with dependency-injector
            directly.
            """
            configuration_provider = providers.Configuration()
            if config:
                configuration_provider.from_dict(dict(config))
            setattr(di_container, "config", configuration_provider)
            return configuration_provider

        @staticmethod
        def bind_configuration_provider(
            configuration_provider: providers.Configuration,
            config: t.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration directly to an existing provider."""
            if config:
                configuration_provider.from_dict(dict(config))
            return configuration_provider

        @staticmethod
        def register_object(
            di_container: containers.DynamicContainer,
            name: str,
            instance: T,
        ) -> providers.Provider[T]:
            """Register a concrete instance using ``providers.Object``.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = providers.Object(instance)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_factory(
            di_container: containers.DynamicContainer,
            name: str,
            factory: Callable[[], T],
            *,
            cache: bool = True,
        ) -> providers.Provider[T]:
            """Register a factory using Singleton/Factory providers.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = (
                providers.Singleton(factory) if cache else providers.Factory(factory)
            )
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def register_resource(
            di_container: containers.DynamicContainer,
            name: str,
            factory: Callable[[], T],
        ) -> providers.Provider[T]:
            """Register a resource provider for lifecycle-managed dependencies.

            Args:
                di_container: DynamicContainer instance for provider registration.

            """
            if hasattr(di_container, name):
                msg = f"Provider '{name}' is already registered"
                raise ValueError(msg)
            provider: providers.Provider[T] = providers.Resource(factory)
            setattr(di_container, name, provider)
            return provider

        @staticmethod
        def wire(
            container: containers.DeclarativeContainer | containers.DynamicContainer,
            *,
            modules: Sequence[ModuleType] | None = None,
            packages: Sequence[str] | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules or packages to a DeclarativeContainer or DynamicContainer for @inject usage.

            Accepts both DeclarativeContainer and DynamicContainer since dependency-injector's
            wiring.wire() accepts any container type that implements the container protocol.

            Note: packages parameter is accepted for API compatibility but not used internally.
            wiring.wire's packages parameter expects Iterable[Module] (module objects),
            but we accept Sequence[str] (package names). The actual wiring is handled by modules parameter.
            For now, we pass None for packages when it's a Sequence[str] to avoid type errors.
            The actual wiring will be handled by modules parameter.
            """
            modules_to_wire: list[ModuleType] = list(modules or [])
            if classes:
                for target_class in classes:
                    module = inspect.getmodule(target_class)
                    if module is not None:
                        modules_to_wire.append(module)

            # wiring.wire accepts both DeclarativeContainer and DynamicContainer
            # Both implement the same container interface for wiring purposes
            # Note: wiring.wire's packages parameter expects Iterable[Module] (module objects),
            # but we accept Sequence[str] (package names). We need to convert package names
            # to module objects, or pass None if packages is provided as strings.
            # For now, we pass None for packages when it's a Sequence[str] to avoid type errors.
            # The actual wiring will be handled by modules parameter.
            # packages is intentionally unused - see docstring above
            _ = packages  # Explicitly ignore to avoid unused warning
            wiring.wire(
                modules=modules_to_wire or None,
                packages=None,  # packages parameter expects Iterable[Module], not Sequence[str]
                container=container,
            )

    @staticmethod
    def level_based_context_filter(
        _logger: t.ConfigMapValue,
        method_name: str,
        event_dict: Mapping[str, t.ConfigMapValue],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Filter context variables based on log level.

        Removes context variables that are restricted to specific log levels
        when the current log level doesn't match.

        This processor handles level-prefixed context variables created by
        FlextLogger.bind_context_for_level() and removes them from logs that
        don't meet the required log level.

        Args:
            _logger: Logger instance (unused, required by structlog protocol)
            method_name: Log method name ('debug', 'info', 'warning', 'error', etc.)
            event_dict: Event dictionary with context variables

        Returns:
            Filtered event dictionary

        Example:
            Context bound with:
            >>> FlextLogger.bind_context_for_level("DEBUG", config=config_dict)
            >>> FlextLogger.bind_context_for_level("ERROR", stack_trace=trace)

            Results in:
            - DEBUG logs: include config
            - INFO logs: exclude config
            - ERROR logs: include stack_trace
            - INFO logs: exclude stack_trace

        Note:
            Log level hierarchy: DEBUG < INFO < WARNING < ERROR < CRITICAL

        """
        # Log level hierarchy (lowest to highest)
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }

        # Get current log level from method name
        current_level = level_hierarchy.get(method_name.lower(), 20)  # Default to INFO

        # Process all keys in event_dict
        # Business Rule: Build mutable dict for construction, then return as dict
        # dict is compatible with ConfigMap return type.
        # This pattern is correct: construct mutable dict, return it (dict is Mapping subtype).
        #
        # Audit Implication: This method filters log event data based on log level.
        # Used for conditional inclusion of verbose fields in structured logging.
        # Returns dict that is compatible with Mapping interface for read-only access.
        filtered_dict = {}
        for key, value in event_dict.items():
            # Check if this is a level-prefixed variable
            if key.startswith("_level_"):
                # Extract the required level and actual key
                # Format: _level_debug_config -> required_level='debug', actual_key='config'
                parts = key.split(
                    "_",
                    c.Validation.LEVEL_PREFIX_PARTS_COUNT,
                )  # Split into ['', 'level', 'debug', 'config']
                if len(parts) >= c.Validation.LEVEL_PREFIX_PARTS_COUNT:
                    required_level_name = parts[2]
                    actual_key = parts[3]
                    required_level = level_hierarchy.get(
                        required_level_name.lower(),
                        10,
                    )

                    # Only include if current level >= required level
                    if current_level >= required_level:
                        # Add with actual key (strip prefix)
                        filtered_dict[actual_key] = value
                    # Else: skip this variable (too verbose for current level)
                else:
                    # Malformed prefix, include as-is
                    filtered_dict[key] = value
            else:
                # Not level-prefixed, include as-is
                filtered_dict[key] = value

        return filtered_dict

    @classmethod
    def configure_structlog(
        cls,
        *,
        config: t.ConfigMapValue | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[object] | None = None,
        wrapper_class_factory: Callable[[], type[p.Log.StructlogLogger]] | None = None,
        logger_factory: Callable[[], p.Log.StructlogLogger] | None = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults.

        DEBUG INSTRUMENTATION ACTIVE

        Supports both config object pattern (reduced params) and individual parameters.

        Args:
            config: Optional FlextModels.Config.StructlogConfig for all params
            log_level: Numeric log level (ignored if config provided). Defaults to ``logging.INFO``.
            console_renderer: Use console renderer (ignored if config provided)
            additional_processors: Extra processors (ignored if config provided)
            wrapper_class_factory: Custom wrapper factory (ignored if config provided)
            logger_factory: Custom logger factory (ignored if config provided)
            cache_logger_on_first_use: Cache logger (ignored if config provided)

        Example (config pattern - RECOMMENDED):
            ```python
            from flext_core import FlextModels

            config = FlextModels.Config.StructlogConfig(
                log_level=20, console_renderer=True
            )
            FlextRuntime.configure_structlog(config=config)
            ```

        """
        # Extract config values or use individual parameters
        async_logging = True
        if config is not None:
            log_level = config.log_level if hasattr(config, "log_level") else log_level
            console_renderer = (
                config.console_renderer
                if hasattr(config, "console_renderer")
                else console_renderer
            )
            additional_processors_from_config = (
                config.additional_processors
                if hasattr(config, "additional_processors")
                else None
            )
            if additional_processors_from_config:
                additional_processors = additional_processors_from_config
            wrapper_class_factory = (
                config.wrapper_class_factory
                if hasattr(config, "wrapper_class_factory")
                else wrapper_class_factory
            )
            logger_factory = (
                config.logger_factory
                if hasattr(config, "logger_factory")
                else logger_factory
            )
            cache_logger_on_first_use = (
                config.cache_logger_on_first_use
                if hasattr(config, "cache_logger_on_first_use")
                else cache_logger_on_first_use
            )
            async_logging = (
                config.async_logging if hasattr(config, "async_logging") else True
            )

        # Single guard - no redundant checks
        if cls._structlog_configured:
            return

        level_to_use = log_level if log_level is not None else logging.INFO

        module = structlog

        processors: list[object] = [
            module.contextvars.merge_contextvars,
            add_log_level,
            # CRITICAL: Level-based context filter (must be after merge_contextvars and add_log_level)
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
        ]
        if additional_processors:
            # additional_processors is Sequence[object] - structlog processors are callables
            # Add callable processors to the list
            processors.extend(proc for proc in additional_processors if callable(proc))

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:
            # Tested but not covered: structlog configures once per process
            processors.append(JSONRenderer())

        # Configure structlog with processors and logger factory
        # structlog.configure accepts specific types, but we construct them dynamically

        wrapper_arg: type[p.Log.StructlogLogger] | None = None
        if wrapper_class_factory is not None:
            wrapper_arg = cast("type[p.Log.StructlogLogger]", wrapper_class_factory())
        else:
            wrapper_arg = cast(
                "type[p.Log.StructlogLogger]",
                module.make_filtering_bound_logger(level_to_use),
            )

        # Determine logger factory (handle async buffering)
        # structlog accepts various factory types - we use object to accept all
        factory_to_use: Callable[..., object]
        if logger_factory is not None:
            # Use the provided factory directly (Callable[[], p.Log.StructlogLogger])
            factory_to_use = logger_factory
        elif async_logging:
            # Default factory handling with async buffering
            # Use cached async writer or create new one
            if cls._async_writer is None:
                cls._async_writer = cls._AsyncLogWriter(sys.stdout)
            # PrintLoggerFactory accepts file-like objects with write method
            # _AsyncLogWriter has write/flush methods (duck-typed TextIO)
            # Use getattr to call PrintLoggerFactory with duck-typed file arg
            print_logger_factory = (
                module.PrintLoggerFactory
                if hasattr(module, "PrintLoggerFactory")
                else None
            )
            if print_logger_factory is not None:
                factory_to_use = print_logger_factory(
                    file=cast("Any", cls._async_writer)
                )
            else:
                factory_to_use = module.PrintLoggerFactory()
        else:
            # Default factory without async (PrintLoggerFactory instance)
            factory_to_use = module.PrintLoggerFactory()

        # Call configure directly with constructed arguments
        # Processors are dynamically constructed callables that match structlog's Processor protocol
        # structlog.configure accepts processors as Sequence[Processor] or list[Processor]
        # Our processors list contains valid Processor objects, pass directly
        # Use getattr to call configure with processors as Sequence
        configure_fn = module.configure if hasattr(module, "configure") else None
        if configure_fn is not None and callable(configure_fn):
            _ = configure_fn(
                processors=cast("Any", processors),
                wrapper_class=wrapper_arg,
                logger_factory=factory_to_use,
                cache_logger_on_first_use=cache_logger_on_first_use,
            )

        cls._structlog_configured = True

    @classmethod
    def reconfigure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: list[t.ConfigMapValue] | None = None,
    ) -> None:
        """Force reconfigure structlog (ignores is_configured checks).

        **USE ONLY when CLI params override config defaults.**

        For initial configuration, use configure_structlog().
        This method forces reconfiguration,
        allowing CLI parameters to override the initial configuration.

        **Layer 2 Responsibility (flext-cli MODIFIER)**:
        - flext-core: Uses configure_structlog() for initial setup
        - flext-cli: Uses reconfigure_structlog() to override via CLI params
        - Applications: Inherit automatically, zero manual code

        Args:
            log_level: Numeric log level to reconfigure
            console_renderer: Use console renderer vs JSON
            additional_processors: Extra processors to add

        Example:
            ```python
            # CLI override: User passed --debug flag (use runtime aliases)
            from flext_core import c
            from flext_core.runtime import FlextRuntime

            FlextRuntime.reconfigure_structlog(
                log_level=c.Settings.LogLevel.DEBUG.value,
                console_renderer=True,
            )
            ```

        Notes:
            - Resets structlog state completely
            - Resets FLEXT configuration flags
            - Calls configure_structlog() with fresh state
            - Safe to call multiple times

        """
        # Reset structlog state (makes is_configured() return False)
        module = structlog
        module.reset_defaults()

        # Shutdown async writer if exists
        if cls._async_writer:
            cls._async_writer.shutdown()
            cls._async_writer = None

        # Reset FLEXT configuration flag (single source of truth)
        cls._structlog_configured = False

        # NOTE: FlextLogger no longer maintains its own flag - it checks FlextRuntime directly
        # This eliminates circular import and redundant state tracking

        # Now configure with new settings (guards will be False)
        cls.configure_structlog(
            log_level=log_level,
            console_renderer=console_renderer,
            additional_processors=additional_processors,
        )

    @classmethod
    def reset_structlog_state_for_testing(cls) -> None:
        """Reset structlog configuration state for testing purposes.

        Business Rule: Testing-only method to reset structlog configuration state.
        This allows tests to verify structlog configuration from scratch.
        Should only be used in test fixtures.

        Implications for Audit:
        - Modifies private ClassVar _structlog_configured
        - Resets structlog module defaults
        - Breaks configured state temporarily
        - Must be called before testing structlog configuration

        """
        structlog.reset_defaults()
        cls._structlog_configured = False

    @staticmethod
    def enable_runtime_checking() -> bool:
        """Return whether runtime validation helpers are enabled."""
        return True

    # =========================================================================
    # RESULT FACTORY METHODS (Core implementation - NO FlextResult import)
    # =========================================================================
    # ARCHITECTURE: runtime.py is Tier 0.5, result.py is Tier 1.
    # FlextResult MUST import from runtime, NOT the other way around.
    # RuntimeResult is a lightweight Result implementation that
    # can be used by higher layers without circular imports.
    # =========================================================================

    class RuntimeResult[T]:
        """Lightweight Result implementation for Tier 0.5.

        This class implements p.Result without depending
        on FlextResult, allowing runtime.py to create results that can be used
        by higher layers. FlextResult can wrap or extend this as needed.

        **Type Compatibility**:
        - Implements p.Result[T] protocol (structural typing)
        - Compatible with FlextResult[T] for mypy type checking
        - Both types can be used interchangeably where p.Result[T] is expected

        **Mypy Recognition**:
        - RuntimeResult[T] is recognized by mypy as compatible with p.Result[T]
        - Functions accepting p.Result[T] will accept both RuntimeResult[T] and FlextResult[T]
        - Use p.Result[T] in function signatures to accept both types
        """

        __slots__ = (
            "_error",
            "_error_code",
            "_error_data",
            "_exception",
            "_is_success",
            "_value",
        )

        def __init__(
            self,
            value: T | None = None,
            error: str | None = None,
            error_code: str | None = None,
            error_data: t.ConfigMap | None = None,
            *,
            is_success: bool = True,
        ) -> None:
            """Initialize RuntimeResult."""
            super().__init__()
            self._value = value
            self._error = error
            self._error_code = error_code
            self._error_data = error_data
            self._is_success = is_success
            self._exception: BaseException | None = None

        @property
        def value(self) -> T:
            """Get the success value.

            ARCHITECTURAL NOTE: FlextCore never returns None on success, so this
            property always returns a valid value when is_success is True.

            Raises:
                RuntimeError: If result is failure (consistent with FlextResult)

            """
            if not self._is_success:
                msg = f"Cannot access value of failed result: {self._error}"
                raise RuntimeError(msg)
            # ARCHITECTURAL INVARIANT: FlextCore never returns None on success
            # When is_success=True, _value MUST be non-None and of type T
            # This is enforced by all factory methods (ok(), success(), from_*())
            if self._value is None:
                msg = "Invariant violation: successful result has None value"
                raise RuntimeError(msg)
            return self._value

        @property
        def result(self) -> Self:
            """Access internal result for protocol compatibility.

            RuntimeResult doesn't use an internal Result wrapper like FlextResult,
            so this returns self for protocol compatibility.
            """
            return self

        @property
        def is_success(self) -> bool:
            """Check if result is successful."""
            return self._is_success

        @property
        def is_failure(self) -> bool:
            """Check if result is a failure."""
            return not self._is_success

        @property
        def data(self) -> T:
            """Return success data alias for protocol compatibility."""
            return self.value

        @property
        def error(self) -> str | None:
            """Get the error message."""
            return self._error

        @property
        def error_code(self) -> str | None:
            """Get the error code."""
            return self._error_code

        @property
        def error_data(self) -> t.ConfigMap | None:
            """Get the error data."""
            return self._error_data

        @property
        def exception(self) -> BaseException | None:
            """Get the exception if one was captured."""
            return self._exception

        def unwrap(self) -> T:
            """Unwrap the success value or raise RuntimeError."""
            if not self._is_success:
                msg = f"Cannot unwrap failed result: {self._error}"
                raise RuntimeError(msg)
            return self.value

        def unwrap_or(self, default: T) -> T:
            """Return the success value or the default if failed."""
            if self.is_success:
                return self.value
            return default

        def unwrap_or_else(self, func: Callable[[], T]) -> T:
            """Return the success value or call func if failed."""
            if self.is_success:
                return self.value
            return func()

        def map[U](self, func: Callable[[T], U]) -> FlextRuntime.RuntimeResult[U]:
            """Transform success value using function."""
            if self.is_success:
                try:
                    return FlextRuntime.RuntimeResult(
                        value=func(self.value),
                        is_success=True,
                    )
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as e:
                    _module_logger.debug(
                        "RuntimeResult.map callable failed",
                        exc_info=e,
                    )
                    return FlextRuntime.RuntimeResult(error=str(e), is_success=False)
            return FlextRuntime.RuntimeResult(
                error=self._error or "",
                error_code=self._error_code,
                error_data=self._error_data,
                is_success=False,
            )

        def flat_map[U](
            self,
            func: Callable[[T], FlextRuntime.RuntimeResult[U]],
        ) -> FlextRuntime.RuntimeResult[U]:
            """Chain operations returning RuntimeResult."""
            if self.is_success:
                return func(self.value)
            return FlextRuntime.RuntimeResult(
                error=self._error or "",
                error_code=self._error_code,
                error_data=self._error_data,
                is_success=False,
            )

        def and_then[U](
            self,
            func: Callable[[T], FlextRuntime.RuntimeResult[U]],
        ) -> FlextRuntime.RuntimeResult[U]:
            """Alias for flat_map to support railway naming conventions."""
            return self.flat_map(func)

        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U:
            """Fold result into single value (catamorphism)."""
            if self.is_success:
                return on_success(self.value)
            return on_failure(self._error or "")

        def tap(self, func: Callable[[T], None]) -> FlextRuntime.RuntimeResult[T]:
            """Apply side effect to success value, return unchanged."""
            if self._is_success and self._value is not None:
                func(self._value)
            return self

        def tap_error(
            self,
            func: Callable[[str], None],
        ) -> FlextRuntime.RuntimeResult[T]:
            """Apply side effect to error, return unchanged."""
            if not self._is_success:
                func(self._error or "")
            return self

        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextRuntime.RuntimeResult[T]:
            """Transform error message."""
            if not self._is_success:
                return FlextRuntime.RuntimeResult(
                    error=func(self._error or ""),
                    error_code=self._error_code,
                    error_data=self._error_data,
                    is_success=False,
                )
            return self

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> FlextRuntime.RuntimeResult[T]:
            """Filter success value using predicate."""
            if self.is_success and not predicate(self.value):
                return FlextRuntime.RuntimeResult(
                    error="Filter predicate failed",
                    is_success=False,
                )
            return self

        def alt(self, func: Callable[[str], str]) -> FlextRuntime.RuntimeResult[T]:
            """Apply alternative function on failure."""
            if not self._is_success:
                return FlextRuntime.RuntimeResult(
                    error=func(self._error or ""),
                    error_code=self._error_code,
                    error_data=self._error_data,
                    is_success=False,
                )
            return self

        def lash(
            self,
            func: Callable[[str], FlextRuntime.RuntimeResult[T]],
        ) -> FlextRuntime.RuntimeResult[T]:
            """Apply recovery function on failure."""
            if not self._is_success:
                return func(self._error or "")
            return self

        def recover(self, func: Callable[[str], T]) -> FlextRuntime.RuntimeResult[T]:
            """Recover from failure with fallback value."""
            if not self._is_success:
                fallback_value = func(self._error or "")
                return FlextRuntime.RuntimeResult(value=fallback_value, is_success=True)
            return self

        def flow_through[U](
            self,
            *funcs: Callable[[T | U], FlextRuntime.RuntimeResult[U]],
        ) -> FlextRuntime.RuntimeResult[T] | FlextRuntime.RuntimeResult[U]:
            """Chain multiple operations in sequence.

            Returns:
                RuntimeResult[T] if chain short-circuits on first failure or no funcs,
                RuntimeResult[U] if all funcs applied successfully.

            """
            # Start with self (RuntimeResult[T])
            current: FlextRuntime.RuntimeResult[T] | FlextRuntime.RuntimeResult[U] = (
                self
            )
            for func in funcs:
                if current.is_success:
                    # Use value property - guaranteed to be T | U when is_success is True
                    result_value = current.value
                    if result_value is not None:
                        # func returns RuntimeResult[U]
                        current = func(result_value)
                    else:
                        break
                else:
                    break
            # Return the current result - either T (original/failed) or U (transformed)
            return current

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
            return self.unwrap_or(default)

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            return self.is_success

        @override
        def __repr__(self) -> str:
            """String representation using short alias 'r' for brevity."""
            if self.is_success:
                return f"r.ok({self.value!r})"
            return f"r.fail({self.error!r})"

        def __enter__(self) -> Self:
            """Context manager entry."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Context manager exit."""

        def _protocol_name(self) -> str:
            """Return the protocol name for BaseProtocol compliance.

            Required by FlextProtocols.BaseProtocol to enable protocol introspection
            and ensure FlextResult satisfies the ResultLike protocol.
            """
            return "RuntimeResult"

        @classmethod
        def ok(cls, value: T) -> FlextRuntime.RuntimeResult[T]:
            """Create successful result wrapping data.

            Business Rule: Creates successful RuntimeResult wrapping value. Raises ValueError
            if value is None (None values are not allowed in success results). This enforces
            the same invariant as FlextResult.ok() at the base class level.

            Args:
                value: Value to wrap in success result (must not be None)

            Returns:
                Successful RuntimeResult instance

            Raises:
                ValueError: If value is None

            """
            if value is None:
                msg = "Cannot create success result with None value"
                raise ValueError(msg)
            return cls(value=value, is_success=True)

        @classmethod
        def fail[U](
            cls: type[FlextRuntime.RuntimeResult[U]],
            error: str | None,
            error_code: str | None = None,
            error_data: t.ConfigMap | None = None,
            expected_type: type[U] | None = None,
        ) -> FlextRuntime.RuntimeResult[U]:
            """Create failed result with error message.

            Business Rule: Creates failed RuntimeResult with error message, optional error
            code, and optional error metadata. Converts None error to empty string for
            consistency. This matches the API of FlextResult.fail() for compatibility.

            Args:
                error: Error message (None will be converted to empty string)
                error_code: Optional error code for categorization
                error_data: Optional error metadata

            Returns:
                Failed RuntimeResult instance

            """
            _ = expected_type
            error_msg = error if error is not None else ""
            return cls(
                error=error_msg,
                error_code=error_code,
                error_data=error_data,
                is_success=False,
            )

    # =========================================================================
    # APPLICATION LAYER INTEGRATION (Using structlog directly - Layer 0.5)
    # =========================================================================
    # DESIGN: Integration uses structlog directly without importing from
    # Infrastructure layer (FlextContext, FlextLogger), avoiding circular imports.
    # USAGE: Opt-in helpers for APPLICATION/SERVICE layer only.
    # =========================================================================

    class Integration:
        """Application-layer integration helpers using structlog directly (Layer 0.5).

        **DESIGN**: These methods use structlog directly without importing from
        higher layers (FlextContext, FlextLogger), avoiding all circular imports.

        **USAGE**: Opt-in helpers for application/service layer to integrate
        foundation components with context tracking.

        **CORRECT USAGE** (Application Layer):
            ```python
            from flext_core import FlextContainer
            from flext_core.runtime import FlextRuntime

            container = FlextContainer.get_global()
            result = container.get("database")

            # Opt-in integration at application layer
            FlextRuntime.Integration.track_service_resolution(
                "database", resolved=result.is_success
            )
            ```

        **NOTES**:
            - Uses structlog directly (single source of truth for context)
            - No imports from Infrastructure layer (context.py, loggings.py)
            - Pure Layer 0.5 implementation
        """

        @staticmethod
        def track_service_resolution(
            service_name: str,
            *,
            resolved: bool = True,
            error_message: str | None = None,
        ) -> None:
            """Track service resolution with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service being resolved
                resolved: Whether resolution was successful
                error_message: Error message if resolution failed

            """
            # Get correlation_id directly from structlog (single source of truth)
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly (no FlextLogger wrapper needed)
            logger = structlog.get_logger(__name__)

            if resolved:
                logger.info(
                    "Service resolved",
                    service_name=service_name,
                    correlation_id=correlation_id,
                )
            else:
                logger.error(
                    "Service resolution failed",
                    service_name=service_name,
                    error=error_message,
                    correlation_id=correlation_id,
                )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: t.ConfigMap | None = None,
        ) -> None:
            """Track domain event with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                event_name: Name of the domain event
                aggregate_id: ID of the aggregate root
                event_data: Additional event data

            """
            # Get correlation_id directly from structlog
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly
            logger = structlog.get_logger(__name__)

            logger.info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Setup complete service infrastructure.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service
                service_version: Version of the service
                enable_context_correlation: Whether to enable correlation

            """
            # Set service context directly in structlog contextvars
            _ = structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = structlog.contextvars.bind_contextvars(
                    service_version=service_version,
                )

            # Generate correlation ID if enabled
            if enable_context_correlation:
                # Use secrets directly to avoid circular import (runtime.py is Layer 0.5, utilities.py is Layer 2)
                alphabet = string.ascii_letters + string.digits
                correlation_id = (
                    f"flext-{''.join(secrets.choice(alphabet) for _ in range(12))}"
                )
                _ = structlog.contextvars.bind_contextvars(
                    correlation_id=correlation_id,
                )

            # Use structlog directly
            logger = structlog.get_logger(__name__)
            logger.info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )

    # =========================================================================
    # MODEL SUPPORT METHODS (Tier 0.5 - used by _models to avoid circular imports)
    # =========================================================================

    @staticmethod
    def generate_datetime_utc() -> datetime:
        """Generate current UTC datetime for _models/container.py timestamps."""
        return datetime.now(UTC)

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID using UUID4 for _models."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_prefixed_id(prefix: str, length: int | None = None) -> str:
        """Generate prefixed ID for _models (e.g., 'query_abc123')."""
        base_id = str(uuid.uuid4()).replace("-", "")
        if length is not None:
            base_id = base_id[:length]
        return f"{prefix}_{base_id}" if prefix else base_id

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.ConfigMapValue,
        entity_b: t.ConfigMapValue,
        id_attr: str = "unique_id",
    ) -> bool:
        """Compare two entities by unique ID attribute."""
        if FlextRuntime._is_scalar(entity_a):
            return False
        match entity_a:
            case Sequence() | Mapping():
                return False
            case _:
                pass
        if FlextRuntime._is_scalar(entity_b):
            return False
        match entity_b:
            case Sequence() | Mapping():
                return False
            case _:
                pass
        if (
            not isinstance(entity_b, type(entity_a))
            and type(entity_a) not in type(entity_b).__mro__
        ):
            return False
        id_a = getattr(entity_a, id_attr) if hasattr(entity_a, id_attr) else None
        id_b = getattr(entity_b, id_attr) if hasattr(entity_b, id_attr) else None
        return id_a is not None and id_a == id_b

    @staticmethod
    def hash_entity_by_id(
        entity: t.ConfigMapValue,
        id_attr: str = "unique_id",
    ) -> int:
        """Hash entity based on unique ID and type."""
        if FlextRuntime._is_scalar(entity):
            return hash(entity)
        # Now entity is a complex object with potential id_attr
        entity_id = getattr(entity, id_attr) if hasattr(entity, id_attr) else None
        if entity_id is None:
            return hash(id(entity))
        # Complex objects always have __class__
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.ConfigMapValue,
        obj_b: t.ConfigMapValue,
    ) -> bool:
        """Compare value objects by their values (all attributes)."""
        if FlextRuntime._is_scalar(obj_a):
            return obj_a == obj_b
        if FlextRuntime._is_scalar(obj_b):
            return False
        if hasattr(obj_a, "__iter__") and not hasattr(obj_a, "model_dump"):
            return obj_a == obj_b
        if hasattr(obj_b, "__iter__") and not hasattr(obj_b, "model_dump"):
            return obj_a == obj_b
        if (
            not isinstance(obj_b, type(obj_a))
            and type(obj_a) not in type(obj_b).__mro__
        ):
            return False
        if isinstance(obj_a, BaseModel) and isinstance(obj_b, BaseModel):
            dump_a = obj_a.model_dump()
            dump_b = obj_b.model_dump()
            return dump_a == dump_b
        # datetime, Path, and other objects - compare by repr
        return repr(obj_a) == repr(obj_b)

    @staticmethod
    def hash_value_object_by_value(obj: t.ConfigMapValue) -> int:
        """Hash value object based on all attribute values."""
        if FlextRuntime._is_scalar(obj):
            return hash(obj)
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            # Ensure all values are hashable by converting to tuple
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        # For other objects (datetime, Path, custom objects), use repr
        return hash(repr(obj))

    # =========================================================================
    # Configuration Bridge Methods (for _models)
    # =========================================================================

    @staticmethod
    def get_log_level_from_config() -> int:
        """Get log level from default constant (bridge for _models).

        Returns:
            int: Numeric logging level (e.g., logging.INFO = 20)

        """
        default_log_level = c.Logging.DEFAULT_LEVEL.upper()
        return int(
            getattr(logging, default_log_level)
            if hasattr(logging, default_log_level)
            else logging.INFO,
        )

    @staticmethod
    def ensure_trace_context(
        context: Mapping[str, str] | t.ConfigMapValue,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> Mapping[str, str]:
        """Ensure context dict has distributed tracing fields (bridge for _models).

        Args:
            context: Context dictionary or object to enrich
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists

        Returns:
            Mapping[str, str]: Enriched context with trace fields

        """
        context_dict = t.ConfigMap(root={})
        if FlextRuntime._is_scalar(context):
            context_dict = t.ConfigMap(root={})
        elif isinstance(context, BaseModel):
            context_dict.update(context.model_dump())
        elif FlextRuntime.is_dict_like(context):
            try:
                for k_obj, v_obj in context.items():
                    key_str = str(k_obj)
                    val_typed = FlextRuntime.normalize_to_general_value(v_obj)
                    context_dict[key_str] = val_typed
            except (TypeError, ValueError, AttributeError) as exc:
                _module_logger.debug(
                    "Failed to normalize mapping context fields",
                    exc_info=exc,
                )
                context_dict = t.ConfigMap(root={})
        elif hasattr(context, "items"):
            context_dict = t.ConfigMap(root={})

        # Convert all values to strings for trace context
        result: MutableMapping[str, str] = {}
        for key, value in context_dict.items():
            result[key] = str(value)

        # Ensure trace fields
        if "trace_id" not in result:
            result["trace_id"] = FlextRuntime.generate_id()
        if "span_id" not in result:
            result["span_id"] = FlextRuntime.generate_id()

        # Optional fields
        if include_correlation_id and "correlation_id" not in result:
            result["correlation_id"] = FlextRuntime.generate_id()

        if include_timestamp and "timestamp" not in result:
            result["timestamp"] = FlextRuntime.generate_datetime_utc().isoformat()

        return result

    @staticmethod
    def validate_http_status_codes(
        codes: list[int] | list[str] | list[t.ConfigMapValue],
        min_code: int | None = None,
        max_code: int | None = None,
    ) -> RuntimeResult[list[int]]:
        """Validate and normalize HTTP status codes (bridge for _models).

        Args:
            codes: List of status codes (int or str) to validate
            min_code: Minimum valid HTTP status code (default: 100)
            max_code: Maximum valid HTTP status code (default: 599)

        Returns:
            RuntimeResult[list[int]]: Success with normalized codes or failure

        """
        min_val = min_code if min_code is not None else c.Network.HTTP_STATUS_MIN
        max_val = max_code if max_code is not None else c.Network.HTTP_STATUS_MAX

        validated_codes: list[int] = []
        for code in codes:
            try:
                match code:
                    case int() | str():
                        code_int = int(str(code))
                        if not min_val <= code_int <= max_val:
                            msg = (
                                f"Invalid HTTP status code: {code} "
                                f"(must be {min_val}-{max_val})"
                            )
                            return FlextRuntime.RuntimeResult[list[int]].fail(msg)
                        validated_codes.append(code_int)
                    case _:
                        return FlextRuntime.RuntimeResult[list[int]].fail(
                            f"Invalid HTTP status code type: {code.__class__.__name__}",
                        )
            except ValueError:
                return FlextRuntime.RuntimeResult[list[int]].fail(
                    f"Cannot convert to integer: {code}",
                )

        return FlextRuntime.RuntimeResult[list[int]].ok(validated_codes)


__all__ = ["FlextRuntime"]
