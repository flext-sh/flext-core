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

**Usage Patterns**:
1. **Type Guards**: Use is_valid_phone(), is_valid_json() for pattern validation
4. **Structured Logging**: Use configure_structlog() once at startup
5. **Service Integration**: Use FlextRuntime.Integration.track_service_resolution()
6. **Domain Events**: Use FlextRuntime.Integration.track_domain_event()

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
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Self, TypeGuard

import structlog
from beartype import BeartypeConf, BeartypeStrategy
from beartype.claw import beartype_package
from dependency_injector import containers, providers, wiring
from pydantic import BaseModel

from flext_core.constants import c

if typing.TYPE_CHECKING:
    from types import ModuleType, TracebackType

    from structlog.typing import BindableLogger

    from flext_core.protocols import p
    from flext_core.typings import T, t


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


    **Usage Patterns**:
    1. **Type Validation**: `if FlextRuntime.is_valid_phone(value): ...`
    4. **Logging Setup**: `FlextRuntime.configure_structlog(console_renderer=True)`
    5. **Service Tracking**: `FlextRuntime.Integration.track_service_resolution(name)`
    6. **Event Logging**: `FlextRuntime.Integration.track_domain_event(event_name)`

    **Design Principles**:
    - Circular import prevention through foundation + bridge layers only
    - No imports from higher layers (result.py, container.py, context.py, loggings.py)
    - Direct structlog usage as single source of truth for context
    - Safe fallback strategies for all risky operations (serialization)
    - Opt-in integration helpers (not forced on all modules)
    - Pattern-based validation using c (single source of truth)
    """

    _structlog_configured: ClassVar[bool] = False

    class _AsyncLogWriter:
        """Background log writer using a queue and a separate thread.

        Provides non-blocking logging by buffering log messages to a queue
        and writing them to the destination stream in a background thread.
        """

        def __init__(self, stream: typing.TextIO) -> None:
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
            atexit.register(self.shutdown)

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
            self.queue.put(None)  # Sentinel
            self.thread.join(timeout=2.0)
            self.flush()

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get(timeout=0.1)
                    if msg is None:
                        break
                    self.stream.write(msg)
                    if hasattr(self.stream, "flush"):
                        self.stream.flush()
                    self.queue.task_done()
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                except Exception:
                    # Fallback to direct write if queue processing fails
                    with contextlib.suppress(Exception):
                        self.stream.write("Error in async log writer\n")

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
        # object.__new__ returns object, but we know it's actually T
        # Use explicit type narrowing via isinstance check after creation
        instance = object.__new__(class_type)
        # Type assertion: object.__new__(class_type) always returns an instance of class_type
        # This is a fundamental Python guarantee, so we can safely assert the type
        if not isinstance(instance, class_type):
            class_name = getattr(class_type, "__name__", str(class_type))
            msg = (
                f"object.__new__({class_name}) did not return instance of {class_name}"
            )
            raise TypeError(msg)
        return instance

    # =========================================================================
    # TYPE GUARD UTILITIES
    # =========================================================================

    @staticmethod
    def is_dict_like(
        value: t.GeneralValueType,
    ) -> TypeGuard[t.ConfigurationMapping]:
        """Type guard to check if value is dict-like.

        Args:
            value: Value to check

        Returns:
            True if value is a t.ConfigurationMapping or dict-like object, False otherwise

        """
        # #region agent log
        # import json
        # with open('/home/marlonsc/flext/.cursor/debug.log', 'a') as f:
        #     f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "runtime.py:317", "message": "Entering is_dict_like", "data": {"value": str(value)}, "timestamp": Date.now()}) + '\n')
        # #endregion
        if isinstance(value, dict):
            return True
        # Check for dict-like objects (UserDict, etc.)
        if hasattr(value, "keys") and hasattr(value, "items") and hasattr(value, "get"):
            # Verify it's actually dict-like by checking if it has dict methods
            try:
                # Try to access items to verify it's dict-like
                items_method = getattr(value, "items", None)
                if callable(items_method):
                    _ = items_method()
                return True
            except (AttributeError, TypeError):
                return False
        return False

    @staticmethod
    def is_list_like(
        value: t.GeneralValueType,
    ) -> TypeGuard[Sequence[t.GeneralValueType]]:
        """Type guard to check if value is list-like.

        Args:
            value: Value to check

        Returns:
            True if value is a list[t.GeneralValueType] or list-like sequence, False otherwise

        """
        # #region agent log
        # import json
        # with open('/home/marlonsc/flext/.cursor/debug.log', 'a') as f:
        #     f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "runtime.py:343", "message": "Entering is_list_like", "data": {"value": str(value)}, "timestamp": Date.now()}) + '\n')
        # #endregion
        return isinstance(value, list)

    @staticmethod
    def normalize_to_general_value(
        val: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Normalize any value to t.GeneralValueType recursively.

        Converts arbitrary objects, dict[str, t.GeneralValueType], list[object], and other types
        to t.ConfigurationMapping, Sequence[t.GeneralValueType], etc.
        This is the central conversion function for type safety.

        Args:
            val: Value to normalize (accepts object for flexibility with generics)

        Returns:
            Normalized value compatible with t.GeneralValueType

        Examples:
            >>> FlextRuntime.normalize_to_general_value({"key": "value"})
            {'key': 'value'}
            >>> FlextRuntime.normalize_to_general_value({"nested": {"inner": 123}})
            {'nested': {'inner': 123}}
            >>> FlextRuntime.normalize_to_general_value([1, 2, {"a": "b"}])
            [1, 2, {'a': 'b'}]

        """
        # #region agent log
        # import json
        # with open('/home/marlonsc/flext/.cursor/debug.log', 'a') as f:
        #     f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "runtime.py:366", "message": "Entering normalize_to_general_value", "data": {"val": str(val)}, "timestamp": Date.now()}) + '\n')
        # #endregion
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if FlextRuntime.is_dict_like(val):
            # Business Rule: Convert to dict[str, t.GeneralValueType] recursively
            # dict is compatible with Mapping[str, t.GeneralValueType] in t.GeneralValueType union.
            # This pattern is correct: construct mutable dict, return it (dict is Mapping subtype).
            #
            # Audit Implication: Normalizes nested data structures for type safety.
            # Used throughout FLEXT for converting arbitrary data to t.GeneralValueType.
            # Returns dict that is compatible with Mapping interface for read-only access.
            dict_v = dict(val.items()) if hasattr(val, "items") else dict(val)
            # Type narrowing: dict_v is dict[object, object] from dict() constructor
            # We need to filter only str keys to build ConfigurationDict
            # Use direct dict comprehension to avoid circular dependency with mapper
            result: dict[str, t.GeneralValueType] = {}
            for k, v in dict_v.items():
                # Type narrowing: k is object from dict.items()
                # ConfigurationDict requires str keys - convert non-str keys to str
                key_str = str(k)
                result[key_str] = FlextRuntime.normalize_to_general_value(v)
            return result
        if FlextRuntime.is_list_like(val):
            # Convert to list[t.GeneralValueType] recursively
            return [FlextRuntime.normalize_to_general_value(item) for item in val]
        # For arbitrary objects that don't match known types, convert to string representation
        # This ensures we always return a t.GeneralValueType-compatible value
        return str(val)

    @staticmethod
    def normalize_to_metadata_value(
        val: t.GeneralValueType,
    ) -> t.MetadataAttributeValue:
        """Normalize any value to t.MetadataAttributeValue.

        t.MetadataAttributeValue is more restrictive than t.GeneralValueType,
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
        if isinstance(val, (str, int, float, bool, type(None))):
            # Return type is t.MetadataAttributeValue (scalar types)
            result_scalar: t.MetadataAttributeValue = val
            return result_scalar
        if FlextRuntime.is_dict_like(val):
            # Business Rule: Serialize dict to JSON string for Metadata.attributes compatibility.
            # Metadata.attributes only accepts flat scalar values (str, int, float, bool, None, list).
            # Nested dicts must be serialized to JSON strings to be valid attribute values.
            #
            # Audit Implication: Nested structures are serialized to JSON for metadata storage.
            # This ensures all metadata values are JSON-serializable and flat.
            result_json: t.MetadataAttributeValue = json.dumps(
                dict(val.items()) if hasattr(val, "items") else dict(val),
            )
            return result_json
        if FlextRuntime.is_list_like(val):
            # Convert to list of MetadataAttributeValue scalars (including datetime)
            result_list: list[str | int | float | bool | datetime | None] = []
            for item in val:
                if isinstance(item, (str, int, float, bool, datetime, type(None))):
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
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def is_valid_identifier(
        value: t.GeneralValueType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is a valid Python identifier.

        Args:
            value: Value to check

        Returns:
            True if value is a valid Python identifier, False otherwise

        """
        if not isinstance(value, str):
            return False
        return value.isidentifier()

    @staticmethod
    def is_base_model(obj: object) -> TypeGuard[BaseModel]:
        """Type guard to narrow object to BaseModel (part of GeneralValueType).

        This allows isinstance checks to narrow types for FlextRuntime methods
        that accept GeneralValueType (which includes BaseModel).
        """
        return isinstance(obj, BaseModel)

    # =========================================================================
    # SERIALIZATION UTILITIES (No flext_core imports)
    # =========================================================================

    @staticmethod
    def safe_get_attribute(
        obj: t.GeneralValueType,
        attr: str,
        default: t.GeneralValueType = None,
    ) -> t.GeneralValueType:
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
        return getattr(obj, attr, default)

    @staticmethod
    def extract_generic_args(
        type_hint: t.TypeHintSpecifier,
    ) -> tuple[t.GenericTypeArgument, ...]:
        """Extract generic type arguments from a type hint.

        Business Rule: Extracts generic type arguments from type hints using
        typing.get_args() for standard generics, and fallback mapping for type
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
                type_name = getattr(type_hint, "__name__", "")
                # Handle common type aliases - use actual type objects
                # GenericTypeArgument = str | type[t.GeneralValueType]
                # Type objects (str, int, float, bool) are valid GenericTypeArgument
                # since they represent type[T] where T is a scalar t.GeneralValueType
                type_mapping: t.StringGenericTypeArgumentTupleDict = {
                    "StringList": (str,),
                    "IntList": (int,),
                    "FloatList": (float,),
                    "BoolList": (bool,),
                    "Dict": (str, str),
                    "List": (str,),
                    "StringDict": (str, str),
                    "IntDict": (str, int),
                    "FloatDict": (str, float),
                    "BoolDict": (str, bool),
                    # NestedDict: str key, nested value (recursive dict structure)
                    "NestedDict": (str, dict),
                }
                if type_name in type_mapping:
                    return type_mapping[type_name]

            return ()
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):  # pragma: no cover
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
            if origin is not None and isinstance(origin, type):
                return issubclass(origin, Sequence)

            # Check if the type itself is a sequence subclass (for type aliases)
            if isinstance(type_hint, type) and issubclass(type_hint, Sequence):
                return True

            # Check __name__ for type aliases like StringList
            # Type narrowing: Check if type_hint has __name__ attribute (must be a type, not callable)
            # Use isinstance to narrow to type before accessing __name__
            if isinstance(type_hint, type) and hasattr(type_hint, "__name__"):
                type_name: str = type_hint.__name__
                # Common sequence type aliases
                if type_name in {
                    "StringList",
                    "IntList",
                    "FloatList",
                    "BoolList",
                    "List",
                }:
                    return True

            return False
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):  # pragma: no cover
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
            config: t.ConfigurationMapping | None = None,
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
            services_provider.override(service_module)
            resources_provider.override(resource_module)
            cls.bind_configuration_provider(bridge.config, config)
            return bridge, service_module, resource_module

        @classmethod
        def create_container(
            cls,
            *,
            config: t.ConfigurationMapping | None = None,
            services: Mapping[
                str,
                t.GeneralValueType | BaseModel | p.VariadicCallable[t.GeneralValueType],
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
                cls.bind_configuration(di_container, config)

            if services:
                for name, instance in services.items():
                    cls.register_object(di_container, name, instance)

            if factories:
                for name, factory in factories.items():
                    cls.register_factory(
                        di_container,
                        name,
                        factory,
                        cache=factory_cache,
                    )

            if resources:
                for name, resource_factory in resources.items():
                    # register_resource[T] accepts Callable[[], T] and infers T
                    # resource_factory is Callable[[], t.GeneralValueType] from resources
                    # T is inferred as t.GeneralValueType, which is valid
                    cls.register_resource(
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
            config: t.ConfigurationMapping | None,
        ) -> providers.Configuration:
            """Bind configuration mapping to the DI container.

            Uses ``providers.Configuration`` to expose values to downstream
            providers without higher layers interacting with dependency-injector
            directly.
            """
            configuration_provider = providers.Configuration()
            if config:
                configuration_provider.from_dict(dict(config))
            di_container.config = configuration_provider
            return configuration_provider

        @staticmethod
        def bind_configuration_provider(
            configuration_provider: providers.Configuration,
            config: t.ConfigurationMapping | None,
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
        _logger: p.Log,
        method_name: str,
        event_dict: t.ConfigurationMapping,
    ) -> t.ConfigurationMapping:
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
        # dict is compatible with t.ConfigurationMapping return type.
        # This pattern is correct: construct mutable dict, return it (dict is Mapping subtype).
        #
        # Audit Implication: This method filters log event data based on log level.
        # Used for conditional inclusion of verbose fields in structured logging.
        # Returns dict that is compatible with Mapping interface for read-only access.
        filtered_dict: dict[str, t.GeneralValueType] = {}
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
        config: t.GeneralValueType | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[object] | None = None,
        wrapper_class_factory: Callable[[], type[BindableLogger]] | None = None,
        logger_factory: Callable[[], BindableLogger] | None = None,
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
            log_level = getattr(config, "log_level", log_level)
            console_renderer = getattr(config, "console_renderer", console_renderer)
            additional_processors_from_config = getattr(
                config,
                "additional_processors",
                None,
            )
            if additional_processors_from_config:
                additional_processors = additional_processors_from_config
            wrapper_class_factory = getattr(
                config,
                "wrapper_class_factory",
                wrapper_class_factory,
            )
            logger_factory = getattr(config, "logger_factory", logger_factory)
            cache_logger_on_first_use = getattr(
                config,
                "cache_logger_on_first_use",
                cache_logger_on_first_use,
            )
            async_logging = getattr(config, "async_logging", True)

        # Single guard - no redundant checks
        if cls._structlog_configured:
            return

        level_to_use = log_level if log_level is not None else logging.INFO

        module = structlog

        # structlog processors have specific signatures - use object to accept any processor type
        # structlog processors are callables with varying signatures, so we use object for flexibility
        processors: list[object] = [
            module.contextvars.merge_contextvars,
            module.processors.add_log_level,
            # CRITICAL: Level-based context filter (must be after merge_contextvars and add_log_level)
            cls.level_based_context_filter,
            module.processors.TimeStamper(fmt="iso"),
            module.processors.StackInfoRenderer(),
        ]
        if additional_processors:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.extend(additional_processors)

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.append(module.processors.JSONRenderer())

        # Configure structlog with processors and logger factory
        # structlog.configure accepts specific types, but we construct them dynamically

        wrapper_arg: type[BindableLogger] | None = (
            wrapper_class_factory()
            if wrapper_class_factory is not None
            else module.make_filtering_bound_logger(level_to_use)
        )

        # Determine logger factory (handle async buffering)
        # structlog accepts various factory types - we use object to accept all
        factory_to_use: object
        if logger_factory is not None:
            # Use the provided factory directly (Callable[[], BindableLogger])
            factory_to_use = logger_factory
        elif async_logging:
            # Default factory handling with async buffering
            # Use cached async writer or create new one
            if cls._async_writer is None:
                cls._async_writer = cls._AsyncLogWriter(sys.stdout)
            # PrintLoggerFactory accepts file-like objects with write method
            # _AsyncLogWriter has write/flush methods (duck-typed TextIO)
            # Use getattr to call PrintLoggerFactory with duck-typed file arg
            print_logger_factory = getattr(module, "PrintLoggerFactory", None)
            if print_logger_factory is not None:
                factory_to_use = print_logger_factory(file=cls._async_writer)
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
        configure_fn = getattr(module, "configure", None)
        if configure_fn is not None and callable(configure_fn):
            configure_fn(
                processors=processors,
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
        additional_processors: list[object] | None = None,
    ) -> None:
        """Force reconfigure structlog (ignores is_configured checks).

        **USE ONLY when CLI params override config defaults.**

        For initial configuration, use configure_structlog().
        This method bypasses the guards and forces reconfiguration,
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
            # CLI override: User passed --debug flag
            from flext_core.runtime import FlextRuntime
            from flext_core.constants import c

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

    # =========================================================================
    # =========================================================================
    # RUNTIME TYPE CHECKING (Beartype Integration - Python 3.13 Strict Typing)
    # =========================================================================

    @staticmethod
    def enable_runtime_checking() -> bool:
        """Enable beartype runtime type checking for entire flext_core package.

        This applies @beartype decorator to ALL functions/methods in flext_core
        modules automatically using beartype.claw. Provides O(log n) runtime
        validation with minimal overhead.

        **WARNING**: This is global package-wide activation. Use with caution
        in production environments. Consider enabling only in development/test.

        **Architecture**: Layer 0.5 - Runtime Integration Bridge
        - Integrates beartype external library with flext_core
        - Zero circular import risk (foundation layer only)
        - Complements static type checking (pyright strict mode)

        Returns:
            True if enabled successfully, False if beartype not available.

        Raises:
            None - Warnings issued if beartype unavailable.

        Example:
            >>> from flext_core import FlextRuntime
            >>> FlextRuntime.enable_runtime_checking()
            True
            >>> # Now all flext_core calls have runtime type validation
            >>> from flext_core.result import r
            >>> result = r[int].ok("not an int")  # Raises BeartypeError

        Notes:
            - Beartype uses O(log n) strategy for thorough validation
            - Adds minimal runtime overhead (~5-10% typically)
            - All type annotations in flext_core are validated at runtime
            - Pydantic models continue using their own validation
            - Static type checking (pyright) always active regardless

        """
        # Configure beartype with O(log n) strategy
        conf = BeartypeConf(
            strategy=BeartypeStrategy.Ologn,
            is_color=True,  # Colored error messages
            claw_is_pep526=False,  # Disable variable annotation checking
            warning_cls_on_decorator_exception=UserWarning,  # Warn on failures
        )

        # Apply beartype to entire flext_core package
        beartype_package("flext_core", conf=conf)

        # Log activation using structlog directly
        logger = structlog.get_logger(__name__)
        logger.info(
            "Runtime type checking enabled",
            package="flext_core",
            strategy="Ologn",
        )

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

        __slots__ = ("_error", "_error_code", "_error_data", "_is_success", "_value")

        def __init__(
            self,
            value: T | None = None,
            error: str | None = None,
            error_code: str | None = None,
            error_data: t.ConfigurationMapping | None = None,
            *,
            is_success: bool = True,
        ) -> None:
            """Initialize RuntimeResult."""
            self._value = value
            self._error = error
            self._error_code = error_code
            self._error_data = error_data
            self._is_success = is_success

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
        def data(self) -> T:
            """Alias for value - backward compatibility with older API."""
            return self.value

        @property
        def result(self) -> object:
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
        def error(self) -> str | None:
            """Get the error message."""
            return self._error

        @property
        def error_code(self) -> str | None:
            """Get the error code."""
            return self._error_code

        @property
        def error_data(self) -> t.ConfigurationMapping | None:
            """Get the error data."""
            return self._error_data

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
                except Exception as e:
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
            """RFC-compliant alias for flat_map."""
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

        @classmethod
        def ok(cls, value: T) -> FlextRuntime.RuntimeResult[T]:
            """Create successful result wrapping data.

            Business Rule: Creates successful RuntimeResult wrapping value. For None values,
            creates a result with value=None and is_success=True (special case for r[None]).
            This matches the API of FlextResult.ok() for compatibility, but allows None
            when T is None type.

            Args:
                value: Value to wrap in success result (None allowed for r[None] type)

            Returns:
                Successful RuntimeResult instance

            """
            # Allow None values for r[None] type compatibility
            return cls(value=value, is_success=True)

        @classmethod
        def fail(
            cls,
            error: str | None,
            error_code: str | None = None,
            error_data: t.ConfigurationMapping | None = None,
        ) -> FlextRuntime.RuntimeResult[T]:
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
            event_data: t.EventDataMapping | None = None,
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
            structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                structlog.contextvars.bind_contextvars(service_version=service_version)

            # Generate correlation ID if enabled
            if enable_context_correlation:
                # Use secrets directly to avoid circular import (runtime.py is Layer 0.5, utilities.py is Layer 2)
                alphabet = string.ascii_letters + string.digits
                correlation_id = (
                    f"flext-{''.join(secrets.choice(alphabet) for _ in range(12))}"
                )
                structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

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
        entity_a: t.GeneralValueType,
        entity_b: t.GeneralValueType,
        id_attr: str = "unique_id",
    ) -> bool:
        """Compare two entities by unique ID attribute."""
        # Type narrowing: Filter out types that don't support getattr
        # Scalars, sequences, mappings don't have entity IDs
        if isinstance(entity_a, (str, int, float, bool, type(None), Sequence, Mapping)):
            return False
        if isinstance(entity_b, (str, int, float, bool, type(None), Sequence, Mapping)):
            return False
        # Now both are objects that support getattr (BaseModel, datetime, Path, or custom objects)
        if not isinstance(entity_b, entity_a.__class__):
            return False
        id_a = getattr(entity_a, id_attr, None)
        id_b = getattr(entity_b, id_attr, None)
        return id_a is not None and id_a == id_b

    @staticmethod
    def hash_entity_by_id(
        entity: t.GeneralValueType,
        id_attr: str = "unique_id",
    ) -> int:
        """Hash entity based on unique ID and type."""
        # Type narrowing: Filter out scalars (they don't have id_attr)
        if isinstance(entity, (str, int, float, bool, type(None))):
            return hash(entity)
        # Now entity is a complex object with potential id_attr
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))
        # Complex objects always have __class__
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.GeneralValueType,
        obj_b: t.GeneralValueType,
    ) -> bool:
        """Compare value objects by their values (all attributes)."""
        # Type narrowing: Filter out scalars (compare directly)
        if isinstance(obj_a, (str, int, float, bool, type(None))):
            return obj_a == obj_b
        if isinstance(obj_b, (str, int, float, bool, type(None))):
            return False
        # Filter out sequences and mappings (use equality operator)
        if isinstance(obj_a, (Sequence, Mapping)) or isinstance(
            obj_b, (Sequence, Mapping),
        ):
            # Use equality instead of repr for sequences/mappings
            return obj_a == obj_b
        # Now both are objects that support comparison (BaseModel, datetime, Path, or custom)
        if not isinstance(obj_b, obj_a.__class__):
            return False
        # Use isinstance for proper type narrowing with BaseModel
        if isinstance(obj_a, BaseModel) and isinstance(obj_b, BaseModel):
            dump_a = obj_a.model_dump()
            dump_b = obj_b.model_dump()
            return dump_a == dump_b
        # datetime, Path, and other objects - compare by repr
        return repr(obj_a) == repr(obj_b)

    @staticmethod
    def hash_value_object_by_value(obj: t.GeneralValueType) -> int:
        """Hash value object based on all attribute values."""
        # Type narrowing: Filter out scalars (hash directly)
        if isinstance(obj, (str, int, float, bool, type(None))):
            return hash(obj)
        # Use isinstance for proper type narrowing with BaseModel
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            # Ensure all values are hashable by converting to tuple
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))
        # Check dict first (most specific) before Mapping
        if isinstance(obj, dict):
            # dict might have Unknown types - use repr to avoid type issues
            # repr() accepts object, which includes Unknown types
            # Type assertion: dict is always a valid object for repr()
            obj_as_object: object = obj
            return hash(repr(obj_as_object))
        # Check list first (most specific) before Sequence
        if isinstance(obj, list):
            # list might have Unknown types - use repr to avoid type issues
            # Type assertion: list is always a valid object for repr()
            obj_as_object = obj
            return hash(repr(obj_as_object))
        # Generic Mapping - use repr to avoid Unknown types
        if isinstance(obj, Mapping):
            # Type assertion: Mapping is always a valid object for repr()
            obj_as_object = obj
            return hash(repr(obj_as_object))
        # Generic Sequence - use repr to avoid Unknown types
        if isinstance(obj, Sequence):
            # Type assertion: Sequence is always a valid object for repr()
            obj_as_object = obj
            return hash(repr(obj_as_object))
        # For other objects (datetime, Path, custom objects), use repr
        return hash(repr(obj))

    class Bootstrap:
        """Bootstrap utility for creating instances via object.__new__.

        Provides convenient access to FlextRuntime.create_instance() method
        for type-safe instance creation without calling __init__.
        """

        @staticmethod
        def create_instance[T](class_type: type[T]) -> T:
            """Type-safe factory for creating instances via object.__new__.

            Args:
                class_type: The class to instantiate

            Returns:
                An instance of type T

            Raises:
                TypeError: If object.__new__() does not return instance of
                    expected type

            Example:
                >>> instance = FlextRuntime.Bootstrap.create_instance(MyClass)
                >>> # instance is properly typed as MyClass

            """
            return FlextRuntime.create_instance(class_type)

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
        return int(getattr(logging, default_log_level, logging.INFO))

    @staticmethod
    def ensure_trace_context(
        context: t.StringMapping | t.GeneralValueType,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> dict[str, str]:
        """Ensure context dict has distributed tracing fields (bridge for _models).

        Args:
            context: Context dictionary or object to enrich
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists

        Returns:
            dict[str, str]: Enriched context with trace fields

        """
        # Normalize context to dict with explicit type narrowing
        # Filter out scalars first (they don't have __dict__ or items())
        if isinstance(context, (str, int, float, bool, type(None))):
            context_dict: dict[str, t.GeneralValueType] = {}
        elif isinstance(context, BaseModel):
            # BaseModel has model_dump() - use that first
            context_dict = context.model_dump()
        elif isinstance(context, dict):
            # dict might have Unknown types - avoid iteration
            # Convert to str representation and parse back if needed
            # For trace context, we just need string keys - use __str__ fallback
            context_dict = {}
            # Try to iterate, but handle potential Unknown types
            try:
                # Iterate over items directly to preserve key types
                # Type assertion: cast items to object to handle Unknown types
                for k_obj, v_obj in context.items():
                    # Safe: str() works on any object, including Unknown
                    key_str: str = str(k_obj)
                    # Convert value to GeneralValueType using isinstance checks
                    val_typed: t.GeneralValueType
                    if v_obj is None:
                        val_typed = None
                    elif isinstance(
                        v_obj, (str, int, float, bool, datetime, BaseModel, Path),
                    ):
                        val_typed = v_obj
                    elif isinstance(v_obj, (list, tuple)):
                        val_typed = v_obj  # Sequence[GeneralValueType]
                    elif isinstance(v_obj, dict):
                        val_typed = v_obj  # Mapping[str, GeneralValueType]
                    elif callable(v_obj):
                        val_typed = v_obj  # Callable
                    else:
                        val_typed = str(v_obj)  # Fallback to string
                    context_dict[key_str] = val_typed
            except Exception:
                # If iteration fails, use empty dict
                context_dict = {}
        elif isinstance(context, Mapping):
            # Generic Mapping - avoid iteration to prevent Unknown type errors
            context_dict = {}
        else:
            # All other types - use empty dict
            context_dict = {}

        # Convert all values to strings for trace context
        result: dict[str, str] = {
            k: v if isinstance(v, str) else str(v) for k, v in context_dict.items()
        }

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
        codes: list[t.GeneralValueType],
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
                if isinstance(code, (int, str)):
                    code_int = int(str(code))
                    if not min_val <= code_int <= max_val:
                        return FlextRuntime.RuntimeResult[list[int]].fail(
                            f"Invalid HTTP status code: {code} "
                            f"(must be {min_val}-{max_val})",
                        )
                    validated_codes.append(code_int)
                else:
                    return FlextRuntime.RuntimeResult[list[int]].fail(
                        f"Invalid HTTP status code type: {type(code).__name__}",
                    )
            except ValueError:
                return FlextRuntime.RuntimeResult[list[int]].fail(
                    f"Cannot convert to integer: {code}",
                )

        return FlextRuntime.RuntimeResult[list[int]].ok(validated_codes)


__all__ = ["FlextRuntime"]
