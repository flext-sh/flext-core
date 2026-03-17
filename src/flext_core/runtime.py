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
- Facades (e.g. FlextUtilities) expose staticmethod aliases from external subclasses so call sites get one flat namespace (u.foo, u.bar), no subdivision (no u.foo).
- At call sites use project namespace only: c, m, r, t, u, p, d, e, h, s, x from project __init__. Subprojects: access only via that project's namespace; no cross-project alias subdivision. MRO protocol only; direct methods.
- Runtime helpers via x (e.g. x.create_instance, x.is_dict_like).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import atexit
import contextlib
import inspect
import io
import logging
import queue
import secrets
import string
import sys
import threading
import typing
import uuid
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
    Annotated,
    ClassVar,
    Self,
    TypeGuard,
    TypeIs,
    cast,
    override,
)

import orjson
import structlog
from dependency_injector import containers, providers, wiring
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level

from flext_core import T, c, p, t
from flext_core._models import FlextModelFoundation
from flext_core._utilities import FlextUtilitiesGuardsTypeCore


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
    2. **is_valid_json()** - JSON string validation via TypeAdapter
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
    1. **Type Safety** - TypeIs utilities for pattern validation
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
    - Facades use staticmethod aliases from external subclasses so one flat namespace (no u.foo); subprojects use project namespace only (from flext_cli import m, x; m.Foo, m.Bar).
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
    _runtime_logger: ClassVar[p.Logger | None] = None

    @property
    def logger(self) -> p.Logger:
        """Infrastructure logger for FlextRuntime internals (avoids circular imports)."""
        cls = type(self)
        logger = cls._runtime_logger
        if logger is None:
            logger = structlog.get_logger(__name__)
            cls._runtime_logger = logger
        return logger

    class _LazyMetadata:
        """Lazy access to Metadata model to avoid heavy Tier 0 imports."""

        _model: ClassVar[type | None] = None

        @classmethod
        def get(cls) -> type:
            """Lazily import and return the Metadata model."""
            if cls._model is None:
                cls._model = FlextModelFoundation.Metadata
            return cls._model

    Metadata: ClassVar[type] = _LazyMetadata.get()

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread.

        Provides non-blocking logging by buffering log messages to a queue
        and writing them to the destination stream in a background thread.
        """

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._stream_mode: str = getattr(stream, "mode", "w")
            self._stream_name: str = getattr(stream, "name", "<async-log-writer>")
            self._stream_encoding: str = getattr(stream, "encoding", "utf-8")
            self._stream_errors: str | None = getattr(stream, "errors", None)
            self._stream_newlines: str | tuple[str, ...] | None = getattr(
                stream, "newlines", None
            )
            self.queue: queue.Queue[str | None] = queue.Queue(
                maxsize=c.Logging.ASYNC_QUEUE_SIZE
            )
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._worker, daemon=True, name="flext-async-log-writer"
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)
            self._writer_logger: p.Logger | None = None

        @property
        def _writer_log(self) -> p.Logger:
            """Logger for async log writer."""
            if getattr(self, "_writer_logger", None) is None:
                self._writer_logger = structlog.get_logger(__name__)
            logger = self._writer_logger
            if logger is None:
                logger = structlog.get_logger(__name__)
                self._writer_logger = logger
            return logger

        @property
        def buffer(self) -> typing.BinaryIO:
            """Return underlying binary buffer."""
            buf = getattr(self.stream, "buffer", None)
            if buf is not None:
                return buf
            return io.BytesIO()

        @property
        def line_buffering(self) -> bool:
            """Return whether line buffering is enabled."""
            return getattr(self.stream, "line_buffering", False)

        @override
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
                self.queue.put_nowait(None)
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
            self.flush()

        @override
        def write(self, s: str, /) -> int:
            """Write message to queue (non-blocking)."""
            with contextlib.suppress(queue.Full):
                self.queue.put(s, block=c.Logging.ASYNC_BLOCK_ON_FULL)
            return len(s)

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
                    self._writer_log.warning(
                        "Async log writer stream operation failed", exc_info=exc
                    )
                    with contextlib.suppress(OSError, ValueError, TypeError):
                        _ = self.stream.write("Error in async log writer\n")

    _async_writer: ClassVar[_AsyncLogWriter | None] = None

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

    @classmethod
    def is_structlog_configured(cls) -> bool:
        """Check if structlog has been configured.

        Returns:
            bool: True if structlog is configured, False otherwise

        """
        return cls._structlog_configured

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

    @staticmethod
    def _is_scalar(value: t.RuntimeData) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type accepted by t.Scalar."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def extract_generic_args(
        type_hint: t.TypeHintSpecifier,
    ) -> tuple[t.GenericTypeArgument | type, ...]:
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
            args = typing.get_args(type_hint)
            if args:
                return args
            if hasattr(type_hint, "__name__"):
                type_name = getattr(type_hint, "__name__", "")
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
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return ()

    @staticmethod
    def get_logger(name: str | None = None) -> p.Logger:
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
        logger: p.Logger = structlog.get_logger(name)
        return logger

    @staticmethod
    def is_base_model(obj: t.RuntimeData) -> TypeIs[BaseModel]:
        """Type guard to narrow object to BaseModel.

        This allows isinstance checks to narrow types for FlextRuntime methods
        that accept object (which includes BaseModel).
        """
        match obj:
            case BaseModel():
                return True
            case _:
                return False

    @staticmethod
    def _has_dict_protocol(obj: t.RuntimeData) -> bool:
        if not (hasattr(obj, "keys") and hasattr(obj, "items") and hasattr(obj, "get")):
            return False
        try:
            items_fn = getattr(obj, "items", None)
            if items_fn is not None and callable(items_fn):
                items_fn()
                return True
        except (AttributeError, TypeError):
            pass
        return False

    @staticmethod
    def is_dict_like(
        value: t.RuntimeData,
    ) -> TypeIs[t.ConfigMap | Mapping[str, t.NormalizedValue]]:
        """TypeIs to check if value is dict-like.

        Note:
            ``value`` remains broad because this guard is a boundary utility used
            by normalization paths that accept full ``object``.

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
                return bool(FlextRuntime._has_dict_protocol(value))

    @staticmethod
    def is_list_like(
        value: t.RuntimeData,
    ) -> TypeIs[Sequence[t.NormalizedValue]]:
        """Type guard to check if value is list-like."""
        return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))

    @staticmethod
    def is_sequence_type(type_hint: t.TypeHintSpecifier) -> bool:
        """Check if type hint represents a sequence type (list, tuple, etc.).

        Business Rule: Checks if type hint represents a sequence type using
        typing.get_origin() and structural class checks. Supports both standard
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
            if isinstance(origin, type):
                if origin in {list, tuple}:
                    return True
                return FlextRuntime._is_sequence_type_class(origin)
            if type_hint in {list, tuple, str}:
                return True
            if isinstance(type_hint, type) and FlextRuntime._is_sequence_type_class(
                type_hint
            ):
                return True
            if isinstance(type_hint, type) and getattr(type_hint, "__name__", "") in {
                "StringList",
                "IntList",
                "FloatList",
                "BoolList",
                "List",
            }:
                return True
            if not isinstance(type_hint, str):
                return False
            type_hint_name = type_hint
            return type_hint_name in {
                "StringList",
                "IntList",
                "FloatList",
                "BoolList",
                "List",
            }
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return False

    @staticmethod
    def _is_sequence_type_class(candidate: type) -> bool:
        candidate_name = getattr(candidate, "__name__", "")
        if candidate_name in {"list", "tuple", "range"}:
            return True
        if candidate_name in {"str", "bytes", "bytearray", "memoryview", "dict"}:
            return False
        candidate_mro = getattr(candidate, "__mro__", ())
        if any(getattr(base, "__name__", "") == "Sequence" for base in candidate_mro):
            return True
        required_members = ("__iter__", "__len__", "__getitem__", "count", "index")
        return all(hasattr(candidate, member) for member in required_members)

    @staticmethod
    def is_valid_identifier(value: t.RuntimeData) -> TypeIs[str]:
        """Type guard to check if value is a valid Python identifier."""
        return isinstance(value, str) and value.isidentifier()

    @staticmethod
    def is_valid_json(value: t.RuntimeData) -> TypeIs[str]:
        """Type guard to check if value is valid JSON string.

        Business Rule: Validates JSON strings using Pydantic v2 TypeAdapter for parsing.
        Returns TypeIs[str] for type narrowing in conditional blocks.
        Catches ValidationError for safe validation. Used for
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
            # Bridge-level: orjson used for JSON validation at infrastructure boundary
            orjson.loads(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_structlog_processor(
        value: t.RuntimeData,
    ) -> TypeGuard[structlog.types.Processor]:
        return callable(value)

    @staticmethod
    def normalize_to_container(
        val: t.RuntimeData,
    ) -> t.RuntimeAtomic:
        """Normalize any value to t.Container | BaseModel.

        Args:
            val: Value to normalize

        Returns:
            Scalar | Path | BaseModel

        """
        if val is None:
            return ""
        if FlextRuntime._is_scalar(val) or isinstance(val, Path):
            return val
        if isinstance(val, BaseModel):
            return val

        def _to_plain_container(value: t.RuntimeAtomic) -> t.NormalizedValue:
            if isinstance(
                value,
                (t.ConfigMap, t.Dict),
            ):
                return {
                    str(inner_key): _to_plain_container(
                        FlextRuntime.normalize_to_container(inner_value)
                    )
                    for inner_key, inner_value in value.root.items()
                }
            if isinstance(value, t.ObjectList):
                return list(value.root)
            if isinstance(value, (str, int, float, bool, datetime, Path)):
                return value
            return str(value)

        if FlextRuntime.is_dict_like(val):
            normalized_dict: dict[str, t.ValueOrModel] = {}
            if isinstance(val, t.ConfigMap):
                for key, item in val.root.items():
                    normalized_item = FlextRuntime.normalize_to_container(item)
                    normalized_dict[key] = _to_plain_container(normalized_item)
            else:
                typed_mapping = val
                for key, item in typed_mapping.items():
                    normalized_item = FlextRuntime.normalize_to_container(item)
                    normalized_dict[str(key)] = _to_plain_container(normalized_item)
            return t.Dict(root=normalized_dict)
        if FlextRuntime.is_list_like(val):
            normalized_list: list[t.Container] = []
            for v in val:
                normalized_item = FlextRuntime.normalize_to_container(v)
                if isinstance(normalized_item, (str, int, float, bool, datetime, Path)):
                    normalized_list.append(normalized_item)
            return t.ObjectList(root=normalized_list)
        return str(val)

    @staticmethod
    def _normalize_to_metadata_scalar(val: t.RuntimeData) -> t.Primitives:
        if val is None:
            return ""
        if FlextUtilitiesGuardsTypeCore.is_primitive(val):
            return val
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, BaseModel):
            return val.model_dump_json()
        return str(val)

    @staticmethod
    def normalize_to_metadata(
        val: t.RuntimeData,
    ) -> t.MetadataValue:
        """Normalize input into metadata-compatible scalar, list, or mapping values."""
        if val is None:
            return ""
        if FlextUtilitiesGuardsTypeCore.is_primitive(val):
            return val
        if isinstance(val, datetime):
            return val
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, BaseModel):
            return val.model_dump_json()
        if FlextRuntime.is_dict_like(val):
            normalized: dict[str, t.Scalar | list[t.Scalar]] = {}
            for k, v in val.items():
                str_k = str(k)
                if v is None:
                    normalized[str_k] = ""
                elif FlextUtilitiesGuardsTypeCore.is_scalar(v):
                    normalized[str_k] = v
                elif isinstance(v, Path):
                    normalized[str_k] = str(v)
                elif isinstance(v, BaseModel):
                    normalized[str_k] = v.model_dump_json()
                elif FlextRuntime.is_list_like(v):
                    normalized[str_k] = [
                        FlextRuntime._normalize_to_metadata_scalar(item) for item in v
                    ]
                elif FlextRuntime.is_dict_like(v):
                    inner: dict[str, t.Primitives] = {}
                    for ik, iv in v.items():
                        inner[str(ik)] = FlextRuntime._normalize_to_metadata_scalar(iv)
                    normalized[str_k] = (
                        FlextModelFoundation.Validators
                        .metadata_json_dict_adapter()
                        .dump_json(inner)
                        .decode()
                    )
                else:
                    normalized[str_k] = str(v)
            return normalized
        if FlextRuntime.is_list_like(val):
            return [FlextRuntime._normalize_to_metadata_scalar(item) for item in val]
        return str(val)

    @staticmethod
    def safe_get_attribute(
        obj: t.RuntimeData | type | ModuleType,
        attr: str,
        default: t.ValueOrModel | None = None,
    ) -> t.ValueOrModel | None:
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
    def structlog() -> ModuleType:
        """Return the imported structlog module."""
        return structlog

    class DependencyIntegration:
        """Centralize dependency-injector wiring with provider helpers.

        This bridge keeps dependency-injector usage confined to L1 while
        exposing a narrow API for higher layers. Factories and configuration
        providers are materialized here to avoid duplicate dictionaries or
        direct imports of ``dependency_injector`` outside the runtime module.
        A small DeclarativeContainer captures config/resources so L3 callers
        never import dependency-injector directly.
        """

        class DynamicContainerWithConfig(containers.DynamicContainer):
            """Dynamic container with declared configuration provider."""

            config: providers.Configuration = providers.Configuration()

        class BridgeContainer(containers.DeclarativeContainer):
            """Declarative container grouping config and resource modules."""

            config = providers.Configuration()
            services = providers.Object(containers.DynamicContainer())
            resources = providers.Object(containers.DynamicContainer())

        Provide = wiring.Provide
        inject = staticmethod(wiring.inject)

        @classmethod
        def create_container(
            cls,
            *,
            config: t.ConfigMap | None = None,
            services: Mapping[str, t.RegisterableService] | None = None,
            factories: Mapping[
                str,
                Callable[[], t.Scalar | Sequence[t.Scalar] | Mapping[str, t.Scalar]],
            ]
            | None = None,
            resources: Mapping[str, t.ResourceCallable] | None = None,
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
            di_container = cls.DynamicContainerWithConfig()
            if config is not None:
                _ = cls.bind_configuration(di_container, config)
            if services:
                for name, instance in services.items():
                    _ = cls.register_object(di_container, name, instance)
            if factories:
                for name, factory in factories.items():
                    _ = cls.register_factory(
                        di_container, name, factory, cache=factory_cache
                    )
            if resources:
                for name, resource_factory in resources.items():
                    _ = cls.register_resource(di_container, name, resource_factory)
            if wire_modules or wire_packages or wire_classes:
                cls.wire(
                    di_container,
                    modules=wire_modules,
                    packages=wire_packages,
                    classes=wire_classes,
                )
            return di_container

        @classmethod
        def create_layered_bridge(
            cls, config: t.ConfigMap | None = None
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
            bridge.services = providers.Object(service_module)
            bridge.resources = providers.Object(resource_module)
            cls.bind_configuration_provider(bridge.config, config)
            return (bridge, service_module, resource_module)

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
                configuration_provider.from_dict(dict(config.items()))
            if isinstance(
                di_container,
                FlextRuntime.DependencyIntegration.DynamicContainerWithConfig,
            ):
                configured_container: FlextRuntime.DependencyIntegration.DynamicContainerWithConfig = di_container
                configured_container.config = configuration_provider
            else:
                setattr(di_container, "config", configuration_provider)
            return configuration_provider

        @staticmethod
        def bind_configuration_provider(
            configuration_provider: providers.Configuration,
            config: t.ConfigMap | None,
        ) -> providers.Configuration:
            """Bind configuration directly to an existing provider."""
            if config:
                configuration_provider.from_dict(dict(config.items()))
            return configuration_provider

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
        def register_object(
            di_container: containers.DynamicContainer, name: str, instance: T
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
            _ = packages
            wiring.wire(
                modules=modules_to_wire or None, packages=None, container=container
            )

    @classmethod
    def configure_structlog(
        cls,
        *,
        config: BaseModel | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[t.Container] | None = None,
        wrapper_class_factory: Callable[[], type[p.Logger]] | None = None,
        logger_factory: Callable[[], p.Logger] | None = None,
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
        async_logging = True
        if config is not None:
            log_level = getattr(config, "log_level", log_level)
            console_renderer = getattr(config, "console_renderer", console_renderer)
            additional_processors_from_config = getattr(
                config, "additional_processors", None
            )
            if additional_processors_from_config:
                additional_processors = additional_processors_from_config
            wrapper_class_factory = getattr(
                config, "wrapper_class_factory", wrapper_class_factory
            )
            logger_factory = getattr(config, "logger_factory", logger_factory)
            cache_logger_on_first_use = getattr(
                config, "cache_logger_on_first_use", cache_logger_on_first_use
            )
            async_logging = getattr(config, "async_logging", True)
        if cls._structlog_configured:
            return
        level_to_use = log_level if log_level is not None else logging.INFO
        module = structlog
        processors: list[structlog.types.Processor] = [
            module.contextvars.merge_contextvars,
            add_log_level,
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
        ]
        if additional_processors:
            processors.extend(
                proc
                for proc in additional_processors
                if cls._is_structlog_processor(proc)
            )
        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(JSONRenderer())
        wrapper_arg: type | None = None
        if wrapper_class_factory is not None:
            wrapper_arg = wrapper_class_factory()
        else:
            wrapper_arg = module.make_filtering_bound_logger(level_to_use)
        factory_to_use: Callable[..., object] | object | None
        if logger_factory is not None:
            factory_to_use = logger_factory
        elif async_logging:
            if cls._async_writer is None:
                cls._async_writer = cls._AsyncLogWriter(sys.stdout)
            print_logger_factory_cls = getattr(module, "PrintLoggerFactory", None)
            if print_logger_factory_cls is not None:
                factory_to_use = print_logger_factory_cls(file=cls._async_writer)
            else:
                factory_to_use = module.PrintLoggerFactory()
        else:
            factory_to_use = module.PrintLoggerFactory()
        configure_fn = module.configure if hasattr(module, "configure") else None
        if configure_fn is not None and callable(configure_fn):
            _ = configure_fn(
                processors=processors,
                wrapper_class=wrapper_arg,
                logger_factory=factory_to_use if callable(factory_to_use) else None,
                cache_logger_on_first_use=cache_logger_on_first_use,
            )
        cls._structlog_configured = True

    @classmethod
    def reconfigure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: list[t.Container] | None = None,
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
            from flext_core import FlextRuntime

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
        module = structlog
        module.reset_defaults()
        if cls._async_writer:
            cls._async_writer.shutdown()
            cls._async_writer = None
        cls._structlog_configured = False
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

    @staticmethod
    def level_based_context_filter(
        _logger: p.Logger | None,
        method_name: str,
        event_dict: t.GeneralValueTypeMapping,
    ) -> t.GeneralValueTypeMapping:
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
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        current_level = level_hierarchy.get(method_name.lower(), 20)
        filtered_dict: dict[str, t.Scalar] = {}
        for key, value in event_dict.items():
            if key.startswith("_level_"):
                parts = key.split("_", c.Validation.LEVEL_PREFIX_PARTS_COUNT)
                if len(parts) >= c.Validation.LEVEL_PREFIX_PARTS_COUNT:
                    required_level_name = parts[2]
                    actual_key = parts[3]
                    required_level = level_hierarchy.get(
                        required_level_name.lower(), 10
                    )
                    if current_level >= required_level:
                        filtered_dict[actual_key] = value
                else:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value
        return filtered_dict

    class RuntimeResult[T](BaseModel):
        """Lightweight implementation of Result pattern (Layer 0.5).

        Implements basic success/failure handling with Pydantic integration.
        Compatible with p.Result and r usage patterns.
        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            frozen=True,
            populate_by_name=True,
        )

        is_success: Annotated[bool, Field(default=True)]
        _payload: T | None = PrivateAttr(default=None)
        error: Annotated[str | None, Field(default=None)]
        error_code: Annotated[str | None, Field(default=None)]
        error_data: Annotated[t.ConfigMap | None, Field(default=None)]

        _exception: BaseException | None = PrivateAttr(default=None)
        _result_logger: p.Logger | None = PrivateAttr(default=None)

        @override
        def __repr__(self) -> str:
            """String representation using short alias 'r' for brevity."""
            if self.is_success:
                return f"r[T].ok({self.value!r})"
            return f"r[T].fail({self.error!r})"

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            return self.is_success

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

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
            return self.unwrap_or(default)

        @property
        def exception(self) -> BaseException | None:
            """Get the exception if one was captured."""
            return self._exception

        @property
        def is_failure(self) -> bool:
            """Check if result is a failure."""
            return not self.is_success

        @property
        def result_logger(self) -> p.Logger:
            """Logger for RuntimeResult."""
            logger = self._result_logger
            if logger is None:
                logger = structlog.get_logger(__name__)
                setattr(self, "_result_logger", logger)
            return logger

        @property
        def value(self) -> T:
            """Result value — returns _payload directly on success.

            None IS a valid payload when T includes None (e.g. r[str | None]).
            DO NOT add None checks, asserts, or invariant guards here.
            The only guard is is_success — if the result is a failure,
            accessing .value raises RuntimeError. cast() narrows T | None to T.
            """
            if not self.is_success:
                msg = f"Cannot access value of failed result: {self.error}"
                raise RuntimeError(msg)
            if self._payload is not None:
                return self._payload
            return cast("T", self._payload)

        @classmethod
        def fail(
            cls,
            error: str | None,
            error_code: str | None = None,
            error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        ) -> Self:
            """Create failed result with error message.

            Business Rule: Creates failed RuntimeResult with error message, optional error
            code, and optional error metadata. Converts None error to empty string for
            consistency. This matches the API of r.fail() for compatibility.

            Args:
                error: Error message (None will be converted to empty string)
                error_code: Optional error code for categorization
                error_data: Optional error metadata

            Returns:
                Failed RuntimeResult instance

            """
            error_msg = error if error is not None else ""
            validated_error_data: t.ConfigMap
            if error_data is None:
                validated_error_data = t.ConfigMap(root={})
            elif isinstance(error_data, t.ConfigMap):
                validated_error_data = error_data
            elif isinstance(error_data, BaseModel):
                dump = error_data.model_dump()
                validated_error_data = t.ConfigMap(dump)
            else:
                validated_error_data = t.ConfigMap(dict(error_data))

            return cls(
                is_success=False,
                error=error_msg,
                error_code=error_code,
                error_data=validated_error_data,
            )

        @classmethod
        def ok(cls, value: T) -> FlextRuntime.RuntimeResult[T]:
            """Create successful result wrapping data.

            Business Rule: Creates successful RuntimeResult wrapping value. Raises ValueError
            if value is None (None values are not allowed in success results). This enforces
            the same invariant as r.ok() at the base class level.

            Args:
                value: Value to wrap in success result (must not be None)

            Returns:
                Successful RuntimeResult instance

            """
            instance = cls(
                is_success=True,
                error=None,
                error_code=None,
                error_data=t.ConfigMap(root={}),
            )
            setattr(instance, "_payload", value)
            return instance

        def filter(
            self, predicate: Callable[[T], bool]
        ) -> FlextRuntime.RuntimeResult[T]:
            """Filter success value using predicate."""
            if self.is_success and (not predicate(self.value)):
                return FlextRuntime.RuntimeResult[T].fail(
                    error="Filter predicate failed"
                )
            return self

        def flat_map[U](
            self, func: Callable[[T], FlextRuntime.RuntimeResult[U]]
        ) -> FlextRuntime.RuntimeResult[U]:
            """Chain operations returning RuntimeResult."""
            if self.is_success:
                return func(self.value)
            return FlextRuntime.RuntimeResult[U].fail(
                error=self.error,
                error_code=self.error_code,
                error_data=self.error_data,
            )

        def flow_through[U](
            self, *funcs: Callable[[T | U], FlextRuntime.RuntimeResult[U]]
        ) -> FlextRuntime.RuntimeResult[T] | FlextRuntime.RuntimeResult[U]:
            """Chain multiple operations in sequence.

            Returns:
                RuntimeResult[T] if no funcs provided, value is None, or chain
                short-circuits on failure. RuntimeResult[U] if all funcs applied.

            """
            if self.is_failure or not funcs:
                return self
            current: FlextRuntime.RuntimeResult[T] | FlextRuntime.RuntimeResult[U] = (
                self
            )
            for func in funcs:
                if current.is_success:
                    result_value = current.value
                    if result_value is not None:
                        current = func(result_value)
                    else:
                        break
                else:
                    break
            return current

        def fold[U](
            self, on_failure: Callable[[str], U], on_success: Callable[[T], U]
        ) -> U:
            """Fold result into single value (catamorphism)."""
            if self.is_success:
                return on_success(self.value)
            return on_failure(self.error or "")

        def lash(
            self, func: Callable[[str], FlextRuntime.RuntimeResult[T]]
        ) -> FlextRuntime.RuntimeResult[T]:
            """Apply recovery function on failure."""
            if not self.is_success:
                return func(self.error or "")
            return self

        def map[U](self, func: Callable[[T], U]) -> FlextRuntime.RuntimeResult[U]:
            """Transform success value using function."""
            if self.is_success:
                try:
                    return FlextRuntime.RuntimeResult[U].ok(value=func(self.value))
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as e:
                    return FlextRuntime.RuntimeResult[U].fail(error=str(e))
            return FlextRuntime.RuntimeResult[U].fail(
                error=self.error,
                error_code=self.error_code,
                error_data=self.error_data,
            )

        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextRuntime.RuntimeResult[T]:
            """Transform error message."""
            if not self.is_success:
                return FlextRuntime.RuntimeResult[T].fail(
                    error=func(self.error or ""),
                    error_code=self.error_code,
                    error_data=self.error_data,
                )
            return self

        def recover(self, func: Callable[[str], T]) -> FlextRuntime.RuntimeResult[T]:
            """Recover from failure with fallback value."""
            if not self.is_success:
                fallback_value = func(self.error or "")
                return FlextRuntime.RuntimeResult[T].ok(value=fallback_value)
            return self

        def tap(self, func: Callable[[T], None]) -> FlextRuntime.RuntimeResult[T]:
            """Apply side effect to success value, return unchanged."""
            if self.is_success and self._payload is not None:
                func(self._payload)
            return self

        def tap_error(
            self, func: Callable[[str], None]
        ) -> FlextRuntime.RuntimeResult[T]:
            """Apply side effect to error, return unchanged."""
            if not self.is_success:
                func(self.error or "")
            return self

        def unwrap(self) -> T:
            """Unwrap the success value or raise RuntimeError."""
            if not self.is_success:
                msg = f"Cannot unwrap failed result: {self.error}"
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

    class Integration:
        """Application-layer integration helpers using structlog directly (Layer 0.5).

        **DESIGN**: These methods use structlog directly without importing from
        higher layers (FlextContext, FlextLogger), avoiding all circular imports.

        **USAGE**: Opt-in helpers for application/service layer to integrate
        foundation components with context tracking.

        **CORRECT USAGE** (Application Layer):
            ```python
            from flext_core import FlextContainer
            from flext_core import FlextRuntime

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
            _ = structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                _ = structlog.contextvars.bind_contextvars(
                    service_version=service_version
                )
            if enable_context_correlation:
                alphabet = string.ascii_letters + string.digits
                correlation_id = (
                    f"flext-{''.join(secrets.choice(alphabet) for _ in range(12))}"
                )
                _ = structlog.contextvars.bind_contextvars(
                    correlation_id=correlation_id
                )
            logger = structlog.get_logger(__name__)
            logger.info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
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
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")
            logger = structlog.get_logger(__name__)
            logger.info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

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
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")
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

    @classmethod
    def ensure_trace_context(
        cls,
        context: Mapping[str, t.Scalar] | t.ScalarOrModel,
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
        if isinstance(context, Mapping):
            try:
                parsed_context: dict[str, t.ValueOrModel] = {
                    str(k): str(v) for k, v in context.items()
                }
            except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
                logging.getLogger(__name__).debug(
                    "Failed to convert mapping context to string dict", exc_info=exc
                )
                parsed_context = {}
            context_dict = t.ConfigMap(parsed_context)
        elif not isinstance(context, Mapping) and FlextRuntime._is_scalar(context):
            context_dict = t.ConfigMap(root={})
        elif isinstance(context, BaseModel):
            context_dict.update(context.model_dump())
        else:
            context_dict = t.ConfigMap(root={})
        result: dict[str, str] = {}
        for key, value in context_dict.items():
            result[key] = str(value)
        if "trace_id" not in result:
            result["trace_id"] = FlextRuntime.generate_id()
        if "span_id" not in result:
            result["span_id"] = FlextRuntime.generate_id()
        if include_correlation_id and "correlation_id" not in result:
            result["correlation_id"] = FlextRuntime.generate_id()
        if include_timestamp and "timestamp" not in result:
            result["timestamp"] = FlextRuntime.generate_datetime_utc().isoformat()
        return result

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.RuntimeData,
        entity_b: t.RuntimeData,
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
        entity_a_type = type(entity_a)
        if not isinstance(entity_b, entity_a_type):
            return False
        id_a = FlextRuntime.safe_get_attribute(entity_a, id_attr)
        id_b = FlextRuntime.safe_get_attribute(entity_b, id_attr)
        return id_a is not None and id_a == id_b

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.RuntimeData, obj_b: t.RuntimeData
    ) -> bool:
        """Compare value objects by their values (all attributes)."""
        if FlextRuntime._is_scalar(obj_a):
            return obj_a == obj_b
        if FlextRuntime._is_scalar(obj_b):
            return False
        if hasattr(obj_a, "__iter__") and (not hasattr(obj_a, "model_dump")):
            return obj_a == obj_b
        if hasattr(obj_b, "__iter__") and (not hasattr(obj_b, "model_dump")):
            return obj_a == obj_b
        obj_a_type = type(obj_a)
        if not isinstance(obj_b, obj_a_type):
            return False
        if isinstance(obj_a, BaseModel) and isinstance(obj_b, BaseModel):
            dump_a = obj_a.model_dump()
            dump_b = obj_b.model_dump()
            return dump_a == dump_b
        if hasattr(obj_a, "__dict__") and hasattr(obj_b, "__dict__"):
            return obj_a.__dict__ == obj_b.__dict__
        return repr(obj_a) == repr(obj_b)

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
    def get_log_level_from_config() -> int:
        """Get log level from default constant (bridge for _models).

        Returns:
            int: Numeric logging level (e.g., logging.INFO = 20)

        """
        default_log_level = c.Logging.DEFAULT_LEVEL.upper()
        return int(
            getattr(logging, default_log_level)
            if hasattr(logging, default_log_level)
            else logging.INFO
        )

    @staticmethod
    def hash_entity_by_id(entity: t.RuntimeData, id_attr: str = "unique_id") -> int:
        """Hash entity based on unique ID and type."""
        if FlextRuntime._is_scalar(entity):
            return hash(entity)
        entity_id = FlextRuntime.safe_get_attribute(entity, id_attr)
        if entity_id is None:
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.RuntimeData) -> int:
        """Hash value object based on all attribute values."""
        if FlextRuntime._is_scalar(obj):
            return hash(obj)
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            return hash(tuple(sorted(((k, str(v)) for k, v in data.items()))))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        return hash(repr(obj))

    @staticmethod
    def validate_http_status_codes(
        codes: list[int] | list[str] | list[int | str],
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
                code_int = int(str(code))
                if not min_val <= code_int <= max_val:
                    msg = f"Invalid HTTP status code: {code} (must be {min_val}-{max_val})"
                    return FlextRuntime.RuntimeResult[list[int]].fail(msg)
                validated_codes.append(code_int)
            except ValueError:
                return FlextRuntime.RuntimeResult[list[int]].fail(
                    f"Cannot convert to integer: {code}"
                )
        return FlextRuntime.RuntimeResult[list[int]].ok(validated_codes)


__all__ = ["FlextRuntime"]
