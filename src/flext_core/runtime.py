"""Runtime bridge exposing external libraries with dispatcher-safe boundaries.

**ARCHITECTURE LAYER 0.5** - Integration Bridge (Minimal Dependencies)

This module provides runtime utilities that consume patterns from c and
expose external library APIs to higher-level modules, maintaining proper dependency
hierarchy while eliminating code duplication. Implements structural typing via
p (duck typing - no inheritance required).

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
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import (
    ClassVar,
    TypeIs,
    override,
)

import orjson
import structlog
from dependency_injector import containers, providers, wiring
from pydantic import BaseModel, ValidationError
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level
from structlog.types import Processor as _StructlogProcessor

from flext_core import T, c, p, t


class FlextRuntime:
    """Expose structlog, DI providers, and validation helpers to higher layers."""

    _structlog_configured: ClassVar[bool] = False
    _runtime_logger: ClassVar[p.Logger | None] = None
    Metadata: ClassVar[type[p.Metadata] | None] = None

    @classmethod
    def _require_metadata_model(cls) -> type[p.Metadata]:
        """Return the bound metadata model class or raise a runtime contract error."""
        metadata_cls = cls.Metadata
        if metadata_cls is None:
            msg = "FlextRuntime.Metadata is not bound to a concrete model"
            raise RuntimeError(msg)
        return metadata_cls

    @property
    def logger(self) -> p.Logger:
        """Infrastructure logger for FlextRuntime internals (avoids circular imports)."""
        cls = type(self)
        logger = cls._runtime_logger
        if logger is None:
            logger = structlog.get_logger(__name__)
            cls._runtime_logger = logger
        return logger

    class _AsyncLogWriter(io.TextIOBase):
        """Background log writer using a queue and a separate thread.

        Provides non-blocking logging by buffering log messages to a queue
        and writing them to the destination stream in a background thread.
        """

        def __init__(self, stream: typing.TextIO) -> None:
            super().__init__()
            self.stream = stream
            self._encoding: str = str(
                getattr(self.stream, "encoding", c.DEFAULT_ENCODING)
            )
            self._errors: str | None = getattr(self.stream, "errors", None)
            self._newlines: str | tuple[str, ...] | None = getattr(
                self.stream, "newlines", None
            )
            self.queue: queue.Queue[str | None] = queue.Queue(
                maxsize=c.MAX_ITEMS,
            )
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="flext-async-log-writer",
            )
            self.thread.start()
            _ = atexit.register(self.shutdown)
            self._writer_logger: p.Logger | None = None

        @property
        def _writer_log(self) -> p.Logger:
            """Logger for async log writer."""
            logger = self._writer_logger
            if logger is None:
                logger = structlog.get_logger(__name__)
                self._writer_logger = logger
            return logger

        @property
        def mode(self) -> str:
            """Expose text stream mode for TextIO compatibility."""
            return str(getattr(self.stream, "mode", "w"))

        @property
        def name(self) -> str:
            """Expose text stream name for TextIO compatibility."""
            return str(getattr(self.stream, "name", "<async-log-writer>"))

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
            try:
                self.stream.flush()
            except (OSError, ValueError):
                return

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
                self.queue.put(s, block=c.ASYNC_BLOCK_ON_FULL)
            return len(s)

        def _worker(self) -> None:
            """Worker thread processing log queue."""
            while True:
                try:
                    msg = self.queue.get(timeout=0.1)
                    if msg is None:
                        break
                    _ = self.stream.write(msg)
                    _ = self.stream.flush()
                    self.queue.task_done()
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                except (OSError, ValueError, TypeError) as exc:
                    self._writer_log.warning(
                        "Async log writer stream operation failed",
                        exc_info=exc,
                    )
                    with contextlib.suppress(OSError, ValueError, TypeError):
                        _ = self.stream.write("Error in async log writer\n")

    _async_writer: ClassVar[_AsyncLogWriter | None] = None

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
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
    def is_primitive(value: t.GuardInput) -> TypeIs[t.Primitives]:
        """Check if value is a primitive type accepted by t.Primitives."""
        return isinstance(value, t.PRIMITIVES_TYPES)

    @staticmethod
    def is_scalar(value: t.GuardInput) -> TypeIs[t.Scalar]:
        """Check if value is a scalar type accepted by t.Scalar."""
        return isinstance(value, t.SCALAR_TYPES)

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    _GENERIC_LIST_ALIASES: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "StringList": (str,),
        "List": (str,),
        "IntList": (int,),
        "FloatList": (float,),
        "BoolList": (bool,),
    }

    _GENERIC_DICT_ALIASES: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "Dict": (str, str),
        "StringDict": (str, str),
        "NestedDict": (str, dict),
        "IntDict": (str, int),
        "FloatDict": (str, float),
        "BoolDict": (str, bool),
    }

    @staticmethod
    def extract_generic_args(
        type_hint: t.TypeHintSpecifier,
    ) -> tuple[t.GenericTypeArgument | type, ...]:
        """Extract generic type arguments from a type hint.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            resolved_hint = getattr(type_hint, "__value__", type_hint)
            args = typing.get_args(resolved_hint)
            if args:
                return args
            type_name = getattr(type_hint, "__name__", "")
            if not type_name:
                type_name = getattr(resolved_hint, "__name__", "")
            if not type_name:
                return ()
            return (
                FlextRuntime._GENERIC_LIST_ALIASES.get(type_name)
                or FlextRuntime._GENERIC_DICT_ALIASES.get(type_name)
                or ()
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return ()

    @classmethod
    def get_logger(cls, name: str | None = None) -> p.Logger:
        """Get structlog logger instance - same structure/config used by FlextLogger.

        Args:
            name: Logger name (module name). Defaults to __name__ of caller.

        Returns:
            Logger: Typed structlog logger instance (same as FlextLogger.logger).

        Note:
            FlextLogger internally uses: FlextRuntime.get_logger(name).bind(**context)
            This method returns the base logger before context binding.

        """
        cls.ensure_structlog_configured()
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
        """Type guard to narrow broad runtime data to BaseModel.

        This allows isinstance checks to narrow types for FlextRuntime methods
        that accept broad runtime contracts that may include BaseModel.
        """
        match obj:
            case BaseModel():
                return True
            case _:
                return False

    @staticmethod
    def _has_dict_protocol(obj: t.RuntimeData) -> bool:
        if not isinstance(obj, Mapping):
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
    ) -> TypeIs[t.ConfigMap | t.ContainerMapping]:
        """TypeIs to check if value is dict-like.

        Note:
            ``value`` remains broad because this guard is a boundary utility used
            by normalization paths that accept full runtime payload contracts.

        Args:
            value: Value to check

        Returns:
            True if value is a ConfigMap or dict-like recursive payload, False otherwise

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
    ) -> TypeIs[t.ContainerList]:
        """Type guard to check if value is list-like."""
        return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))

    @staticmethod
    def is_sequence_type(type_hint: t.TypeHintSpecifier) -> bool:
        """Check if type hint represents a sequence type (list, tuple, etc.).

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
                type_hint,
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
        value: object,
    ) -> TypeIs[_StructlogProcessor]:
        return callable(value)

    @staticmethod
    def _to_plain_container(value: t.RuntimeAtomic) -> t.RecursiveContainer:
        """Flatten a runtime atomic value to plain Python types."""
        match value:
            case t.ConfigMap() | t.Dict():
                return {
                    str(k): FlextRuntime._to_plain_container(
                        FlextRuntime.normalize_to_container(v),
                    )
                    for k, v in value.root.items()
                }
            case t.ObjectList():
                return list(value.root)
            case bool() | str() | int() | float() | datetime() | Path():
                return value
            case _:
                return str(value)

    @staticmethod
    def _normalize_dict_entries(
        items: Sequence[tuple[str, t.RuntimeData]],
    ) -> MutableMapping[str, t.ValueOrModel]:
        """Normalize key-value pairs for container dict construction."""
        result: MutableMapping[str, t.ValueOrModel] = {}
        for key, item in items:
            normalized = FlextRuntime.normalize_to_container(item)
            result[key] = FlextRuntime._to_plain_container(normalized)
        return result

    @staticmethod
    def resolve_nested_model_class[TModel: BaseModel](
        *,
        module_name: str,
        qualname: str,
        models_module_name: str,
        attribute_name: str,
        fallback: type[TModel],
    ) -> type[TModel]:
        """Resolve a nested override model class from a facade module path."""
        min_qualname_parts = 2
        if module_name != models_module_name or "." not in qualname:
            return fallback
        parts = qualname.split(".")
        models_module = sys.modules.get(models_module_name)
        if not isinstance(models_module, ModuleType) or len(parts) < min_qualname_parts:
            return fallback
        obj: object | None = models_module.__dict__.get(parts[0])
        for part in parts[1:-1]:
            if isinstance(obj, ModuleType):
                obj = obj.__dict__.get(part)
                continue
            if isinstance(obj, type):
                obj = obj.__dict__.get(part)
                continue
            obj = None
            break
        if not isinstance(obj, type):
            return fallback
        resolved_attr = obj.__dict__.get(attribute_name)
        if isinstance(resolved_attr, type) and issubclass(resolved_attr, fallback):
            return resolved_attr
        return fallback

    @staticmethod
    def normalize_model_input_mapping(
        value: BaseModel | t.Dict | t.ScalarMapping | None,
    ) -> Mapping[str, t.ValueOrModel] | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        if isinstance(value, t.Dict):
            return dict(value.root)
        if isinstance(value, BaseModel):
            return value.model_dump()
        return dict(value)

    @staticmethod
    def validate_model_sequence[TModel: BaseModel](
        values: Sequence[t.ValueOrModel],
        model_cls: type[TModel],
    ) -> Sequence[TModel]:
        """Validate a heterogeneous input sequence into one Pydantic model type."""
        validated_items: list[TModel] = []
        item_errors: list[str] = []
        for index, value in enumerate(values):
            try:
                if isinstance(value, model_cls):
                    validated_items.append(value)
                elif isinstance(value, BaseModel):
                    validated_items.append(model_cls.model_validate(value.model_dump()))
                elif isinstance(value, Mapping):
                    validated_items.append(model_cls.model_validate(dict(value)))
                else:
                    validated_items.append(model_cls.model_validate(value))
            except ValidationError as exc:
                item_errors.extend(
                    f"{index}.{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', 'validation error')}"
                    for err in exc.errors()
                )
        if item_errors:
            msg = f"Batch validation failed: {'; '.join(item_errors)}"
            raise TypeError(msg)
        return validated_items

    @classmethod
    def normalize_metadata_input(cls, value: t.MetadataInput) -> p.Metadata:
        """Normalize metadata input into the bound metadata model."""
        metadata_cls = typing.cast("type[BaseModel]", cls._require_metadata_model())
        if value is None:
            return typing.cast(
                "p.Metadata",
                metadata_cls.model_validate({c.FIELD_ATTRIBUTES: {}}),
            )
        if isinstance(value, metadata_cls):
            return typing.cast("p.Metadata", value)
        if not isinstance(value, Mapping):
            msg = (
                "metadata must be None, dict, or bound metadata model, got "
                f"{value.__class__.__name__}"
            )
            raise TypeError(msg)
        return typing.cast(
            "p.Metadata",
            metadata_cls.model_validate({c.FIELD_ATTRIBUTES: dict(value)}),
        )

    @staticmethod
    def normalize_recursive_metadata_value(
        item: t.MetadataOrValue | None,
    ) -> t.RecursiveContainer:
        """Normalize metadata-like values while preserving recursive shape."""
        if item is None:
            return None
        if isinstance(item, (bool, str, int, float, datetime)):
            return item
        if isinstance(item, Mapping):
            normalized_map: t.MutableContainerMapping = {}
            for key, value in item.items():
                normalized_map[str(key)] = (
                    FlextRuntime.normalize_recursive_metadata_value(
                        value if FlextRuntime.is_scalar(value) else str(value)
                    )
                )
            return normalized_map
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [
                FlextRuntime.normalize_recursive_metadata_value(
                    value if FlextRuntime.is_scalar(value) else str(value)
                )
                for value in item
            ]
        return str(item)

    @staticmethod
    def normalize_domain_event_data(
        value: t.ValueOrModel | Mapping[str, t.ValueOrModel] | None,
    ) -> Mapping[str, t.RecursiveContainer]:
        """Normalize domain event payloads into comparable plain mappings."""
        if value is None:
            return {}
        if not isinstance(value, (t.ConfigMap, Mapping)):
            msg = "Domain event data must be a dictionary or None"
            raise TypeError(msg)
        raw_source = value.root if isinstance(value, t.ConfigMap) else value
        typed_source = t.dict_str_metadata_adapter().validate_python(raw_source)
        return {
            str(key): FlextRuntime.normalize_recursive_metadata_value(item)
            for key, item in typed_source.items()
        }

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
        match val:
            case None:
                return ""
            case BaseModel():
                return val
            case Path():
                return val
            case _ if FlextRuntime.is_scalar(val):
                return val
            case _ if FlextRuntime.is_dict_like(val):
                if isinstance(val, t.ConfigMap):
                    entries = [(k, v) for k, v in val.root.items()]
                else:
                    entries = [(str(k), v) for k, v in val.items()]
                return t.Dict(root=FlextRuntime._normalize_dict_entries(entries))
            case _ if FlextRuntime.is_list_like(val):
                normalized_list: t.FlatContainerList = [
                    item
                    for v in val
                    if isinstance(
                        item := FlextRuntime.normalize_to_container(v),
                        (str, int, float, bool, datetime, Path),
                    )
                ]
                return t.ObjectList(root=normalized_list)
            case _:
                return str(val)

    @staticmethod
    def _normalize_to_metadata_scalar(val: t.RuntimeData) -> t.Primitives:
        if val is None:
            return ""
        if FlextRuntime.is_primitive(val):
            return val
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, BaseModel):
            return val.model_dump_json()
        return str(val)

    @staticmethod
    def _normalize_metadata_dict_value(
        v: t.RuntimeData,
    ) -> t.Scalar | t.ScalarList:
        """Normalize a single dict value for metadata context."""
        match v:
            case None:
                return ""
            case Path():
                return str(v)
            case BaseModel():
                return v.model_dump_json()
            case _ if FlextRuntime.is_scalar(v):
                return v
            case _ if FlextRuntime.is_list_like(v):
                return [FlextRuntime._normalize_to_metadata_scalar(item) for item in v]
            case _ if FlextRuntime.is_dict_like(v):
                inner: MutableMapping[str, t.Primitives] = {}
                for ik, iv in v.items():
                    inner[str(ik)] = FlextRuntime._normalize_to_metadata_scalar(iv)
                return orjson.dumps(inner).decode()
            case _:
                return str(v)

    @staticmethod
    def normalize_to_metadata(
        val: t.RuntimeData,
    ) -> t.MetadataValue:
        """Normalize input into metadata-compatible scalar, list, or mapping values."""
        match val:
            case None:
                return ""
            case Path():
                return str(val)
            case BaseModel():
                return val.model_dump_json()
            case datetime():
                return val
            case _ if FlextRuntime.is_primitive(val):
                return val
            case _ if FlextRuntime.is_dict_like(val):
                normalized: MutableMapping[str, t.Scalar | t.ScalarList] = {}
                for k, v in val.items():
                    normalized[str(k)] = FlextRuntime._normalize_metadata_dict_value(v)
                return normalized
            case _ if FlextRuntime.is_list_like(val):
                return [
                    FlextRuntime._normalize_to_metadata_scalar(item) for item in val
                ]
            case _:
                return str(val)

    @staticmethod
    def safe_get_attribute(
        obj: t.RuntimeData | type | ModuleType,
        attr: str,
        default: t.ValueOrModel | None = None,
    ) -> t.ValueOrModel | None:
        """Safe attribute access without raising AttributeError.

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
        """Centralize dependency-injector wiring with provider helpers."""

        class DynamicContainerWithConfig(containers.DynamicContainer):
            """Dynamic container with declared configuration provider."""

            config: providers.Configuration = providers.Configuration()

        class BridgeContainer(containers.DeclarativeContainer):
            """Declarative container grouping config and resource modules."""

            config = providers.Configuration()
            services = providers.Object(containers.DynamicContainer())
            resources = providers.Object(containers.DynamicContainer())

        ContainerCreationOptionsModel: ClassVar[
            p.ContainerCreationOptionsType | None
        ] = None

        Provide = wiring.Provide
        inject = staticmethod(wiring.inject)

        @classmethod
        def _require_container_creation_options_model(
            cls,
        ) -> p.ContainerCreationOptionsType:
            """Return the bound container options model or raise a contract error."""
            options_model = cls.ContainerCreationOptionsModel
            if options_model is None:
                msg = (
                    "FlextRuntime.DependencyIntegration.ContainerCreationOptionsModel "
                    "is not bound to a concrete implementation"
                )
                raise RuntimeError(msg)
            return options_model

        _OPTION_FIELDS: ClassVar[t.StrSequence] = (
            "config",
            "services",
            "factories",
            "resources",
            "wire_modules",
            "wire_packages",
            "wire_classes",
        )

        @classmethod
        def _parse_options(
            cls,
            container_options: p.ContainerCreationOptions
            | Mapping[str, t.RuntimeData]
            | None,
        ) -> p.ContainerCreationOptions:
            """Parse raw container options into a validated model."""
            options_model = cls._require_container_creation_options_model()
            match container_options:
                case None:
                    return options_model.model_validate({})
                case Mapping():
                    return options_model.model_validate(dict(container_options))
                case _:
                    return options_model.model_validate(
                        {
                            field: getattr(container_options, field)
                            for field in cls._OPTION_FIELDS
                        }
                        | {"factory_cache": container_options.factory_cache}
                    )

        @classmethod
        def _merge_options(
            cls,
            base: p.ContainerCreationOptions,
            overrides: Mapping[str, t.RuntimeData],
        ) -> p.ContainerCreationOptions:
            """Merge runtime kwargs over base options (override wins if not None)."""
            options_model = cls._require_container_creation_options_model()
            override_opts = options_model.model_validate(overrides)
            merged: MutableMapping[str, t.RuntimeData] = {
                field: (
                    getattr(override_opts, field)
                    if getattr(override_opts, field) is not None
                    else getattr(base, field)
                )
                for field in cls._OPTION_FIELDS
            }
            merged["factory_cache"] = override_opts.factory_cache
            return options_model.model_validate(merged)

        @classmethod
        def _populate_container(
            cls,
            di_container: containers.DynamicContainer,
            opts: p.ContainerCreationOptions,
        ) -> None:
            """Register config, services, factories, resources, and wiring."""
            if opts.config is not None:
                _ = cls.bind_configuration(di_container, opts.config)
            if opts.services:
                for name, instance in opts.services.items():
                    _ = cls.register_object(di_container, name, instance)
            if opts.factories:
                for name, factory in opts.factories.items():
                    _ = cls.register_factory(
                        di_container,
                        name,
                        factory,
                        cache=opts.factory_cache,
                    )
            if opts.resources:
                for name, resource_factory in opts.resources.items():
                    _ = cls.register_resource(di_container, name, resource_factory)
            if opts.wire_modules or opts.wire_packages or opts.wire_classes:
                cls.wire(
                    di_container,
                    modules=opts.wire_modules,
                    packages=opts.wire_packages,
                    classes=opts.wire_classes,
                )

        @classmethod
        def create_container(
            cls,
            container_options: p.ContainerCreationOptions
            | Mapping[str, t.RuntimeData]
            | None = None,
            **runtime_kwargs: t.RuntimeData,
        ) -> containers.DynamicContainer:
            """Create a DynamicContainer with optional pre-registration and wiring.

            Args:
                container_options: Options as protocol, mapping, or None.
                **runtime_kwargs: Override individual option fields.

            Returns:
                A dynamic container ready for immediate ``@inject`` consumption
                without manual follow-up registration calls.

            """
            base = cls._parse_options(container_options)
            opts = cls._merge_options(base, runtime_kwargs) if runtime_kwargs else base
            di_container = cls.DynamicContainerWithConfig()
            cls._populate_container(di_container, opts)
            return di_container

        @classmethod
        def create_layered_bridge(
            cls,
            config: t.ConfigMap | None = None,
        ) -> tuple[
            containers.DeclarativeContainer,
            containers.DynamicContainer,
            containers.DynamicContainer,
        ]:
            """Create a DeclarativeContainer bridged to dynamic modules."""
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
                configuration_provider.from_dict(dict(config))
            if isinstance(
                di_container,
                FlextRuntime.DependencyIntegration.DynamicContainerWithConfig,
            ):
                configured_container: FlextRuntime.DependencyIntegration.DynamicContainerWithConfig = di_container
                configured_container.config = configuration_provider
            else:
                setattr(di_container, c.DIR_CONFIG, configuration_provider)
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
            packages: t.StrSequence | None = None,
            classes: Sequence[type] | None = None,
        ) -> None:
            """Wire modules or packages to a DeclarativeContainer or DynamicContainer for @inject usage."""
            modules_to_wire: MutableSequence[ModuleType] = list(modules or [])
            if classes:
                for target_class in classes:
                    module = inspect.getmodule(target_class)
                    if module is not None:
                        modules_to_wire.append(module)
            _ = packages
            wiring.wire(
                modules=modules_to_wire or None,
                packages=None,
                container=container,
            )

    @staticmethod
    def _resolve_structlog_params(
        config: BaseModel | None,
        *,
        log_level: int | None,
        console_renderer: bool,
        additional_processors: Sequence[t.StructlogProcessor] | None,
        wrapper_class_factory: Callable[[], type[p.Logger]] | None,
        logger_factory: t.LoggerFactory,
        cache_logger_on_first_use: bool,
    ) -> tuple[
        int,
        bool,
        Sequence[t.StructlogProcessor] | None,
        Callable[[], type[p.Logger]] | None,
        t.LoggerFactory,
        bool,
        bool,
    ]:
        """Extract structlog params from config model or pass-through individual args.

        Returns (log_level, console_renderer, additional_processors,
                 wrapper_class_factory, logger_factory, cache_logger_on_first_use,
                 async_logging).
        """
        async_logging = True
        if config is not None:
            log_level = getattr(config, "log_level", log_level)
            console_renderer = getattr(config, "console_renderer", console_renderer)
            cfg_processors = getattr(config, "additional_processors", None)
            if cfg_processors:
                additional_processors = cfg_processors
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
        level = log_level if log_level is not None else logging.INFO
        return (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        )

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        additional_processors: Sequence[t.StructlogProcessor] | None,
    ) -> MutableSequence[_StructlogProcessor]:
        """Assemble the structlog processor chain."""
        processors: MutableSequence[_StructlogProcessor] = [
            structlog.contextvars.merge_contextvars,
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
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(JSONRenderer())
        return processors

    @classmethod
    def _resolve_logger_factory(
        cls,
        *,
        logger_factory: t.LoggerFactory,
        async_logging: bool,
    ) -> t.LoggerFactory | None:
        """Resolve the logger factory, enabling async output when requested."""
        if logger_factory is not None:
            return logger_factory
        if async_logging:
            try:
                print_logger_factory = structlog.PrintLoggerFactory
            except AttributeError:
                pass
            else:
                if callable(print_logger_factory):
                    return cls._build_async_logger_factory(print_logger_factory)
            try:
                write_logger_factory = structlog.WriteLoggerFactory
            except AttributeError:
                pass
            else:
                if callable(write_logger_factory):
                    return cls._build_async_logger_factory(write_logger_factory)
        return None

    @classmethod
    def _build_async_logger_factory(
        cls,
        factory_builder: Callable[..., t.LoggerFactory],
    ) -> t.LoggerFactory:
        """Build a structlog logger factory bound to the shared async writer."""
        if cls._async_writer is None:
            cls._async_writer = cls._AsyncLogWriter(sys.stdout)
        return factory_builder(file=cls._async_writer)

    @classmethod
    def configure_structlog(
        cls,
        *,
        config: BaseModel | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[t.StructlogProcessor] | None = None,
        wrapper_class_factory: Callable[[], type[p.Logger]] | None = None,
        logger_factory: t.LoggerFactory = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults.

        Supports both config model pattern (reduced params) and individual parameters.

        Args:
            config: Optional FlextModels.Config.StructlogConfig for all params
            log_level: Numeric log level (ignored if config provided). Defaults to ``logging.INFO``.
            console_renderer: Use console renderer (ignored if config provided)
            additional_processors: Extra processors (ignored if config provided)
            wrapper_class_factory: Custom wrapper factory (ignored if config provided)
            logger_factory: Custom logger factory (ignored if config provided)
            cache_logger_on_first_use: Cache logger (ignored if config provided)

        """
        if cls._structlog_configured:
            return
        (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        ) = cls._resolve_structlog_params(
            config,
            log_level=log_level,
            console_renderer=console_renderer,
            additional_processors=additional_processors,
            wrapper_class_factory=wrapper_class_factory,
            logger_factory=logger_factory,
            cache_logger_on_first_use=cache_logger_on_first_use,
        )
        processors = cls._build_structlog_processors(
            console_renderer=console_renderer,
            additional_processors=additional_processors,
        )
        wrapper_arg = (
            wrapper_class_factory()
            if wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(level)
        )
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory,
            async_logging=async_logging,
        )
        configure_fn = getattr(structlog, "configure", None)
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
        additional_processors: Sequence[t.StructlogProcessor] | None = None,
    ) -> None:
        """Force reconfigure structlog (ignores is_configured checks).

        Args:
            log_level: Numeric log level to reconfigure
            console_renderer: Use console renderer vs JSON
            additional_processors: Extra processors to add

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
        event_dict: t.ScalarMapping,
    ) -> t.ScalarMapping:
        """Filter context variables based on log level.

        Args:
            _logger: Logger instance (unused, required by structlog protocol)
            method_name: Log method name ('debug', 'info', 'warning', 'error', etc.)
            event_dict: Event dictionary with context variables

        Returns:
            Filtered event dictionary

        """
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            c.WarningLevel.ERROR: 40,
            "critical": 50,
        }
        current_level = level_hierarchy.get(method_name.lower(), 20)
        filtered_dict: t.MutableConfigurationMapping = {}
        for key, value in event_dict.items():
            if key.startswith("_level_"):
                parts = key.split("_", c.DEFAULT_MAX_WORKERS)
                if len(parts) >= c.DEFAULT_MAX_WORKERS:
                    required_level_name = parts[2]
                    actual_key = parts[3]
                    required_level = level_hierarchy.get(
                        required_level_name.lower(),
                        10,
                    )
                    if current_level >= required_level:
                        filtered_dict[actual_key] = value
                else:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value
        return filtered_dict

    class Integration:
        """Application-layer integration helpers using structlog directly (Layer 0.5)."""

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
                    service_version=service_version,
                )
            if enable_context_correlation:
                alphabet = string.ascii_letters + string.digits
                correlation_id = (
                    f"flext-{''.join(secrets.choice(alphabet) for _ in range(12))}"
                )
                _ = structlog.contextvars.bind_contextvars(
                    correlation_id=correlation_id,
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
            correlation_id = context_vars.get(c.KEY_CORRELATION_ID)
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
            correlation_id = context_vars.get(c.KEY_CORRELATION_ID)
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
        context: t.ScalarMapping | t.ScalarOrModel,
        *,
        include_correlation_id: bool = False,
        include_timestamp: bool = False,
    ) -> t.StrMapping:
        """Ensure context dict has distributed tracing fields (bridge for _models).

        Args:
            context: Context dictionary or recursive payload to enrich
            include_correlation_id: If True, ensure correlation_id exists
            include_timestamp: If True, ensure timestamp exists

        Returns:
            t.StrMapping: Enriched context with trace fields

        """
        context_dict = t.ConfigMap(root={})
        if isinstance(context, Mapping):
            try:
                parsed_context: MutableMapping[str, t.ValueOrModel] = {
                    str(k): str(v) for k, v in context.items()
                }
            except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
                logging.getLogger(__name__).debug(
                    "Failed to convert mapping context to string dict",
                    exc_info=exc,
                )
                parsed_context = {}
            context_dict = t.ConfigMap(root=parsed_context)
        elif not isinstance(context, Mapping) and FlextRuntime.is_scalar(context):
            context_dict = t.ConfigMap(root={})
        elif isinstance(context, BaseModel):
            context_dict.update(context.model_dump())
        else:
            context_dict = t.ConfigMap(root={})
        result: t.MutableStrMapping = {}
        for key, value in context_dict.items():
            result[key] = str(value)
        if "trace_id" not in result:
            result["trace_id"] = cls.generate_id()
        if "span_id" not in result:
            result["span_id"] = cls.generate_id()
        if include_correlation_id and c.KEY_CORRELATION_ID not in result:
            result[c.KEY_CORRELATION_ID] = cls.generate_id()
        if include_timestamp and "timestamp" not in result:
            result["timestamp"] = cls.generate_datetime_utc().isoformat()
        return result

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.RuntimeData,
        entity_b: t.RuntimeData,
        id_attr: str = "unique_id",
    ) -> bool:
        """Compare two entities by unique ID attribute."""
        if FlextRuntime.is_scalar(entity_a):
            return False
        match entity_a:
            case Sequence() | Mapping():
                return False
            case _:
                pass
        if FlextRuntime.is_scalar(entity_b):
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
        obj_a: t.RuntimeData,
        obj_b: t.RuntimeData,
    ) -> bool:
        """Compare value objects by their values (all attributes)."""
        if FlextRuntime.is_scalar(obj_a):
            return obj_a == obj_b
        if FlextRuntime.is_scalar(obj_b):
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
        default_log_level = c.DEFAULT_LEVEL.upper()
        return int(
            getattr(logging, default_log_level)
            if hasattr(logging, default_log_level)
            else logging.INFO,
        )

    @staticmethod
    def hash_entity_by_id(entity: t.RuntimeData, id_attr: str = "unique_id") -> int:
        """Hash entity based on unique ID and type."""
        if FlextRuntime.is_scalar(entity):
            return hash(entity)
        entity_id = FlextRuntime.safe_get_attribute(entity, id_attr)
        if entity_id is None:
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.RuntimeData) -> int:
        """Hash runtime value object based on all attribute values."""
        if FlextRuntime.is_scalar(obj):
            return hash(obj)
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            return hash(tuple(sorted(((k, str(v)) for k, v in data.items()))))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        return hash(repr(obj))


__all__ = ["FlextRuntime"]
