"""Type guard utilities split from guards.py."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard, TypeIs

from pydantic import BaseModel

from flext_core import p, t


class FlextUtilitiesGuardsType:
    type _GuardInput = (
        t.Scalar
        | Path
        | list[t.NormalizedValue]
        | Mapping[str, t.NormalizedValue | BaseModel]
        | tuple[object, ...]
        | BaseModel
        | p.ResultLike[t.Container | BaseModel]
        | t.ConfigMap
        | p.HasModelDump
        | p.ValidatorSpec
        | t.RegistrablePlugin
        | p.Logger
        | t.FactoryCallable
        | p.Settings
        | p.Context
        | t.RegisterableService
        | object
        | None
    )

    _PROTOCOL_SPECS: Mapping[str, Callable[[t.NormalizedValue], bool]] = (
        MappingProxyType({
            "config": lambda v: isinstance(v, p.Settings),
            "context": lambda v: isinstance(v, p.Context),
            "container": lambda v: isinstance(v, p.Container),
            "command_bus": lambda v: isinstance(v, p.Dispatcher),
            "handler": lambda v: isinstance(v, p.Handler),
            "logger": lambda v: isinstance(v, p.Logger),
            "result": lambda v: isinstance(v, p.Result),
            "service": lambda v: isinstance(v, p.Service),
            "middleware": lambda v: isinstance(v, p.Middleware),
        })
    )
    _PROTOCOL_TYPE_MAP: Mapping[type, str] = MappingProxyType({
        p.Settings: "config",
        p.Context: "context",
        p.Container: "container",
        p.Dispatcher: "command_bus",
        p.Handler: "handler",
        p.Logger: "logger",
        p.Result: "result",
        p.Service: "service",
        p.Middleware: "middleware",
    })
    _STRING_METHOD_MAP: frozenset[str] = frozenset({
        "str",
        "dict",
        "list",
        "tuple",
        "sequence",
        "mapping",
        "list_or_tuple",
        "sequence_not_str",
        "sequence_not_str_bytes",
        "sized",
        "callable",
        "bytes",
        "int",
        "float",
        "bool",
        "none",
        "string_non_empty",
        "dict_non_empty",
        "list_non_empty",
    })

    @staticmethod
    def _is_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
        return isinstance(value, Mapping)

    @staticmethod
    def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
        return isinstance(value, (list, tuple))

    @staticmethod
    def _all_container_sequence(value: Sequence[object]) -> bool:
        for item in value:
            if not FlextUtilitiesGuardsType.is_container(item):
                return False
        return True

    @staticmethod
    def _all_container_mapping_values(value: Mapping[str, object]) -> bool:
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsType.is_container(mapped_value):
                return False
        return True

    @staticmethod
    def is_object_list(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[list[t.NormalizedValue]]:
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[tuple[t.NormalizedValue, ...]]:
        return isinstance(value, tuple)

    @staticmethod
    def is_config_value(value: t.NormalizedValue) -> TypeGuard[t.NormalizedValue]:
        if value is None or isinstance(value, (str, int, float, bool, datetime)):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if not (
                    item is None or isinstance(item, (str, int, float, bool, datetime))
                ):
                    return False
            return True
        if isinstance(value, Mapping):
            for item in value.values():
                if not (
                    item is None or isinstance(item, (str, int, float, bool, datetime))
                ):
                    return False
            return True
        return False

    @staticmethod
    def is_configuration_dict(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[t.Dict]:
        if isinstance(value, t.Dict):
            for item_value in value.root.values():
                if not FlextUtilitiesGuardsType.is_container(item_value):
                    return False
            return True
        return FlextUtilitiesGuardsType._is_object_mapping(
            value
        ) and FlextUtilitiesGuardsType._all_container_mapping_values(value)

    @staticmethod
    def is_configuration_mapping(
        value: Mapping[str, t.NormalizedValue] | t.ConfigMap | t.Dict,
    ) -> TypeGuard[t.ConfigMap]:
        candidate: Mapping[str, t.NormalizedValue | BaseModel] = (
            value.root if isinstance(value, (t.ConfigMap, t.Dict)) else value
        )
        for item_value in candidate.values():
            if not FlextUtilitiesGuardsType.is_container(item_value):
                return False
        return True

    @staticmethod
    def is_context(value: FlextUtilitiesGuardsType._GuardInput) -> TypeGuard[p.Context]:
        return bool(
            hasattr(value, "get") and hasattr(value, "set") and hasattr(value, "clone")
        )

    @staticmethod
    def is_dict_non_empty(value: t.NormalizedValue) -> bool:
        return isinstance(value, Mapping) and bool(value)

    @staticmethod
    def is_flexible_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        if value is None or FlextUtilitiesGuardsType.is_scalar(value):
            return True
        if isinstance(value, (list, tuple)):
            return all(
                item is None or FlextUtilitiesGuardsType.is_scalar(item)
                for item in value
            )
        if isinstance(value, Mapping):
            return all(
                item is None or FlextUtilitiesGuardsType.is_scalar(item)
                for item in value.values()
            )
        return False

    @staticmethod
    def is_container(
        value: object,
    ) -> TypeGuard[str | int | float | bool | datetime | Path]:
        if value is None or isinstance(value, (str, int, float, bool, datetime)):
            return True
        if FlextUtilitiesGuardsType._is_object_sequence(value):
            return FlextUtilitiesGuardsType._all_container_sequence(value)
        if FlextUtilitiesGuardsType._is_object_mapping(value):
            return FlextUtilitiesGuardsType._all_container_mapping_values(value)
        return isinstance(value, Path)

    @staticmethod
    def is_general_value_type(value: t.NormalizedValue) -> bool:
        warnings.warn(
            "is_general_value_type is deprecated; use is_container. Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return callable(value) or FlextUtilitiesGuardsType.is_container(value)

    @staticmethod
    def is_handler_callable(value: t.NormalizedValue) -> TypeGuard[t.HandlerCallable]:
        return callable(value)

    @staticmethod
    def is_handler_type(
        value: t.NormalizedValue | t.HandlerCallable,
    ) -> TypeGuard[t.HandlerLike]:
        return (
            callable(value)
            or isinstance(value, Mapping)
            or (
                isinstance(value, BaseModel)
                and hasattr(value, "model_dump")
                and callable(value.model_dump)
            )
            or hasattr(value, "handle")
            or hasattr(value, "can_handle")
        )

    @staticmethod
    def is_list(value: t.NormalizedValue) -> TypeGuard[list[t.NormalizedValue]]:
        return isinstance(value, list)

    @staticmethod
    def is_list_non_empty(value: t.NormalizedValue) -> bool:
        return (
            isinstance(value, Sequence)
            and (not isinstance(value, (str, bytes)))
            and len(value) > 0
        )

    @staticmethod
    def is_mapping(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[Mapping[str, t.NormalizedValue]]:
        return isinstance(value, Mapping)

    @staticmethod
    def is_registerable(value: t.NormalizedValue) -> bool:
        return (
            isinstance(
                value, (str, int, float, bool, type(None), BaseModel, Path, Mapping)
            )
            or callable(value)
            or (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            )
            or FlextUtilitiesGuardsType.is_context(value)
            or hasattr(value, "__dict__")
            or bool(hasattr(value, "bind") and hasattr(value, "info"))
        )

    @staticmethod
    def is_factory(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[t.FactoryCallable]:
        return callable(value)

    @staticmethod
    def is_resource(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[t.ResourceCallable]:
        return callable(value)

    @staticmethod
    def is_primitive(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[t.Primitives]:
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def is_result_like(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[p.ResultLike[t.Container | BaseModel]]:
        return (
            hasattr(value, "is_success")
            and hasattr(value, "error")
            and hasattr(type(value), "value")
        )

    @staticmethod
    def is_scalar(value: FlextUtilitiesGuardsType._GuardInput) -> TypeGuard[t.Scalar]:
        return isinstance(value, (str, int, float, bool, datetime))

    @staticmethod
    def is_string_non_empty(value: t.NormalizedValue) -> TypeGuard[str]:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_registerable_service(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[t.RegisterableService]:
        return (
            value is None
            or isinstance(value, (str, int, float, bool, BaseModel, Path, Mapping))
            or callable(value)
            or (
                isinstance(value, Sequence)
                and (not isinstance(value, (str, bytes, bytearray)))
            )
            or FlextUtilitiesGuardsType.is_context(value)
            or hasattr(value, "__dict__")
            or bool(hasattr(value, "bind") and hasattr(value, "info"))
        )

    @staticmethod
    def is_instance_of[T](
        value: FlextUtilitiesGuardsType._GuardInput, type_cls: type[T]
    ) -> TypeGuard[T]:
        return isinstance(value, getattr(type_cls, "__origin__", None) or type_cls)

    @staticmethod
    def is_pydantic_model(value: t.NormalizedValue) -> TypeGuard[p.HasModelDump]:
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        if value is None:
            msg = f"{name} is not initialized"
            raise AttributeError(msg)
        return value

    @staticmethod
    def _run_string_type_check(type_name: str, value: t.NormalizedValue) -> bool:
        match type_name:
            case "str":
                return bool(isinstance(value, str))
            case "dict":
                return bool(isinstance(value, dict))
            case "list":
                return bool(FlextUtilitiesGuardsType.is_list(value))
            case "tuple":
                return bool(isinstance(value, tuple))
            case "sequence":
                return bool(isinstance(value, (list, tuple, range)))
            case "mapping":
                return bool(isinstance(value, Mapping))
            case "list_or_tuple":
                return bool(isinstance(value, (list, tuple)))
            case "sequence_not_str":
                return bool(
                    isinstance(value, (list, tuple, range))
                    and (not isinstance(value, str))
                )
            case "sequence_not_str_bytes":
                return bool(
                    isinstance(value, (list, tuple, range))
                    and (not isinstance(value, (str, bytes)))
                )
            case "sized":
                return bool(hasattr(value, "__len__"))
            case "callable":
                return bool(callable(value))
            case "bytes":
                return bool(isinstance(value, bytes))
            case "int":
                return bool(isinstance(value, int))
            case "float":
                return bool(isinstance(value, float))
            case "bool":
                return bool(isinstance(value, bool))
            case "none":
                return bool(value is None)
            case "string_non_empty":
                return bool(FlextUtilitiesGuardsType.is_string_non_empty(value))
            case "dict_non_empty":
                return bool(FlextUtilitiesGuardsType.is_dict_non_empty(value))
            case "list_non_empty":
                return bool(FlextUtilitiesGuardsType.is_list_non_empty(value))
            case _:
                return False

    @staticmethod
    def _check_protocol(value: t.NormalizedValue, name: str) -> bool:
        if name == "context":
            return FlextUtilitiesGuardsType.is_context(value)
        try:
            return FlextUtilitiesGuardsType._PROTOCOL_SPECS[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _is_type_tuple(
        value: FlextUtilitiesGuardsType._GuardInput,
    ) -> TypeGuard[tuple[type, ...]]:
        return FlextUtilitiesGuardsType.is_object_tuple(value) and all(
            isinstance(item, type) for item in value
        )

    @staticmethod
    def has(obj: t.NormalizedValue, key: str) -> bool:
        return key in obj if isinstance(obj, dict) else hasattr(obj, key)

    @staticmethod
    def in_(value: t.NormalizedValue, container: t.NormalizedValue) -> bool:
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False

    @staticmethod
    def is_type(
        value: t.NormalizedValue, type_spec: str | type | tuple[type, ...]
    ) -> bool:
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            if type_name in FlextUtilitiesGuardsType._PROTOCOL_SPECS:
                return FlextUtilitiesGuardsType._check_protocol(value, type_name)
            if type_name in FlextUtilitiesGuardsType._STRING_METHOD_MAP:
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    return (
                        FlextUtilitiesGuardsType._run_string_type_check(
                            type_name, value
                        )
                        if FlextUtilitiesGuardsType.is_container(value)
                        else False
                    )
                return FlextUtilitiesGuardsType._run_string_type_check(type_name, value)
            return False
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)
        if type_spec in FlextUtilitiesGuardsType._PROTOCOL_TYPE_MAP:
            return FlextUtilitiesGuardsType._check_protocol(
                value, FlextUtilitiesGuardsType._PROTOCOL_TYPE_MAP[type_spec]
            )
        try:
            return isinstance(
                value, getattr(type_spec, "__origin__", None) or type_spec
            )
        except TypeError:
            return False

    @staticmethod
    def none_(*values: t.NormalizedValue) -> bool:
        return all(v is None for v in values)


__all__ = ["FlextUtilitiesGuardsType"]
