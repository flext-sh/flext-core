from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard

from pydantic import BaseModel

from flext_core import c, p, t
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore


class FlextUtilitiesGuardsTypeProtocol:
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

    @staticmethod
    def is_context(
        value: object,
    ) -> TypeGuard[p.Context]:
        return isinstance(value, p.Context)

    @staticmethod
    def is_handler_callable(
        value: t.NormalizedValue,
    ) -> TypeGuard[t.HandlerCallable]:
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
    def is_registerable(
        value: t.NormalizedValue,
    ) -> TypeGuard[t.RegisterableService]:
        return FlextUtilitiesGuardsTypeProtocol.is_registerable_service(value)

    @staticmethod
    def is_factory(
        value: object,
    ) -> TypeGuard[t.FactoryCallable]:
        return callable(value)

    @staticmethod
    def is_resource(
        value: object,
    ) -> TypeGuard[t.ResourceCallable]:
        return callable(value)

    @staticmethod
    def is_result_like(
        value: object,
    ) -> TypeGuard[p.ResultLike[t.Container | BaseModel]]:
        return isinstance(value, p.ResultLike)

    @staticmethod
    def is_registerable_service(
        value: object,
    ) -> TypeGuard[t.RegisterableService]:
        return (
            value is None
            or isinstance(value, (str, int, float, bool, BaseModel, Path, Mapping))
            or callable(value)
            or (
                isinstance(value, Sequence)
                and (not isinstance(value, (str, bytes, bytearray)))
            )
            or FlextUtilitiesGuardsTypeProtocol.is_context(value)
            or hasattr(value, "__dict__")
            or bool(hasattr(value, "bind") and hasattr(value, "info"))
        )

    @staticmethod
    def _run_string_type_check(type_name: str, value: t.NormalizedValue) -> bool:
        match type_name:
            case "str":
                return bool(isinstance(value, str))
            case "dict":
                return bool(isinstance(value, dict))
            case "list":
                return bool(FlextUtilitiesGuardsTypeCore.is_list(value))
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
                return bool(FlextUtilitiesGuardsTypeCore.is_string_non_empty(value))
            case "dict_non_empty":
                return bool(FlextUtilitiesGuardsTypeCore.is_dict_non_empty(value))
            case "list_non_empty":
                return bool(FlextUtilitiesGuardsTypeCore.is_list_non_empty(value))
            case _:
                return False

    @staticmethod
    def _check_protocol(value: t.NormalizedValue, name: str) -> bool:
        if name == "context":
            return FlextUtilitiesGuardsTypeProtocol.is_context(value)
        try:
            return FlextUtilitiesGuardsTypeProtocol._PROTOCOL_SPECS[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _is_type_tuple(
        value: t.GuardInput,
    ) -> TypeGuard[tuple[type, ...]]:
        return isinstance(value, tuple)

    @staticmethod
    def is_type_tuple(
        value: t.GuardInput,
    ) -> TypeGuard[tuple[type, ...]]:
        return FlextUtilitiesGuardsTypeProtocol._is_type_tuple(value)

    @staticmethod
    def is_type(
        value: t.NormalizedValue,
        type_spec: str | type | tuple[type, ...],
    ) -> bool:
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            if type_name in FlextUtilitiesGuardsTypeProtocol._PROTOCOL_SPECS:
                return FlextUtilitiesGuardsTypeProtocol._check_protocol(
                    value, type_name
                )
            if type_name in c.Guards.STRING_METHOD_MAP:
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    return (
                        FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                            type_name, value
                        )
                        if FlextUtilitiesGuardsTypeCore.is_container(value)
                        else False
                    )
                return FlextUtilitiesGuardsTypeProtocol._run_string_type_check(
                    type_name, value
                )
            return False
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)
        if type_spec in FlextUtilitiesGuardsTypeProtocol._PROTOCOL_TYPE_MAP:
            return FlextUtilitiesGuardsTypeProtocol._check_protocol(
                value, FlextUtilitiesGuardsTypeProtocol._PROTOCOL_TYPE_MAP[type_spec]
            )
        try:
            return isinstance(
                value, getattr(type_spec, "__origin__", None) or type_spec
            )
        except TypeError:
            return False


__all__ = ["FlextUtilitiesGuardsTypeProtocol"]
