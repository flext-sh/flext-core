"""Runtime type guard helpers for dispatcher-safe validations.

The utilities provide runtime type checking functions that use structural typing
to keep handler and service checks lightweight while staying compatible with
duck-typed inputs used throughout the CQRS pipeline.

TypeGuard functions enable type narrowing without  - the preferred pattern
for FLEXT codebase to achieve zero-tolerance typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard, TypeIs

from pydantic import BaseModel, ValidationError

from flext_core import m, p, r, t


class FlextUtilitiesGuards:
    """Runtime type checking utilities for FLEXT ecosystem.

    Provides type guard functions for common validation patterns used throughout
    the FLEXT framework, implementing structural typing for duck-typed interfaces.

    Core Features:
    - String validation guards (non-empty, etc.)
    - Collection validation guards (dict, list)
    - Type-safe runtime checking
    - Consistent error handling patterns
    - Metadata value normalization
    """

    @staticmethod
    def _is_bool(value: t.NormalizedValue) -> TypeGuard[bool]:
        """Check if value is bool."""
        return isinstance(value, bool)

    @staticmethod
    def _is_bytes(value: t.NormalizedValue) -> TypeGuard[bytes]:
        """Check if value is bytes."""
        return isinstance(value, bytes)

    @staticmethod
    def _is_callable_key_func(
        func: t.NormalizedValue,
    ) -> TypeGuard[Callable[[t.NormalizedValue], t.NormalizedValue]]:
        """Check if value is callable and can be used as key function for sorted()."""
        return callable(func)

    @staticmethod
    def _is_dict(value: t.NormalizedValue) -> TypeGuard[m.Dict]:
        """Check if value is a dict-like mapping."""
        return isinstance(value, dict)

    @staticmethod
    def _is_float(value: t.NormalizedValue) -> TypeGuard[float]:
        """Check if value is float."""
        return isinstance(value, float)

    @staticmethod
    def _is_int(value: t.NormalizedValue) -> TypeGuard[int]:
        """Check if value is int."""
        return isinstance(value, int)

    type _GuardInput = (
        t.RegisterableService
        | t.NormalizedValue
        | p.Context
        | p.ValidatorSpec
        | tuple[type, ...]
        | None
    )

    @staticmethod
    def _is_list_or_tuple(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[list[t.NormalizedValue] | tuple[t.NormalizedValue, ...]]:
        """Check if value is list or tuple."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_mapping(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[Mapping[str, t.NormalizedValue]]:
        return isinstance(value, Mapping)

    @staticmethod
    def is_object_list(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[list[t.NormalizedValue]]:
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[tuple[t.NormalizedValue, ...]]:
        return isinstance(value, tuple)

    @staticmethod
    def _is_none(value: t.NormalizedValue) -> TypeGuard[None]:
        """Check if value is None."""
        return value is None

    @staticmethod
    def _is_sequence(
        value: t.NormalizedValue,
    ) -> TypeGuard[Sequence[t.NormalizedValue]]:
        """Check if value is Sequence of NormalizedValue."""
        return isinstance(value, (list, tuple, range))

    @staticmethod
    def _is_sequence_not_str(
        value: t.NormalizedValue,
    ) -> TypeGuard[Sequence[t.NormalizedValue]]:
        """Check if value is Sequence and not str."""
        return isinstance(value, (list, tuple, range)) and (not isinstance(value, str))

    @staticmethod
    def _is_sequence_not_str_bytes(
        value: t.NormalizedValue,
    ) -> TypeGuard[Sequence[t.NormalizedValue]]:
        """Check if value is Sequence and not str or bytes."""
        return isinstance(value, (list, tuple, range)) and (
            not isinstance(value, (str, bytes))
        )

    @staticmethod
    def _is_sized(value: t.NormalizedValue) -> TypeGuard[Sized]:
        """Check if value has __len__ (str, bytes, Sequence, Mapping)."""
        if isinstance(value, (str, bytes, list, tuple, dict)):
            return True
        return hasattr(value, "__len__") and callable(getattr(value, "__len__", None))

    @staticmethod
    def _is_str(value: t.NormalizedValue) -> TypeGuard[str]:
        """Check if value is str."""
        return isinstance(value, str)

    @staticmethod
    def _is_tuple(value: t.NormalizedValue) -> TypeGuard[tuple[t.NormalizedValue, ...]]:
        """Check if value is tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def is_config_value(value: t.NormalizedValue) -> TypeGuard[t.NormalizedValue]:
        """Check if value is a valid config value.

        ConfigValue = str | int | float | bool | datetime | None |
                      Sequence[scalar] | Mapping[str, scalar]

        This TypeGuard enables type narrowing for simple config values.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.NormalizedValue]: True if value matches config value type

        """
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool, datetime)):
            return True
        if FlextUtilitiesGuards._is_list_or_tuple(value):
            sequence_value: Sequence[t.NormalizedValue] = value
            for item in sequence_value:
                if not (
                    item is None or isinstance(item, (str, int, float, bool, datetime))
                ):
                    return False
            return True
        if FlextUtilitiesGuards._is_mapping(value):
            for v in value.values():
                if not (v is None or isinstance(v, (str, int, float, bool, datetime))):
                    return False
            return True
        return False

    @staticmethod
    def is_configuration_dict(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[m.Dict]:
        """Check if value is a valid m.Dict mapping.

        This TypeGuard enables explicit narrowing for m.Dict values.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[m.Dict]: True if value matches ConfigurationDict structure

        Example:
            >>> from flext_core import u
            >>> if u.is_configuration_dict(config):
            ...     # config is now typed as m.Dict
            ...     config["key"] = "value"

        """
        if isinstance(value, m.Dict):
            candidate: Mapping[str, t.NormalizedValue | BaseModel] = value.root
            for item_key, item_value in candidate.items():
                if not isinstance(item_key, str):
                    return False
                if not FlextUtilitiesGuards.is_container(item_value):
                    return False
            return True
        if not FlextUtilitiesGuards._is_mapping(value):
            return False
        for mk, mv in value.items():
            if not isinstance(mk, str):
                return False
            if not FlextUtilitiesGuards.is_container(mv):
                return False
        return True

    @staticmethod
    def is_configuration_mapping(
        value: Mapping[str, t.NormalizedValue] | m.ConfigMap | m.Dict,
    ) -> TypeGuard[m.ConfigMap]:
        """Check if value is a valid m.ConfigMap.

        m.ConfigMap = Mapping[str, t.NormalizedValue | BaseModel]

        This TypeGuard enables explicit narrowing for m.ConfigMap.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[m.ConfigMap]: True if value matches ConfigurationMapping structure

        Example:
            >>> from flext_core import u
            >>> if u.is_configuration_mapping(config):
            ...     # config is now typed as m.ConfigMap
            ...     items = config.items()

        """
        if isinstance(value, (m.ConfigMap, m.Dict)):
            candidate: Mapping[str, t.NormalizedValue | BaseModel] = value.root
        elif isinstance(value, Mapping):
            candidate = value
        else:
            return False
        for item_key, item_value in candidate.items():
            if not isinstance(item_key, str):
                return False
            if not FlextUtilitiesGuards.is_container(item_value):
                return False
        return True

    @staticmethod
    def is_context(value: FlextUtilitiesGuards._GuardInput) -> TypeGuard[p.Context]:
        """Check if *value* satisfies ``p.Context`` structurally."""
        return bool(
            hasattr(value, "get") and hasattr(value, "set") and hasattr(value, "clone")
        )

    @staticmethod
    def is_dict_non_empty(value: t.NormalizedValue) -> bool:
        """Check if value is a non-empty dictionary using duck typing."""
        return FlextUtilitiesGuards._is_mapping(value) and bool(value)

    @staticmethod
    def is_flexible_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        if value is None or FlextUtilitiesGuards.is_scalar(value):
            return True
        if FlextUtilitiesGuards._is_list_or_tuple(value):
            sequence_value: Sequence[t.NormalizedValue] = value
            for item in sequence_value:
                if item is not None and (not FlextUtilitiesGuards.is_scalar(item)):
                    return False
            return True
        if FlextUtilitiesGuards._is_mapping(value):
            for item in value.values():
                if item is not None and (not FlextUtilitiesGuards.is_scalar(item)):
                    return False
            return True
        return False

    @staticmethod
    def is_container(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[str | int | float | bool | datetime | Path]:
        """Check if value is a valid Container type.

        Container = Scalar | Path
        Scalar = str | int | float | bool | datetime

        This TypeGuard enables type narrowing to t.Container.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard narrowing to Container union members

        """
        if value is None or isinstance(value, (str, int, float, bool, datetime)):
            return True
        if value is True or value is False:
            return True
        if FlextUtilitiesGuards._is_list_or_tuple(value):
            return all(FlextUtilitiesGuards.is_container(item) for item in value)
        if FlextUtilitiesGuards._is_mapping(value):
            return all(FlextUtilitiesGuards.is_container(v) for v in value.values())
        return isinstance(value, Path)

    @staticmethod
    def is_general_value_type(value: t.NormalizedValue) -> bool:
        """Deprecated alias; use is_container."""
        warnings.warn(
            "is_general_value_type is deprecated; use is_container. "
            "Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return callable(value) or FlextUtilitiesGuards.is_container(value)

    @staticmethod
    def is_handler_callable(
        value: t.NormalizedValue,
    ) -> TypeGuard[t.HandlerCallable]:
        """Check if value is a valid t.HandlerCallable."""
        return callable(value)

    @staticmethod
    def is_handler_type(
        value: t.NormalizedValue | t.HandlerCallable,
    ) -> TypeGuard[t.HandlerLike]:
        """Check if value is a valid t.HandlerLike.

        t.HandlerLike = Callable[..., BaseModel]

        This TypeGuard enables type narrowing for t.HandlerLike.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.HandlerLike]: True if value matches t.HandlerLike structure

        Example:
            >>> from flext_core import u
            >>> if u.is_handler_type(handler):
            ...     # handler is now typed as t.HandlerLike
            ...     result = container.register("my_handler", handler)

        """
        if callable(value):
            return True
        if isinstance(value, Mapping):
            return True
        if (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        ):
            return True
        return hasattr(value, "handle") or hasattr(value, "can_handle")

    @staticmethod
    def is_list(value: t.NormalizedValue) -> TypeGuard[list[t.NormalizedValue]]:
        """Check if value is a list (type guard)."""
        return isinstance(value, list)

    @staticmethod
    def is_list_non_empty(value: t.NormalizedValue) -> bool:
        """Check if value is a non-empty list using duck typing."""
        if not isinstance(value, Sequence):
            return False
        if isinstance(value, (str, bytes)):
            return False
        sequence_value: Sequence[t.NormalizedValue] = value
        return len(sequence_value) > 0

    @staticmethod
    def is_mapping(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[Mapping[str, t.NormalizedValue]]:
        """Check if *value* is a mapping instance."""
        return isinstance(value, Mapping)

    @staticmethod
    def is_registerable(value: t.NormalizedValue) -> bool:
        """Check if *value* can be registered in FlextContainer."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        if isinstance(value, (BaseModel, Path)):
            return True
        if callable(value):
            return True
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return True
        if FlextUtilitiesGuards.is_context(value):
            return True
        if hasattr(value, "__dict__"):
            return True
        return bool(hasattr(value, "bind") and hasattr(value, "info"))

    @staticmethod
    def is_factory(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[t.FactoryCallable]:
        """Check if *value* is a factory callable."""
        return callable(value)

    @staticmethod
    def is_resource(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[t.ResourceCallable]:
        """Check if *value* is a resource callable."""
        return callable(value)

    @staticmethod
    def is_primitive(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[t.Primitives]:
        """Check if value is a primitive type (str, int, float, bool)."""
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def is_result_like(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[p.ResultLike[t.Container | BaseModel]]:
        """Check if value implements ResultLike protocol (has is_success, value, error).

        Checks the class MRO rather than instance attribute access to avoid
        triggering property getters that raise RuntimeError (e.g., r.value on failure).
        """
        if not (hasattr(value, "is_success") and hasattr(value, "error")):
            return False
        return hasattr(type(value), "value")

    @staticmethod
    def is_scalar(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[t.Scalar]:
        """Check if value is a scalar type (str, int, float, bool, datetime)."""
        return isinstance(value, (str, int, float, bool, datetime))

    @staticmethod
    def is_string_non_empty(value: t.NormalizedValue) -> TypeGuard[str]:
        """Check if value is a non-empty string using duck typing."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_registerable_service(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[t.RegisterableService]:
        """Check if value is a registerable service for DI container.

        Matches logic from FlextContainer._is_registerable_service using structural typing.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.RegisterableService]: True if value can be registered in the container

        """
        # scalars/none
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        # models/paths
        if isinstance(value, (BaseModel, Path)):
            return True
        # callables
        if callable(value):
            return True
        # mappings/sequences
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and (
            not isinstance(value, (str, bytes, bytearray))
        ):
            return True
        # protocols via duck typing
        if FlextUtilitiesGuards.is_context(value):
            return True
        # generic objects with dict
        if hasattr(value, "__dict__"):
            return True
        # fallback for logger-like objects (bind/info)
        return bool(hasattr(value, "bind") and hasattr(value, "info"))

    @staticmethod
    def is_instance_of[T](
        value: FlextUtilitiesGuards._GuardInput, type_cls: type[T]
    ) -> TypeGuard[T]:
        """Check if value is an instance of type_cls, handling generics.

        Args:
            value: Object to check
            type_cls: Target type/class

        Returns:
            TypeGuard[T]: True if value is an instance of type_cls

        """
        origin = getattr(type_cls, "__origin__", None)
        check_type: type = origin if origin is not None else type_cls
        return isinstance(value, check_type)

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        """Guard that a service attribute was initialized.

        Args:
            value: The potentially uninitialized attribute.
            name: Human-readable name for the error message.

        Returns:
            The value if it is not None.

        Raises:
            AttributeError: If the value is None (uninitialized).

        """
        if value is None:
            msg = f"{name} is not initialized"
            raise AttributeError(msg)
        return value

    _PROTOCOL_SPECS: Mapping[str, Callable[[t.NormalizedValue], bool]] = (
        MappingProxyType({
            "config": lambda v: (
                hasattr(v, "app_name") and getattr(v, "app_name", None) is not None
            ),
            "context": lambda v: (
                hasattr(v, "request_id") or hasattr(v, "correlation_id")
            ),
            "container": lambda v: (
                hasattr(v, "register") and callable(getattr(v, "register", None))
            ),
            "command_bus": lambda v: (
                hasattr(v, "dispatch") and callable(getattr(v, "dispatch", None))
            ),
            "handler": lambda v: (
                hasattr(v, "handle") and callable(getattr(v, "handle", None))
            ),
            "logger": lambda v: all(
                hasattr(v, a)
                for a in ("debug", "info", "warning", "error", "exception")
            ),
            "result": lambda v: all(
                hasattr(v, a) for a in ("is_success", "is_failure", "value", "error")
            ),
            "service": lambda v: (
                hasattr(v, "run") and callable(getattr(v, "run", None))
            ),
            "middleware": lambda v: (
                hasattr(v, "before_dispatch")
                and callable(getattr(v, "before_dispatch", None))
            ),
        })
    )
    _PROTOCOL_TYPE_MAP: Mapping[type, str] = MappingProxyType({
        p.Config: "config",
        p.Context: "context",
        p.DI: "container",
        p.CommandBus: "command_bus",
        p.Handler: "handler",
        p.Log.StructlogLogger: "logger",
        p.Result: "result",
        p.Service: "service",
        p.Middleware: "middleware",
    })
    _STRING_METHOD_MAP: Mapping[str, str] = MappingProxyType({
        "str": "_is_str",
        "dict": "_is_dict",
        "list": "is_list",
        "tuple": "_is_tuple",
        "sequence": "_is_sequence",
        "mapping": "_is_mapping",
        "list_or_tuple": "_is_list_or_tuple",
        "sequence_not_str": "_is_sequence_not_str",
        "sequence_not_str_bytes": "_is_sequence_not_str_bytes",
        "sized": "_is_sized",
        "callable": "_is_callable_key_func",
        "bytes": "_is_bytes",
        "int": "_is_int",
        "float": "_is_float",
        "bool": "_is_bool",
        "none": "_is_none",
        "string_non_empty": "is_string_non_empty",
        "dict_non_empty": "is_dict_non_empty",
        "list_non_empty": "is_list_non_empty",
    })

    @staticmethod
    def _check_protocol(value: t.NormalizedValue, name: str) -> bool:
        """Check protocol via _PROTOCOL_SPECS mapping."""
        if name == "context":
            return FlextUtilitiesGuards.is_context(value)
        try:
            return FlextUtilitiesGuards._PROTOCOL_SPECS[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _ensure_to_dict(
        value: t.NormalizedValue | None,
        default: Mapping[str, t.NormalizedValue] | None,
    ) -> Mapping[str, t.NormalizedValue]:
        if value is None:
            return default if default is not None else {}
        if isinstance(value, Mapping):
            mapping_value: Mapping[str, t.NormalizedValue] = value
            normalized: dict[str, t.NormalizedValue] = {}
            for key, item_value in mapping_value.items():
                normalized[str(key)] = item_value
            return normalized
        wrapped_dict: Mapping[str, t.NormalizedValue] = {"value": value}
        return wrapped_dict

    @staticmethod
    def _ensure_to_list(
        value: t.NormalizedValue | list[t.NormalizedValue] | None,
        default: list[t.NormalizedValue] | None,
    ) -> list[t.NormalizedValue]:
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            return list(value)
        single_item_list: list[t.NormalizedValue] = [value]
        return single_item_list

    @staticmethod
    def _guard_check_condition[T: FlextUtilitiesGuards._GuardInput](
        value: T,
        condition: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
        | str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if isinstance(condition, type):
            if FlextUtilitiesGuards.is_container(value):
                return FlextUtilitiesGuards._guard_check_type(
                    value, condition, context_name, error_msg
                )
            return (
                error_msg
                or f"{context_name} must be {condition.__name__}, got {type(value).__name__}"
            )
        if FlextUtilitiesGuards._is_type_tuple(condition):
            if FlextUtilitiesGuards.is_container(value):
                return FlextUtilitiesGuards._guard_check_type(
                    value, condition, context_name, error_msg
                )
            return (
                error_msg
                or f"{context_name} type check failed for {type(value).__name__}"
            )
        if isinstance(condition, p.ValidatorSpec):
            if not FlextUtilitiesGuards.is_container(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value: t.NormalizedValue = value
            return FlextUtilitiesGuards._guard_check_validator(
                typed_value, condition, context_name, error_msg
            )
        if isinstance(condition, str):
            if not FlextUtilitiesGuards.is_container(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value_s: t.NormalizedValue = value
            return FlextUtilitiesGuards._guard_check_string_shortcut(
                typed_value_s, condition, context_name, error_msg
            )
        if callable(condition):
            return FlextUtilitiesGuards._guard_check_predicate(
                value, condition, context_name, error_msg
            )
        return error_msg or f"{context_name} invalid guard condition type"

    @staticmethod
    def _guard_check_predicate(
        value: FlextUtilitiesGuards._GuardInput,
        condition: Callable[..., FlextUtilitiesGuards._GuardInput | bool],
        context_name: str,
        error_msg: str | None,
    ) -> str:
        try:
            if not bool(condition(value)):
                if error_msg is None:
                    func_name = (
                        condition.__name__
                        if hasattr(condition, "__name__")
                        else "custom"
                    )
                    return f"{context_name} failed {func_name} check"
                return error_msg
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            if error_msg is None:
                return f"{context_name} guard check raised: {e}"
            return error_msg
        return ""

    @staticmethod
    def _guard_check_string_shortcut(
        value: t.NormalizedValue,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        shortcut_lower = condition.lower()
        if shortcut_lower == "non_empty":
            if isinstance(value, str) and bool(value):
                return ""
            if isinstance(value, list) and len(value) > 0:
                return ""
            if isinstance(value, dict) and len(value) > 0:
                return ""
            return error_msg or f"{context_name} must be non-empty"
        if shortcut_lower == "positive":
            if (
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and (value > 0)
            ):
                return ""
            return error_msg or f"{context_name} must be positive number"
        if shortcut_lower == "non_negative":
            if (
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and (value >= 0)
            ):
                return ""
            return error_msg or f"{context_name} must be non-negative number"
        if shortcut_lower == "dict":
            if hasattr(value, "items") and (not isinstance(value, (str, bytes))):
                return ""
            return error_msg or f"{context_name} must be dict-like"
        if shortcut_lower == "list":
            if (
                hasattr(value, "__iter__")
                and (not isinstance(value, (str, bytes)))
                and (not hasattr(value, "items"))
            ):
                return ""
            return error_msg or f"{context_name} must be list-like"
        if shortcut_lower == "string":
            if isinstance(value, str):
                return ""
            return error_msg or f"{context_name} must be string"
        if shortcut_lower == "int":
            if isinstance(value, int) and (not isinstance(value, bool)):
                return ""
            return error_msg or f"{context_name} must be int"
        if shortcut_lower == "float":
            if isinstance(value, int | float) and (not isinstance(value, bool)):
                return ""
            return error_msg or f"{context_name} must be float"
        if shortcut_lower == "bool":
            if isinstance(value, bool):
                return ""
            return error_msg or f"{context_name} must be bool"
        return error_msg or f"{context_name} unknown guard shortcut: {condition}"

    @staticmethod
    def _guard_check_type(
        value: t.NormalizedValue,
        condition: type | tuple[type, ...],
        context_name: str,
        error_msg: str | None,
    ) -> str:
        type_match = isinstance(value, condition)
        if not type_match:
            if error_msg is None:
                type_name = (
                    condition.__name__
                    if isinstance(condition, type)
                    else " | ".join(c.__name__ for c in condition)
                )
                return f"{context_name} must be {type_name}, got {value.__class__.__name__}"
            return error_msg
        return ""

    @staticmethod
    def _guard_check_validator(
        value: t.NormalizedValue,
        condition: p.ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if not FlextUtilitiesGuards.is_container(value):
            return error_msg or f"{context_name} must be a valid configuration value"
        if not condition(value):
            if error_msg is None:
                desc = (
                    getattr(condition, "description", "validation")
                    if hasattr(condition, "description")
                    else "validation"
                )
                return f"{context_name} failed {desc} check"
            return error_msg
        return ""

    @staticmethod
    def _guard_handle_failure[T](
        error_message: str, *, return_value: bool, default: T | None
    ) -> r[T] | T:
        if return_value:
            if default is not None:
                return default
            return r[T].fail(error_message)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(error_message)

    @staticmethod
    def _is_type_tuple(
        value: FlextUtilitiesGuards._GuardInput,
    ) -> TypeGuard[tuple[type, ...]]:
        if not FlextUtilitiesGuards.is_object_tuple(value):
            return False
        return all(isinstance(item, type) for item in value)

    @staticmethod
    def chk(
        value: t.NormalizedValue,
        *,
        eq: t.NormalizedValue | None = None,
        ne: t.NormalizedValue | None = None,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
        is_: type | None = None,
        not_: type | None = None,
        in_: Sequence[t.NormalizedValue] | None = None,
        not_in: Sequence[t.NormalizedValue] | None = None,
        none: bool | None = None,
        empty: bool | None = None,
        match: str | None = None,
        contains: t.NormalizedValue | None = None,
        starts: str | None = None,
        ends: str | None = None,
    ) -> bool:
        """Universal check - single method for ALL validation scenarios.

        Args:
            value: Value to check
            eq: Check value == eq
            ne: Check value != ne
            gt/gte/lt/lte: Numeric comparisons (works with len for sequences)
            is_: Check isinstance(value, is_)
            not_: Check not isinstance(value, not_)
            in_: Check value in in_
            not_in: Check value not in not_in
            none: Check value is None (True) or is not None (False)
            empty: Check if empty (True) or not empty (False)
            match: Check regex pattern match (strings)
            contains: Check if value contains item
            starts/ends: Check string prefix/suffix

        Returns:
            True if ALL conditions pass, False otherwise.

        Examples:
            u.chk(x, gt=0, lt=100)             # 0 < x < 100
            u.chk(s, empty=False, match="[0-9]+")  # non-empty and has digits
            u.chk(lst, gte=1, lte=10)          # 1 <= len(lst) <= 10
            u.chk(v, is_=str, none=False)      # is string and not None

        """
        if none is True and value is not None:
            return False
        if none is False and value is None:
            return False
        if is_ is not None and (not isinstance(value, is_)):
            return False
        if not_ is not None and isinstance(value, not_):
            return False
        if eq is not None and value != eq:
            return False
        if ne is not None and value == ne:
            return False
        if in_ is not None and value not in in_:
            return False
        if not_in is not None and value in not_in:
            return False
        check_val: int | float = 0
        if isinstance(value, (int, float)):
            check_val = value
        elif isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            sized_value: Sized = value
            check_val = len(sized_value)
        elif hasattr(value, "__len__"):
            try:
                len_method = getattr(value, "__len__", None)
                if callable(len_method):
                    length = len_method()
                    if isinstance(length, int):
                        check_val = length
            except (TypeError, AttributeError):
                check_val = 0
        if gt is not None and check_val <= gt:
            return False
        if gte is not None and check_val < gte:
            return False
        if lt is not None and check_val >= lt:
            return False
        if lte is not None and check_val > lte:
            return False
        if empty is True and check_val != 0:
            return False
        if empty is False and check_val == 0:
            return False
        if isinstance(value, str):
            if match is not None and (not re.search(match, value)):
                return False
            if starts is not None and (not value.startswith(starts)):
                return False
            if ends is not None and (not value.endswith(ends)):
                return False
            if (
                contains is not None
                and isinstance(contains, str)
                and (contains not in value)
            ):
                return False
        elif contains is not None:
            if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
                iterable_value: Iterable[t.NormalizedValue] = value
                found = False
                for item in iterable_value:
                    item_value: t.NormalizedValue = item
                    if item_value == contains:
                        found = True
                        break
                if not found:
                    return False
            else:
                return False
        return True

    @staticmethod
    def empty(items: t.NormalizedValue | None) -> bool:
        """Check if items is empty or None.

        Args:
            items: Value to check (None, Sized, or other value)

        Returns:
            True if items is None, empty, or falsy

        """
        if items is None:
            return True
        if FlextUtilitiesGuards._is_sized(items):
            return len(items) == 0
        return not bool(items)

    @staticmethod
    def ensure(
        value: t.NormalizedValue,
        *,
        target_type: str = "auto",
        default: str | list[t.NormalizedValue] | t.NormalizedValue | None = None,
    ) -> (
        str
        | list[t.NormalizedValue]
        | t.NormalizedValue
        | Mapping[str, t.NormalizedValue]
    ):
        if target_type == "str":
            str_default = default if isinstance(default, str) else ""
            return (
                value
                if isinstance(value, str)
                else str(value)
                if value is not None
                else str_default
            )
        if target_type == "str_list":
            str_list_default: list[str] | None = None
            if isinstance(default, list):
                default_values: list[t.NormalizedValue] = default
                str_list_default = [str(item) for item in default_values]
            if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                seq_value: Sequence[t.NormalizedValue] = value
                return list(seq_value)
            if value is None:
                result_str_list: list[t.NormalizedValue] = (
                    list(str_list_default) if str_list_default else []
                )
                return result_str_list
            return [value]
        if target_type == "dict":
            dict_default: Mapping[str, t.NormalizedValue] | None = None
            if FlextUtilitiesGuards._is_mapping(default):
                dict_default = default
            return FlextUtilitiesGuards._ensure_to_dict(value, dict_default)
        if target_type == "auto" and isinstance(value, Mapping):
            mapping_value: Mapping[str, t.NormalizedValue] = value
            normalized_auto: dict[str, t.NormalizedValue] = {}
            for key, item_value in mapping_value.items():
                normalized_auto[str(key)] = item_value
            return normalized_auto
        list_default: list[t.NormalizedValue] | None = None
        if FlextUtilitiesGuards.is_object_list(default):
            list_default = default
        return FlextUtilitiesGuards._ensure_to_list(value, list_default)

    @staticmethod
    def extract_mapping_or_none(value: t.NormalizedValue) -> r[m.ConfigMap]:
        """Extract a mapping from a value or return None.

        Used for type narrowing when a generic parameter could be a Mapping
        or another type. Returns the value as ConfigurationMapping if it's
        a Mapping, otherwise returns None.

        Args:
            value: Value that may or may not be a Mapping

        Returns:
            r[m.ConfigMap] containing mapping on success, failure otherwise

        """
        if FlextUtilitiesGuards._is_mapping(
            value
        ) and FlextUtilitiesGuards.is_configuration_mapping(value):
            return r[m.ConfigMap].ok(value)
        return r[m.ConfigMap].fail("Value is not a configuration mapping")

    @staticmethod
    def guard(
        value: t.NormalizedValue,
        validator: Callable[[t.NormalizedValue], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.NormalizedValue | None = None,
        return_value: bool = False,
    ) -> t.Container | bool | r[t.Container]:
        guarded_value: t.NormalizedValue = value
        try:
            if isinstance(validator, type):
                if isinstance(value, validator):
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuards.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif FlextUtilitiesGuards.is_object_tuple(validator):
                tuple_types = tuple(
                    item for item in validator if isinstance(item, type)
                )
                if len(tuple_types) == len(validator) and isinstance(
                    value, tuple_types
                ):
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuards.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif callable(validator):
                if validator(value):
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuards.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif value:
                if return_value:
                    return (
                        guarded_value
                        if FlextUtilitiesGuards.is_container(guarded_value)
                        else str(guarded_value)
                    )
                return True
            if default is not None:
                return (
                    default
                    if FlextUtilitiesGuards.is_container(default)
                    else str(default)
                )
            return (
                r[t.Container].fail("Guard validation failed")
                if return_value
                else False
            )
        except (TypeError, ValueError, AttributeError):
            if default is not None:
                return (
                    default
                    if FlextUtilitiesGuards.is_container(default)
                    else str(default)
                )
            return (
                r[t.Container].fail("Guard validation raised an exception")
                if return_value
                else False
            )

    @staticmethod
    def guard_result[T: FlextUtilitiesGuards._GuardInput](
        value: T,
        *conditions: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
        | str,
        error_message: str | None = None,
        context: str | None = None,
        default: T | None = None,
        return_value: bool = False,
    ) -> r[T] | T:
        context_name = context or "Value"
        if len(conditions) == 0:
            if bool(value):
                return value if return_value else r[T].ok(value)
            failure_message = error_message or f"{context_name} guard failed"
            return FlextUtilitiesGuards._guard_handle_failure(
                failure_message, return_value=return_value, default=default
            )
        for condition in conditions:
            condition_error = FlextUtilitiesGuards._guard_check_condition(
                value, condition, context_name, error_message
            )
            if condition_error:
                return FlextUtilitiesGuards._guard_handle_failure(
                    condition_error, return_value=return_value, default=default
                )
        return value if return_value else r[T].ok(value)

    @staticmethod
    def has(obj: t.NormalizedValue, key: str) -> bool:
        """Check if object has attribute/key."""
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    @staticmethod
    def in_(value: t.NormalizedValue, container: t.NormalizedValue) -> bool:
        """Check if value is in container."""
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False

    @staticmethod
    def is_pydantic_model(value: t.NormalizedValue) -> TypeGuard[p.HasModelDump]:
        """Type guard to check if value is a Pydantic model with model_dump method."""
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )

    @staticmethod
    def is_type(
        value: t.NormalizedValue, type_spec: str | type | tuple[type, ...]
    ) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. Uses centralized _PROTOCOL_SPECS mapping
        for protocol validation to eliminate repeated if/type_spec branches.

        Args:
            value: Object to check
            type_spec: Type specification as:
                - String name: "config", "str", "dict", "list", "sequence",
                  "mapping", "callable", "sized", "list_or_tuple", "sequence_not_str",
                  "string_non_empty", "dict_non_empty", "list_non_empty"
                - Type/class: str, dict, list, tuple, Sequence, Mapping, etc.
                - Protocol: p.Config, p.Context, etc.

        Returns:
            bool: True if value matches the type specification

        Examples:
            >>> from flext_core import u
            >>> # String-based checks
            >>> u.is_type(obj, "config")
            >>> u.is_type(obj, "str")
            >>> u.is_type(obj, "dict")
            >>> u.is_type(obj, "string_non_empty")

            >>> # Direct type checks
            >>> u.is_type(obj, str)
            >>> u.is_type(obj, dict)
            >>> u.is_type(obj, list)

            >>> # Tuple of types checks
            >>> u.is_type(obj, (int, float))
            >>> u.is_type(obj, (str, bytes))

            >>> # Protocol checks
            >>> u.is_type(obj, p.Config)
            >>> u.is_type(obj, p.Context)

        """
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            if type_name in FlextUtilitiesGuards._PROTOCOL_SPECS:
                return FlextUtilitiesGuards._check_protocol(value, type_name)
            if type_name in FlextUtilitiesGuards._STRING_METHOD_MAP:
                method_name = FlextUtilitiesGuards._STRING_METHOD_MAP[type_name]
                method = getattr(FlextUtilitiesGuards, method_name)
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    if FlextUtilitiesGuards.is_container(value):
                        return bool(method(value))
                    return False
                return bool(method(value))
            return False
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)
        # Check if type_spec is a protocol we have specialized handlers for
        if type_spec in FlextUtilitiesGuards._PROTOCOL_TYPE_MAP:
            protocol_name = FlextUtilitiesGuards._PROTOCOL_TYPE_MAP[type_spec]
            return FlextUtilitiesGuards._check_protocol(value, protocol_name)

        # Handle direct type check with support for generic origins
        check_type = getattr(type_spec, "__origin__", None) or type_spec
        try:
            return isinstance(value, check_type)
        except TypeError:
            # PEP 695 TypeAliasType cannot be used with isinstance()
            return False

    @staticmethod
    def none_(*values: t.NormalizedValue) -> bool:
        """Check if all values are None.

        Args:
            *values: Values to check

        Returns:
            True if all values are None, False otherwise

        Example:
            if u.none_(name, email):
                return r[str].fail("Name and email are required")

        """
        return all(v is None for v in values)

    @staticmethod
    def validate_hostname(hostname: str, field_name: str = "hostname") -> r[str]:
        """Return success when ``hostname`` is a valid hostname or FQDN."""
        hostname_pattern = (
            "^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\\.[a-zA-Z0-9-]{1,63}(?<!-))*$"
        )
        if re.search(hostname_pattern, hostname) is None:
            return r[str].fail(f"{field_name} has invalid hostname format")
        return r[str].ok(hostname)

    @staticmethod
    def validate_length[T: Sized](
        value: T,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        field_name: str = "value",
    ) -> r[T]:
        """Return success when ``value`` length is within bounds."""
        try:
            length = len(value)
        except (TypeError, ValueError):
            return r[T].fail(f"{field_name} length is invalid")
        if min_length is not None and length < min_length:
            return r[T].fail(
                f"{field_name} must have at least {min_length} characters/items"
            )
        if max_length is not None and length > max_length:
            return r[T].fail(
                f"{field_name} must have at most {max_length} characters/items"
            )
        return r[T].ok(value)

    @staticmethod
    def validate_pattern(value: str, pattern: str, field_name: str = "value") -> r[str]:
        """Return success when ``value`` matches ``pattern``."""
        if re.search(pattern, value) is None:
            return r[str].fail(f"{field_name} has invalid format")
        return r[str].ok(value)

    @staticmethod
    def validate_port_number(port: int, field_name: str = "port") -> r[int]:
        """Return success when ``port`` is a valid port number (1-65535)."""
        if isinstance(port, bool):
            return r[int].fail(f"{field_name} must be an integer")
        max_port = 65535
        if port < 1 or port > max_port:
            return r[int].fail(f"{field_name} must be between 1 and 65535")
        return r[int].ok(port)

    @staticmethod
    def validate_positive(value: float, field_name: str = "value") -> r[int | float]:
        """Return success when numeric ``value`` is greater than zero."""
        if isinstance(value, bool) or value <= 0:
            return r[int | float].fail(f"{field_name} must be positive")
        return r[int | float].ok(value)

    @staticmethod
    def validate_uri(uri: str, field_name: str = "uri") -> r[str]:
        """Return success when ``uri`` is a valid URI/URL format."""
        uri_pattern = "^[a-zA-Z][a-zA-Z0-9+.-]*://[^\\s]+$"
        if re.search(uri_pattern, uri) is None:
            return r[str].fail(f"{field_name} has invalid URI format")
        return r[str].ok(uri)

    @staticmethod
    def validate_pydantic_model[T: BaseModel](
        model_class: type[T], data: Mapping[str, t.NormalizedValue]
    ) -> r[T]:
        """Validate data using Pydantic v2 model and return r[T]."""
        try:
            validated = model_class.model_validate(data)
            return r[T].ok(validated)
        except (ValidationError, TypeError, ValueError) as exc:
            return r[T].fail(f"Validation failed: {exc}")


validate_pydantic_model = FlextUtilitiesGuards.validate_pydantic_model

__all__ = ["FlextUtilitiesGuards", "validate_pydantic_model"]
