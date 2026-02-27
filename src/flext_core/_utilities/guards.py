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
from collections.abc import Callable, Mapping, Sequence, Sized
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard, TypeIs

from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t


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

    @staticmethod
    def is_string_non_empty(value: t.GuardInputValue) -> TypeGuard[str]:
        """Check if value is a non-empty string using duck typing."""
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_dict_non_empty(value: t.GuardInputValue) -> bool:
        """Check if value is a non-empty dictionary using duck typing."""
        return isinstance(value, Mapping) and bool(value)

    @staticmethod
    def is_list_non_empty(value: t.GuardInputValue) -> bool:
        """Check if value is a non-empty list using duck typing."""
        return (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and bool(value)
        )

    @staticmethod
    def is_result_like(value: object) -> TypeGuard[p.ResultLike[t.ConfigMapValue]]:
        """Check if value implements ResultLike protocol (has is_success, value, error).

        Uses try/except to avoid triggering property getters that may raise
        (e.g., FlextResult.value on failure raises RuntimeError, not AttributeError).
        """
        try:
            return (
                hasattr(value, "is_success")
                and hasattr(value, "value")
                and hasattr(value, "error")
            )
        except (RuntimeError, TypeError, ValueError):
            cls = type(value)
            return (
                hasattr(cls, "is_success")
                and hasattr(cls, "value")
                and hasattr(cls, "error")
            )

    @staticmethod
    def is_flexible_value(value: t.FlexibleValue) -> TypeIs[t.FlexibleValue]:
        if value is None or isinstance(value, str | int | float | bool | datetime):
            return True
        if isinstance(value, list | tuple):
            for item in value:
                if item is not None and not isinstance(
                    item,
                    str | int | float | bool | datetime,
                ):
                    return False
            return True
        if isinstance(value, Mapping):
            for item in value.values():
                if item is not None and not isinstance(
                    item,
                    str | int | float | bool | datetime,
                ):
                    return False
            return True
        return False

    # =========================================================================
    # TypeGuard Functions for FLEXT Core Types
    # =========================================================================
    # These functions enable type narrowing without  - zero tolerance typing

    @staticmethod
    def is_general_value_type(value: object) -> TypeGuard[t.GuardInputValue]:
        """Check if value is a valid t.GuardInputValue.

        t.GuardInputValue = ScalarValue | Sequence[t.GuardInputValue] | Mapping[str, t.GuardInputValue]
        ScalarValue = str | int | float | bool | datetime | None

        This TypeGuard enables type narrowing for t.GuardInputValue.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.GuardInputValue]: True if value matches t.GuardInputValue structure

        """
        # Check scalar types first (most common case)
        if value is None or isinstance(value, (str, int, float, bool, datetime)):
            return True
        # Check for bool before int (bool is subclass of int in Python)
        if value is True or value is False:
            return True
        # Check sequence types (list/tuple can never be str/bytes)
        if isinstance(value, (list, tuple)):
            # Iterate with explicit type annotation to satisfy pyright
            item: t.GuardInputValue
            for item in value:
                if not FlextUtilitiesGuards.is_general_value_type(item):
                    return False
            return True
        # Check mapping types (structural)
        if isinstance(value, Mapping):
            # Iterate with explicit type annotations to satisfy pyright
            v: str | int | float | bool | None
            for v in value.values():
                if not FlextUtilitiesGuards.is_general_value_type(v):
                    return False
            return True
        # Check callable types
        if callable(value):
            return True
        # Check BaseModel or Path instances (structural for BaseModel)
        return hasattr(value, "model_dump") or isinstance(value, Path)

    @staticmethod
    def is_handler_type(
        value: t.GuardInputValue | t.HandlerCallable,
    ) -> TypeGuard[t.HandlerType]:
        """Check if value is a valid t.HandlerType.

        t.HandlerType = HandlerCallable | Mapping[str, t.ConfigMapValue] | BaseModel

        This TypeGuard enables type narrowing for t.HandlerType.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.HandlerType]: True if value matches t.HandlerType structure

        Example:
            >>> from flext_core.utilities import u
            >>> if u.Guards.is_handler_type(handler):
            ...     # handler is now typed as t.HandlerType
            ...     result = container.register("my_handler", handler)

        """
        # Check if callable (most common case - HandlerCallable)
        if callable(value):
            return True
        # Check if Mapping (handler mapping) - structural
        if isinstance(value, Mapping):
            return True
        # Check if BaseModel instance or class
        if isinstance(value, BaseModel) and hasattr(value, "model_dump") and callable(value.model_dump):
            return True
        # Check for handler protocol methods (duck typing)
        # All values are objects in Python, so type check (value, object) is always True
        return hasattr(value, "handle") or hasattr(value, "can_handle")

    @staticmethod
    def is_handler_callable(value: t.GuardInputValue) -> TypeGuard[t.HandlerCallable]:
        """Check if value is a valid t.HandlerCallable."""
        return callable(value)

    @staticmethod
    def is_configuration_mapping(
        value: t.GuardInputValue,
    ) -> TypeGuard[m.ConfigMap]:
        """Check if value is a valid m.ConfigMap.

        m.ConfigMap = Mapping[str, t.GuardInputValue]

        This TypeGuard enables explicit narrowing for m.ConfigMap.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[m.ConfigMap]: True if value matches ConfigurationMapping structure

        Example:
            >>> from flext_core.utilities import u
            >>> if u.Guards.is_configuration_mapping(config):
            ...     # config is now typed as m.ConfigMap
            ...     items = config.items()

        """
        # Check if it's a Mapping (structural)
        if not isinstance(value, Mapping):
            return False
        # Check all keys are strings and values are ConfigMapValue
        for item_value in value.values():
            if not FlextUtilitiesGuards.is_general_value_type(item_value):
                return False
        return True

    @staticmethod
    def is_configuration_dict(
        value: t.GuardInputValue,
    ) -> TypeGuard[m.Dict]:
        """Check if value is a valid m.Dict mapping.

        This TypeGuard enables explicit narrowing for m.Dict values.
        Uses structural typing to validate at runtime.

        Args:
            value: Object to check

        Returns:
            TypeGuard[m.Dict]: True if value matches ConfigurationDict structure

        Example:
            >>> from flext_core.utilities import u
            >>> if u.Guards.is_configuration_dict(config):
            ...     # config is now typed as m.Dict
            ...     config["key"] = "value"

        """
        if not isinstance(value, dict):
            return False
        for item_value in value.values():
            if not FlextUtilitiesGuards.is_general_value_type(item_value):
                return False
        return True

    @staticmethod
    def is_config_value(value: t.GuardInputValue) -> TypeGuard[t.GuardInputValue]:
        """Check if value is a valid t.GuardInputValue.

        t.GuardInputValue = str | int | float | bool | datetime | None |
                          Sequence[scalar] | Mapping[str, scalar]

        This TypeGuard enables type narrowing for simple config values.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.GuardInputValue]: True if value matches config value type

        """
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool, datetime)):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if not (
                    item is None or isinstance(item, (str, int, float, bool, datetime))
                ):
                    return False
            return True
        if isinstance(value, Mapping):
            for v in value.values():
                if not (v is None or isinstance(v, (str, int, float, bool, datetime))):
                    return False
            return True
        return False

    @staticmethod
    def is_context(obj: t.GuardInputValue) -> TypeGuard[p.Context]:
        """Check if object satisfies the Context protocol."""
        return (
            hasattr(obj, "clone")
            and callable(obj.clone)
            and hasattr(obj, "set")
            and callable(obj.set)
            and hasattr(obj, "get")
            and callable(obj.get)
        )

    # =========================================================================
    # Generic Type Guards for Collections and Sequences
    # =========================================================================

    @staticmethod
    def _is_sequence_not_str(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence and not str."""
        # Runtime check needed: type checker sees str | Sequence[T] as Sequence[Unknown]
        # but runtime can be either str or Sequence[str]
        # Type check is necessary for runtime type distinction
        return isinstance(value, (list, tuple, range)) and not isinstance(value, str)

    @staticmethod
    def is_mapping(
        value: t.GuardInputValue,
    ) -> TypeGuard[Mapping[str, t.GuardInputValue]]:
        """Check if value is ConfigurationMapping (Mapping[str, t.ConfigMapValue])."""
        return isinstance(value, Mapping)

    @staticmethod
    def _is_callable_key_func(
        func: t.GuardInputValue,
    ) -> TypeGuard[Callable[[t.GuardInputValue], t.GuardInputValue]]:
        """Check if value is callable and can be used as key function for sorted()."""
        return callable(func)

    @staticmethod
    def _is_sequence(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence of ConfigMapValue."""
        return isinstance(value, (list, tuple, range))

    @staticmethod
    def _is_str(value: t.GuardInputValue) -> TypeGuard[str]:
        """Check if value is str."""
        return isinstance(value, str)

    @staticmethod
    def _is_dict(value: t.GuardInputValue) -> TypeGuard[m.Dict]:
        """Check if value is a dict-like mapping."""
        return isinstance(value, dict)

    @staticmethod
    def _is_mapping(
        value: t.GuardInputValue,
    ) -> TypeGuard[Mapping[str, t.GuardInputValue]]:
        """Check if value is a Mapping (dict-like)."""
        return isinstance(value, Mapping)

    @staticmethod
    def _is_int(value: t.GuardInputValue) -> TypeGuard[int]:
        """Check if value is int."""
        return isinstance(value, int)

    @staticmethod
    def _is_list_or_tuple(
        value: t.GuardInputValue,
    ) -> TypeGuard[list[t.GuardInputValue] | tuple[t.GuardInputValue, ...]]:
        """Check if value is list or tuple."""
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_sized(value: t.GuardInputValue) -> TypeGuard[Sized]:
        """Check if value has __len__ (str, bytes, Sequence, Mapping)."""
        if isinstance(value, (str, bytes, list, tuple, dict)):
            return True
        return hasattr(value, "__len__") and callable(getattr(value, "__len__", None))

    @staticmethod
    def is_list(value: t.GuardInputValue) -> TypeGuard[list[t.GuardInputValue]]:
        """Check if value is list of ConfigMapValue."""
        return isinstance(value, list)

    @staticmethod
    def _is_float(value: t.GuardInputValue) -> TypeGuard[float]:
        """Check if value is float."""
        return isinstance(value, float)

    @staticmethod
    def _is_bool(value: t.GuardInputValue) -> TypeGuard[bool]:
        """Check if value is bool."""
        return isinstance(value, bool)

    @staticmethod
    def _is_none(value: t.GuardInputValue) -> TypeGuard[None]:
        """Check if value is None."""
        return value is None

    @staticmethod
    def _is_tuple(value: t.GuardInputValue) -> TypeGuard[tuple[t.GuardInputValue, ...]]:
        """Check if value is tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def _is_bytes(value: t.GuardInputValue) -> TypeGuard[bytes]:
        """Check if value is bytes."""
        return isinstance(value, bytes)

    @staticmethod
    def _is_sequence_not_str_bytes(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence and not str or bytes."""
        return isinstance(value, (list, tuple, range)) and not isinstance(
            value,
            (str, bytes),
        )

    # =========================================================================
    # Generic is_type() Function - Unified Type Checking
    # =========================================================================

    # Protocol specs: name -> check function (returns bool)
    # Replaces 9 TypeCheck* Pydantic classes + _PROTOCOL_CATEGORY_MAP + _is_* methods
    _PROTOCOL_SPECS: Mapping[str, Callable[[object], bool]] = MappingProxyType({
        "config": lambda v: hasattr(v, "app_name") and getattr(v, "app_name", None) is not None,
        "context": lambda v: hasattr(v, "request_id") or hasattr(v, "correlation_id"),
        "container": lambda v: hasattr(v, "register") and callable(getattr(v, "register", None)),
        "command_bus": lambda v: hasattr(v, "dispatch") and callable(getattr(v, "dispatch", None)),
        "handler": lambda v: hasattr(v, "handle") and callable(getattr(v, "handle", None)),
        "logger": lambda v: all(
            hasattr(v, a) for a in ("debug", "info", "warning", "error", "exception")
        ),
        "result": lambda v: all(
            hasattr(v, a) for a in ("is_success", "is_failure", "value", "error")
        ),
        "service": lambda v: hasattr(v, "run") and callable(getattr(v, "run", None)),
        "middleware": lambda v: (
            hasattr(v, "before_dispatch") and callable(getattr(v, "before_dispatch", None))
        ),
    })

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
        # Collection checks
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
        # Primitive type checks
        "int": "_is_int",
        "float": "_is_float",
        "bool": "_is_bool",
        "none": "_is_none",
        # Non-empty checks
        "string_non_empty": "is_string_non_empty",
        "dict_non_empty": "is_dict_non_empty",
        "list_non_empty": "is_list_non_empty",
    })

    @staticmethod
    def is_type(
        value: t.FlexibleValue,
        type_spec: str | type | tuple[type, ...],
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
            >>> from flext_core.utilities import u
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
        # String-based type names (delegate to specific guard functions or centralized models)
        if isinstance(type_spec, str):
            type_name = type_spec.lower()

            # Protocol checks via _PROTOCOL_SPECS mapping
            if type_name in FlextUtilitiesGuards._PROTOCOL_SPECS:
                return FlextUtilitiesGuards._check_protocol(value, type_name)

            # Non-protocol string-based checks
            if type_name in FlextUtilitiesGuards._STRING_METHOD_MAP:
                method_name = FlextUtilitiesGuards._STRING_METHOD_MAP[type_name]
                method = getattr(FlextUtilitiesGuards, method_name)
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    if FlextUtilitiesGuards.is_general_value_type(value):
                        return bool(method(value))
                    return False
                return bool(method(value))

            return False

        # Tuple of types check
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)

        # Fallback: type check for any other type specification
        try:
            return isinstance(value, type_spec)
        except TypeError:
            return False

    @staticmethod
    def _check_protocol(value: t.FlexibleValue, name: str) -> bool:
        """Check protocol via _PROTOCOL_SPECS mapping."""
        if name == "context":
            return FlextUtilitiesGuards.is_context(value)
        try:
            return FlextUtilitiesGuards._PROTOCOL_SPECS[name](value)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def is_pydantic_model(value: t.GuardInputValue) -> TypeGuard[p.HasModelDump]:
        """Type guard to check if value is a Pydantic model with model_dump method."""
        return isinstance(value, BaseModel) and hasattr(value, "model_dump") and callable(
            value.model_dump,
        )

    @staticmethod
    def extract_mapping_or_none(
        value: t.GuardInputValue,
    ) -> m.ConfigMap | None:
        """Extract a mapping from a value or return None.

        Used for type narrowing when a generic parameter could be a Mapping
        or another type. Returns the value as ConfigurationMapping if it's
        a Mapping, otherwise returns None.

        Args:
            value: Value that may or may not be a Mapping

        Returns:
            The value as ConfigurationMapping if it's a Mapping, None otherwise

        """
        if (
            isinstance(value, dict)
            or (hasattr(value, "keys") and hasattr(value, "__getitem__"))
        ) and FlextUtilitiesGuards.is_configuration_mapping(
            value,
        ):
            return value
        return None

    @staticmethod
    def _guard_check_type(
        value: t.FlexibleValue,
        condition: type | tuple[type, ...],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
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
        return None

    @staticmethod
    def _is_type_tuple(value: object) -> TypeGuard[tuple[type, ...]]:
        return isinstance(value, tuple) and all(
            isinstance(item, type) for item in value
        )

    @staticmethod
    def _guard_check_validator(
        value: t.ConfigMapValue,
        condition: p.ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        if not condition(value):
            if error_msg is None:
                desc = (
                    getattr(condition, "description", "validation")
                    if hasattr(condition, "description")
                    else "validation"
                )
                return f"{context_name} failed {desc} check"
            return error_msg
        return None

    @staticmethod
    def _guard_check_string_shortcut(
        value: t.ConfigMapValue,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        shortcut_lower = condition.lower()
        if shortcut_lower == "non_empty":
            if isinstance(value, str | list | dict) and bool(value):
                return None
            return error_msg or f"{context_name} must be non-empty"
        if shortcut_lower == "positive":
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and value > 0
            ):
                return None
            return error_msg or f"{context_name} must be positive number"
        if shortcut_lower == "non_negative":
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and value >= 0
            ):
                return None
            return error_msg or f"{context_name} must be non-negative number"
        if shortcut_lower == "dict":
            if hasattr(value, "items") and value.__class__ not in {str, bytes}:
                return None
            return error_msg or f"{context_name} must be dict-like"
        if shortcut_lower == "list":
            if (
                hasattr(value, "__iter__")
                and value.__class__ not in {str, bytes}
                and not hasattr(value, "items")
            ):
                return None
            return error_msg or f"{context_name} must be list-like"
        if shortcut_lower == "string":
            if isinstance(value, str):
                return None
            return error_msg or f"{context_name} must be string"
        if shortcut_lower == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return None
            return error_msg or f"{context_name} must be int"
        if shortcut_lower == "float":
            if isinstance(value, int | float) and not isinstance(value, bool):
                return None
            return error_msg or f"{context_name} must be float"
        if shortcut_lower == "bool":
            if isinstance(value, bool):
                return None
            return error_msg or f"{context_name} must be bool"
        return error_msg or f"{context_name} unknown guard shortcut: {condition}"

    @staticmethod
    def _guard_check_predicate[T](
        value: T,
        condition: Callable[[T], bool],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
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
        return None

    @staticmethod
    def _guard_check_condition[T](
        value: T,
        condition: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
        | str,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        if isinstance(condition, type):
            return FlextUtilitiesGuards._guard_check_type(
                value,
                condition,
                context_name,
                error_msg,
            )
        if FlextUtilitiesGuards._is_type_tuple(condition):
            return FlextUtilitiesGuards._guard_check_type(
                value,
                condition,
                context_name,
                error_msg,
            )
        if isinstance(condition, p.ValidatorSpec):
            if not FlextUtilitiesGuards.is_general_value_type(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value: t.ConfigMapValue = value
            return FlextUtilitiesGuards._guard_check_validator(
                typed_value,
                condition,
                context_name,
                error_msg,
            )
        if isinstance(condition, str):
            if not FlextUtilitiesGuards.is_general_value_type(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value_s: t.ConfigMapValue = value
            return FlextUtilitiesGuards._guard_check_string_shortcut(
                typed_value_s,
                condition,
                context_name,
                error_msg,
            )
        if callable(condition):
            return FlextUtilitiesGuards._guard_check_predicate(
                value,
                condition,
                context_name,
                error_msg,
            )
        return error_msg or f"{context_name} invalid guard condition type"

    @staticmethod
    def _guard_handle_failure[T](
        error_message: str,
        *,
        return_value: bool,
        default: T | None,
    ) -> r[T] | T | None:
        if return_value:
            return default
        if default is not None:
            return r.ok(default)
        return r.fail(error_message)

    @staticmethod
    def guard_result[T](
        value: T,
        *conditions: (
            type[T] | tuple[type[T], ...] | Callable[[T], bool] | p.ValidatorSpec | str
        ),
        error_message: str | None = None,
        context: str | None = None,
        default: T | None = None,
        return_value: bool = False,
    ) -> r[T] | T | None:
        context_name = context or "Value"
        if len(conditions) == 0:
            if bool(value):
                return value if return_value else r.ok(value)
            failure_message = error_message or f"{context_name} guard failed"
            return FlextUtilitiesGuards._guard_handle_failure(
                failure_message,
                return_value=return_value,
                default=default,
            )

        for condition in conditions:
            condition_error = FlextUtilitiesGuards._guard_check_condition(
                value,
                condition,
                context_name,
                error_message,
            )
            if condition_error is not None:
                return FlextUtilitiesGuards._guard_handle_failure(
                    condition_error,
                    return_value=return_value,
                    default=default,
                )

        return value if return_value else r.ok(value)

    @staticmethod
    def guard(
        value: t.GuardInputValue,
        validator: Callable[[t.GuardInputValue], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.GuardInputValue | None = None,
        return_value: bool = False,
    ) -> t.GuardInputValue | bool | None:
        guarded_value: t.GuardInputValue = value
        try:
            if isinstance(validator, type):
                if isinstance(value, validator):
                    return guarded_value if return_value else True
            elif isinstance(validator, tuple):
                tuple_types = tuple(
                    item for item in validator if isinstance(item, type)
                )
                if len(tuple_types) == len(validator) and isinstance(
                    value,
                    tuple_types,
                ):
                    return guarded_value if return_value else True
            elif callable(validator):
                if validator(value):
                    return guarded_value if return_value else True
            elif value:
                return guarded_value if return_value else True
            return default
        except (TypeError, ValueError, AttributeError):
            return default

    @staticmethod
    def _ensure_to_list(
        value: t.ConfigMapValue | list[t.ConfigMapValue] | None,
        default: list[t.ConfigMapValue] | None,
    ) -> list[t.ConfigMapValue]:
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            return value
        single_item_list: list[t.ConfigMapValue] = [value]
        return single_item_list

    @staticmethod
    def _ensure_to_dict(
        value: t.ConfigMapValue | Mapping[str, t.ConfigMapValue] | None,
        default: Mapping[str, t.ConfigMapValue] | None,
    ) -> Mapping[str, t.ConfigMapValue]:
        if value is None:
            return default if default is not None else {}
        if isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}
        wrapped_dict: Mapping[str, t.ConfigMapValue] = {"value": value}
        return wrapped_dict

    @staticmethod
    def ensure(
        value: t.ConfigMapValue,
        *,
        target_type: str = "auto",
        default: str
        | list[t.ConfigMapValue]
        | Mapping[str, t.ConfigMapValue]
        | None = None,
    ) -> str | list[t.ConfigMapValue] | Mapping[str, t.ConfigMapValue]:
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
                str_list_default = [str(x) for x in default]
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                return list(value)
            if value is None:
                return list(str_list_default) if str_list_default else []
            return [value]

        if target_type == "dict":
            dict_default: Mapping[str, t.ConfigMapValue] | None = (
                default if isinstance(default, Mapping) else None
            )
            return FlextUtilitiesGuards._ensure_to_dict(value, dict_default)

        if target_type == "auto" and isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}

        list_default: list[t.ConfigMapValue] | None = (
            default if isinstance(default, list) else None
        )
        return FlextUtilitiesGuards._ensure_to_list(value, list_default)

    @staticmethod
    def in_(value: t.GuardInputValue, container: t.GuardInputValue) -> bool:
        """Check if value is in container."""
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False

    @staticmethod
    def has(obj: t.GuardInputValue, key: str) -> bool:
        """Check if object has attribute/key."""
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    @staticmethod
    def empty(items: t.GuardInputValue | None) -> bool:
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
    def none_(*values: t.GuardInputValue) -> bool:
        """Check if all values are None.

        Args:
            *values: Values to check

        Returns:
            True if all values are None, False otherwise

        Example:
            if u.none_(name, email):
                return r.fail("Name and email are required")

        """
        return all(v is None for v in values)

    @staticmethod
    def chk(
        value: t.GuardInputValue,
        *,
        eq: t.GuardInputValue | None = None,
        ne: t.GuardInputValue | None = None,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
        is_: type | None = None,
        not_: type | None = None,
        in_: Sequence[t.GuardInputValue] | None = None,
        not_in: Sequence[t.GuardInputValue] | None = None,
        none: bool | None = None,
        empty: bool | None = None,
        match: str | None = None,
        contains: t.GuardInputValue | None = None,
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
        # None checks
        if none is True and value is not None:
            return False
        if none is False and value is None:
            return False

        # Type checks
        # is_ and not_ are type[ConfigMapValue] which can be generic
        # Check if the type is a plain type (not generic) before using type check
        if is_ is not None and not isinstance(value, is_):
            return False
        if not_ is not None and isinstance(value, not_):
            return False

        # Equality checks
        if eq is not None and value != eq:
            return False
        if ne is not None and value == ne:
            return False

        # Membership checks
        if in_ is not None and value not in in_:
            return False
        if not_in is not None and value in not_in:
            return False

        # Length/numeric checks - use len() for sequences, direct for numbers
        check_val: int | float = 0
        if isinstance(value, (int, float)):
            check_val = value
        elif isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            check_val = len(value)
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

        # Empty checks (after len is computed)
        if empty is True and check_val != 0:
            return False
        if empty is False and check_val == 0:
            return False

        # String-specific checks
        if isinstance(value, str):
            if match is not None and not re.search(match, value):
                return False
            if starts is not None and not value.startswith(starts):
                return False
            if ends is not None and not value.endswith(ends):
                return False
            if (
                contains is not None
                and isinstance(contains, str)
                and contains not in value
            ):
                return False
        elif contains is not None:
            if isinstance(value, (dict, list, tuple, set, frozenset)):
                if contains not in value:
                    return False
            # Other types - use hasattr/getattr with explicit type narrowing
            elif hasattr(value, "__contains__"):
                contains_method = getattr(value, "__contains__", None)
                if callable(contains_method):
                    try:
                        if not contains_method(contains):
                            return False
                    except (TypeError, ValueError):
                        return False

        return True

    # =========================================================================
    # Validation Methods
    # =========================================================================

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
                f"{field_name} must have at least {min_length} characters/items",
            )
        if max_length is not None and length > max_length:
            return r[T].fail(
                f"{field_name} must have at most {max_length} characters/items",
            )
        return r[T].ok(value)

    @staticmethod
    def validate_pattern(
        value: str,
        pattern: str,
        field_name: str = "value",
    ) -> r[str]:
        """Return success when ``value`` matches ``pattern``."""
        if re.search(pattern, value) is None:
            return r[str].fail(f"{field_name} has invalid format")
        return r[str].ok(value)

    @staticmethod
    def validate_positive(
        value: float,
        field_name: str = "value",
    ) -> r[int | float]:
        """Return success when numeric ``value`` is greater than zero."""
        if isinstance(value, bool) or value <= 0:
            return r[int | float].fail(f"{field_name} must be positive")
        return r[int | float].ok(value)

    @staticmethod
    def validate_uri(
        uri: str,
        field_name: str = "uri",
    ) -> r[str]:
        """Return success when ``uri`` is a valid URI/URL format."""
        uri_pattern = r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+$"
        if re.search(uri_pattern, uri) is None:
            return r[str].fail(f"{field_name} has invalid URI format")
        return r[str].ok(uri)

    @staticmethod
    def validate_port_number(
        port: int,
        field_name: str = "port",
    ) -> r[int]:
        """Return success when ``port`` is a valid port number (1-65535)."""
        if not isinstance(port, int) or isinstance(port, bool):
            return r[int].fail(f"{field_name} must be an integer")
        max_port = 65535
        if port < 1 or port > max_port:
            return r[int].fail(f"{field_name} must be between 1 and 65535")
        return r[int].ok(port)

    @staticmethod
    def validate_hostname(
        hostname: str,
        field_name: str = "hostname",
    ) -> r[str]:
        """Return success when ``hostname`` is a valid hostname or FQDN."""
        hostname_pattern = (
            r"^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\.[a-zA-Z0-9-]{1,63}(?<!-))*$"
        )
        if re.search(hostname_pattern, hostname) is None:
            return r[str].fail(f"{field_name} has invalid hostname format")
        return r[str].ok(hostname)


__all__ = [
    "FlextUtilitiesGuards",
]
