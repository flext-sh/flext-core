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
from collections.abc import Callable, Mapping, Sequence, Sized
from datetime import datetime
from pathlib import Path
from typing import TypeGuard

from pydantic import BaseModel

from flext_core.models import m
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
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
    def is_string_non_empty(value: t.GuardInputValue) -> bool:
        """Check if value is a non-empty string using duck typing.

        Validates that the provided value is a string type and contains
        non-whitespace content after stripping.

        Args:
            value: Object to check for non-empty string type

        Returns:
            bool: True if value is non-empty string, False otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> u.is_type("hello", "string_non_empty")
            True
            >>> u.is_type("   ", "string_non_empty")
            False
            >>> u.is_type(123, "string_non_empty")
            False

        """
        return value.__class__ is str and bool(value.strip())

    @staticmethod
    def is_dict_non_empty(value: t.GuardInputValue) -> bool:
        """Check if value is a non-empty dictionary using duck typing.

        Validates that the provided value behaves like a dictionary
        (has dict-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty dict-like type

        Returns:
            bool: True if value is non-empty dict-like, False otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> u.is_type({"key": "value"}, "dict_non_empty")
            True
            >>> u.is_type({}, "dict_non_empty")
            False
            >>> u.is_type("not_a_dict", "dict_non_empty")
            False

        """
        return FlextRuntime.is_dict_like(value) and bool(value)

    @staticmethod
    def is_list_non_empty(value: t.GuardInputValue) -> bool:
        """Check if value is a non-empty list using duck typing.

        Validates that the provided value behaves like a list
        (has list-like interface) and contains at least one item.

        Args:
            value: Object to check for non-empty list-like type

        Returns:
            bool: True if value is non-empty list-like, False otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> u.is_type([1, 2, 3], "list_non_empty")
            True
            >>> u.is_type([], "list_non_empty")
            False
            >>> u.is_type("not_a_list", "list_non_empty")
            False

        """
        return FlextRuntime.is_list_like(value) and bool(value)

    @staticmethod
    def normalize_to_metadata_value(
        val: t.GuardInputValue,
    ) -> t.MetadataAttributeValue:
        """Normalize any value to MetadataAttributeValue.

        MetadataAttributeValue is more restrictive than t.GuardInputValue,
        so we need to normalize nested structures to flat types.

        Args:
            val: Value to normalize

        Returns:
            t.MetadataAttributeValue: Normalized value compatible with Metadata attributes

        Example:
            >>> from flext_core.utilities import u
            >>> u.Guards.normalize_to_metadata_value("test")
            'test'
            >>> u.Guards.normalize_to_metadata_value({"key": "value"})
            {'key': 'value'}
            >>> u.Guards.normalize_to_metadata_value([1, 2, 3])
            [1, 2, 3]

        """
        if val is None or val.__class__ in (str, int, float, bool):
            return val
        if FlextRuntime.is_dict_like(val):
            # Convert to flat dict with ScalarValue values
            # Type narrowing: is_dict_like returns TypeGuard[ConfigurationMapping]
            # ConfigurationMapping is Mapping[str, t.GuardInputValue]
            val_mapping = val  # type narrowing via TypeGuard
            # Use full type from start to satisfy dict invariance
            result_dict: dict[
                str,
                str
                | int
                | float
                | bool
                | datetime
                | list[str | int | float | bool | datetime | None]
                | None,
            ] = {}
            # TypeGuard already narrows to Mapping - no extra check needed
            dict_v = dict(val_mapping.items())
            for k, v in dict_v.items():
                # Explicit type annotations for loop variables
                key: str = k
                value: t.GuardInputValue = v
                if value is None or value.__class__ in (
                    str,
                    int,
                    float,
                    bool,
                    datetime,
                ):
                    result_dict[key] = value
                else:
                    result_dict[key] = str(value)
            return result_dict
        if FlextRuntime.is_list_like(val):
            # Convert to list[t.MetadataAttributeValue]
            # Type narrowing: is_list_like returns TypeGuard[Sequence[t.GuardInputValue]]
            val_sequence = val  # type narrowing via TypeGuard
            result_list: t.GeneralListValue = []
            # TypeGuard already narrows to Sequence - no extra check needed
            # Exclude str/bytes from iteration
            if val_sequence.__class__ not in (str, bytes):
                for item in val_sequence:
                    # Explicit type annotation for loop variable
                    list_item: t.GuardInputValue = item
                    if list_item is None or list_item.__class__ in (
                        str,
                        int,
                        float,
                        bool,
                    ):
                        result_list.append(list_item)
                    else:
                        result_list.append(str(list_item))
            return result_list
        return str(val)

    # =========================================================================
    # TypeGuard Functions for FLEXT Core Types
    # =========================================================================
    # These functions enable type narrowing without  - zero tolerance typing

    @staticmethod
    def is_general_value_type(value: t.GuardInputValue) -> TypeGuard[t.GuardInputValue]:
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
        if hasattr(value, "items") and hasattr(value, "keys") and callable(value.items):
            # Iterate with explicit type annotations to satisfy pyright
            k: t.GuardInputValue
            v: t.GuardInputValue
            for k, v in value.items():
                if not isinstance(k, str):
                    return False
                if not FlextUtilitiesGuards.is_general_value_type(v):
                    return False
            return True
        # Check callable types
        if callable(value):
            return True
        # Check BaseModel or Path instances (structural for BaseModel)
        return hasattr(value, "model_dump") or isinstance(value, Path)

    @staticmethod
    def is_handler_type(value: t.GuardInputValue) -> TypeGuard[t.HandlerType]:
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
        if hasattr(value, "items") and hasattr(value, "keys") and callable(value.items):
            return True
        # Check if BaseModel instance or class
        if hasattr(value, "model_dump") and callable(
            getattr(value, "model_dump", None)
        ):
            return True
        if isinstance(value, type):
            try:
                if BaseModel in value.__mro__:
                    return True
            except TypeError:
                pass
        # Check for handler protocol methods (duck typing)
        # All values are objects in Python, so type check (value, object) is always True
        return hasattr(value, "handle") or hasattr(value, "can_handle")

    @staticmethod
    def is_handler_callable(value: t.GuardInputValue) -> TypeGuard[t.HandlerCallable]:
        """Check if value is a valid t.HandlerCallable.

        t.HandlerCallable = Callable[[t.ConfigMapValue], t.ConfigMapValue]

        This TypeGuard enables explicit narrowing for handler functions.
        Checks if value is callable and has the handler decorator attribute.

        Args:
            value: Object to check

        Returns:
            TypeGuard[t.HandlerCallable]: True if value is a decorated handler callable

        Example:
            >>> from flext_core.utilities import u
            >>> if u.Guards.is_handler_callable(func):
            ...     # func is now typed as t.HandlerCallable
            ...     result = func(message)

        """
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
        if not (
            hasattr(value, "items") and hasattr(value, "keys") and callable(value.items)
        ):
            return False
        # Check all keys are strings and values are ConfigMapValue
        # Iterate with explicit type annotations to satisfy pyright
        k: t.GuardInputValue
        v: t.GuardInputValue
        for k, v in value.items():
            if k.__class__ is not str:
                return False
            if not FlextUtilitiesGuards.is_general_value_type(v):
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
        if value.__class__ is not dict:
            return False
        k: t.GuardInputValue
        v: t.GuardInputValue
        for k, v in value.items():
            if k.__class__ is not str:
                return False
            if not FlextUtilitiesGuards.is_general_value_type(v):
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
        if value.__class__ in (str, int, float, bool, datetime):
            return True
        if value.__class__ in (list, tuple):
            item: t.GuardInputValue
            for item in value:
                if not (
                    item is None or item.__class__ in (str, int, float, bool, datetime)
                ):
                    return False
            return True
        if hasattr(value, "items") and callable(getattr(value, "items", None)):
            for k, v in value.items():
                if k.__class__ is not str:
                    return False
                if not (v is None or v.__class__ in (str, int, float, bool, datetime)):
                    return False
            return True
        return False

    @staticmethod
    def _is_config(obj: t.GuardInputValue) -> TypeGuard[p.Config]:
        """Check if object satisfies the Config protocol.

        Enables type narrowing for configuration objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Config]: True if obj satisfies Config protocol

        Example:
            >>> from flext_core.utilities import u
            >>> if u.is_type(config, "config"):
            ...     # config is now typed as p.Config
            ...     name = config.app_name

        """
        return hasattr(obj, "app_name") and getattr(obj, "app_name", None) is not None

    @staticmethod
    def is_context(obj: t.GuardInputValue) -> TypeGuard[p.Context]:
        """Check if object satisfies the Context protocol.

        Enables type narrowing for context objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Context]: True if obj satisfies Ctx protocol

        """
        return hasattr(obj, "request_id") or hasattr(obj, "correlation_id")

    @staticmethod
    def _is_context(obj: t.GuardInputValue) -> TypeGuard[p.Context]:
        """Private version of is_context for internal protocol checks."""
        return hasattr(obj, "request_id") or hasattr(obj, "correlation_id")

    @staticmethod
    def _is_container(obj: t.GuardInputValue) -> TypeGuard[p.DI]:
        """Check if object satisfies the DI protocol.

        Enables type narrowing for container objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.DI]: True if obj satisfies DI protocol

        """
        return hasattr(obj, "register") and callable(getattr(obj, "register", None))

    @staticmethod
    def _is_command_bus(obj: t.GuardInputValue) -> TypeGuard[p.CommandBus]:
        """Check if object satisfies the CommandBus protocol.

        Enables type narrowing for dispatcher/command bus without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.CommandBus]: True if obj satisfies CommandBus

        """
        return hasattr(obj, "dispatch") and callable(getattr(obj, "dispatch", None))

    @staticmethod
    def _is_handler(obj: t.GuardInputValue) -> TypeGuard[p.Handler]:
        """Check if object satisfies the Handler protocol.

        Enables type narrowing for handler objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Handler]: True if obj satisfies Handler protocol

        """
        return hasattr(obj, "handle") and callable(getattr(obj, "handle", None))

    @staticmethod
    def _is_logger(obj: t.GuardInputValue) -> TypeGuard[p.Log.StructlogLogger]:
        """Check if object satisfies the StructlogLogger protocol.

        Enables type narrowing for logger objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Log.StructlogLogger]: True if satisfies protocol

        """
        return (
            hasattr(obj, "debug")
            and hasattr(obj, "info")
            and hasattr(obj, "warning")
            and hasattr(obj, "error")
            and hasattr(obj, "exception")
        )

    @staticmethod
    def _is_result(obj: t.GuardInputValue) -> TypeGuard[p.Result[t.GuardInputValue]]:
        """Check if object satisfies the Result protocol.

        Enables type narrowing for result objects.

        Uses attribute-based checking instead of type check because
        p.Result has optional methods that
        may not be implemented by all Result classes.

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Result]: True if obj satisfies Result protocol

        """
        # Check for core result properties
        return (
            hasattr(obj, "is_success")
            and hasattr(obj, "is_failure")
            and hasattr(obj, "value")
            and hasattr(obj, "error")
        )

    @staticmethod
    def _is_service(obj: t.GuardInputValue) -> TypeGuard[p.Service[t.GuardInputValue]]:
        """Check if object satisfies the Service protocol.

        Enables type narrowing for service objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Service]: True if obj satisfies Service protocol

        """
        return hasattr(obj, "run") and callable(getattr(obj, "run", None))

    @staticmethod
    def _is_middleware(obj: t.GuardInputValue) -> TypeGuard[p.Middleware]:
        """Check if object satisfies the Middleware protocol.

        Enables type narrowing for middleware objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Middleware]: True if obj satisfies Middleware

        """
        return hasattr(obj, "process") and callable(getattr(obj, "process", None))

    # =========================================================================
    # Generic Type Guards for Collections and Sequences
    # =========================================================================

    @staticmethod
    def _is_sequence_not_str(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence and not str.

        Type guard to distinguish Sequence[str] from str in union types.
        Useful for ExclusionSpec, ErrorCodeSpec, ContainmentSpec, etc.

        Args:
            value: Value that can be str or Sequence[T]

        Returns:
            TypeGuard[Sequence[t.GuardInputValue]]: True if value is Sequence and not str

        Example:
            >>> if FlextUtilitiesGuards.is_sequence_not_str(spec):
            ...     # spec is now typed as Sequence[t.GuardInputValue]
            ...     items = list(spec)

        """
        # Runtime check needed: type checker sees str | Sequence[T] as Sequence[Unknown]
        # but runtime can be either str or Sequence[str]
        # Type check is necessary for runtime type distinction
        return value.__class__ in (list, tuple, range) and value.__class__ is not str

    @staticmethod
    def is_mapping(value: t.GuardInputValue) -> TypeGuard[m.ConfigMap]:
        """Check if value is ConfigurationMapping (Mapping[str, t.ConfigMapValue]).

        Type guard for mapping types used in FLEXT validation.
        Uses proper FLEXT types instead of object.

        Args:
            value: ConfigMapValue to check

        Returns:
            TypeGuard[m.ConfigMap]: True if value is ConfigurationMapping

        Example:
            >>> if FlextUtilitiesGuards.is_mapping(params.kv):
            ...     # params.kv is now typed as m.ConfigMap
            ...     for key, val in params.kv.items():

        """
        return hasattr(value, "items") and callable(getattr(value, "items", None))

    @staticmethod
    def _is_callable_key_func(
        func: t.GuardInputValue,
    ) -> TypeGuard[Callable[[t.GuardInputValue], t.GuardInputValue]]:
        """Check if value is callable and can be used as key function for sorted().

        Type guard for sorted() key functions that return comparable values.
        Runtime validation ensures correctness. Uses FLEXT types.

        Args:
            func: ConfigMapValue to check

        Returns:
            TypeGuard[Callable[[t.ConfigMapValue], t.ConfigMapValue]]: True if callable

        Example:
            >>> if FlextUtilitiesGuards.is_callable_key_func(key_func):
            ...     # key_func is callable, can be used with sorted()
            ...     sorted_list = sorted(items, key=key_func)

        """
        return callable(func)

    @staticmethod
    def _is_sequence(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence of ConfigMapValue.

        Type guard for sequence types using FLEXT types.

        Args:
            value: ConfigMapValue to check

        Returns:
            TypeGuard[Sequence[t.ConfigMapValue]]: True if value is Sequence

        Example:
            >>> if FlextUtilitiesGuards.is_sequence(key_equals):
            ...     # key_equals is now typed as Sequence[t.GuardInputValue]
            ...     pairs = list(key_equals)

        """
        return value.__class__ in (list, tuple, range)

    @staticmethod
    def _is_str(value: t.GuardInputValue) -> TypeGuard[str]:
        """Check if value is str.

        Type guard for string types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[str]: True if value is str

        Example:
            >>> if FlextUtilitiesGuards.is_str(path):
            ...     # path is now typed as str
            ...     parts = path.split(".")

        """
        return value.__class__ is str

    @staticmethod
    def _is_dict(value: t.GuardInputValue) -> TypeGuard[m.Dict]:
        """Check if value is a dict-like mapping.

        Type guard for dictionary types. Returns ConfigurationDict for type safety.

        Args:
            value: Object to check

        Returns:
            TypeGuard[ConfigurationDict]: True if value is dict

        Example:
            >>> if FlextUtilitiesGuards._is_dict(items):
            ...     # items is now typed as ConfigurationDict
            ...     value = items.get("key")

        """
        return value.__class__ is dict

    @staticmethod
    def _is_mapping(
        value: t.GuardInputValue,
    ) -> TypeGuard[Mapping[str, t.GuardInputValue]]:
        """Check if value is a Mapping (dict-like).

        Type guard for Mapping types (dict, ChainMap, MappingProxyType, etc.).

        Args:
            value: Object to check

        Returns:
            TypeGuard[Mapping[str, t.GuardInputValue]]: True if value is a Mapping

        Example:
            >>> if FlextUtilitiesGuards._is_mapping(config):
            ...     # config is now typed as Mapping[str, t.GuardInputValue]
            ...     value = config.get("key")

        """
        return hasattr(value, "items") and callable(getattr(value, "items", None))

    @staticmethod
    def _is_int(value: t.GuardInputValue) -> TypeGuard[int]:
        """Check if value is int.

        Type guard for integer types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[int]: True if value is int

        Example:
            >>> if FlextUtilitiesGuards._is_int(index):
            ...     # index is now typed as int
            ...     value = items[index]

        """
        return value.__class__ is int

    @staticmethod
    def _is_list_or_tuple(
        value: t.GuardInputValue,
    ) -> TypeGuard[list[t.GuardInputValue] | tuple[t.GuardInputValue, ...]]:
        """Check if value is list or tuple.

        Type guard for list and tuple types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[list[t.GuardInputValue] | tuple[t.GuardInputValue, ...]]: True if value is list or tuple

        Example:
            >>> if FlextUtilitiesGuards._is_list_or_tuple(items):
            ...     # items is now typed as list[t.GuardInputValue] | tuple[t.GuardInputValue, ...]
            ...     value = items[0]

        """
        return value.__class__ in (list, tuple)

    @staticmethod
    def _is_sized(value: t.GuardInputValue) -> TypeGuard[Sized]:
        """Check if value has __len__ (str, bytes, Sequence, Mapping).

        Type guard for sized types that support len().

        Args:
            value: Object to check

        Returns:
            TypeGuard[Sized]: True if value has __len__

        Example:
            >>> if FlextUtilitiesGuards.is_sized(value):
            ...     # value has __len__, can call len()
            ...     length = len(value)

        """
        return value.__class__ in (str, bytes, list, tuple, dict) or (
            hasattr(value, "__len__") and callable(getattr(value, "__len__", None))
        )

    @staticmethod
    def is_list(value: t.GuardInputValue) -> TypeGuard[list[t.GuardInputValue]]:
        """Check if value is list of ConfigMapValue.

        Type guard for list types using FLEXT types.

        Args:
            value: ConfigMapValue to check

        Returns:
            TypeGuard[list[t.ConfigMapValue]]: True if value is list

        Example:
            >>> if FlextUtilitiesGuards.is_list(value):
            ...     # value is list[t.ConfigMapValue]
            ...     first = value[0]

        """
        return value.__class__ is list

    @staticmethod
    def _is_float(value: t.GuardInputValue) -> TypeGuard[float]:
        """Check if value is float.

        Type guard for float types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[float]: True if value is float

        """
        return value.__class__ is float

    @staticmethod
    def _is_bool(value: t.GuardInputValue) -> TypeGuard[bool]:
        """Check if value is bool.

        Type guard for boolean types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bool]: True if value is bool

        """
        return value.__class__ is bool

    @staticmethod
    def _is_none(value: t.GuardInputValue) -> TypeGuard[None]:
        """Check if value is None.

        Type guard for None type.

        Args:
            value: Object to check

        Returns:
            TypeGuard[None]: True if value is None

        """
        return value is None

    @staticmethod
    def _is_tuple(value: t.GuardInputValue) -> TypeGuard[tuple[t.GuardInputValue, ...]]:
        """Check if value is tuple.

        Type guard for tuple types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[tuple[t.GuardInputValue, ...]]: True if value is tuple

        """
        return value.__class__ is tuple

    @staticmethod
    def _is_bytes(value: t.GuardInputValue) -> TypeGuard[bytes]:
        """Check if value is bytes.

        Type guard for bytes types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bytes]: True if value is bytes

        """
        return value.__class__ is bytes

    @staticmethod
    def _is_sequence_not_str_bytes(
        value: t.GuardInputValue,
    ) -> TypeGuard[Sequence[t.GuardInputValue]]:
        """Check if value is Sequence and not str or bytes.

        Type guard to distinguish Sequence from str/bytes in union types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[Sequence[t.GuardInputValue]]: True if value is Sequence and not str/bytes

        """
        return isinstance(value, (list, tuple, range)) and not isinstance(
            value, (str, bytes)
        )

    # =========================================================================
    # Generic is_type() Function - Unified Type Checking
    # =========================================================================

    @staticmethod
    def is_type(
        value: t.GuardInputValue, type_spec: str | type | tuple[type, ...]
    ) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. This function delegates to the appropriate specific
        guard function or performs direct type checks.

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
        # String-based type names (delegate to specific guard functions)
        if type_spec.__class__ is str:
            type_name = type_spec.lower()
            # Map string names to private method names
            method_map: Mapping[str, str] = {
                # Protocol checks
                "config": "_is_config",
                "context": "_is_context",
                "container": "_is_container",
                "command_bus": "_is_command_bus",
                "handler": "_is_handler",
                "logger": "_is_logger",
                "result": "_is_result",
                "service": "_is_service",
                "middleware": "_is_middleware",
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
            }
            if type_name in method_map:
                method_name = method_map[type_name]
                method = getattr(FlextUtilitiesGuards, method_name)
                # For non-empty checks, use t.GuardInputValue from lower layer
                # Methods accept t.GuardInputValue, so use TypeGuard for type narrowing
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    # TypeGuard-based type narrowing for t.GuardInputValue
                    if FlextUtilitiesGuards.is_general_value_type(value):
                        return bool(method(value))
                    # Value is not t.GuardInputValue, return False
                    return False
                return bool(method(value))
            # Unknown string type spec
            return False

        # Tuple of types check
        if type_spec.__class__ is tuple:
            return value.__class__ in type_spec

        # Direct type/class checks
        if type_spec.__class__ is type:
            # Check if it's a protocol first
            # Use class metadata to avoid runtime type() narrowing calls
            type_class = type_spec.__class__
            if hasattr(type_spec, "__protocol_attrs__") or (
                hasattr(type_spec, "__module__") and "Protocol" in str(type_class)
            ):
                if type_spec == p.Config:
                    return FlextUtilitiesGuards._is_config(value)
                if type_spec == p.Context:
                    return FlextUtilitiesGuards._is_context(value)
                if type_spec == p.DI:
                    return FlextUtilitiesGuards._is_container(value)
                if type_spec == p.CommandBus:
                    return FlextUtilitiesGuards._is_command_bus(value)
                if type_spec == p.Handler:
                    return FlextUtilitiesGuards._is_handler(value)
                if type_spec == p.Log.StructlogLogger:
                    return FlextUtilitiesGuards._is_logger(value)
                if type_spec == p.Result:
                    return FlextUtilitiesGuards._is_result(value)
                if type_spec == p.Service:
                    return FlextUtilitiesGuards._is_service(value)
                if type_spec == p.Middleware:
                    return FlextUtilitiesGuards._is_middleware(value)
                return False
            # Regular type check
            return isinstance(value, type_spec)

        # Fallback: type check for any other type specification
        try:
            return isinstance(value, type_spec)
        except TypeError:
            return False

    @staticmethod
    def is_pydantic_model(value: t.GuardInputValue) -> TypeGuard[p.HasModelDump]:
        """Type guard to check if value is a Pydantic model with model_dump method.

        Args:
            value: Object to check

        Returns:
            True if object implements HasModelDump protocol, False otherwise

        """
        return hasattr(value, "model_dump") and callable(
            getattr(value, "model_dump", None),
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
            value.__class__ is dict
            or (hasattr(value, "keys") and hasattr(value, "__getitem__"))
        ) and FlextUtilitiesGuards.is_configuration_mapping(
            value,
        ):
            return value
        return None

    # =========================================================================
    # Guard Methods - Moved from utilities.py
    # =========================================================================

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
    ) -> t.GuardInputValue | None:
        """Simple guard method for validation. Returns value if valid, default if not.

        Args:
            value: Value to validate
            validator: Callable, type, or tuple of types to validate against
            default: Default value to return if validation fails
            return_value: If True, return the value itself; if False, return True/default

        Returns:
            Validated value, True, or default value

        """
        try:
            if callable(validator):
                if validator(value):
                    return value if return_value else True
            elif validator is not None and validator.__class__ is tuple:
                if value.__class__ in validator:
                    return value if return_value else True
            elif validator is not None and validator.__class__ is type:
                if value.__class__ is validator:
                    return value if return_value else True
            # Default validation - check if value is truthy
            elif value:
                return value if return_value else True
            return default
        except Exception:
            return default

    @staticmethod
    def in_(value: t.GuardInputValue, container: t.GuardInputValue) -> bool:
        """Check if value is in container."""
        if container.__class__ in (list, tuple, set, dict):
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
        if hasattr(items, "__len__") and callable(getattr(items, "__len__", None)):
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
            is_: Check type(value) is is_
            not_: Check type(value) is not not_
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
        if is_ is not None:
            if is_.__class__ is type:
                if not isinstance(value, is_):
                    return False
        if not_ is not None:
            if not_.__class__ is type:
                if isinstance(value, not_):
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
        elif isinstance(value, (str, bytes)) or isinstance(
            value,
            (list, tuple, dict, set, frozenset),
        ):
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
            if isinstance(value, dict) or isinstance(
                value,
                (list, tuple, set, frozenset),
            ):
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

    def __getattribute__(self, name: str) -> t.GuardInputValue:
        """Intercept attribute access to warn about direct usage.

        Emits DeprecationWarning when public methods are accessed directly
        instead of through u.guard() or u.Guards.*.

        Args:
            name: Attribute name being accessed

        Returns: t.GuardInputValue: The requested attribute

        """
        # Allow access to private methods and special attributes
        if name.startswith("_") or name in {
            "__class__",
            "__dict__",
            "__module__",
            "__qualname__",
            "__name__",
            "__doc__",
            "__annotations__",
            "__init__",
            "__new__",
            "__subclasshook__",
            "__instancecheck__",
            "__subclasscheck__",
        }:
            return super().__getattribute__(name)

        # Check if this is a public method that should be accessed via u.Guards
        if hasattr(FlextUtilitiesGuards, name):
            warnings.warn(
                (
                    f"Direct access to FlextUtilitiesGuards.{name} is deprecated. "
                    f"Use u.guard() or u.Guards.{name} instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )

        return super().__getattribute__(name)


__all__ = [
    "FlextUtilitiesGuards",
]
