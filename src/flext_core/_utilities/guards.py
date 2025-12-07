"""Runtime type guard helpers for dispatcher-safe validations.

The utilities provide runtime type checking functions that use structural typing
to keep handler and service checks lightweight while staying compatible with
duck-typed inputs used throughout the CQRS pipeline.

TypeGuard functions enable type narrowing without cast() - the preferred pattern
for FLEXT codebase to achieve zero-tolerance typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard, TypeVar, cast

from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

T = TypeVar("T")


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
    def _is_string_non_empty(value: t.GeneralValueType) -> bool:
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
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _is_dict_non_empty(value: t.GeneralValueType) -> bool:
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
    def _is_list_non_empty(value: t.GeneralValueType) -> bool:
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
        val: t.GeneralValueType,
    ) -> t.MetadataAttributeValue:
        """Normalize any value to MetadataAttributeValue.

        MetadataAttributeValue is more restrictive than t.GeneralValueType,
        so we need to normalize nested structures to flat types.

        Args:
            val: Value to normalize

        Returns:
            t.MetadataAttributeValue: Normalized value compatible with Metadata attributes

        Example:
            >>> FlextUtilitiesGuards.normalize_to_metadata_value("test")
            'test'
            >>> FlextUtilitiesGuards.normalize_to_metadata_value({"key": "value"})
            {'key': 'value'}
            >>> FlextUtilitiesGuards.normalize_to_metadata_value([1, 2, 3])
            [1, 2, 3]

        """
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        if FlextRuntime.is_dict_like(val):
            # Convert to flat dict with ScalarValue values
            result_dict: dict[str, t.ScalarValue] = {}
            dict_v = dict(val.items()) if hasattr(val, "items") else dict(val)
            for k, v in dict_v.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    result_dict[k] = v
                else:
                    result_dict[k] = str(v)
            # Return as Mapping[str, ScalarValue] - compatible with MetadataAttributeValue
            return result_dict
        if FlextRuntime.is_list_like(val):
            # Convert to list[t.MetadataAttributeValue]
            result_list: list[str | int | float | bool | None] = []
            for item in val:
                if isinstance(item, (str, int, float, bool, type(None))):
                    result_list.append(item)
                else:
                    result_list.append(str(item))
            return result_list
        return str(val)

    # =========================================================================
    # TypeGuard Functions for FLEXT Core Types
    # =========================================================================
    # These functions enable type narrowing without cast() - zero tolerance typing

    @staticmethod
    def _is_config(obj: object) -> TypeGuard[p.Configuration.Config]:
        """Check if object satisfies the Config protocol.

        Enables type narrowing for configuration objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Configuration.Config]: True if obj satisfies Config protocol

        Example:
            >>> from flext_core.utilities import u
            >>> if u.is_type(config, "config"):
            ...     # config is now typed as p.Configuration.Config
            ...     name = config.app_name

        """
        return isinstance(obj, p.Configuration.Config)

    @staticmethod
    def _is_context(obj: object) -> TypeGuard[p.Context.Ctx]:
        """Check if object satisfies the Context protocol.

        Enables type narrowing for context objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Context.Ctx]: True if obj satisfies Ctx protocol

        """
        return isinstance(obj, p.Context.Ctx)

    @staticmethod
    def _is_container(obj: object) -> TypeGuard[p.Container.DI]:
        """Check if object satisfies the Container DI protocol.

        Enables type narrowing for container objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Container.DI]: True if obj satisfies DI protocol

        """
        return isinstance(obj, p.Container.DI)

    @staticmethod
    def _is_command_bus(obj: object) -> TypeGuard[p.Application.CommandBus]:
        """Check if object satisfies the CommandBus protocol.

        Enables type narrowing for dispatcher/command bus without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Application.CommandBus]: True if obj satisfies CommandBus

        """
        return isinstance(obj, p.Application.CommandBus)

    @staticmethod
    def _is_handler(obj: object) -> TypeGuard[p.Application.Handler]:
        """Check if object satisfies the Handler protocol.

        Enables type narrowing for handler objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Application.Handler]: True if obj satisfies Handler protocol

        """
        return isinstance(obj, p.Application.Handler)

    @staticmethod
    def _is_logger(obj: object) -> TypeGuard[p.Infrastructure.Logger.StructlogLogger]:
        """Check if object satisfies the StructlogLogger protocol.

        Enables type narrowing for logger objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Infrastructure.Logger.StructlogLogger]: True if satisfies protocol

        """
        return isinstance(obj, p.Infrastructure.Logger.StructlogLogger)

    @staticmethod
    def _is_result(obj: object) -> TypeGuard[p.Foundation.Result[t.GeneralValueType]]:
        """Check if object satisfies the Result protocol.

        Enables type narrowing for result objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Foundation.Result]: True if obj satisfies Result protocol

        """
        return isinstance(obj, p.Foundation.Result)

    @staticmethod
    def _is_service(obj: object) -> TypeGuard[p.Domain.Service[t.GeneralValueType]]:
        """Check if object satisfies the Service protocol.

        Enables type narrowing for service objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Domain.Service]: True if obj satisfies Service protocol

        """
        return isinstance(obj, p.Domain.Service)

    @staticmethod
    def _is_middleware(obj: object) -> TypeGuard[p.Application.Middleware]:
        """Check if object satisfies the Middleware protocol.

        Enables type narrowing for middleware objects without cast().

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Application.Middleware]: True if obj satisfies Middleware

        """
        return isinstance(obj, p.Application.Middleware)

    # =========================================================================
    # Generic Type Guards for Collections and Sequences
    # =========================================================================

    @staticmethod
    def _is_sequence_not_str(value: object) -> TypeGuard[Sequence[object]]:
        """Check if value is Sequence and not str.

        Type guard to distinguish Sequence[str] from str in union types.
        Useful for ExclusionSpec, ErrorCodeSpec, ContainmentSpec, etc.

        Args:
            value: Value that can be str or Sequence[T]

        Returns:
            TypeGuard[Sequence[object]]: True if value is Sequence and not str

        Example:
            >>> if FlextUtilitiesGuards.is_sequence_not_str(spec):
            ...     # spec is now typed as Sequence[object]
            ...     items = list(spec)

        """
        # Runtime check needed: type checker sees str | Sequence[T] as Sequence[Unknown]
        # but runtime can be either str or Sequence[str]
        # isinstance is necessary for runtime type distinction
        return isinstance(value, Sequence) and not isinstance(value, str)

    @staticmethod
    def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
        """Check if value is Mapping[str, object].

        Type guard for mapping types used in validation.

        Args:
            value: Object to check

        Returns:
            TypeGuard[Mapping[str, object]]: True if value is Mapping

        Example:
            >>> if FlextUtilitiesGuards.is_mapping(params.kv):
            ...     # params.kv is now typed as Mapping[str, object]
            ...     for key, val in params.kv.items():

        """
        return isinstance(value, Mapping)

    @staticmethod
    def _is_callable_key_func(
        func: object,
    ) -> TypeGuard[Callable[[object], object]]:
        """Check if value is callable and can be used as key function for sorted().

        Type guard for sorted() key functions that return comparable values.
        Runtime validation ensures correctness.

        Args:
            func: Object to check

        Returns:
            TypeGuard[Callable[[object], object]]: True if func is callable

        Example:
            >>> if FlextUtilitiesGuards.is_callable_key_func(key_func):
            ...     # key_func is callable, can be used with sorted()
            ...     sorted_list = sorted(items, key=key_func)

        """
        return callable(func)

    @staticmethod
    def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
        """Check if value is Sequence.

        Type guard for sequence types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[Sequence[object]]: True if value is Sequence

        Example:
            >>> if FlextUtilitiesGuards.is_sequence(key_equals):
            ...     # key_equals is now typed as Sequence[object]
            ...     pairs = list(key_equals)

        """
        return isinstance(value, Sequence)

    @staticmethod
    def _is_str(value: object) -> TypeGuard[str]:
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
        return isinstance(value, str)

    @staticmethod
    def _is_sized(value: object) -> TypeGuard[object]:
        """Check if value has __len__ (str, bytes, Sequence, Mapping).

        Type guard for sized types that support len().

        Args:
            value: Object to check

        Returns:
            TypeGuard[object]: True if value has __len__

        Example:
            >>> if FlextUtilitiesGuards.is_sized(value):
            ...     # value has __len__, can call len()
            ...     length = len(value)

        """
        return isinstance(value, (str, bytes, Sequence, Mapping))

    @staticmethod
    def _is_list_or_tuple(
        value: object,
    ) -> TypeGuard[list[object] | tuple[object, ...]]:
        """Check if value is list or tuple.

        Type guard for list/tuple types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[list[object] | tuple[object, ...]]: True if value is list or tuple

        Example:
            >>> if FlextUtilitiesGuards.is_list_or_tuple(value):
            ...     # value is list or tuple
            ...     first = value[0]

        """
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_dict(value: object) -> TypeGuard[t.Types.ConfigurationDict]:
        """Check if value is dict.

        Type guard for dict types. Returns ConfigurationDict for type safety.

        Args:
            value: Object to check

        Returns:
            TypeGuard[ConfigurationDict]: True if value is dict

        Example:
            >>> if FlextUtilitiesGuards.is_dict(value):
            ...     # value is ConfigurationDict
            ...     keys = list(value.keys())

        """
        return isinstance(value, dict)

    @staticmethod
    def _is_list(value: object) -> TypeGuard[list[object]]:
        """Check if value is list.

        Type guard for list types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[list[object]]: True if value is list

        Example:
            >>> if FlextUtilitiesGuards.is_list(value):
            ...     # value is list
            ...     first = value[0]

        """
        return isinstance(value, list)

    @staticmethod
    def _is_int(value: object) -> TypeGuard[int]:
        """Check if value is int.

        Type guard for integer types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[int]: True if value is int

        """
        return isinstance(value, int)

    @staticmethod
    def _is_float(value: object) -> TypeGuard[float]:
        """Check if value is float.

        Type guard for float types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[float]: True if value is float

        """
        return isinstance(value, float)

    @staticmethod
    def _is_bool(value: object) -> TypeGuard[bool]:
        """Check if value is bool.

        Type guard for boolean types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bool]: True if value is bool

        """
        return isinstance(value, bool)

    @staticmethod
    def _is_none(value: object) -> TypeGuard[None]:
        """Check if value is None.

        Type guard for None type.

        Args:
            value: Object to check

        Returns:
            TypeGuard[None]: True if value is None

        """
        return value is None

    @staticmethod
    def _is_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
        """Check if value is tuple.

        Type guard for tuple types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[tuple[object, ...]]: True if value is tuple

        """
        return isinstance(value, tuple)

    @staticmethod
    def _is_bytes(value: object) -> TypeGuard[bytes]:
        """Check if value is bytes.

        Type guard for bytes types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bytes]: True if value is bytes

        """
        return isinstance(value, bytes)

    @staticmethod
    def _is_sequence_not_str_bytes(value: object) -> TypeGuard[Sequence[object]]:
        """Check if value is Sequence and not str or bytes.

        Type guard to distinguish Sequence from str/bytes in union types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[Sequence[object]]: True if value is Sequence and not str/bytes

        """
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    # =========================================================================
    # Generic is_type() Function - Unified Type Checking
    # =========================================================================

    @staticmethod
    def is_type(value: object, type_spec: str | type | tuple[type, ...]) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. This function delegates to the appropriate specific
        guard function or performs direct isinstance checks.

        Args:
            value: Object to check
            type_spec: Type specification as:
                - String name: "config", "str", "dict", "list", "sequence",
                  "mapping", "callable", "sized", "list_or_tuple", "sequence_not_str",
                  "string_non_empty", "dict_non_empty", "list_non_empty"
                - Type/class: str, dict, list, tuple, Sequence, Mapping, etc.
                - Protocol: p.Configuration.Config, p.Context.Ctx, etc.

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
            >>> u.is_type(obj, p.Configuration.Config)
            >>> u.is_type(obj, p.Context.Ctx)

        """
        # String-based type names (delegate to specific guard functions)
        if isinstance(type_spec, str):
            type_name = type_spec.lower()
            # Map string names to private method names
            method_map: dict[str, str] = {
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
                "list": "_is_list",
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
                "string_non_empty": "_is_string_non_empty",
                "dict_non_empty": "_is_dict_non_empty",
                "list_non_empty": "_is_list_non_empty",
            }
            if type_name in method_map:
                method_name = method_map[type_name]
                method = getattr(FlextUtilitiesGuards, method_name)
                # For non-empty checks, cast to GeneralValueType
                if type_name in {
                    "string_non_empty",
                    "dict_non_empty",
                    "list_non_empty",
                }:
                    return bool(method(cast("t.GeneralValueType", value)))
                return bool(method(value))
            # Unknown string type spec
            return False

        # Tuple of types check
        if isinstance(type_spec, tuple):
            return isinstance(value, type_spec)

        # Direct type/class checks
        if type(type_spec) is type:
            # Check if it's a protocol first
            if hasattr(type_spec, "__protocol_attrs__") or (
                hasattr(type_spec, "__module__")
                and "Protocol" in str(type_spec.__class__)
            ):
                # Protocol check - try isinstance first
                try:
                    return isinstance(value, type_spec)
                except TypeError:
                    # Protocol runtime check failed, try specific protocol checks
                    if type_spec == p.Configuration.Config:
                        return FlextUtilitiesGuards._is_config(value)
                    if type_spec == p.Context.Ctx:
                        return FlextUtilitiesGuards._is_context(value)
                    if type_spec == p.Container.DI:
                        return FlextUtilitiesGuards._is_container(value)
                    if type_spec == p.Application.CommandBus:
                        return FlextUtilitiesGuards._is_command_bus(value)
                    if type_spec == p.Application.Handler:
                        return FlextUtilitiesGuards._is_handler(value)
                    if type_spec == p.Infrastructure.Logger.StructlogLogger:
                        return FlextUtilitiesGuards._is_logger(value)
                    if type_spec == p.Foundation.Result:
                        return FlextUtilitiesGuards._is_result(value)
                    if type_spec == p.Domain.Service:
                        return FlextUtilitiesGuards._is_service(value)
                    if type_spec == p.Application.Middleware:
                        return FlextUtilitiesGuards._is_middleware(value)
                    return False
            # Regular type check
            return isinstance(value, type_spec)

        # Fallback: try isinstance for any other type specification
        try:
            return isinstance(value, type_spec)
        except TypeError:
            return False


__all__ = [
    "FlextUtilitiesGuards",
]
