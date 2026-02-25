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
from types import MappingProxyType
from typing import Annotated, Literal, TypeGuard, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t

# =============================================================================
# CENTRALIZED TYPE CHECK SPECIFICATIONS (Pydantic v2 Discriminated Unions)
# =============================================================================


class TypeCheckConfig(BaseModel):
    """Type check specification for Config protocol."""

    category: Literal["config"] = Field(
        default="config", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Config protocol."""
        return (
            hasattr(self.value, "app_name")
            and getattr(self.value, "app_name", None) is not None
        )


class TypeCheckContext(BaseModel):
    """Type check specification for Context protocol."""

    category: Literal["context"] = Field(
        default="context", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Context protocol."""
        return hasattr(self.value, "request_id") or hasattr(
            self.value, "correlation_id"
        )


class TypeCheckContainer(BaseModel):
    """Type check specification for DI/Container protocol."""

    category: Literal["container"] = Field(
        default="container", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies DI protocol."""
        return hasattr(self.value, "register") and callable(
            getattr(self.value, "register", None)
        )


class TypeCheckCommandBus(BaseModel):
    """Type check specification for CommandBus protocol."""

    category: Literal["command_bus"] = Field(
        default="command_bus", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies CommandBus protocol."""
        return hasattr(self.value, "dispatch") and callable(
            getattr(self.value, "dispatch", None)
        )


class TypeCheckHandler(BaseModel):
    """Type check specification for Handler protocol."""

    category: Literal["handler"] = Field(
        default="handler", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Handler protocol."""
        return hasattr(self.value, "handle") and callable(
            getattr(self.value, "handle", None)
        )


class TypeCheckLogger(BaseModel):
    """Type check specification for StructlogLogger protocol."""

    category: Literal["logger"] = Field(
        default="logger", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies StructlogLogger protocol."""
        return (
            hasattr(self.value, "debug")
            and hasattr(self.value, "info")
            and hasattr(self.value, "warning")
            and hasattr(self.value, "error")
            and hasattr(self.value, "exception")
        )


class TypeCheckResult(BaseModel):
    """Type check specification for Result protocol."""

    category: Literal["result"] = Field(
        default="result", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Result protocol."""
        return (
            hasattr(self.value, "is_success")
            and hasattr(self.value, "is_failure")
            and hasattr(self.value, "value")
            and hasattr(self.value, "error")
        )


class TypeCheckService(BaseModel):
    """Type check specification for Service protocol."""

    category: Literal["service"] = Field(
        default="service", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Service protocol."""
        return hasattr(self.value, "run") and callable(getattr(self.value, "run", None))


class TypeCheckMiddleware(BaseModel):
    """Type check specification for Middleware protocol."""

    category: Literal["middleware"] = Field(
        default="middleware", description="Protocol category discriminator"
    )
    value: object = Field(description="Value to check")

    model_config = ConfigDict(
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    def matches(self) -> bool:
        """Validate if value satisfies Middleware protocol."""
        return hasattr(self.value, "before_dispatch") and callable(
            getattr(self.value, "before_dispatch", None),
        )


TypeCheckSpec = Annotated[
    TypeCheckConfig
    | TypeCheckContext
    | TypeCheckContainer
    | TypeCheckCommandBus
    | TypeCheckHandler
    | TypeCheckLogger
    | TypeCheckResult
    | TypeCheckService
    | TypeCheckMiddleware,
    Discriminator("category"),
]


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
    def is_string_non_empty(value: t.GuardInputValue) -> TypeGuard[str]:
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
        return isinstance(value, Mapping) and bool(value)

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
        return (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and bool(value)
        )

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
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        if isinstance(val, Mapping):
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
                if value is None or isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        datetime,
                    ),
                ):
                    result_dict[key] = value
                else:
                    result_dict[key] = str(value)
            return result_dict
        if isinstance(val, list):
            # Convert to list[t.MetadataAttributeValue]
            # Type narrowing: is_list_like returns TypeGuard[Sequence[t.GuardInputValue]]
            val_sequence = val  # type narrowing via TypeGuard
            result_list: t.GeneralListValue = []
            # TypeGuard already narrows to Sequence - no extra check needed
            # Exclude str/bytes from iteration
            if not isinstance(val_sequence, (str, bytes)):
                for item in val_sequence:
                    # Explicit type annotation for loop variable
                    list_item: t.GuardInputValue = item
                    if list_item is None or isinstance(
                        list_item,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    ):
                        result_list.append(list_item)
                    else:
                        result_list.append(str(list_item))
            return result_list
        return str(val)

    @staticmethod
    def is_flexible_value(value: object) -> bool:
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
            v: object
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
        if isinstance(value, Mapping):
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
            item: t.GuardInputValue
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
    def is_context(obj: object) -> TypeGuard[p.Context]:
        """Check if object satisfies the Context protocol.

        Enables type narrowing for context objects without .

        Args:
            obj: Object to check

        Returns:
            TypeGuard[p.Context]: True if obj satisfies Ctx protocol

        """
        return (
            hasattr(obj, "clone")
            and callable(getattr(obj, "clone", None))
            and hasattr(obj, "set")
            and callable(getattr(obj, "set", None))
            and hasattr(obj, "get")
            and callable(getattr(obj, "get", None))
        )

    @staticmethod
    def _is_context(obj: object) -> TypeGuard[p.Context]:
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
        return isinstance(value, (list, tuple, range)) and not isinstance(value, str)

    @staticmethod
    def is_mapping(
        value: t.GuardInputValue,
    ) -> TypeGuard[Mapping[str, t.GuardInputValue]]:
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
        return isinstance(value, Mapping)

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
        return isinstance(value, (list, tuple, range))

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
        return isinstance(value, str)

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
        return isinstance(value, dict)

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
        return isinstance(value, Mapping)

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
        return isinstance(value, int)

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
        return isinstance(value, (list, tuple))

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
        return isinstance(value, (str, bytes, list, tuple, dict)) or (
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
        return isinstance(value, list)

    @staticmethod
    def _is_float(value: t.GuardInputValue) -> TypeGuard[float]:
        """Check if value is float.

        Type guard for float types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[float]: True if value is float

        """
        return isinstance(value, float)

    @staticmethod
    def _is_bool(value: t.GuardInputValue) -> TypeGuard[bool]:
        """Check if value is bool.

        Type guard for boolean types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bool]: True if value is bool

        """
        return isinstance(value, bool)

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
        return isinstance(value, tuple)

    @staticmethod
    def _is_bytes(value: t.GuardInputValue) -> TypeGuard[bytes]:
        """Check if value is bytes.

        Type guard for bytes types.

        Args:
            value: Object to check

        Returns:
            TypeGuard[bytes]: True if value is bytes

        """
        return isinstance(value, bytes)

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

    # Module-level immutable maps to avoid RUF012 (mutable class attribute)
    _PROTOCOL_CATEGORY_MAP: Mapping[str, str] = MappingProxyType({
        "config": "config",
        "context": "context",
        "container": "container",
        "command_bus": "command_bus",
        "handler": "handler",
        "logger": "logger",
        "result": "result",
        "service": "service",
        "middleware": "middleware",
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
    def is_type(value: object, type_spec: str | type | tuple[type, ...]) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. Uses centralized Pydantic v2 discriminated union models
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

            # Protocol checks via centralized TypeCheckSpec models
            if type_name in FlextUtilitiesGuards._PROTOCOL_CATEGORY_MAP:
                return FlextUtilitiesGuards._check_protocol_via_model(value, type_name)

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
    def _check_protocol_via_model(value: object, type_name: str) -> bool:
        """Check protocol via centralized Pydantic v2 model.

        Creates the appropriate TypeCheckSpec variant and validates the value.

        Args:
            value: Value to validate against protocol
            type_name: Protocol name from _PROTOCOL_CATEGORY_MAP

        Returns:
            bool: True if value satisfies the protocol

        """
        category = FlextUtilitiesGuards._PROTOCOL_CATEGORY_MAP[type_name]
        try:
            spec: TypeCheckSpec | None = None
            if category == "config":
                spec = TypeCheckConfig(value=value)
            elif category == "context":
                return FlextUtilitiesGuards.is_context(value)
            elif category == "container":
                spec = TypeCheckContainer(value=value)
            elif category == "command_bus":
                spec = TypeCheckCommandBus(value=value)
            elif category == "handler":
                spec = TypeCheckHandler(value=value)
            elif category == "logger":
                spec = TypeCheckLogger(value=value)
            elif category == "result":
                spec = TypeCheckResult(value=value)
            elif category == "service":
                spec = TypeCheckService(value=value)
            elif category == "middleware":
                spec = TypeCheckMiddleware(value=value)
            return spec.matches() if spec else False
        except Exception:
            return False

    @staticmethod
    def _check_protocol_type(value: object, type_spec: type) -> bool:
        """Check protocol by type via centralized models.

        Maps protocol types to appropriate TypeCheckSpec variants.

        Args:
            value: Value to validate against protocol
            type_spec: Protocol type (p.Config, p.Context, etc.)

        Returns:
            bool: True if value satisfies the protocol

        """
        try:
            spec: TypeCheckSpec | None = None
            if type_spec == p.Config:
                spec = TypeCheckConfig(value=value)
            elif type_spec == p.Context:
                spec = TypeCheckContext(value=value)
            elif type_spec == p.DI:
                spec = TypeCheckContainer(value=value)
            elif type_spec == p.CommandBus:
                spec = TypeCheckCommandBus(value=value)
            elif type_spec == p.Handler:
                spec = TypeCheckHandler(value=value)
            elif type_spec == p.Log.StructlogLogger:
                spec = TypeCheckLogger(value=value)
            elif type_spec == p.Result:
                spec = TypeCheckResult(value=value)
            elif type_spec == p.Service:
                spec = TypeCheckService(value=value)
            elif type_spec == p.Middleware:
                spec = TypeCheckMiddleware(value=value)
            return spec.matches() if spec else False
        except Exception:
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
            isinstance(value, dict)
            or (hasattr(value, "keys") and hasattr(value, "__getitem__"))
        ) and FlextUtilitiesGuards.is_configuration_mapping(
            value,
        ):
            return value
        return None

    @staticmethod
    def _guard_check_type(
        value: object,
        condition: type | tuple[type, ...],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        type_match = (
            value.__class__ is condition
            if isinstance(condition, type)
            else any(value.__class__ is c for c in condition)
        )
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
                desc = getattr(condition, "description", "validation")
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
            if value.__class__ is str:
                return None
            return error_msg or f"{context_name} must be string"
        if shortcut_lower == "int":
            if value.__class__ is int and value.__class__ is not bool:
                return None
            return error_msg or f"{context_name} must be int"
        if shortcut_lower == "float":
            if value.__class__ in {int, float} and value.__class__ is not bool:
                return None
            return error_msg or f"{context_name} must be float"
        if shortcut_lower == "bool":
            if value.__class__ is bool:
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
                    func_name = getattr(condition, "__name__", "custom")
                    return f"{context_name} failed {func_name} check"
                return error_msg
        except Exception as e:
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
            typed_value: t.ConfigMapValue = cast("t.ConfigMapValue", value)
            return FlextUtilitiesGuards._guard_check_validator(
                typed_value,
                condition,
                context_name,
                error_msg,
            )
        if isinstance(condition, str):
            typed_value = cast("t.ConfigMapValue", value)
            return FlextUtilitiesGuards._guard_check_string_shortcut(
                typed_value,
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
        try:
            if isinstance(validator, type | tuple):
                if isinstance(value, validator):
                    return value if return_value else True
            elif callable(validator):
                if validator(value):
                    return value if return_value else True
            elif value:
                return value if return_value else True
            return default
        except Exception:
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
