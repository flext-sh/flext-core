"""String parsing helpers for deterministic CQRS utility flows.

These helpers centralize delimiter handling, whitespace normalization, and
escaped character parsing so dispatcher handlers and services receive
predictable ``r`` outcomes instead of ad-hoc string handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import overload

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import FlextRuntime, c, m, p, r, t
from flext_core._utilities import FlextUtilitiesGuards, FlextUtilitiesModel


class FlextUtilitiesParser:
    r"""Parse delimited and structured strings with predictable results.

    The parser consolidates delimiter handling, escape-aware splits, and
    normalization routines behind ``r`` so callers can compose
    parsing logic in dispatcher pipelines without manual error handling.

    Examples:
        >>> parser = FlextUtilitiesParser()
        >>> parser.parse_delimited("a, b, c", ",").value
        ['a', 'b', 'c']
        >>> parser.split_on_char_with_escape(
        ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\\\,dc=com", ",", "\\\\"
        ... ).value
        ['cn=REDACTED_LDAP_BIND_PASSWORD', 'dc=com']

    """

    PATTERN_TUPLE_MIN_LENGTH: int = c.Processing.PATTERN_TUPLE_MIN_LENGTH
    PATTERN_TUPLE_MAX_LENGTH: int = c.Processing.PATTERN_TUPLE_MAX_LENGTH
    TUPLE_LENGTH_2: int = 2
    TUPLE_LENGTH_3: int = 3

    def __init__(self) -> None:
        """Initialize string parser with logging."""
        super().__init__()
        self._parser_log = FlextRuntime.get_logger(__name__)

    @staticmethod
    def _coerce_to_bool(value: t.NormalizedValue) -> r[bool]:
        """Coerce value to bool. Returns None if not coercible."""
        if FlextUtilitiesGuards.is_type(value, str):
            normalized_val = FlextUtilitiesParser._parse_normalize_str(
                value,
                case="lower",
            )
            if normalized_val in {"true", "1", "yes", "on"}:
                return r[bool].ok(value=True)
            if normalized_val in {"false", "0", "no", "off"}:
                return r[bool].ok(False)
            return r[bool].fail(f"Cannot coerce '{value}' to bool")
        return r[bool].ok(bool(value))

    @staticmethod
    def _coerce_to_float(value: t.NormalizedValue) -> r[float]:
        """Coerce value to float. Returns None if not coercible."""
        if isinstance(value, (str, int)):
            return r[float].create_from_callable(
                lambda: float(str(value)),
                error_code="FLOAT_COERCE_ERROR",
            )
        return r[float].fail(
            f"Cannot coerce {value.__class__.__name__} to float",
            error_code="FLOAT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_int(value: t.NormalizedValue) -> r[int]:
        """Coerce value to int. Returns None if not coercible."""
        if isinstance(value, (str, float)):
            return r[int].create_from_callable(
                lambda: int(float(str(value))),
                error_code="INT_COERCE_ERROR",
            )
        return r[int].fail(
            f"Cannot coerce {value.__class__.__name__} to int",
            error_code="INT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_str(value: t.NormalizedValue) -> r[str]:
        """Coerce value to string - returns r[str]."""
        return r[str].ok(str(value))

    @staticmethod
    def _convert_to_bool(value: t.NormalizedValue, *, default: bool) -> bool:
        """Convert value to bool with fallback."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = FlextUtilitiesParser._parse_normalize_str(value, case="lower")
            return normalized in {"true", "1", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _convert_to_float(value: t.NormalizedValue, *, default: float) -> float:
        """Convert value to float with fallback."""
        if isinstance(value, float):
            return value
        if isinstance(value, int | str) and (not isinstance(value, bool)):
            return (
                r[float].create_from_callable(lambda: float(value)).unwrap_or(default)
            )
        return default

    @staticmethod
    def _convert_to_int(value: t.NormalizedValue, *, default: int) -> int:
        """Convert value to int with fallback."""
        if isinstance(value, int) and (not isinstance(value, bool)):
            return value
        if isinstance(value, str):
            return r[int].create_from_callable(lambda: int(value)).unwrap_or(default)
        if isinstance(value, float):
            return int(value)
        return default

    @staticmethod
    def _convert_to_str(value: t.NormalizedValue, *, default: str) -> str:
        """Convert value to str with fallback."""
        if isinstance(value, str):
            return value
        if value is None:
            return default
        return r[str].create_from_callable(lambda: str(value)).unwrap_or(default)

    @staticmethod
    def _extract_key_from_attributes(
        obj: t.TypeHintSpecifier | t.NormalizedValue,
    ) -> r[str]:
        """Extract key from object attributes (Strategy 3).

        Args:
            obj: Object to extract key from.

        Returns:
            String key if found, None otherwise.

        """
        for attr in ("name", "id"):
            obj_vars: dict[str, t.NormalizedValue] = (
                vars(obj) if hasattr(obj, "__dict__") else {}
            )
            if attr not in obj_vars:
                continue
            attr_value = obj_vars[attr]
            if isinstance(attr_value, str):
                return r[str].ok(attr_value)
        return r[str].fail("No key attribute found")

    @staticmethod
    def _extract_key_from_mapping(
        obj: Mapping[str, t.NormalizedValue] | t.NormalizedValue,
    ) -> r[str]:
        """Extract key from mapping object (Strategy 2).

        Args:
            obj: Mapping object to extract key from.

        Returns:
            String key if found, None otherwise.

        """
        if not FlextUtilitiesGuards.is_mapping(obj):
            return r[str].fail("Object is not a valid mapping")
        for key in ("name", "id"):
            if key in obj:
                map_value = obj[key]
                if isinstance(map_value, str):
                    return r[str].ok(map_value)
        return r[str].fail("No key field found in mapping")

    @staticmethod
    def _extract_key_from_str_conversion(
        obj: t.TypeHintSpecifier | t.NormalizedValue,
    ) -> r[str]:
        """Extract key from string conversion (Strategy 5).

        Args:
            obj: Object to convert to string.

        Returns:
            String key if valid, None otherwise.

        """
        str_result = r[str].create_from_callable(lambda: str(obj))
        if str_result.is_failure:
            return r[str].fail(str_result.error or "String conversion failed")
        str_repr = str_result.value
        obj_class_name: str = type(obj).__name__
        if str_repr and str_repr != f"<{obj_class_name} object>":
            return r[str].ok(str_repr)
        return r[str].fail("String conversion did not yield a usable key")

    @staticmethod
    def _is_primitive_type(target: type) -> bool:
        """Check if target is a primitive type."""
        return target in {int, float, str, bool}

    @staticmethod
    def _parse_enum[T: StrEnum](
        value: str,
        target: type[T],
        *,
        case_insensitive: bool,
    ) -> r[T]:
        """Parse StrEnum with optional case-insensitivity. Returns None if not enum."""
        if StrEnum not in target.__mro__:
            return r[T].fail(
                f"Target {target.__name__} is not a StrEnum",
                error_code="TARGET_NOT_ENUM",
            )
        members_proxy: Mapping[str, T] = target.__members__
        members: Mapping[str, T] = dict(members_proxy)
        if case_insensitive:
            for member_name, member_value in members.items():
                name_matches = FlextUtilitiesParser._parse_normalize_compare(
                    member_name,
                    value,
                )
                value_attr = getattr(member_value, "value", None)
                value_matches = (
                    value_attr is not None
                    and FlextUtilitiesParser._parse_normalize_compare(value_attr, value)
                )
                if name_matches or value_matches:
                    return r[T].ok(member_value)
        if value in members:
            return r[T].ok(members[value])
        for member_instance in members.values():
            member_val = getattr(member_instance, "value", None)
            if member_val == value:
                return r[T].ok(member_instance)
        target_name = target.__name__ if hasattr(target, "__name__") else "Unknown"
        return r[T].fail(f"Cannot parse '{value}' as {target_name}")

    @staticmethod
    def _parse_find_first[T](items: list[T], predicate: Callable[[T], bool]) -> r[T]:
        """Find first item matching predicate (avoids circular import)."""
        for item in items:
            item_typed: T = item
            if predicate(item_typed):
                return r[T].ok(item_typed)
        return r[T].fail("No matching item found")

    @staticmethod
    def _parse_get_attr(
        obj: t.ValueOrModel,
        attr: str,
        default: t.NormalizedValue = None,
    ) -> t.NormalizedValue:
        """Get attribute safely (avoids circular import with u.get)."""
        obj_vars: dict[str, t.NormalizedValue] = (
            vars(obj) if hasattr(obj, "__dict__") else {}
        )
        if attr not in obj_vars:
            return default
        attr_value = obj_vars[attr]
        return (
            attr_value
            if FlextUtilitiesGuards.is_container(attr_value)
            else str(attr_value)
        )

    @staticmethod
    def _parse_model[TModel: BaseModel](
        value: t.NormalizedValue,
        target: type[TModel],
        field_prefix: str,
        *,
        strict: bool,
    ) -> r[TModel]:
        """Parse Pydantic BaseModel. Returns None if not model."""
        if not FlextUtilitiesGuards.is_mapping(value):
            return r[TModel].fail(
                f"{field_prefix}Expected dict for model, got {value.__class__.__name__}",
            )
        value_dict_data: dict[str, t.NormalizedValue] = {
            str(k): v for k, v in value.items()
        }
        try:
            return r[TModel].ok(target.model_validate(value_dict_data, strict=strict))
        except (ValidationError, TypeError, ValueError) as exc:
            return r[TModel].fail(f"Model parse failed: {exc}")

    @staticmethod
    def _parse_normalize_compare(a: t.NormalizedValue, b: t.NormalizedValue) -> bool:
        """Case-insensitive string comparison (avoids circular import)."""
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        return a.lower() == b.lower()

    @staticmethod
    def _parse_normalize_str(value: t.NormalizedValue, *, case: str = "lower") -> str:
        """Normalize string value (avoids circular import with u.normalize)."""
        if not isinstance(value, str):
            return str(value)
        value_str: str = value
        if case == "lower":
            return value_str.lower()
        if case == "upper":
            return value_str.upper()
        return value_str

    @staticmethod
    def _parse_result_error[T](result: r[T], default: str = "") -> str:
        """Extract error from result (avoids circular import with u.err)."""
        return result.fold(
            on_failure=lambda e: e or default,
            on_success=lambda _: default,
        )

    @staticmethod
    def _parse_try_direct[T](
        value: t.NormalizedValue,
        target: type[T],
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Helper: Try direct type call."""
        if target is object or str(target) == "typing.Any":
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot construct '{getattr(target, '__name__', 'Any')}' type directly",
            )
        try:
            parsed_value = TypeAdapter(target).validate_python(value)
            return r[T].ok(parsed_value)
        except (ValidationError, TypeError, ValueError) as e:
            target_name = target.__name__ if hasattr(target, "__name__") else "type"
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot parse {value.__class__.__name__} to {target_name}: {e}",
            )

    @staticmethod
    def _parse_try_enum[T](
        value: t.NormalizedValue,
        target: type[T],
        *,
        case_insensitive: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Helper: Try enum parsing, return None if not enum."""
        if not issubclass(target, StrEnum):
            return r[T].fail(
                f"{field_prefix}Target is not a StrEnum",
                error_code="TARGET_NOT_ENUM",
            )
        members = TypeAdapter(dict[str, T]).validate_python(target.__members__)
        value_str = str(value)
        if case_insensitive:
            value_lower = value_str.lower()
            for member_name, member_value in members.items():
                if member_name.lower() == value_lower:
                    return r[T].ok(member_value)
                member_val = getattr(member_value, "value", None)
                if member_val is not None and str(member_val).lower() == value_lower:
                    return r[T].ok(member_value)
        else:
            if value_str in members:
                return r[T].ok(members[value_str])
            for member_value in members.values():
                member_val = getattr(member_value, "value", None)
                if member_val == value_str:
                    return r[T].ok(member_value)
        target_name = target.__name__ if hasattr(target, "__name__") else "Unknown"
        error_msg = f"Cannot parse '{value_str}' as {target_name}"
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            f"{field_prefix}{error_msg}",
        )

    @staticmethod
    def _parse_try_model[T](
        value: t.NormalizedValue,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
    ) -> r[T]:
        """Helper: Try model parsing, return None if not model."""
        if not issubclass(target, BaseModel):
            return r[T].fail(
                f"{field_prefix}Target is not a BaseModel",
                error_code="TARGET_NOT_MODEL",
            )
        model_result = FlextUtilitiesParser._parse_model(
            value,
            target,
            field_prefix,
            strict=strict,
        )
        if model_result.is_success:
            validated_model = TypeAdapter(target).validate_python(model_result.value)
            return r[T].ok(validated_model)
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            FlextUtilitiesParser._parse_result_error(model_result, ""),
        )

    @staticmethod
    def _parse_try_primitive(
        value: t.NormalizedValue,
        target: type,
        default: float | str | bool | None,
        default_factory: Callable[[], t.Numeric | str | bool] | None,
        field_prefix: str,
    ) -> r[t.Primitives]:
        """Helper: Try primitive coercion."""
        if not FlextUtilitiesParser._is_primitive_type(target):
            return r[t.Primitives].fail(
                f"{field_prefix}Target is not primitive",
                error_code="TARGET_NOT_PRIMITIVE",
            )
        try:
            if target is int:
                int_result = FlextUtilitiesParser._coerce_to_int(value)
                if int_result.is_failure:
                    return r[t.Primitives].fail(int_result.error)
                int_val = int_result.value
                return r[t.Primitives].ok(int_val)
            if target is float:
                float_result = FlextUtilitiesParser._coerce_to_float(value)
                if float_result.is_failure:
                    return r[t.Primitives].fail(float_result.error)
                float_val = float_result.value
                if not isinstance(float_val, float):
                    return r[t.Primitives].fail(
                        f"Expected float, got {type(float_val).__name__}",
                    )
                return r[t.Primitives].ok(float_val)
            if target is str:
                str_result = FlextUtilitiesParser._coerce_to_str(value)
                if str_result.is_failure:
                    return r[t.Primitives].fail(str_result.error)
                str_val = str_result.value
                return r[t.Primitives].ok(str_val)
            if target is bool:
                bool_result = FlextUtilitiesParser._coerce_to_bool(value)
                if bool_result.is_failure:
                    return r[t.Primitives].fail(bool_result.error)
                bool_val = bool_result.value
                return r[t.Primitives].ok(bool_val)
        except (ValueError, TypeError) as e:
            target_name = getattr(target, "__name__", "type")
            if (
                target is int
                and isinstance(default, int)
                and (not isinstance(default, bool))
            ):
                return r[t.Primitives].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is float and isinstance(default, float):
                return r[t.Primitives].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is str and isinstance(default, str):
                return r[t.Primitives].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is bool and isinstance(default, bool):
                return r[t.Primitives].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
        return r[t.Primitives].fail(
            f"{field_prefix}Unsupported primitive target: {target.__name__}",
            error_code="UNSUPPORTED_PRIMITIVE_TARGET",
        )

    @staticmethod
    def _parse_with_default[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        error_msg: str,
    ) -> r[T]:
        """Return default or error for parse failures."""
        if default is not None:
            return r[T].ok(default)
        if default_factory is not None:
            return r[T].ok(default_factory())
        return r[T].fail(error_msg)

    @staticmethod
    def _safe_text_length(text: t.NormalizedValue) -> str | int:
        """Safely get text length for logging."""
        if isinstance(text, str | bytes):
            text_length_result = r[int].create_from_callable(lambda: len(text))
            if text_length_result.is_success:
                return text_length_result.value
            return c.Mixins.IDENTIFIER_UNKNOWN
        text_adapter: TypeAdapter[str | bytes] = TypeAdapter(str | bytes)
        try:
            text_value: str | bytes = text_adapter.validate_python(text)
        except ValidationError:
            return c.Mixins.IDENTIFIER_UNKNOWN
        text_length_result = r[int].create_from_callable(lambda: len(text_value))
        if text_length_result.is_success:
            return text_length_result.value
        return c.Mixins.IDENTIFIER_UNKNOWN

    @staticmethod
    def _validate_split_inputs(split_char: str, escape_char: str) -> r[bool]:
        """Validate inputs for split operation.

        Args:
            split_char: Character to split on
            escape_char: Escape character

        Returns:
            r[bool]: True if valid, failure with error message

        """
        if not split_char:
            return r[bool].fail("Split character cannot be empty")
        if not escape_char:
            return r[bool].fail("Escape character cannot be empty")
        if split_char == escape_char:
            return r[bool].fail(
                "Split character and escape character cannot be the same",
            )
        return r[bool].ok(value=True)

    @staticmethod
    def conv_int(value: t.NormalizedValue, *, default: int = 0) -> int:
        """Convert to int (builder: conv().int()).

        Mnemonic: conv = convert, int = integer

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            int: Converted integer

        """
        return FlextUtilitiesParser.convert(value, int, default)

    @staticmethod
    def conv_str(value: t.NormalizedValue, *, default: str = "") -> str:
        """Convert to string (builder: conv().str()).

        Mnemonic: conv = convert, str = string

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            str: Converted string

        """
        if value is None:
            return default
        if isinstance(value, str):
            return value
        return r[str].create_from_callable(lambda: str(value)).unwrap_or(default)

    @staticmethod
    def conv_str_list(
        value: t.NormalizedValue,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert to str_list (builder: conv().str_list()).

        Mnemonic: conv = convert, str_list = list[str]

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted list

        """
        if default is None:
            default = list[str]()
        if value is None:
            return default
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value] if value else default
        if isinstance(value, (tuple, set, frozenset)):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def conv_str_list_safe(value: t.NormalizedValue | None) -> list[str]:
        """Safe str_list conversion.

        Mnemonic: conv_str_list_safe = convert + safe mode

        Args:
            value: Value to convert (can be None)

        Returns:
            list[str]: Converted list or []

        """
        if value is None:
            return []
        return FlextUtilitiesParser.conv_str_list(value, default=[])

    @staticmethod
    def conv_str_list_truthy(
        value: t.NormalizedValue | None,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert to str_list and filter truthy.

        Mnemonic: conv_str_list_truthy = convert + filter truthy

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted and filtered list

        """
        if value is None:
            return list(default) if default is not None else []
        result = FlextUtilitiesParser.conv_str_list(value, default=default)
        return [item for item in result if item]

    @overload
    @staticmethod
    def convert(
        value: t.NormalizedValue,
        target_type: type[bool],
        default: bool,
    ) -> bool: ...

    @overload
    @staticmethod
    def convert(
        value: t.NormalizedValue,
        target_type: type[int],
        default: int,
    ) -> int: ...

    @overload
    @staticmethod
    def convert(
        value: t.NormalizedValue,
        target_type: type[float],
        default: float,
    ) -> float: ...

    @overload
    @staticmethod
    def convert(
        value: t.NormalizedValue,
        target_type: type[str],
        default: str,
    ) -> str: ...

    @staticmethod
    def convert(
        value: t.NormalizedValue,
        target_type: type[t.Numeric | str | bool],
        default: t.Numeric | str | bool,
    ) -> t.Numeric | str | bool:
        """Unified type conversion with safe fallback.

        Automatically handles common type conversions (int, str, float, bool) with
        safe fallback to default value on conversion failure.

        Args:
            value: Value to convert
            target_type: Target type (int, str, float, bool)
            default: Default value to return on conversion failure

        Returns:
            Converted value or default

        Example:
            # Convert to int
            result = FlextUtilitiesParser.convert("123", int, 0)
            # → 123

            # Convert to int (invalid)
            result = FlextUtilitiesParser.convert("invalid", int, 0)
            # → 0

            # Convert to float
            result = FlextUtilitiesParser.convert("3.14", float, 0.0)
            # → 3.14

        """
        if (
            target_type is int
            and isinstance(value, int)
            and (not isinstance(value, bool))
        ):
            return value
        if target_type is float and isinstance(value, float):
            return value
        if target_type is str and isinstance(value, str):
            return value
        if target_type is bool and isinstance(value, bool):
            return value
        if (
            target_type is int
            and isinstance(default, int)
            and (not isinstance(default, bool))
        ):
            return FlextUtilitiesParser._convert_to_int(value, default=default)
        if target_type is float and isinstance(default, float):
            return FlextUtilitiesParser._convert_to_float(value, default=default)
        if target_type is str and isinstance(default, str):
            return FlextUtilitiesParser._convert_to_str(value, default=default)
        if target_type is bool and isinstance(default, bool):
            return FlextUtilitiesParser._convert_to_bool(value, default=default)
        return default

    @staticmethod
    def norm_in(
        value: str,
        items: p.HasModelDump | list[str] | t.ConfigMap | Mapping[str, t.Container],
        *,
        case: str | None = None,
    ) -> bool:
        """Normalized membership check (builder: norm().in_()).

        Mnemonic: norm = normalize, in_ = membership check

        Canonical path: pass t.ConfigMap or any Pydantic BaseModel (p.HasModelDump).
        Legacy path: raw Mapping or list[str] — emits DeprecationWarning.

        Args:
            value: Value to check
            items: Items to check against (ConfigMap/BaseModel preferred; raw Mapping deprecated)
            case: Case normalization

        Returns:
            bool: True if normalized value in normalized items

        """
        items_to_check: list[str]
        if isinstance(items, t.ConfigMap):
            items_to_check = [str(k) for k in items.root]
        elif isinstance(items, p.HasModelDump):
            items_to_check = list(items.model_dump().keys())
        elif isinstance(items, Mapping):
            warnings.warn(
                "Passing raw Mapping to norm_in() is deprecated. "
                "Use t.ConfigMap or a Pydantic BaseModel (p.HasModelDump) instead. "
                "Will be removed in v0.13.",
                DeprecationWarning,
                stacklevel=2,
            )
            items_to_check = [str(k) for k in items]
        else:
            items_to_check = items
        normalized_value = FlextUtilitiesParser.norm_str(value, case=case or "lower")
        normalized_result = [
            FlextUtilitiesParser.norm_str(item, case=case or "lower")
            for item in items_to_check
        ]
        return normalized_value in normalized_result

    @staticmethod
    def norm_join(items: list[str], *, case: str | None = None, sep: str = " ") -> str:
        """Normalize and join (builder: norm().join()).

        Mnemonic: norm = normalize, join = string join

        Args:
            items: Items to normalize and join
            case: Case normalization
            sep: Separator

        Returns:
            str: Normalized and joined string

        """
        if case:
            normalized = [FlextUtilitiesParser.norm_str(v, case=case) for v in items]
        else:
            normalized = items
        return sep.join(normalized)

    @staticmethod
    def norm_list(
        items: t.ConfigModelInput | list[str] | Mapping[str, t.NormalizedValue],
        *,
        case: str | None = None,
        filter_truthy: bool = False,
        to_set: bool = False,
    ) -> list[str] | set[str] | dict[str, str]:
        """Normalize list/dict (builder: norm().list())."""
        if isinstance(items, t.ConfigMap):
            dict_items: Mapping[str, t.ValueOrModel] = items.root
            if filter_truthy:
                dict_items = {k: v for k, v in dict_items.items() if v}
            return {
                k: FlextUtilitiesParser.norm_str(
                    str(v) if isinstance(v, BaseModel) else v,
                    case=case,
                )
                for k, v in dict_items.items()
            }
        if isinstance(items, BaseModel):
            dumped: dict[str, t.RuntimeAtomic] = {
                str(k): FlextRuntime.normalize_to_container(v)
                for k, v in items.model_dump().items()
            }
            if filter_truthy:
                dumped = {k: v for k, v in dumped.items() if v}
            return {
                k: FlextUtilitiesParser.norm_str(
                    str(v) if isinstance(v, BaseModel) else v,
                    case=case,
                )
                for k, v in dumped.items()
            }
        if isinstance(items, Mapping):
            warnings.warn(
                "Passing raw Mapping to norm_list() is deprecated. "
                "Use t.ConfigMap or a Pydantic BaseModel instead. "
                "Will be removed in v0.13.",
                DeprecationWarning,
                stacklevel=2,
            )
            dict_items_raw: dict[str, t.RuntimeAtomic] = {
                str(k): FlextRuntime.normalize_to_container(v) for k, v in items.items()
            }
            if filter_truthy:
                dict_items_raw = {k: v for k, v in dict_items_raw.items() if v}
            return {
                k: FlextUtilitiesParser.norm_str(
                    str(v) if isinstance(v, BaseModel) else v,
                    case=case,
                )
                for k, v in dict_items_raw.items()
            }
        list_items: list[str] = items
        if filter_truthy:
            list_items = [item for item in list_items if item]
        normalized = [
            FlextUtilitiesParser.norm_str(item, case=case) for item in list_items
        ]
        if to_set:
            return set(normalized)
        return normalized

    @staticmethod
    def norm_str(
        value: t.NormalizedValue,
        *,
        case: str | None = None,
        default: str = "",
    ) -> str:
        """Normalize string (builder: norm().str()).

        Mnemonic: norm = normalize, str = string

        Args:
            value: Value to normalize
            case: Case normalization ("lower", "upper", "title")
            default: Default if None

        Returns:
            str: Normalized string

        """
        str_value = FlextUtilitiesParser.conv_str(value, default=default)
        if case:
            return FlextUtilitiesParser._parse_normalize_str(str_value, case=case)
        return str_value

    @staticmethod
    def parse[T](
        value: t.NormalizedValue,
        target: type[T],
        *,
        strict: bool = False,
        coerce: bool = True,
        case_insensitive: bool = False,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        field_name: str | None = None,
    ) -> r[T]:
        """Universal type parser supporting enums, models, and primitives.

        Parsing order: enum → model → primitive coercion → direct type call.

        Args:
            value: The value to parse.
            target: Target type to parse into.
            strict: If True, disable type coercion (exact match only).
            coerce: If True (default), allow type coercion.
            case_insensitive: For enums, match case-insensitively.
            default: Default value to return on parse failure.
            default_factory: Callable to create default on failure.
            field_name: Field name for error messages.

        Returns:
            r[T]: Ok(parsed_value) or Fail with error message.

        Examples:
            >>> result = FlextUtilitiesParser.parse("ACTIVE", Status)
            >>> result = FlextUtilitiesParser.parse("42", int)  # Ok(42)
            >>> result = FlextUtilitiesParser.parse("invalid", int, default=c.ZERO)

        """
        field_prefix = f"{field_name}: " if field_name else ""
        if value is None:
            if default is not None:
                return r[T].ok(default)
            if default_factory is not None:
                return r[T].ok(default_factory())
            return r[T].fail(field_prefix or "Value is None")
        if isinstance(value, target):
            return r[T].ok(value)
        if issubclass(target, StrEnum):
            return FlextUtilitiesParser._parse_try_enum(
                value,
                target,
                case_insensitive=case_insensitive,
                default=default,
                default_factory=default_factory,
                field_prefix=field_prefix,
            )
        if issubclass(target, BaseModel):
            return FlextUtilitiesParser._parse_try_model(
                value,
                target,
                field_prefix,
                strict=strict,
                default=default,
                default_factory=default_factory,
            )
        if FlextUtilitiesParser._is_primitive_type(target):
            primitive_result = FlextUtilitiesParser._parse_try_primitive(
                value,
                target,
                default=None,
                default_factory=None,
                field_prefix=field_prefix,
            )
            if primitive_result.is_success:
                validated_primitive = TypeAdapter(target).validate_python(
                    primitive_result.value,
                )
                return r[T].ok(validated_primitive)
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                primitive_result.error or f"{field_prefix}Primitive coercion failed",
            )
        return FlextUtilitiesParser._parse_try_direct(
            value,
            target,
            default,
            default_factory,
            field_prefix,
        )

    def apply_regex_pipeline(
        self,
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[str]:
        r"""Apply sequence of regex substitutions to text.

        **Generic replacement for**: Multiple regex.sub() calls

        Args:
            text: Text to transform
            patterns: List of (pattern, replacement) or (pattern, replacement, flags) tuples

        Returns:
            r with transformed text or error

        Example:
            >>> patterns = [
            ...     (r"\\\\s+=", "="),  # Remove spaces before =
            ...     (r",\\\\s+", ","),  # Remove spaces after ,
            ...     (r"\\\\s+", " "),  # Normalize whitespace
            ... ]
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.apply_regex_pipeline(
            ...     "cn = REDACTED_LDAP_BIND_PASSWORD , ou = users", patterns
            ... )
            >>> cleaned = result.value  # "cn=REDACTED_LDAP_BIND_PASSWORD,ou=users"

        """
        text_len = self._get_safe_text_length(text)
        self._parser_log.debug(
            "Starting regex pipeline application",
            operation="apply_regex_pipeline",
            text_length=text_len,
            patterns_count=len(patterns),
        )
        edge_result = self._handle_pipeline_edge_cases(text, patterns)
        if edge_result.is_success or edge_result.error_code != "PIPELINE_CONTINUE":
            return edge_result
        if text is None:
            return r[str].fail("Text cannot be None for regex pipeline")
        try:
            self._parser_log.debug(
                "Applying regex patterns sequentially",
                operation="apply_regex_pipeline",
                patterns_count=len(patterns),
            )
            process_result = self._process_all_patterns(text, patterns)
            if process_result.is_failure:
                return r[str].fail(
                    process_result.error or "Unknown error in pattern processing",
                )
            proc_val = process_result.value
            result_text, applied_patterns = proc_val
            final_result = result_text.strip()
            self._parser_log.debug(
                "Regex pipeline completed successfully",
                operation="apply_regex_pipeline",
                patterns_applied=applied_patterns,
                original_length=len(text),
                final_length=len(final_result),
                total_replacements=len(text) - len(final_result),
            )
            return r[str].ok(final_result)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            text_len = self._get_safe_text_length(text)
            self._parser_log.exception(
                "FATAL ERROR during regex pipeline application - PIPELINE ABORTED",
                operation="apply_regex_pipeline",
                error=str(e),
                error_type=e.__class__.__name__,
                patterns_count=len(patterns),
                text_length=text_len,
                consequence="Cannot apply regex transformations - invalid pattern or internal error",
            )
            return r[str].fail(f"Failed to apply regex pipeline: {e}")

    def get_object_key(self, obj: t.TypeHintSpecifier | t.NormalizedValue) -> str:
        """Get comparable string key from object (generic helper).

        This generic helper consolidates object-to-key conversion logic from
        dispatcher.py (_normalize_command_key) and provides flexible key extraction
        strategies for objects.

        Extraction Strategy (in order):
            1. Try __name__ attribute (for types, classes, functions)
            2. Try dict 'name' or 'id' key values (for dict-like objects)
            3. Try 'name' or 'id' attribute on instances
            4. Try object class name
            5. Try str conversion
            6. Use type name as final fallback

        Args:
            obj: Object to extract key from (type, class, instance, etc.)

        Returns:
            String key for object (comparable, hashable)

        Example:
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
        from flext_core import p
            >>> parser = u()
            >>> # Class/Type
            >>> parser.get_object_key(int)
            'int'
            >>> # Function
            >>> parser.get_object_key(len)
            'len'
            >>> # Dict with name key
            >>> parser.get_object_key({"name": "MyObj"})
            'MyObj'
            >>> # Instance
            >>> obj = object()
            >>> key = parser.get_object_key(obj)
            >>> isinstance(key, str)
            True

        """
        obj_type_name: str = type(obj).__name__
        self._parser_log.debug(
            "Starting object key extraction",
            operation="get_object_key",
            obj_type=obj_type_name,
            has_name_attr=hasattr(obj, "__name__"),
        )
        if isinstance(obj, str):
            key = obj
        elif hasattr(obj, "__name__"):
            dunder_name = getattr(obj, "__name__", None)
            key = dunder_name if isinstance(dunder_name, str) else obj_type_name
        elif not callable(obj) and isinstance(obj, Mapping):
            narrowed_map: Mapping[str, t.NormalizedValue] = obj
            str_keyed: dict[str, t.NormalizedValue] = {
                str(mk): mv for mk, mv in narrowed_map.items()
            }
            mapping_key = self._extract_key_from_mapping(str_keyed)
            key = mapping_key.unwrap_or(obj_type_name)
        elif (attr_key := self._extract_key_from_attributes(obj)).is_success:
            ak = attr_key.value
            key = ak
        elif hasattr(obj, "__class__"):
            key = type(obj).__name__
        elif (str_key := self._extract_key_from_str_conversion(obj)).is_success:
            sk = str_key.value
            key = sk
        else:
            key = type(obj).__name__
        return key

    def normalize_whitespace(
        self,
        text: str,
        pattern: str = "\\s+",
        replacement: str = " ",
    ) -> r[str]:
        r"""Normalize whitespace in text using regex pattern.

            **Generic replacement for**: Multiple spaces to single space normalization

        Args:
            text: Text to normalize
            pattern: Regex pattern to match (default: one or more whitespace)
            replacement: Replacement string (default: single space)

        Returns:
            r with normalized text or error

        Example:
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.normalize_whitespace("hello    world\\\\t\\\\nfoo")
            >>> normalized = result.value  # "hello world foo"

        """
        text_len = self._get_safe_text_length(text)
        self._parser_log.debug(
            "Starting whitespace normalization",
            operation="normalize_whitespace",
            text_length=text_len,
            pattern=pattern,
            replacement=replacement,
        )
        if not text:
            self._parser_log.debug(
                "Empty text provided, returning unchanged",
                operation="normalize_whitespace",
            )
            return r[str].ok(text)
        try:
            self._parser_log.debug(
                "Applying regex pattern for whitespace normalization",
                operation="normalize_whitespace",
                pattern=pattern,
                replacement=replacement,
            )
            normalized = re.sub(pattern, replacement, text).strip()
            self._parser_log.debug(
                "Whitespace normalization completed",
                operation="normalize_whitespace",
                original_length=len(text),
                normalized_length=len(normalized),
                replacements_made=len(text) - len(normalized),
            )
            return r[str].ok(normalized)
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
            re.error,
        ) as e:
            self._parser_log.exception(
                "FATAL ERROR during whitespace normalization - NORMALIZATION ABORTED",
                operation="normalize_whitespace",
                error=str(e),
                error_type=e.__class__.__name__,
                pattern=pattern,
                replacement=replacement,
                consequence="Cannot normalize whitespace - invalid pattern or internal error",
            )
            return r[str].fail(f"Failed to normalize whitespace: {e}")

    def parse_delimited(
        self,
        text: str,
        delimiter: str,
        *,
        options: m.ParseOptions | None = None,
    ) -> r[list[str]]:
        """Parse delimited string into list of components.

        **Generic replacement for**: DN.split(), CSV parsing, config parsing

        Args:
            text: String to parse
            delimiter: Delimiter character/string
            options: ParseOptions object with parsing configuration

        Returns:
            r with list of parsed components or error

        Example:
            >>> from flext_core._models.collections import m
            >>> opts = m.ParseOptions(strip=True, remove_empty=True)
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com",
            ...     ",",
            ...     options=opts,
            ... )
            >>> components = result.value
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD", "ou=users", "dc=example", "dc=com"]

        """
        text_len = self._get_safe_text_length(text)
        parse_opts = options if options is not None else m.ParseOptions()
        strip = parse_opts.strip
        remove_empty = parse_opts.remove_empty
        validator = parse_opts.validator
        self._parser_log.debug(
            "Starting delimited string parsing",
            operation="parse_delimited",
            text_length=text_len,
            delimiter=delimiter,
            has_options=options is not None,
            strip=strip,
            remove_empty=remove_empty,
            has_validator=validator is not None,
        )
        if not text:
            self._parser_log.debug(
                "Empty text provided, returning empty list",
                operation="parse_delimited",
            )
            return r[list[str]].ok([])
        if not delimiter or len(delimiter) != 1:
            return r[list[str]].fail(
                f"Delimiter must be exactly one character, got '{delimiter}'",
            )
        if delimiter.isspace() or not delimiter.isprintable():
            return r[list[str]].fail(
                f"Delimiter cannot be a whitespace or control character: '{delimiter}'",
            )
        try:
            self._parser_log.debug(
                "Splitting text by delimiter",
                operation="parse_delimited",
                delimiter=delimiter,
            )
            components = text.split(delimiter)
            self._parser_log.debug(
                "Initial split completed",
                operation="parse_delimited",
                raw_components_count=len(components),
            )
            result = self._process_components(
                components,
                strip=strip,
                remove_empty=remove_empty,
                validator=validator,
            )
            if result.is_failure:
                return result
            components_val = result.value
            components = components_val
            self._parser_log.debug(
                "Delimited parsing completed successfully",
                operation="parse_delimited",
                final_components_count=len(components),
            )
            return r[list[str]].ok(components)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            text_len = self._get_safe_text_length(text)
            self._parser_log.exception(
                "FATAL ERROR during delimited parsing - PARSING ABORTED",
                operation="parse_delimited",
                error=str(e),
                error_type=e.__class__.__name__,
                text_length=text_len,
                delimiter=delimiter,
                consequence="Cannot parse delimited string - invalid input or internal error",
            )
            return r[list[str]].fail(f"Failed to parse delimited string: {e}")

    def split_on_char_with_escape(
        self,
        text: str,
        split_char: str,
        escape_char: str = "\\",
    ) -> r[list[str]]:
        r"""Split string on character, respecting escape sequences.

        **Generic replacement for**: DN parsing with escapes, CSV with quotes

        Args:
            text: String to split
            split_char: Character to split on
            escape_char: Escape character (default: backslash)

        Returns:
            r with list of split components or error

        Example:
            >>> # Parse DN with escaped commas
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.split_on_char_with_escape(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\\\,user,ou=users", ","
            ... )
            >>> parts = result.value
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD\\\\,user", "ou=users"]

        """
        validation_result = self._validate_split_inputs(split_char, escape_char)
        if validation_result.is_failure:
            return r[list[str]].fail(validation_result.error or "Validation failed")
        text_is_empty_result = r[bool].create_from_callable(lambda: not text)
        if text_is_empty_result.is_success and text_is_empty_result.value:
            self._parser_log.debug(
                "Empty text provided, returning list with empty string",
                operation="split_on_char_with_escape",
            )
            return r[list[str]].ok([""])
        return self._execute_escape_splitting(text, split_char, escape_char)

    def _apply_single_pattern(self, params: m.PatternApplicationParams) -> r[str]:
        """Apply a single regex pattern to text."""
        self._parser_log.debug(
            "Applying regex pattern",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            total_patterns=params.total_patterns,
            pattern=params.pattern,
            replacement=params.replacement,
            flags=params.flags,
        )
        before_length = len(params.text)
        try:
            result_text = re.sub(
                params.pattern,
                params.replacement,
                params.text,
                flags=params.flags,
            )
        except (re.PatternError, ValueError) as e:
            return r[str].fail(f"Invalid regex pattern '{params.pattern}': {e}")
        after_length = len(result_text)
        replacements = before_length - after_length
        self._parser_log.debug(
            "Pattern applied",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            replacements_made=replacements,
        )
        return r[str].ok(result_text)

    def _execute_escape_splitting(
        self,
        text: str,
        split_char: str,
        escape_char: str,
    ) -> r[list[str]]:
        """Execute escape-aware splitting with logging and error handling.

        Args:
            text: String to split
            split_char: Character to split on
            escape_char: Escape character

        Returns:
            r with list of split components or error

        """
        text_len = self._get_safe_text_length(text)
        self._parser_log.debug(
            "Starting escape-aware string splitting",
            operation="split_on_char_with_escape",
            text_length=text_len,
            split_char=split_char,
            escape_char=escape_char,
        )
        try:
            self._parser_log.debug(
                "Processing text with escape character handling",
                operation="split_on_char_with_escape",
                text_length=text_len,
            )
            split_result = self._process_escape_splitting(text, split_char, escape_char)
            if split_result.is_failure:
                return r[list[str]].fail(
                    split_result.error or "Unknown error in escape splitting",
                )
            split_val = split_result.value
            components, escape_count = split_val
            self._parser_log.debug(
                "Escape-aware splitting completed successfully",
                operation="split_on_char_with_escape",
                components_count=len(components),
                escape_sequences_found=escape_count,
            )
            return r[list[str]].ok(components)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            text_len = self._get_safe_text_length(text)
            self._parser_log.exception(
                "FATAL ERROR during escape-aware splitting - SPLITTING ABORTED",
                operation="split_on_char_with_escape",
                error=str(e),
                error_type=e.__class__.__name__,
                text_length=text_len,
                split_char=split_char,
                escape_char=escape_char,
                consequence="Cannot split string with escape handling - invalid input or internal error",
            )
            return r[list[str]].fail(f"Failed to split with escape: {e}")

    @overload
    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str],
    ) -> r[tuple[str, str, int]]: ...

    @overload
    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str, int],
    ) -> r[tuple[str, str, int]]: ...

    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str] | tuple[str, str, int],
    ) -> r[tuple[str, str, int]]:
        """Extract pattern, replacement, and flags from tuple."""
        tuple_len = len(pattern_tuple)
        try:
            if tuple_len == self.PATTERN_TUPLE_MIN_LENGTH:
                pattern_val, replacement_val = TypeAdapter(
                    tuple[str, str],
                ).validate_python(pattern_tuple)
                return r[tuple[str, str, int]].ok((pattern_val, replacement_val, 0))
            if tuple_len == self.PATTERN_TUPLE_MAX_LENGTH:
                pattern_val, replacement_val, flags_val = TypeAdapter(
                    tuple[str, str, int],
                ).validate_python(pattern_tuple)
                return r[tuple[str, str, int]].ok((
                    pattern_val,
                    replacement_val,
                    flags_val,
                ))
        except ValidationError:
            if tuple_len == self.PATTERN_TUPLE_MAX_LENGTH:
                return r[tuple[str, str, int]].fail(
                    "validation error: pattern/replacement must be strings and flags must be integer",
                )
            return r[tuple[str, str, int]].fail(
                "validation error: pattern and replacement must be strings",
            )
        return r[tuple[str, str, int]].fail(
            f"Invalid pattern tuple length {tuple_len}, expected 2 or 3",
        )

    def _get_safe_text_length(self, text: str | None) -> int:
        """Get text length safely, handling non-string objects in tests.

        Args:
            text: Text to measure

        Returns:
            Text length or -1 if measurement fails

        """
        if text is None:
            return -1
        try:
            result = self._safe_text_length(text)
        except (TypeError, ValueError, AttributeError):
            return -1
        if isinstance(result, int):
            return result
        return -1

    def _handle_pipeline_edge_cases(
        self,
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[str]:
        """Handle edge cases for regex pipeline application.

        Returns:
            r if edge case handled, None to continue processing

        """
        if text is None:
            self._parser_log.debug(
                "None text provided, returning failure",
                operation="apply_regex_pipeline",
            )
            return r[str].fail("Text cannot be None")
        if not text:
            self._parser_log.debug(
                "Empty text provided, returning unchanged",
                operation="apply_regex_pipeline",
            )
            return r[str].ok(text)
        if not patterns:
            self._parser_log.warning(
                "No patterns provided for regex pipeline",
                operation="apply_regex_pipeline",
                text_length=self._safe_text_length(text),
            )
            return r[str].ok(text)
        return r[str].fail("Continue pipeline", error_code="PIPELINE_CONTINUE")

    def _process_all_patterns(
        self,
        text: str,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[tuple[str, int]]:
        """Process all regex patterns and return final text and count."""
        result_text = text
        applied_patterns = 0
        for i, pattern_tuple in enumerate(patterns):
            tuple_len = len(pattern_tuple)
            if tuple_len in {self.TUPLE_LENGTH_2, self.TUPLE_LENGTH_3}:
                pattern_result = self._extract_pattern_components(pattern_tuple)
            else:
                msg = f"Pattern tuple must have 2 or 3 elements, got {tuple_len}"
                return r[tuple[str, int]].fail(msg)
            if pattern_result.is_failure:
                return r[tuple[str, int]].fail(
                    pattern_result.error
                    or "Unknown error extracting pattern components",
                )
            pattern_val = pattern_result.value
            pattern, replacement, flags = pattern_val
            params_result = FlextUtilitiesModel.from_kwargs(
                m.PatternApplicationParams,
                text=result_text,
                pattern=pattern,
                replacement=replacement,
                flags=flags,
                pattern_index=i,
                total_patterns=len(patterns),
            )
            if params_result.is_failure:
                return r[tuple[str, int]].fail(
                    params_result.error or "Unknown error creating params",
                )
            params_val = params_result.value
            apply_result = self._apply_single_pattern(params_val)
            if apply_result.is_failure:
                return r[tuple[str, int]].fail(
                    apply_result.error or "Unknown error applying pattern",
                )
            result_text_val = apply_result.value
            result_text = result_text_val
            applied_patterns += 1
        return r[tuple[str, int]].ok((result_text, applied_patterns))

    def _process_components(
        self,
        components: list[str],
        *,
        strip: bool,
        remove_empty: bool,
        validator: Callable[[str], bool] | None,
    ) -> r[list[str]]:
        """Process components with strip, remove_empty, and validator."""
        if strip:
            self._parser_log.debug(
                "Stripping whitespace from components",
                operation="parse_delimited",
            )
            components = [c.strip() for c in components]
        if remove_empty:
            self._parser_log.debug(
                "Removing empty components",
                operation="parse_delimited",
            )
            components = [c for c in components if c.strip()]
        if validator:
            self._parser_log.debug(
                "Validating components with custom validator",
                operation="parse_delimited",
            )
            valid_components: list[str] = []
            for comp in components:
                if validator(comp):
                    valid_components.append(comp)
                else:
                    self._parser_log.debug(
                        "Component filtered out by validator",
                        operation="parse_delimited",
                        invalid_component=comp,
                        validator_type=validator.__class__.__name__,
                    )
            components = valid_components
        return r[list[str]].ok(components)

    def _process_escape_splitting(
        self,
        text: str,
        split_char: str,
        escape_char: str,
    ) -> r[tuple[list[str], int]]:
        """Process text with escape character handling and return components."""
        components: list[str] = []
        current: list[str] = []
        i = 0
        escape_count = 0
        while i < len(text):
            if text[i] == escape_char and i + 1 < len(text):
                self._parser_log.debug(
                    "Found escape sequence",
                    operation="split_on_char_with_escape",
                    position=i,
                    escaped_char=text[i + 1],
                )
                current.append(text[i + 1])
                escape_count += 1
                i += 2
            elif text[i] == split_char:
                self._parser_log.debug(
                    "Found unescaped delimiter",
                    operation="split_on_char_with_escape",
                    position=i,
                    current_component_length=len(current),
                )
                components.append("".join(current))
                current = []
                i += 1
            else:
                current.append(text[i])
                i += 1
        self._parser_log.debug(
            "Adding final component",
            operation="split_on_char_with_escape",
            final_component_length=len(current),
        )
        components.append("".join(current))
        return r[tuple[list[str], int]].ok((components, escape_count))


__all__ = ["FlextUtilitiesParser"]
