"""String parsing helpers for deterministic CQRS utility flows.

These helpers centralize delimiter handling, whitespace normalization, and
escaped character parsing so dispatcher handlers and services receive
predictable ``p.Result`` outcomes instead of ad-hoc string handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import TypeAliasType

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.protocols import p
from flext_core.result import FlextResult as r
from flext_core.typings import t


class FlextUtilitiesParser:
    """Parse delimited and structured strings with predictable results.

    The parser consolidates delimiter handling, escape-aware splits, and
    normalization routines behind ``p.Result`` so callers can compose
    parsing logic in dispatcher pipelines without manual error handling.

    """

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
        """Coerce value to string - returns p.Result[str]."""
        return r[str].ok(str(value))

    @staticmethod
    def _is_primitive_type(target: type) -> bool:
        """Check if target is a primitive type."""
        return target in {int, float, str, bool}

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
        value_dict_data: t.ContainerMapping = {str(k): v for k, v in value.items()}
        try:
            return r[TModel].ok(target.model_validate(value_dict_data, strict=strict))
        except (ValidationError, TypeError, ValueError) as exc:
            return r[TModel].fail(f"Model parse failed: {exc}")

    _CASE_OPS: Mapping[str, Callable[[str], str]] = {
        "lower": str.lower,
        "upper": str.upper,
        "title": str.title,
    }

    @staticmethod
    def _parse_normalize_str(value: t.NormalizedValue, *, case: str = "lower") -> str:
        """Normalize string value (avoids circular import with u.normalize)."""
        value_str = value if isinstance(value, str) else str(value)
        op = FlextUtilitiesParser._CASE_OPS.get(case)
        return op(value_str) if op else value_str

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
        if type(target) is TypeAliasType or str(target) == "typing.Any":
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
    def _find_enum_member_insensitive[T](
        members: Mapping[str, T],
        value_lower: str,
    ) -> T | None:
        """Case-insensitive lookup by member name or value."""
        for member_name, member_value in members.items():
            if member_name.lower() == value_lower:
                return member_value
            member_val = getattr(member_value, "value", None)
            if member_val is not None and str(member_val).lower() == value_lower:
                return member_value
        return None

    @staticmethod
    def _find_enum_member_exact[T](
        members: Mapping[str, T],
        value_str: str,
    ) -> T | None:
        """Exact lookup by member name or value."""
        if value_str in members:
            return members[value_str]
        for member_value in members.values():
            member_val = getattr(member_value, "value", None)
            if member_val == value_str:
                return member_value
        return None

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
        members: Mapping[str, T] = TypeAdapter(Mapping[str, T]).validate_python(
            target.__members__
        )
        value_str = str(value)
        match case_insensitive:
            case True:
                found = FlextUtilitiesParser._find_enum_member_insensitive(
                    members,
                    value_str.lower(),
                )
            case False:
                found = FlextUtilitiesParser._find_enum_member_exact(
                    members,
                    value_str,
                )
        if found is not None:
            return r[T].ok(found)
        target_name = target.__name__ if hasattr(target, "__name__") else "Unknown"
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            f"{field_prefix}Cannot parse '{value_str}' as {target_name}",
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
    def _get_primitive_coercers() -> Mapping[
        type, Callable[[t.NormalizedValue], r[int] | r[float] | r[str] | r[bool]]
    ]:
        """Dispatch table for primitive type coercion."""
        return {
            int: FlextUtilitiesParser._coerce_to_int,
            float: FlextUtilitiesParser._coerce_to_float,
            str: FlextUtilitiesParser._coerce_to_str,
            bool: FlextUtilitiesParser._coerce_to_bool,
        }

    @staticmethod
    def _parse_try_primitive(
        value: t.NormalizedValue,
        target: type,
        default: t.Primitives | None,
        default_factory: Callable[[], t.Numeric | str | bool] | None,
        field_prefix: str,
    ) -> r[t.Primitives]:
        """Helper: Try primitive coercion via dispatch table."""
        coercer = FlextUtilitiesParser._get_primitive_coercers().get(target)
        if coercer is None:
            return r[t.Primitives].fail(
                f"{field_prefix}Target is not primitive",
                error_code="TARGET_NOT_PRIMITIVE",
            )
        try:
            coerce_result = coercer(value)
            if coerce_result.is_failure:
                return r[t.Primitives].fail(coerce_result.error)
            return r[t.Primitives].ok(coerce_result.value)
        except (ValueError, TypeError) as e:
            target_name = getattr(target, "__name__", "type")
            return r[t.Primitives].fail(
                f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
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
    def norm_in(
        value: str,
        items: p.HasModelDump | t.StrSequence | t.ConfigMap | t.FlatContainerMapping,
        *,
        case: str | None = None,
    ) -> bool:
        """Normalized membership check (builder: norm().in_()).

        Mnemonic: norm = normalize, in_ = membership check

        Canonical path: pass t.ConfigMap or any Pydantic BaseModel (p.HasModelDump).
        Legacy path: raw Mapping or t.StrSequence — emits DeprecationWarning.

        Args:
            value: Value to check
            items: Items to check against (ConfigMap/BaseModel preferred; raw Mapping deprecated)
            case: Case normalization

        Returns:
            bool: True if normalized value in normalized items

        """
        items_to_check: t.StrSequence
        match items:
            case t.ConfigMap():
                items_to_check = [str(k) for k in items.root]
            case p.HasModelDump():
                items_to_check = list(items.model_dump().keys())
            case Mapping():
                warnings.warn(
                    "Passing raw Mapping to norm_in() is deprecated. "
                    "Use t.ConfigMap or a Pydantic BaseModel (p.HasModelDump) instead. "
                    "Will be removed in v0.13.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                items_to_check = [str(k) for k in items]
            case Sequence():
                items_to_check = [str(i) for i in items]
            case _:
                items_to_check = [str(items)]
        normalized_value = FlextUtilitiesParser.norm_str(value, case=case or "lower")
        normalized_result = [
            FlextUtilitiesParser.norm_str(item, case=case or "lower")
            for item in items_to_check
        ]
        return normalized_value in normalized_result

    @staticmethod
    def norm_join(
        items: t.StrSequence,
        *,
        case: str | None = None,
        sep: str = " ",
    ) -> str:
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
            normalized = list(items)
        return sep.join(normalized)

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
        if value is None:
            str_value = default
        elif isinstance(value, str):
            str_value = value
        else:
            str_value = (
                r[str]
                .create_from_callable(lambda: str(value))
                .unwrap_or(
                    default,
                )
            )
        if case:
            return FlextUtilitiesParser._parse_normalize_str(str_value, case=case)
        return str_value

    @staticmethod
    def _parse_dispatch_primitive[T](
        value: t.NormalizedValue,
        target: type[T],
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Dispatch primitive coercion with validation."""
        prim = FlextUtilitiesParser._parse_try_primitive(
            value,
            target,
            default=None,
            default_factory=None,
            field_prefix=field_prefix,
        )
        if prim.is_success:
            return r[T].ok(TypeAdapter(target).validate_python(prim.value))
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            prim.error or f"{field_prefix}Primitive coercion failed",
        )

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

        Parsing order: enum -> model -> primitive coercion -> direct type call.

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
            p.Result[T]: Ok(parsed_value) or Fail with error message.

        """
        fp = f"{field_name}: " if field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                default, default_factory, fp or "Value is None"
            )
        if isinstance(value, target):
            return r[T].ok(value)
        match target:
            case tgt if issubclass(tgt, StrEnum):
                return FlextUtilitiesParser._parse_try_enum(
                    value,
                    target,
                    case_insensitive=case_insensitive,
                    default=default,
                    default_factory=default_factory,
                    field_prefix=fp,
                )
            case tgt if issubclass(tgt, BaseModel):
                return FlextUtilitiesParser._parse_try_model(
                    value,
                    target,
                    fp,
                    strict=strict,
                    default=default,
                    default_factory=default_factory,
                )
            case tgt if FlextUtilitiesParser._is_primitive_type(tgt):
                return FlextUtilitiesParser._parse_dispatch_primitive(
                    value,
                    target,
                    default,
                    default_factory,
                    fp,
                )
            case _:
                return FlextUtilitiesParser._parse_try_direct(
                    value, target, default, default_factory, fp
                )


__all__ = ["FlextUtilitiesParser"]
