"""String parsing helpers for deterministic CQRS utility flows.

These helpers centralize delimiter handling, whitespace normalization, and
escaped character parsing so dispatcher handlers and services receive
predictable ``p.Result`` outcomes instead of ad-hoc string handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from flext_core import FlextUtilitiesGuards, m, p, r, t
from flext_core._models.base import FlextModelFoundation
from flext_core._utilities.args import FlextUtilitiesArgs


class ParseOptions[T](FlextModelFoundation.FlexibleInternalModel):
    """Options controlling parsing behavior for string-to-type conversion."""

    strict: bool | None = Field(
        default=None, description="Reject coercions; fail on type mismatch"
    )
    coerce: bool | None = Field(
        default=None, description="Enable implicit type coercion"
    )
    case_insensitive: bool | None = Field(
        default=None, description="Normalize case before parsing"
    )
    default: T | None = Field(
        default=None, description="Fallback value when parsing fails"
    )
    default_factory: Callable[[], T] | None = Field(
        default=None, description="Factory producing fallback value"
    )
    field_name: str | None = Field(
        default=None, description="Source field name for error context"
    )


class FlextUtilitiesParser:
    """Parse delimited and structured strings with predictable results.

    The parser consolidates delimiter handling, escape-aware splits, and
    normalization routines behind ``p.Result`` so callers can compose
    parsing logic in dispatcher pipelines without manual error handling.

    """

    @staticmethod
    def _coerce_to_bool[T](value: t.ValueOrModel) -> r[bool]:
        """Coerce value to bool. Returns None if not coercible."""
        if FlextUtilitiesGuards.is_type(value, str):
            normalized_val = FlextUtilitiesParser._parse_normalize_str(
                value,
                case="lower",
            )
            if normalized_val in {"true", "1", "yes", "on"}:
                return r[bool].ok(True)
            if normalized_val in {"false", "0", "no", "off"}:
                return r[bool].ok(False)
            return r[bool].fail(f"Cannot coerce '{value}' to bool")
        return r[bool].ok(bool(value))

    @staticmethod
    def _coerce_to_float[T](value: t.ValueOrModel) -> r[float]:
        """Coerce value to float. Returns None if not coercible."""
        if isinstance(value, (str, int)):
            return r[float].create_from_callable(
                lambda: m.Validators.float_adapter().validate_python(value),
                error_code="FLOAT_COERCE_ERROR",
            )
        return r[float].fail(
            f"Cannot coerce {value.__class__.__name__} to float",
            error_code="FLOAT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_int[T](value: t.ValueOrModel) -> r[int]:
        """Coerce value to int. Returns None if not coercible."""
        if isinstance(value, (str, float)):
            return r[int].create_from_callable(
                lambda: int(m.Validators.float_adapter().validate_python(value)),
                error_code="INT_COERCE_ERROR",
            )
        return r[int].fail(
            f"Cannot coerce {value.__class__.__name__} to int",
            error_code="INT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_str[T](value: t.ValueOrModel) -> r[str]:
        """Coerce value to string - returns p.Result[str]."""
        return r[str].ok(str(value))

    @staticmethod
    def _is_primitive_type(target: type) -> bool:
        """Check if target is a primitive type."""
        return target in {int, float, str, bool}

    @staticmethod
    def _parse_model[TModel: BaseModel](
        value: t.ValueOrModel,
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
    def _parse_normalize_str(value: t.ValueOrModel, *, case: str = "lower") -> str:
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
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try direct type call."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        default = opts.default
        default_factory = opts.default_factory
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{fp}Value is None",
            ).unwrap()
        if isinstance(value, target):
            return value
        try:
            parsed_value = TypeAdapter(target).validate_python(value)
            if not isinstance(parsed_value, target):
                msg = (
                    f"{fp}TypeAdapter returned {type(parsed_value)}, expected {target}"
                )
                raise TypeError(msg)
            return parsed_value
        except (ValidationError, TypeError, ValueError) as e:
            target_name = target.__name__ if hasattr(target, "__name__") else "type"
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{fp}Cannot parse {value.__class__.__name__} to {target_name}: {e}",
            ).unwrap()

    @staticmethod
    @r.safe
    def _parse_try_enum[T](
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try enum parsing, raise ValueError if not enum or invalid."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if not issubclass(target, StrEnum):
            msg = f"{fp}Target is not a StrEnum. Enum mode cannot be used here."
            raise TypeError(
                msg,
            )
        target_name = target.__name__
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                f"{fp}Value is None",
            ).unwrap()
        value_str = FlextUtilitiesParser._coerce_to_str(value).unwrap()
        found: T | None = None
        for member in target:
            member_val = getattr(member, "value", None)
            if member_val is None:
                continue
            if not opts.case_insensitive:
                if str(member_val) == value_str:
                    found = typing.cast("T", member)
                    break
            elif str(member_val).lower() == str(value_str).lower():
                found = typing.cast("T", member)
                break
        if found is not None:
            return found
        msg = f"{fp}Cannot parse '{value_str}' as {target_name} [options: {[e.value for e in target]}]"
        raise ValueError(
            msg,
        )

    @staticmethod
    @r.safe
    def _parse_try_model[T](
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try model parsing, raise ValueError if not model or invalid."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if not issubclass(target, BaseModel):
            msg = f"{fp}Target is not a BaseModel"
            raise TypeError(msg)
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                f"{fp}Value is None",
            ).unwrap()
        if not isinstance(value, (Mapping, BaseModel)):
            msg = f"{fp}Cannot parse scalar '{value}' into {target.__name__}"
            raise TypeError(
                msg,
            )
        if opts.strict:
            return typing.cast("T", target.model_validate(value, strict=True))
        return typing.cast("T", target.model_validate(value))

    @staticmethod
    def _parse_try_primitive[T](
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T | None:
        """Helper function for type primitive parsing fallback."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                f"{fp}Value is None",
            ).unwrap()
        if target is str:
            if isinstance(value, str):
                return typing.cast("T", value)
            try:
                coerced = FlextUtilitiesParser._coerce_to_str(value)
            except (TypeError, ValueError):
                return None
            return typing.cast(
                "T | None", coerced.value if coerced.is_success else None
            )
        if target is int:
            try:
                coerced = FlextUtilitiesParser._coerce_to_int(value)
            except (TypeError, ValueError):
                return None
            return typing.cast(
                "T | None", coerced.value if coerced.is_success else None
            )
        if target is float:
            try:
                coerced = FlextUtilitiesParser._coerce_to_float(value)
            except (TypeError, ValueError):
                return None
            return typing.cast(
                "T | None", coerced.value if coerced.is_success else None
            )
        if target is bool:
            try:
                coerced = FlextUtilitiesParser._coerce_to_bool(value)
            except (TypeError, ValueError):
                return None
            return typing.cast(
                "T | None", coerced.value if coerced.is_success else None
            )
        if target in {int, float, str, bool}:
            try:
                adapter = TypeAdapter(target)
                validated: T = adapter.validate_python(value)
                return validated
            except ValidationError:
                pass
        return None

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
        """Normalized membership check (builder: norm().in_())."""
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
        """Normalize and join (builder: norm().join())."""
        if case:
            normalized = [FlextUtilitiesParser.norm_str(v, case=case) for v in items]
        else:
            normalized = list(items)
        return sep.join(normalized)

    @staticmethod
    def norm_str(
        value: t.ValueOrModel | None,
        *,
        case: str | None = None,
        default: str = "",
    ) -> str:
        """Normalize string (builder: norm().str())."""
        if value is None:
            str_value = default
        elif isinstance(value, str):
            str_value = value
        else:
            str_value = str(value)
        if case:
            return FlextUtilitiesParser._parse_normalize_str(str_value, case=case)
        return str_value

    @staticmethod
    @r.safe
    def _parse_dispatch_primitive[T](
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Dispatch primitive coercion with validation."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        prim = FlextUtilitiesParser._parse_try_primitive(
            value,
            target,
            options=opts,
        )
        if prim is not None:
            return prim
        return FlextUtilitiesParser._parse_with_default(
            opts.default,
            opts.default_factory,
            f"{fp}Failed to parse '{value}' as {target.__name__}",
        ).unwrap()

    @staticmethod
    @r.safe
    def parse[T](
        value: t.ValueOrModel,
        target: type[T],
        options: ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Universal type parser supporting enums, models, and primitives."""
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ParseOptions[T]
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                f"{fp}Value is None",
            ).unwrap()
        if isinstance(value, target):
            return value
        match target:
            case tgt if issubclass(tgt, StrEnum):
                return FlextUtilitiesParser._parse_try_enum(
                    value, target, options=opts, **kwargs
                ).unwrap()
            case tgt if issubclass(tgt, BaseModel):
                res2 = FlextUtilitiesParser._parse_try_model(
                    value, target, options=opts, **kwargs
                )
                return res2.unwrap()
            case tgt if FlextUtilitiesParser._is_primitive_type(tgt):
                prim = FlextUtilitiesParser._parse_try_primitive(
                    value,
                    target,
                    options=opts,
                )
                if prim is not None:
                    return prim
                return FlextUtilitiesParser._parse_with_default(
                    opts.default,
                    opts.default_factory,
                    f"{fp}Failed to parse '{value}' as {target.__name__}",
                ).unwrap()
            case _:
                return FlextUtilitiesParser._parse_try_direct(
                    value, target, options=opts, **kwargs
                )


__all__ = ["FlextUtilitiesParser"]
