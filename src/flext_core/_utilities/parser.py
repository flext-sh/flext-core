"""String parsing helpers for deterministic CQRS utility flows.

These helpers centralize delimiter handling, whitespace normalization, and
escaped character parsing so dispatcher handlers and services receive
predictable ``p.Result`` outcomes instead of ad-hoc string handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
)
from enum import StrEnum

from pydantic import Field

from flext_core import (
    FlextModelsBase,
    FlextUtilitiesArgs,
    FlextUtilitiesGuards,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesModel,
    c,
    p,
    r,
    t,
)


class FlextUtilitiesParser:
    """Parse delimited and structured strings with predictable results.

    The parser consolidates delimiter handling, escape-aware splits, and
    normalization routines behind ``p.Result`` so callers can compose
    parsing logic in dispatcher pipelines without manual error handling.

    """

    class ParseOptions[T](FlextModelsBase.FlexibleInternalModel):
        """Options controlling parsing behavior for string-to-type conversion."""

        strict: bool | None = Field(
            None,
            description="Reject coercions; fail on type mismatch",
        )
        case_insensitive: bool | None = Field(
            None,
            description="Normalize case before parsing",
        )
        default: T | None = Field(
            None,
            description="Fallback value when parsing fails",
        )
        default_factory: Callable[[], T] | None = Field(
            None,
            description="Factory producing fallback value",
        )
        field_name: str | None = Field(
            None,
            description="Source field name for error context",
        )

    @staticmethod
    def _coerce_to_bool(value: t.ValueOrModel) -> p.Result[bool]:
        """Coerce value to bool. Returns None if not coercible."""
        if FlextUtilitiesGuards.matches_type(value, str):
            normalized_val = FlextUtilitiesParser._parse_normalize_str(
                value,
                case="lower",
            )
            if normalized_val in c.PARSER_BOOLEAN_TRUTHY:
                return r[bool].ok(True)
            if normalized_val in c.PARSER_BOOLEAN_FALSY:
                return r[bool].ok(False)
            return r[bool].fail(
                c.ERR_PARSER_COERCE_BOOL_FAILED.format(value=value),
            )
        return r[bool].ok(bool(value))

    @staticmethod
    def _coerce_to_float(value: t.ValueOrModel) -> p.Result[float]:
        """Coerce value to float. Returns None if not coercible."""
        if isinstance(value, (str, int)):
            return r[float].create_from_callable(
                lambda: t.float_adapter().validate_python(value),
                error_code="FLOAT_COERCE_ERROR",
            )
        return r[float].fail(
            c.ERR_PARSER_COERCE_FLOAT_FAILED.format(
                type_name=value.__class__.__name__,
            ),
            error_code="FLOAT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_int(value: t.ValueOrModel) -> p.Result[int]:
        """Coerce value to int. Returns None if not coercible."""
        if isinstance(value, (str, float)):
            return r[int].create_from_callable(
                lambda: int(t.float_adapter().validate_python(value)),
                error_code="INT_COERCE_ERROR",
            )
        return r[int].fail(
            c.ERR_PARSER_COERCE_INT_FAILED.format(
                type_name=value.__class__.__name__,
            ),
            error_code="INT_COERCE_TYPE_ERROR",
        )

    _CASE_OPS: Mapping[str, Callable[[str], str]] = {
        c.ParserCase.LOWER.value: str.lower,
        c.ParserCase.UPPER.value: str.upper,
        c.ParserCase.TITLE.value: str.title,
    }

    @staticmethod
    def _parse_normalize_str(
        value: t.ValueOrModel,
        *,
        case: str = c.ParserCase.LOWER.value,
    ) -> str:
        """Normalize string value (avoids circular import with u.normalize)."""
        value_str = value if isinstance(value, str) else str(value)
        op = FlextUtilitiesParser._CASE_OPS.get(case)
        return op(value_str) if op else value_str

    @staticmethod
    def _parse_try_direct[T](
        value: t.ValueOrModel,
        target: type[T],
        options: FlextUtilitiesParser.ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try direct type call."""
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesParser.ParseOptions[T],
        ).unwrap()
        default = opts.default
        default_factory = opts.default_factory
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if isinstance(value, target):
            return value
        validation_result = FlextUtilitiesModel.validate_value(target, value)
        if validation_result.success:
            return validation_result.value
        target_name = target.__name__ if hasattr(target, "__name__") else "type"
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            c.ERR_PARSER_CANNOT_PARSE_TO_TARGET.format(
                field_prefix=fp,
                source_type=value.__class__.__name__,
                target_name=target_name,
                error=validation_result.error or "",
            ),
        ).unwrap()

    @staticmethod
    @r.safe
    def _parse_try_enum[T](
        value: t.ValueOrModel,
        target: type[T],
        options: FlextUtilitiesParser.ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try enum parsing, raise ValueError if not enum or invalid."""
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesParser.ParseOptions[T],
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if not issubclass(target, StrEnum):
            raise TypeError(
                c.ERR_PARSER_TARGET_NOT_STRENUM.format(field_prefix=fp),
            )
        target_name = target.__name__
        if value is None:
            if opts.default is not None:
                return opts.default
            if opts.default_factory is not None:
                return opts.default_factory()
            raise ValueError(c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp))
        value_str = str(value)
        options_text = [member.value for member in target]
        if not opts.case_insensitive:
            validation_result = FlextUtilitiesModel.validate_value(target, value_str)
            if validation_result.success:
                return validation_result.value
            raise ValueError(
                c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                    field_prefix=fp,
                    value=value_str,
                    target_name=target_name,
                    options=options_text,
                ),
            )
        for member in target:
            member_val = getattr(member, "value", None)
            if member_val is None:
                continue
            if str(member_val).lower() == str(value_str).lower():
                return FlextUtilitiesModel.validate_value(
                    target,
                    str(member_val),
                ).unwrap()
        raise ValueError(
            c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                field_prefix=fp,
                value=value_str,
                target_name=target_name,
                options=options_text,
            ),
        )

    @staticmethod
    @r.safe
    def _parse_try_model[T](
        value: t.ValueOrModel,
        target: type[T],
        options: FlextUtilitiesParser.ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Helper: Try model parsing, raise ValueError if not model or invalid."""
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesParser.ParseOptions[T],
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if not FlextUtilitiesGuardsTypeModel.model_type(target):
            raise TypeError(
                c.ERR_PARSER_TARGET_NOT_BASEMODEL.format(field_prefix=fp),
            )
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if not isinstance(
            value, Mapping
        ) and not FlextUtilitiesGuardsTypeModel.base_model(
            value,
        ):
            raise TypeError(
                c.ERR_PARSER_CANNOT_PARSE_SCALAR_TO_MODEL.format(
                    field_prefix=fp,
                    value=value,
                    target_name=target.__name__,
                ),
            )
        validation_result = FlextUtilitiesModel.validate_value(
            target,
            value,
            strict=opts.strict,
        )
        if validation_result.failure:
            raise ValueError(validation_result.error or "")
        return validation_result.value

    @staticmethod
    def _parse_try_primitive[T](
        value: t.ValueOrModel,
        target: type[T],
        options: FlextUtilitiesParser.ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T | None:
        """Helper function for type primitive parsing fallback."""
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesParser.ParseOptions[T],
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if target is str:
            coerced_value = value if isinstance(value, str) else str(value)
            return FlextUtilitiesModel.validate_value(target, coerced_value).unwrap()
        if target is int:
            try:
                coerced = FlextUtilitiesParser._coerce_to_int(value)
            except (TypeError, ValueError):
                return None
            if coerced.failure:
                return None
            return FlextUtilitiesModel.validate_value(target, coerced.value).unwrap()
        if target is float:
            try:
                coerced = FlextUtilitiesParser._coerce_to_float(value)
            except (TypeError, ValueError):
                return None
            if coerced.failure:
                return None
            return FlextUtilitiesModel.validate_value(target, coerced.value).unwrap()
        if target is bool:
            try:
                coerced = FlextUtilitiesParser._coerce_to_bool(value)
            except (TypeError, ValueError):
                return None
            if coerced.failure:
                return None
            return FlextUtilitiesModel.validate_value(target, coerced.value).unwrap()
        if target in {int, float, str, bool}:
            validated = FlextUtilitiesModel.validate_value(target, value)
            if validated.success:
                return validated.value
        return None

    @staticmethod
    def _parse_with_default[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        error_msg: str,
    ) -> p.Result[T]:
        """Return default or error for parse failures."""
        if default is not None:
            return r[T].ok(default)
        if default_factory is not None:
            return r[T].ok(default_factory())
        return r[T].fail(error_msg)

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
    def parse[T](
        value: t.ValueOrModel,
        target: type[T],
        options: FlextUtilitiesParser.ParseOptions[T] | None = None,
        **kwargs: t.ValueOrModel,
    ) -> T:
        """Universal type parser supporting enums, models, and primitives."""
        opts = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesParser.ParseOptions[T],
        ).unwrap()
        fp = f"{opts.field_name}: " if opts.field_name else ""
        if value is None:
            return FlextUtilitiesParser._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if isinstance(value, target):
            return value
        match target:
            case tgt if issubclass(tgt, StrEnum):
                enum_result = FlextUtilitiesParser._parse_try_enum(
                    value,
                    target,
                    options=opts,
                    **kwargs,
                )
                if enum_result.failure:
                    return FlextUtilitiesParser._parse_with_default(
                        opts.default,
                        opts.default_factory,
                        enum_result.error
                        or c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                            field_prefix=fp,
                            value=value,
                            target_name=target.__name__,
                            options=[],
                        ),
                    ).unwrap()
                return enum_result.value
            case tgt if FlextUtilitiesGuardsTypeModel.model_type(tgt):
                model_result = FlextUtilitiesParser._parse_try_model(
                    value,
                    target,
                    options=opts,
                    **kwargs,
                )
                if model_result.failure:
                    return FlextUtilitiesParser._parse_with_default(
                        opts.default,
                        opts.default_factory,
                        model_result.error
                        or c.ERR_PARSER_CANNOT_PARSE_TO_TARGET.format(
                            field_prefix=fp,
                            source_type=value.__class__.__name__,
                            target_name=target.__name__,
                            error="",
                        ),
                    ).unwrap()
                return model_result.value
            case tgt if tgt in {int, float, str, bool}:
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
                    c.ERR_PARSER_PARSE_FAILED_FOR_TARGET.format(
                        field_prefix=fp,
                        value=value,
                        target_name=target.__name__,
                    ),
                ).unwrap()
            case _:
                return FlextUtilitiesParser._parse_try_direct(
                    value,
                    target,
                    options=opts,
                    **kwargs,
                )


__all__: list[str] = ["FlextUtilitiesParser"]
