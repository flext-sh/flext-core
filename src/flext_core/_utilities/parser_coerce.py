"""Parser coercion primitives + ParseOptions.

Pure value-to-primitive coercion (bool/int/float/str + case normalization)
and the ``ParseOptions`` model. Consumed by the per-target ``_parse_try_*``
helpers in :mod:`parser_targets` and :mod:`parser` via MRO composition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import no_type_check

from flext_core import (
    FlextModelsBase,
    c,
    m,
    p,
    r,
    t,
)


@no_type_check
class FlextUtilitiesParserCoerce:
    """Primitive coercion + string normalization + default fallback."""

    class ParseOptions[T](FlextModelsBase.FlexibleInternalModel):
        """Options controlling parsing behavior for string-to-type conversion."""

        strict: bool | None = m.Field(
            None,
            validate_default=True,
            description="Reject coercions; fail on type mismatch",
        )
        case_insensitive: bool | None = m.Field(
            None,
            validate_default=True,
            description="Normalize case before parsing",
        )
        default: T | None = m.Field(
            None,
            validate_default=True,
            description="Fallback value when parsing fails",
        )
        default_factory: Callable[[], T] | None = m.Field(
            None,
            validate_default=True,
            description="Factory producing fallback value",
        )
        field_name: str | None = m.Field(
            None,
            validate_default=True,
            description="Source field name for error context",
        )

    _CASE_OPS: Mapping[str, Callable[[str], str]] = {
        c.ParserCase.LOWER.value: str.lower,
        c.ParserCase.UPPER.value: str.upper,
        c.ParserCase.TITLE.value: str.title,
    }

    @staticmethod
    def _parse_normalize_str(
        value: t.JsonPayload,
        *,
        case: str = c.ParserCase.LOWER.value,
    ) -> str:
        """Normalize string value (avoids circular import with u.normalize)."""
        value_str = value if isinstance(value, str) else str(value)
        op = FlextUtilitiesParserCoerce._CASE_OPS.get(case)
        return op(value_str) if op else value_str

    @staticmethod
    def _coerce_to_bool(value: t.JsonPayload) -> p.Result[bool]:
        """Coerce value to bool. Returns None if not coercible."""
        if isinstance(value, str):
            normalized_val = FlextUtilitiesParserCoerce._parse_normalize_str(
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
    def _coerce_to_float(value: t.JsonPayload) -> p.Result[float]:
        """Coerce value to float. Returns None if not coercible."""
        if isinstance(value, (str, int)):
            return r[float].create_from_callable(
                lambda: float(value),
                error_code="FLOAT_COERCE_ERROR",
            )
        return r[float].fail(
            c.ERR_PARSER_COERCE_FLOAT_FAILED.format(
                type_name=value.__class__.__name__,
            ),
            error_code="FLOAT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_int(value: t.JsonPayload) -> p.Result[int]:
        """Coerce value to int. Returns None if not coercible."""
        if isinstance(value, (str, float)):
            return r[int].create_from_callable(
                lambda: int(float(value)),
                error_code="INT_COERCE_ERROR",
            )
        return r[int].fail(
            c.ERR_PARSER_COERCE_INT_FAILED.format(
                type_name=value.__class__.__name__,
            ),
            error_code="INT_COERCE_TYPE_ERROR",
        )

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
        value: t.JsonPayload | None,
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
            return FlextUtilitiesParserCoerce._parse_normalize_str(str_value, case=case)
        return str_value


__all__: list[str] = ["FlextUtilitiesParserCoerce"]
