"""Parser coercion primitives.

Pure value-to-primitive coercion (bool/int/float/str + case normalization)
driven by the ``m.ParseOptions`` model. Consumed by the per-target
``_parse_try_*`` helpers in :mod:`parser_targets` and :mod:`parser` via MRO
composition.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, ClassVar

import flext_core

if TYPE_CHECKING:
    pass


class FlextUtilitiesParserCoerce:
    """Primitive coercion + string normalization + default fallback."""

    _CASE_OPS: ClassVar[MutableMapping[str, Callable[[str], str]]] = {
        flext_core.c.ParserCase.LOWER.value: str.lower,
        flext_core.c.ParserCase.UPPER.value: str.upper,
        flext_core.c.ParserCase.TITLE.value: str.title,
    }

    @staticmethod
    def _parse_normalize_str(
        value: flext_core.t.JsonPayload,
        *,
        case: str = flext_core.c.ParserCase.LOWER.value,
    ) -> str:
        """Normalize string value (avoids circular import with u.normalize).

        Returns:
            The resulting ``str``.

        """
        value_str = value if isinstance(value, str) else str(value)
        op = FlextUtilitiesParserCoerce._CASE_OPS.get(case)
        return op(value_str) if op else value_str

    @staticmethod
    def _coerce_to_bool(value: flext_core.t.JsonPayload) -> flext_core.p.Result[bool]:
        """Coerce value to bool. Returns None if not coercible.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        if isinstance(value, str):
            normalized_val = FlextUtilitiesParserCoerce._parse_normalize_str(
                value,
                case="lower",
            )
            if normalized_val in flext_core.c.PARSER_BOOLEAN_TRUTHY:
                return flext_core.FlextResult[bool].ok(value=True)
            if normalized_val in flext_core.c.PARSER_BOOLEAN_FALSY:
                return flext_core.FlextResult[bool].ok(value=False)
            return flext_core.FlextResult[bool].fail(
                flext_core.c.ERR_PARSER_COERCE_BOOL_FAILED.format(value=value)
            )
        return flext_core.FlextResult[bool].ok(bool(value))

    @staticmethod
    def _coerce_to_float(value: flext_core.t.JsonPayload) -> flext_core.p.Result[float]:
        """Coerce value to float. Returns None if not coercible.

        Returns:
            The resulting ``p.Result[float]``.

        """
        if isinstance(value, (str, int)):
            return flext_core.FlextResult[float].create_from_callable(
                lambda: float(value),
                error_code="FLOAT_COERCE_ERROR",
            )
        return flext_core.FlextResult[float].fail(
            flext_core.c.ERR_PARSER_COERCE_FLOAT_FAILED.format(
                type_name=value.__class__.__name__
            ),
            error_code="FLOAT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _coerce_to_int(value: flext_core.t.JsonPayload) -> flext_core.p.Result[int]:
        """Coerce value to int. Returns None if not coercible.

        Returns:
            The resulting ``p.Result[int]``.

        """
        if isinstance(value, (str, float)):
            return flext_core.FlextResult[int].create_from_callable(
                lambda: int(float(value)),
                error_code="INT_COERCE_ERROR",
            )
        return flext_core.FlextResult[int].fail(
            flext_core.c.ERR_PARSER_COERCE_INT_FAILED.format(
                type_name=value.__class__.__name__
            ),
            error_code="INT_COERCE_TYPE_ERROR",
        )

    @staticmethod
    def _parse_with_default[T](
        options: flext_core.p.ParseOptions[T],
        error_msg: str,
    ) -> flext_core.p.Result[T]:
        """Return default or error for parse failures.

        Returns:
            Default or error for parse failures.

        """
        if options.default is not None:
            return flext_core.FlextResult[T].ok(options.default)
        if options.default_factory is not None:
            return flext_core.FlextResult[T].ok(options.default_factory())
        return flext_core.FlextResult[T].fail(error_msg)

    @staticmethod
    def norm_str(
        value: flext_core.t.JsonPayload | None,
        *,
        case: str | None = None,
        default: str = "",
    ) -> str:
        """Normalize string (builder: norm().str()).

        Returns:
            The resulting ``str``.

        """
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
