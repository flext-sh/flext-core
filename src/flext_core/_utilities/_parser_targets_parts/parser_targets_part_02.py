"""Per-target parsing helpers (direct/enum/model/primitive)."""

from __future__ import annotations

from contextlib import suppress

from flext_core import c, t
from flext_core._utilities.model import FlextUtilitiesModel

from .parser_targets_part_01 import (
    FlextUtilitiesParserTargets as FlextUtilitiesParserTargetsPart01,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce


class FlextUtilitiesParserTargets(FlextUtilitiesParserTargetsPart01):
    @staticmethod
    def _parse_try_primitive[T](
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T | None:
        """Fall back to primitive type parsing."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            parsed_default: T = FlextUtilitiesParserTargets._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
            return parsed_default
        if target is str:
            coerced_value = value if isinstance(value, str) else str(value)
            validated_str: T = FlextUtilitiesModel.validate_value(
                target, coerced_value
            ).unwrap()
            return validated_str
        cls = FlextUtilitiesParserTargets
        coerce_map = {
            int: cls._coerce_to_int,
            float: cls._coerce_to_float,
            bool: cls._coerce_to_bool,
        }
        coerce_fn = coerce_map.get(target)
        if coerce_fn is not None:
            coerced_val = None
            with suppress(TypeError, ValueError):
                result = coerce_fn(value)
                if not result.failure:
                    coerced_val = result.value
            return (
                FlextUtilitiesModel.validate_value(target, coerced_val).unwrap()
                if coerced_val is not None
                else None
            )
        if target in {int, float, str, bool}:
            validated_primitive: T | None = FlextUtilitiesModel.validate_value(
                target, value
            ).map_or(None)
            return validated_primitive
        return None


__all__: list[str] = ["FlextUtilitiesParserTargets"]
