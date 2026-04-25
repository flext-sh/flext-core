"""Per-target parsing helpers (direct/enum/model/primitive)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import no_type_check

from flext_core import (
    FlextModelsPydantic,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesModel,
    c,
    p,
    r,
    t,
)
from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce


@no_type_check
class FlextUtilitiesParserTargets(FlextUtilitiesParserCoerce):
    """Per-target ``_parse_try_*`` helpers consumed by :meth:`parse`."""

    @staticmethod
    def resolve_opts[T](
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None,
        kwargs: dict[str, t.JsonPayload],
    ) -> tuple[FlextUtilitiesParserCoerce.ParseOptions[T], str]:
        """Public wrapper around options resolution for external parser dispatch."""
        return FlextUtilitiesParserTargets._resolve_opts(options, kwargs)

    @staticmethod
    def _resolve_opts[T](
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None,
        kwargs: dict[str, t.JsonPayload],
    ) -> tuple[FlextUtilitiesParserCoerce.ParseOptions[T], str]:
        """Resolve options + field-prefix string used by every ``_parse_try_*``."""
        opts: FlextUtilitiesParserCoerce.ParseOptions[T]
        if options is not None:
            opts = options
        else:
            opts = FlextUtilitiesParserCoerce.ParseOptions.model_validate(kwargs)
        return opts, f"{opts.field_name}: " if opts.field_name else ""

    @staticmethod
    def _parse_try_direct[T](
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Helper: Try direct type call."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        default = opts.default
        default_factory = opts.default_factory
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                default,
                default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if isinstance(value, target):
            return value
        validation_result: p.Result[T] = FlextUtilitiesModel.validate_value(
            target, value
        )
        if validation_result.success:
            return validation_result.value
        target_name = target.__name__ if hasattr(target, "__name__") else "type"
        return FlextUtilitiesParserTargets._parse_with_default(
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
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Helper: Try enum parsing, raise ValueError if not enum or invalid."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
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
            validation_result: p.Result[T] = FlextUtilitiesModel.validate_value(
                target, value_str
            )
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
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Helper: Try model parsing, raise ValueError if not model or invalid."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if not FlextUtilitiesGuardsTypeModel.model_type(target):
            raise TypeError(
                c.ERR_PARSER_TARGET_NOT_BASEMODEL.format(field_prefix=fp),
            )
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if not isinstance(value, Mapping) and not isinstance(
            value, FlextModelsPydantic.BaseModel
        ):
            raise TypeError(
                c.ERR_PARSER_CANNOT_PARSE_SCALAR_TO_MODEL.format(
                    field_prefix=fp,
                    value=value,
                    target_name=target.__name__,
                ),
            )
        validation_result: p.Result[T] = FlextUtilitiesModel.validate_value(
            target,
            value,
            strict=opts.strict,
        )
        if validation_result.failure:
            raise ValueError(validation_result.error or "")
        return validation_result.value

    @staticmethod
    def _parse_try_primitive[T](
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserCoerce.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T | None:
        """Helper function for type primitive parsing fallback."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts.default,
                opts.default_factory,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if target is str:
            coerced_value = value if isinstance(value, str) else str(value)
            return FlextUtilitiesModel.validate_value(target, coerced_value).unwrap()
        cls = FlextUtilitiesParserTargets
        coerce_map = {
            int: cls._coerce_to_int,
            float: cls._coerce_to_float,
            bool: cls._coerce_to_bool,
        }
        coerce_fn = coerce_map.get(target)
        if coerce_fn is not None:
            try:
                coerced = coerce_fn(value)
            except (TypeError, ValueError):
                return None
            if coerced.failure:
                return None
            return FlextUtilitiesModel.validate_value(target, coerced.value).unwrap()
        if target in {int, float, str, bool}:
            validated: p.Result[T] = FlextUtilitiesModel.validate_value(target, value)
            if validated.success:
                return validated.value
        return None


__all__: list[str] = ["FlextUtilitiesParserTargets"]
