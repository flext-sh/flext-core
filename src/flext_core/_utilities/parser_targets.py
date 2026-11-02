"""Facade for FlextUtilitiesParserTargets.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from flext_core import FlextResult, c, m, p, t
from flext_core._utilities import (
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesModel,
    FlextUtilitiesParserCoerce,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextUtilitiesParserTargets(FlextUtilitiesParserCoerce):
    """Compose parser targets through their internal part hierarchy."""

    @staticmethod
    def _resolve_opts[T](
        options: p.ParseOptions[T] | None,
        kwargs: Mapping[str, t.JsonPayload],
    ) -> tuple[p.ParseOptions[T], str]:
        """Resolve options + field-prefix string used by every ``_parse_try_*``.

        Returns:
            The resulting ``tuple[p.ParseOptions[T], str]``.

        """
        opts: p.ParseOptions[T]
        if options is not None:
            opts = options
        else:
            opts = cast(
                "m.ParseOptions[T]",
                m.ParseOptions.model_validate(kwargs),
            )
        return opts, f"{opts.field_name}: " if opts.field_name else ""

    @staticmethod
    def _parse_try_direct[T](
        value: t.JsonPayload,
        target: type[T],
        options: p.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Try a direct type call.

        Returns:
            The resulting ``T``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if isinstance(value, target):
            return value
        target_name = target.__name__ if hasattr(target, "__name__") else "type"

        def _on_failure(error: str) -> p.Result[T]:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                c.ERR_PARSER_CANNOT_PARSE_TO_TARGET.format(
                    field_prefix=fp,
                    source_type=value.__class__.__name__,
                    target_name=target_name,
                    error=error,
                ),
            )

        def _on_success(parsed: T) -> p.Result[T]:
            return FlextResult[T].ok(parsed)

        parsed_direct: p.Result[T] = FlextUtilitiesModel.validate_value(
            target,
            value,
        ).fold(
            _on_failure,
            _on_success,
        )
        return parsed_direct.unwrap()

    @staticmethod
    @FlextResult.safe
    def _parse_try_enum[T](
        value: t.JsonPayload,
        target: type[T],
        options: p.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Try enum parsing, raising ValueError if not enum or invalid.

        Returns:
            The resulting ``T``.

        Raises:
            TypeError: If ``not issubclass(target, StrEnum)``.
            ValueError: Always; or if ``value is None``; or if ``not
                opts.case_insensitive``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if not issubclass(target, StrEnum):
            raise TypeError(c.ERR_PARSER_TARGET_NOT_STRENUM.format(field_prefix=fp))
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
                target,
                value_str,
            )
            if validation_result.success:
                validated_enum: T = validation_result.value
                return validated_enum
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
            if str(member_val).lower() == value_str.lower():
                validated_member: T = FlextUtilitiesModel.validate_value(
                    target,
                    str(member_val),
                ).unwrap()
                return validated_member
        raise ValueError(
            c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                field_prefix=fp,
                value=value_str,
                target_name=target_name,
                options=options_text,
            ),
        )

    @staticmethod
    @FlextResult.safe
    def _parse_try_model[T](
        value: t.JsonPayload,
        target: type[T],
        options: p.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Try model parsing, raising ValueError if not model or invalid.

        Returns:
            The resulting ``T``.

        Raises:
            TypeError: If ``not FlextUtilitiesGuardsTypeModel.model_type(target)``; or
                if ``not isinstance(value, Mapping) and (not isinstance(value,
                m.BaseModel))``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if not FlextUtilitiesGuardsTypeModel.model_type(target):
            raise TypeError(c.ERR_PARSER_TARGET_NOT_BASEMODEL.format(field_prefix=fp))
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if not isinstance(value, Mapping) and not isinstance(
            value,
            m.BaseModel,
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
        return validation_result.unwrap()

    @staticmethod
    def _parse_try_primitive[T](
        value: t.JsonPayload,
        target: type[T],
        options: p.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T | None:
        """Fall back to primitive type parsing.

        Returns:
            The resulting ``T | None``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if target is str:
            coerced_value = value if isinstance(value, str) else str(value)
            validated_str: T = FlextUtilitiesModel.validate_value(
                target,
                coerced_value,
            ).unwrap()
            return validated_str
        cls = FlextUtilitiesParserTargets
        coerce_map: Mapping[
            type[T | int | float | bool],
            Callable[[t.JsonPayload], p.Result[int] | p.Result[float] | p.Result[bool]],
        ] = {
            int: cls._coerce_to_int,
            float: cls._coerce_to_float,
            bool: cls._coerce_to_bool,
        }
        coerce_fn = coerce_map.get(target)
        if coerce_fn is not None:
            coerced_val = None
            try:
                result = coerce_fn(value)
                if not result.failure:
                    coerced_val = result.value
            except (TypeError, ValueError):
                coerced_val = None
            return (
                FlextUtilitiesModel.validate_value(target, coerced_val).unwrap()
                if coerced_val is not None
                else None
            )
        if target in {int, float, str, bool}:
            validated_primitive: T | None = FlextUtilitiesModel.validate_value(
                target,
                value,
            ).map_or(None)
            return validated_primitive
        return None


__all__: list[str] = ["FlextUtilitiesParserTargets"]
