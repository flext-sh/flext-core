"""Facade for FlextUtilitiesParserTargets.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import flext_core

if TYPE_CHECKING:
    pass
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
        options: flext_core.p.ParseOptions[T] | None,
        kwargs: Mapping[str, flext_core.t.JsonPayload],
    ) -> tuple[flext_core.p.ParseOptions[T], str]:
        """Resolve options + field-prefix string used by every ``_parse_try_*``.

        Returns:
            The resulting ``tuple[p.ParseOptions[T], str]``.

        """
        opts: flext_core.p.ParseOptions[T]
        if options is not None:
            opts = options
        else:
            opts = cast(
                "flext_core.m.ParseOptions[T]",
                flext_core.m.ParseOptions.model_validate(kwargs),
            )
        return opts, f"{opts.field_name}: " if opts.field_name else ""

    @staticmethod
    def _parse_try_direct[T](
        value: flext_core.t.JsonPayload,
        target: type[T],
        options: flext_core.p.ParseOptions[T] | None = None,
        **kwargs: flext_core.t.JsonPayload,
    ) -> T:
        """Try a direct type call.

        Returns:
            The resulting ``T``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                flext_core.c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if isinstance(value, target):
            return value
        target_name = target.__name__ if hasattr(target, "__name__") else "type"

        def _on_failure(error: str) -> flext_core.p.Result[T]:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                flext_core.c.ERR_PARSER_CANNOT_PARSE_TO_TARGET.format(
                    field_prefix=fp,
                    source_type=value.__class__.__name__,
                    target_name=target_name,
                    error=error,
                ),
            )

        def _on_success(parsed: T) -> flext_core.p.Result[T]:
            return flext_core.FlextResult[T].ok(parsed)

        parsed_direct: flext_core.p.Result[T] = FlextUtilitiesModel.validate_value(
            target,
            value,
        ).fold(
            _on_failure,
            _on_success,
        )
        return parsed_direct.unwrap()

    @staticmethod
    @flext_core.FlextResult.safe
    def _parse_try_enum[T](
        value: flext_core.t.JsonPayload,
        target: type[T],
        options: flext_core.p.ParseOptions[T] | None = None,
        **kwargs: flext_core.t.JsonPayload,
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
            raise TypeError(
                flext_core.c.ERR_PARSER_TARGET_NOT_STRENUM.format(field_prefix=fp)
            )
        target_name = target.__name__
        if value is None:
            if opts.default is not None:
                return opts.default
            if opts.default_factory is not None:
                return opts.default_factory()
            raise ValueError(
                flext_core.c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp)
            )
        value_str = str(value)
        options_text = [member.value for member in target]
        if not opts.case_insensitive:
            validation_result: flext_core.p.Result[T] = (
                FlextUtilitiesModel.validate_value(
                    target,
                    value_str,
                )
            )
            if validation_result.success:
                validated_enum: T = validation_result.value
                return validated_enum
            raise ValueError(
                flext_core.c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
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
            flext_core.c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                field_prefix=fp,
                value=value_str,
                target_name=target_name,
                options=options_text,
            ),
        )

    @staticmethod
    @flext_core.FlextResult.safe
    def _parse_try_model[T](
        value: flext_core.t.JsonPayload,
        target: type[T],
        options: flext_core.p.ParseOptions[T] | None = None,
        **kwargs: flext_core.t.JsonPayload,
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
            raise TypeError(
                flext_core.c.ERR_PARSER_TARGET_NOT_BASEMODEL.format(field_prefix=fp)
            )
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                flext_core.c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
            ).unwrap()
        if not isinstance(value, Mapping) and not isinstance(
            value,
            flext_core.m.BaseModel,
        ):
            raise TypeError(
                flext_core.c.ERR_PARSER_CANNOT_PARSE_SCALAR_TO_MODEL.format(
                    field_prefix=fp,
                    value=value,
                    target_name=target.__name__,
                ),
            )
        validation_result: flext_core.p.Result[T] = FlextUtilitiesModel.validate_value(
            target,
            value,
            strict=opts.strict,
        )
        return validation_result.unwrap()

    @staticmethod
    def _parse_try_primitive[T](
        value: flext_core.t.JsonPayload,
        target: type[T],
        options: flext_core.p.ParseOptions[T] | None = None,
        **kwargs: flext_core.t.JsonPayload,
    ) -> T | None:
        """Fall back to primitive type parsing.

        Returns:
            The resulting ``T | None``.

        """
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)
        if value is None:
            return FlextUtilitiesParserTargets._parse_with_default(
                opts,
                flext_core.c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
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
            Callable[
                [flext_core.t.JsonPayload],
                flext_core.p.Result[int]
                | flext_core.p.Result[float]
                | flext_core.p.Result[bool],
            ],
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
