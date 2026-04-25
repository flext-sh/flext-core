"""String parsing dispatcher — universal type parser entry point.

Composes :class:`FlextUtilitiesParserTargets` (which itself extends
:class:`FlextUtilitiesParserCoerce`) so the public ``parse()`` and
``norm_str()`` surface stays the same while the heavy lifting is
broken into responsibility-scoped layers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import no_type_check

from flext_core import (
    FlextUtilitiesGuardsTypeModel,
    c,
    p,
    r,
    t,
)
from flext_core._utilities.parser_targets import FlextUtilitiesParserTargets


@no_type_check
class FlextUtilitiesParser(FlextUtilitiesParserTargets):
    """Universal type parser dispatch — strings, enums, models, primitives."""

    @staticmethod
    @r.safe
    def parse[T](
        value: t.JsonPayload,
        target: type[T],
        options: FlextUtilitiesParserTargets.ParseOptions[T] | None = None,
        **kwargs: t.JsonPayload,
    ) -> T:
        """Universal type parser supporting enums, models, and primitives."""
        opts, fp = FlextUtilitiesParserTargets._resolve_opts(options, kwargs)  # noqa: SLF001
        return FlextUtilitiesParser._dispatch(value, target, opts, fp, kwargs)

    @staticmethod
    def _dispatch[T](
        value: t.JsonPayload,
        target: type[T],
        opts: FlextUtilitiesParserTargets.ParseOptions[T],
        fp: str,
        kwargs: dict[str, t.JsonPayload],
    ) -> T:
        """Internal dispatcher with pre-resolved options for stable type inference."""
        if value is None:
            default_result_initial: p.Result[T] = (
                FlextUtilitiesParser._parse_with_default(
                    opts.default,
                    opts.default_factory,
                    c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
                )
            )
            return default_result_initial.unwrap()
        if isinstance(value, target):
            return value
        match target:
            case tgt if issubclass(tgt, StrEnum):
                enum_result: p.Result[T] = FlextUtilitiesParser._parse_try_enum(
                    value,
                    target,
                    options=None,
                    **kwargs,
                )
                if enum_result.failure:
                    default_result_enum: p.Result[T] = (
                        FlextUtilitiesParser._parse_with_default(
                            opts.default,
                            opts.default_factory,
                            enum_result.error
                            or c.ERR_PARSER_CANNOT_PARSE_ENUM.format(
                                field_prefix=fp,
                                value=value,
                                target_name=target.__name__,
                                options=[],
                            ),
                        )
                    )
                    return default_result_enum.unwrap()
                return enum_result.value
            case tgt if FlextUtilitiesGuardsTypeModel.model_type(tgt):
                model_result: p.Result[T] = FlextUtilitiesParser._parse_try_model(
                    value,
                    target,
                    options=None,
                    **kwargs,
                )
                if model_result.failure:
                    default_result_model: p.Result[T] = (
                        FlextUtilitiesParser._parse_with_default(
                            opts.default,
                            opts.default_factory,
                            model_result.error
                            or c.ERR_PARSER_CANNOT_PARSE_TO_TARGET.format(
                                field_prefix=fp,
                                source_type=value.__class__.__name__,
                                target_name=target.__name__,
                                error="",
                            ),
                        )
                    )
                    return default_result_model.unwrap()
                return model_result.value
            case tgt if tgt in {int, float, str, bool}:
                prim: T | None = FlextUtilitiesParser._parse_try_primitive(
                    value,
                    target,
                    options=None,
                )
                if prim is not None:
                    return prim
                default_result_prim: p.Result[T] = (
                    FlextUtilitiesParser._parse_with_default(
                        opts.default,
                        opts.default_factory,
                        c.ERR_PARSER_PARSE_FAILED_FOR_TARGET.format(
                            field_prefix=fp,
                            value=value,
                            target_name=target.__name__,
                        ),
                    )
                )
                return default_result_prim.unwrap()
            case _:
                direct_value: T = FlextUtilitiesParser._parse_try_direct(
                    value,
                    target,
                    options=None,
                    **kwargs,
                )
                return direct_value


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesParser"]
