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
        opts, fp = FlextUtilitiesParser._resolve_opts(options, kwargs)
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
        resolved_value: T
        if value is None:
            default_result_initial: p.Result[T] = (
                FlextUtilitiesParser._parse_with_default(
                    opts.default,
                    opts.default_factory,
                    c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp),
                )
            )
            resolved_value = default_result_initial.unwrap()
        elif isinstance(value, target):
            resolved_value = value
        else:
            match target:
                case tgt if issubclass(tgt, StrEnum):
                    enum_result: p.Result[T] = FlextUtilitiesParser._parse_try_enum(
                        value,
                        target,
                        options=None,
                        **kwargs,
                    )
                    resolved_value = (
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
                        ).unwrap()
                        if enum_result.failure
                        else enum_result.value
                    )
                case tgt if FlextUtilitiesGuardsTypeModel.model_type(tgt):
                    model_result: p.Result[T] = FlextUtilitiesParser._parse_try_model(
                        value,
                        target,
                        options=None,
                        **kwargs,
                    )
                    resolved_value = (
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
                        ).unwrap()
                        if model_result.failure
                        else model_result.value
                    )
                case tgt if tgt in {int, float, str, bool}:
                    prim: T | None = FlextUtilitiesParser._parse_try_primitive(
                        value,
                        target,
                        options=None,
                    )
                    resolved_value = (
                        prim
                        if prim is not None
                        else FlextUtilitiesParser._parse_with_default(
                            opts.default,
                            opts.default_factory,
                            c.ERR_PARSER_PARSE_FAILED_FOR_TARGET.format(
                                field_prefix=fp,
                                value=value,
                                target_name=target.__name__,
                            ),
                        ).unwrap()
                    )
                case _:
                    resolved_value = FlextUtilitiesParser._parse_try_direct(
                        value,
                        target,
                        options=None,
                        **kwargs,
                    )
        return resolved_value


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesParser"]
