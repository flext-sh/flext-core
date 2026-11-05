"""Facade for FlextUtilitiesMapperExtract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import flext_core
from flext_core._utilities import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesMapperAccess,
    FlextUtilitiesPydantic,
)


class FlextUtilitiesMapperExtract(FlextUtilitiesMapperAccess):
    """Compose mapper extraction through its internal part hierarchy."""

    @staticmethod
    def _extract_fail_or_default(
        msg: str,
        *,
        default: flext_core.t.JsonPayload | None,
        required: bool,
    ) -> flext_core.p.Result[flext_core.t.JsonPayload]:
        """Return required failure, configured default, or missing-default failure.

        Returns:
            Required failure, configured default, or missing-default failure.

        """
        if not required and default is not None:
            return flext_core.FlextResult[flext_core.t.JsonPayload].ok(default)
        return flext_core.FlextResult[flext_core.t.JsonPayload].fail_op(
            "extract path", msg
        )

    @staticmethod
    def _extract_resolve_result(
        result: flext_core.p.Result[flext_core.t.JsonPayload],
        *,
        default: flext_core.t.JsonPayload | None,
        required: bool,
    ) -> tuple[
        flext_core.t.JsonPayload | None,
        flext_core.p.Result[flext_core.t.JsonPayload] | None,
    ]:
        """Resolve extractor step result into next value or early fallback result.

        Returns:
            The resulting ``tuple[t.JsonPayload | None, p.Result[t.JsonPayload] |
                None]``.

        """
        if result.failure:
            if not required and default is not None:
                return None, flext_core.FlextResult[flext_core.t.JsonPayload].ok(
                    default
                )
            return None, flext_core.FlextResult[flext_core.t.JsonPayload].from_failure(
                result
            )
        return result.unwrap(), None

    @staticmethod
    def _extract_resolve_path_part(
        current: flext_core.t.JsonPayload
        | flext_core.t.JsonMapping
        | flext_core.m.ConfigMap
        | Sequence[flext_core.t.JsonPayload]
        | None,
        part: str,
        *,
        context: flext_core.p.ExtractResolvePathPartContext,
    ) -> tuple[
        flext_core.t.JsonPayload | None,
        flext_core.p.Result[flext_core.t.JsonPayload] | None,
    ]:
        """Resolve one path segment and return its cursor or an early result.

        Returns:
            The resulting ``tuple[t.JsonPayload | None, p.Result[t.JsonPayload] |
                None]``.

        """
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
        else:
            key_part, array_match = part, ""

        get_result = FlextUtilitiesMapperExtract._extract_get_value(current, key_part)
        next_val, early_result = FlextUtilitiesMapperExtract._extract_resolve_result(
            get_result,
            default=context.default,
            required=context.required,
        )
        if early_result is not None:
            return None, early_result

        if array_match and next_val is not None:
            narrowed_for_index = (
                next_val
                if isinstance(next_val, Sequence)
                and not isinstance(next_val, flext_core.c.STR_BYTES_TYPES)
                else FlextUtilitiesPydantic.normalize_to_container(next_val)
            )
            index_result = FlextUtilitiesMapperExtract._extract_handle_array_index(
                narrowed_for_index,
                array_match,
            )
            next_val, early_result = (
                FlextUtilitiesMapperExtract._extract_resolve_result(
                    index_result,
                    default=context.default,
                    required=context.required,
                )
            )
            if early_result is not None:
                return None, early_result

        return next_val, None

    @staticmethod
    def _extract_seed_current(
        data: flext_core.p.AccessibleData,
    ) -> flext_core.t.JsonPayload | None:
        """Build the initial ``current`` cursor for path traversal.

        Returns:
            The original model or the payload cursor for path traversal.

        """
        seed_current: flext_core.t.JsonPayload | None = None
        if isinstance(data, flext_core.m.BaseModel):
            seed_current = data
        elif isinstance(data, Mapping):
            seed_current = flext_core.m.ConfigMap(
                root={
                    k: FlextUtilitiesMapperExtract._normalize_accessible_value(v)
                    for k, v in data.items()
                },
            )
        else:
            model_dump_attr = getattr(data, "model_dump", None)
            if callable(model_dump_attr):
                seed_current = flext_core.m.ConfigMap.model_validate(model_dump_attr())
            elif isinstance(data, flext_core.p.ValidatorSpec):
                seed_current = str(data)
            elif data is None or isinstance(
                data, (*flext_core.c.SCALAR_TYPES, Path, list, tuple)
            ):
                seed_current = data
        return seed_current

    @staticmethod
    def _extract_path_parts(
        data: flext_core.p.AccessibleData,
        path: str,
        *,
        default: flext_core.t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> flext_core.p.Result[flext_core.t.JsonPayload]:
        parts = path.split(separator)
        current = FlextUtilitiesMapperExtract._extract_seed_current(data)
        for i, part in enumerate(parts):
            if current is None:
                return FlextUtilitiesMapperExtract._extract_fail_or_default(
                    flext_core.FlextExceptions.render_template(
                        flext_core.c.ERR_TEMPLATE_PATH_IS_NONE,
                        path=separator.join(parts[:i]),
                    ),
                    default=default,
                    required=required,
                )
            current, early_return = (
                FlextUtilitiesMapperExtract._extract_resolve_path_part(
                    current,
                    part,
                    context=flext_core.m.ExtractResolvePathPartContext.model_validate({
                        "path_context": separator.join(parts[:i]),
                        "default": default,
                        "required": required,
                    }),
                )
            )
            if early_return is not None:
                return early_return
        if current is None:
            return FlextUtilitiesMapperExtract._extract_fail_or_default(
                flext_core.c.ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE,
                default=default,
                required=required,
            )
        value = (
            current
            if FlextUtilitiesGuardsTypeCore.container_value(current)
            else str(current)
        )
        return flext_core.FlextResult[flext_core.t.JsonPayload].ok(value)

    @staticmethod
    def extract(
        data: flext_core.p.AccessibleData,
        path: str,
        *,
        default: flext_core.t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> flext_core.p.Result[flext_core.t.JsonPayload]:
        """Extract nested value via dot-notation path with array index support.

        Returns:
            The resulting ``p.Result[t.JsonPayload]``.

        """
        try:
            return FlextUtilitiesMapperExtract._extract_path_parts(
                data,
                path,
                default=default,
                required=required,
                separator=separator,
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            return flext_core.FlextResult[flext_core.t.JsonPayload].fail_op(
                "extract path",
                flext_core.FlextExceptions.render_template(
                    flext_core.c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=str(exc),
                    params=flext_core.m.OperationErrorParams(
                        operation="extract",
                        reason=str(exc),
                    ),
                ),
            )


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
