"""Facade for FlextUtilitiesMapperExtract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import FlextExceptions, FlextResult, c, m, p, t
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
        default: t.JsonPayload | None,
        required: bool,
    ) -> p.Result[t.JsonPayload]:
        """Return required failure, configured default, or missing-default failure.

        Returns:
            Required failure, configured default, or missing-default failure.

        """
        if not required and default is not None:
            return FlextResult[t.JsonPayload].ok(default)
        return FlextResult[t.JsonPayload].fail_op("extract path", msg)

    @staticmethod
    def _extract_resolve_result(
        result: p.Result[t.JsonPayload],
        *,
        default: t.JsonPayload | None,
        required: bool,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve extractor step result into next value or early fallback result.

        Returns:
            The resulting ``tuple[t.JsonPayload | None, p.Result[t.JsonPayload] |
                None]``.

        """
        if result.failure:
            if not required and default is not None:
                return None, FlextResult[t.JsonPayload].ok(default)
            return None, FlextResult[t.JsonPayload].from_failure(result)
        return result.unwrap(), None

    @staticmethod
    def _extract_resolve_path_part(
        current: t.JsonPayload
        | t.JsonMapping
        | m.ConfigMap
        | Sequence[t.JsonPayload]
        | None,
        part: str,
        *,
        context: p.ExtractResolvePathPartContext,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
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
                and not isinstance(next_val, c.STR_BYTES_TYPES)
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
        data: p.AccessibleData,
    ) -> t.JsonPayload | None:
        """Build the initial ``current`` cursor for path traversal.

        Returns:
            The original model or the payload cursor for path traversal.

        """
        seed_current: t.JsonPayload | None = None
        if isinstance(data, m.BaseModel):
            seed_current = data
        elif isinstance(data, Mapping):
            seed_current = m.ConfigMap(
                root={
                    k: FlextUtilitiesMapperExtract._normalize_accessible_value(v)
                    for k, v in data.items()
                },
            )
        else:
            model_dump_attr = getattr(data, "model_dump", None)
            if callable(model_dump_attr):
                seed_current = m.ConfigMap.model_validate(model_dump_attr())
            elif isinstance(data, p.ValidatorSpec):
                seed_current = str(data)
            elif data is None or isinstance(data, (*c.SCALAR_TYPES, Path, list, tuple)):
                seed_current = data
        return seed_current

    @staticmethod
    def _extract_path_parts(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> p.Result[t.JsonPayload]:
        parts = path.split(separator)
        current = FlextUtilitiesMapperExtract._extract_seed_current(data)
        for i, part in enumerate(parts):
            if current is None:
                return FlextUtilitiesMapperExtract._extract_fail_or_default(
                    FlextExceptions.render_template(
                        c.ERR_TEMPLATE_PATH_IS_NONE,
                        path=separator.join(parts[:i]),
                    ),
                    default=default,
                    required=required,
                )
            current, early_return = (
                FlextUtilitiesMapperExtract._extract_resolve_path_part(
                    current,
                    part,
                    context=m.ExtractResolvePathPartContext.model_validate({
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
                c.ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE,
                default=default,
                required=required,
            )
        value = (
            current
            if FlextUtilitiesGuardsTypeCore.container_value(current)
            else str(current)
        )
        return FlextResult[t.JsonPayload].ok(value)

    @staticmethod
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> p.Result[t.JsonPayload]:
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
            return FlextResult[t.JsonPayload].fail_op(
                "extract path",
                FlextExceptions.render_template(
                    c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=str(exc),
                    params=m.OperationErrorParams(
                        operation="extract",
                        reason=str(exc),
                    ),
                ),
            )


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
