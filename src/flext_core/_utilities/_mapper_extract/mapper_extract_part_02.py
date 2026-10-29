"""Path-based extract pipeline on top of ``FlextUtilitiesMapperAccess``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextExceptions, FlextResult, c, m, p, t
from flext_core._utilities import FlextUtilitiesGuardsTypeCore
from flext_core._utilities._mapper_extract import FlextUtilitiesMapperExtractPart01


class FlextUtilitiesMapperExtractPart02(FlextUtilitiesMapperExtractPart01):
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
        current = FlextUtilitiesMapperExtractPart02._extract_seed_current(data)
        for i, part in enumerate(parts):
            if current is None:
                return FlextUtilitiesMapperExtractPart02._extract_fail_or_default(
                    FlextExceptions.render_template(
                        c.ERR_TEMPLATE_PATH_IS_NONE,
                        path=separator.join(parts[:i]),
                    ),
                    default=default,
                    required=required,
                )
            current, early_return = (
                FlextUtilitiesMapperExtractPart02._extract_resolve_path_part(
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
            return FlextUtilitiesMapperExtractPart02._extract_fail_or_default(
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
            return FlextUtilitiesMapperExtractPart02._extract_path_parts(
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


__all__: list[str] = ["FlextUtilitiesMapperExtractPart02"]
