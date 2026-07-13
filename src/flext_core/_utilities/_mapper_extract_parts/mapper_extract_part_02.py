"""Path-based extract pipeline on top of ``FlextUtilitiesMapperAccess``."""

from __future__ import annotations

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
)
from flext_core._models.exception_params import FlextModelsExceptionParams
from flext_core._utilities.guards import FlextUtilitiesGuards

from .mapper_extract_part_01 import (
    FlextUtilitiesMapperExtract as FlextUtilitiesMapperExtractPart01,
)


class FlextUtilitiesMapperExtract(FlextUtilitiesMapperExtractPart01):
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
                    e.render_template(
                        c.ERR_TEMPLATE_PATH_IS_NONE, path=separator.join(parts[:i])
                    ),
                    default=default,
                    required=required,
                )
            current, early_return = (
                FlextUtilitiesMapperExtract._extract_resolve_path_part(
                    current,
                    part,
                    context=FlextUtilitiesMapperExtract.ExtractResolvePathPartContext.model_validate({
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
        return r[t.JsonPayload].ok(
            current if FlextUtilitiesGuards.container(current) else str(current)
        )

    @staticmethod
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> p.Result[t.JsonPayload]:
        """Extract nested value via dot-notation path with array index support."""
        try:
            return FlextUtilitiesMapperExtract._extract_path_parts(
                data, path, default=default, required=required, separator=separator
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            return r[t.JsonPayload].fail_op(
                "extract path",
                e.render_template(
                    c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=str(exc),
                    params=FlextModelsExceptionParams.OperationErrorParams(
                        operation="extract", reason=str(exc)
                    ),
                ),
            )


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
