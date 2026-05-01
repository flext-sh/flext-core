"""Exception template rendering — render_template, render_error_template.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstants as c,
    FlextModelsContainers as mc,
    FlextModelsPydantic as mp,
    FlextTypes as t,
)
from flext_core.runtime import FlextRuntime


class FlextExceptionsTemplate:
    """Template rendering helpers for exception and result messages."""

    type TemplateValues = t.MappingKV[str, t.JsonPayload | None] | mc.ConfigMap

    @staticmethod
    def template_values(
        params: mp.BaseModel | None,
        values: FlextExceptionsTemplate.TemplateValues,
    ) -> mc.ConfigMap:
        """Build template substitution values using params data and field metadata."""
        payload: dict[str, t.JsonValue] = (
            {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in params.model_dump(exclude_none=True).items()
            }
            if params is not None
            else {}
        )
        if params is not None:
            payload |= {
                f"{field_name}_description": field_help
                for field_name, field_info in params.__class__.model_fields.items()
                if isinstance(
                    (field_help := field_info.description or field_info.title), str
                )
                and field_help
            }
        payload |= {
            key: FlextRuntime.normalize_to_metadata(value)
            for key, value in values.items()
            if value is not None
        }
        return mc.ConfigMap.model_validate(payload)

    @staticmethod
    def render_template(
        template: str,
        *,
        params: mp.BaseModel | None = None,
        **values: t.JsonPayload | None,
    ) -> str:
        """Render a message template from params + explicit values.

        Fail-fast: raises ValueError when any placeholder value is missing.
        """
        payload = FlextExceptionsTemplate.template_values(params, values)
        try:
            return template.format_map(dict(payload))
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(
                c.ERR_TEMPLATE_MISSING_VALUE.format(
                    key=missing_key,
                    template=template,
                ),
            ) from exc

    @staticmethod
    def result_error_data(
        params: mp.BaseModel | None,
        **values: t.JsonPayload | None,
    ) -> mc.ConfigMap | None:
        """Build canonical error_data payload from params and explicit values."""
        payload = FlextExceptionsTemplate.template_values(params, values)
        return payload or None


__all__: list[str] = ["FlextExceptionsTemplate"]
