"""Exception template rendering — render_template, render_error_template.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities import FlextUtilitiesPydantic


class FlextExceptionsTemplate:
    """Template rendering helpers for exception and result messages."""

    type TemplateValues = (
        Mapping[str, flext_core.t.JsonPayload | None] | flext_core.m.ConfigMap
    )

    @staticmethod
    def template_values(
        params: flext_core.p.BaseModel | None,
        values: FlextExceptionsTemplate.TemplateValues,
    ) -> flext_core.p.ConfigMap:
        """Build template substitution values using params data and field metadata.

        Returns:
            The resulting ``m.ConfigMap``.

        """
        payload: flext_core.t.JsonDict = (
            params.model_dump(mode="json", exclude_none=True)
            if params is not None
            else {}
        )
        if isinstance(params, flext_core.m.BaseModel):
            payload |= {
                f"{field_name}_description": field_help
                for field_name, field_info in type(params).model_fields.items()
                if isinstance(
                    (field_help := field_info.description or field_info.title),
                    str,
                )
                and field_help
            }
        payload |= {
            key: FlextUtilitiesPydantic.normalize_to_json_value(value)
            for key, value in values.items()
            if value is not None
        }
        return flext_core.m.ConfigMap.model_validate(payload)

    @staticmethod
    def render_template(
        template: str,
        *,
        params: flext_core.p.BaseModel | None = None,
        **values: flext_core.t.JsonPayload | None,
    ) -> str:
        """Render a message template from params + explicit values.

        Fail-fast: raises ValueError when any placeholder value is missing.

        Returns:
            The resulting ``str``.

        Raises:
            ValueError: If a ``KeyError`` is caught.

        """
        payload = FlextExceptionsTemplate.template_values(params, values)
        try:
            return template.format_map(dict(payload))
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise ValueError(
                flext_core.c.ERR_TEMPLATE_MISSING_VALUE.format(
                    key=missing_key, template=template
                ),
            ) from exc

    @staticmethod
    def result_error_data(
        params: flext_core.p.BaseModel | None,
        **values: flext_core.t.JsonPayload | None,
    ) -> flext_core.p.ConfigMap | None:
        """Build canonical error_data payload from params and explicit values.

        Returns:
            The resulting ``m.ConfigMap | None``.

        """
        payload = FlextExceptionsTemplate.template_values(params, values)
        return payload or None


__all__: list[str] = ["FlextExceptionsTemplate"]
