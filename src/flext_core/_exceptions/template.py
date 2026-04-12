"""Exception template rendering — render_template, render_error_template.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic.fields import FieldInfo

from flext_core import c, t
from flext_core.runtime import FlextRuntime


class FlextExceptionsTemplate:
    """Template rendering helpers for exception and result messages."""

    type TemplateValues = Mapping[str, t.MetadataOrValue | None] | t.ConfigMap

    @staticmethod
    def template_values(
        params: t.ModelCarrier | None,
        values: FlextExceptionsTemplate.TemplateValues,
    ) -> t.ConfigMap:
        """Build template substitution values using params data and field metadata."""
        payload: t.ConfigMap = t.ConfigMap(root={})
        if params is not None:
            model_fields: Mapping[str, FieldInfo] = params.__class__.model_fields
            params_dump = params.model_dump(exclude_none=True)
            for key, value in params_dump.items():
                payload[str(key)] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(value),
                )
            for field_name, field_info in model_fields.items():
                field_help = field_info.description or field_info.title
                if isinstance(field_help, str) and field_help:
                    payload[f"{field_name}_description"] = field_help
        for key, value in values.items():
            payload[str(key)] = FlextRuntime.normalize_to_container(
                FlextRuntime.normalize_to_metadata(value),
            )
        return payload

    @staticmethod
    def render_template(
        template: str,
        *,
        params: t.ModelCarrier | None = None,
        **values: t.MetadataOrValue | None,
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
    def render_error_template(
        template: str,
        *,
        operation: str | None = None,
        error: Exception | str | None = None,
        params: t.ModelCarrier | None = None,
        **values: t.MetadataOrValue | None,
    ) -> str:
        """Render error template with canonical operation/error fields."""
        payload: t.ConfigMap = t.ConfigMap(root={})
        if operation is not None:
            payload[c.HandlerType.OPERATION] = operation
        if error is not None:
            payload["error"] = str(error)
        for key, value in values.items():
            payload[str(key)] = value
        payload_mapping = {
            str(key): FlextRuntime.normalize_to_metadata(value)
            for key, value in payload.items()
        }
        return FlextExceptionsTemplate.render_template(
            template,
            params=params,
            **payload_mapping,
        )

    @staticmethod
    def result_error_data(
        params: t.ModelCarrier | None,
        **values: t.MetadataOrValue | None,
    ) -> t.ConfigMap | None:
        """Build canonical error_data payload from params and explicit values."""
        payload = FlextExceptionsTemplate.template_values(params, values)
        return payload or None


__all__ = ["FlextExceptionsTemplate"]
