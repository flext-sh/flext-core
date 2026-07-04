"""Mapper value-access primitives.

Low-level helpers for reading properties out of mappings, Pydantic models,
and protocol objects. Consumed by the path-based ``extract`` pipeline and
by the public ``prop``/``agg`` helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
)
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities._guards_parts.guards_part_01 import FlextUtilitiesGuards
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore


class FlextUtilitiesMapperAccess:
    """Value-access primitives: normalize / read / array-index."""

    @staticmethod
    def _normalize_accessible_value(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec | None,
    ) -> t.JsonPayload | t.JsonValue:
        """Normalize protocol-accessible values to canonical runtime/container shapes."""
        if value is None:
            return ""
        if isinstance(value, FlextModelsPydantic.BaseModel):
            return value
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return FlextRuntime.normalize_to_container(
                m.ConfigMap.model_validate(model_dump_attr()),
            )
        if isinstance(value, p.ValidatorSpec):
            return str(value)
        if isinstance(value, (*t.SCALAR_TYPES, Path)):
            return value
        if isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value):
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and all(FlextUtilitiesGuardsTypeCore.container(item) for item in value)
        ):
            return value
        return str(value)

    @staticmethod
    def _resolve_raw_value(
        raw: t.JsonPayload | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Wrap raw value in Result: fail on None, keep containers, stringify rest."""
        if raw is None:
            return r[t.JsonPayload].fail_op(
                "resolve extracted value",
                "found_none:"
                + e.render_template(c.ERR_TEMPLATE_FOUND_NONE, key=key_part),
            )
        return r[t.JsonPayload].ok(
            raw if FlextUtilitiesGuards.container(raw) else str(raw),
        )

    @staticmethod
    def _extract_get_value(
        current: t.JsonPayload
        | t.JsonMapping
        | FlextModelsContainers.ConfigMap
        | t.SequenceOf[t.JsonValue | t.JsonPayload]
        | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        not_found_result = r[t.JsonPayload].fail_op(
            "extract key",
            e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )
        result: p.Result[t.JsonPayload]
        mapping_obj: t.MappingKV[str, t.JsonValue | t.JsonPayload] | None = None
        if isinstance(current, FlextModelsContainers.ConfigMap):
            mapping_obj = current.root
        elif isinstance(current, Mapping):
            mapping_obj = current
        if mapping_obj is not None:
            result = (
                FlextUtilitiesMapperAccess._resolve_raw_value(
                    mapping_obj[key_part],
                    key_part,
                )
                if key_part in mapping_obj
                else r[t.JsonPayload].fail_op(
                    "extract mapping key",
                    e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part)
                    + " in Mapping",
                )
            )
        elif hasattr(current, key_part):
            result = FlextUtilitiesMapperAccess._resolve_raw_value(
                getattr(current, key_part),
                key_part,
            )
        else:
            result = not_found_result
        return result


__all__: list[str] = ["FlextUtilitiesMapperAccess"]
