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
    FlextModelsContainers,
    FlextModelsPydantic,
    FlextRuntime,
    FlextUtilitiesGuards,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    c,
    e,
    m,
    p,
    r,
    t,
)


class FlextUtilitiesMapperAccess:
    """Value-access primitives: normalize / read / array-index."""

    @staticmethod
    def _normalize_accessible_value(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec | None,
    ) -> t.JsonPayload | t.JsonValue:
        """Normalize protocol-accessible values to canonical runtime/container shapes."""
        normalized_value: t.JsonPayload | t.JsonValue
        if value is None:
            normalized_value = ""
        elif isinstance(value, FlextModelsPydantic.BaseModel):
            normalized_value = value
        else:
            model_dump_attr = getattr(value, "model_dump", None)
            if callable(model_dump_attr):
                normalized_value = FlextRuntime.normalize_to_container(
                    m.ConfigMap.model_validate(model_dump_attr()),
                )
            elif isinstance(value, p.ValidatorSpec):
                normalized_value = str(value)
            elif (
                isinstance(value, (*t.SCALAR_TYPES, Path))
                or (
                    isinstance(value, Mapping)
                    and (
                        FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
                    )
                )
                or (
                    isinstance(value, (list, tuple))
                    and all(
                        FlextUtilitiesGuardsTypeCore.container(item) for item in value
                    )
                )
            ):
                normalized_value = value
            else:
                normalized_value = str(value)
        return normalized_value

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
        | Sequence[t.JsonValue | t.JsonPayload]
        | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        not_found_result = r[t.JsonPayload].fail_op(
            "extract key",
            e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )
        result: p.Result[t.JsonPayload]
        if isinstance(current, Mapping):
            mapping_obj: Mapping[str, t.JsonValue | t.JsonPayload] = current
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
        elif FlextUtilitiesGuardsTypeModel.pydantic_model(current):
            model_dump_attr = getattr(current, "model_dump", None)
            if callable(model_dump_attr):
                model_dict = m.ConfigMap.model_validate(model_dump_attr()).root
                has_key = key_part in model_dict
                val = model_dict[key_part] if has_key else None
                result = (
                    r[t.JsonPayload].fail_op(
                        "extract model key value",
                        "found_none:"
                        + e.render_template(
                            c.ERR_TEMPLATE_FOUND_NONE,
                            key=key_part,
                        ),
                    )
                    if has_key and val is None
                    else r[t.JsonPayload].ok(val)
                    if has_key
                    else not_found_result
                )
            else:
                result = not_found_result
        else:
            result = not_found_result
        return result

    @staticmethod
    def _extract_handle_array_index(
        current: t.JsonPayload
        | t.JsonMapping
        | Sequence[t.JsonValue | t.JsonPayload]
        | FlextModelsContainers.ObjectList,
        array_match: str,
    ) -> p.Result[t.JsonPayload]:
        """Handle array indexing with negative index support."""
        if isinstance(current, FlextModelsContainers.ObjectList):
            sequence: Sequence[t.JsonValue | t.JsonPayload] = current.root
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            sequence = current
        else:
            return r[t.JsonPayload].fail_op(
                "extract array index",
                c.ERR_MAPPER_NOT_A_SEQUENCE,
            )
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                item = sequence[idx]
                if item is None:
                    return r[t.JsonPayload].fail_op(
                        "extract array index value",
                        c.ERR_MAPPER_FOUND_NONE_INDEX,
                    )
                return r[t.JsonPayload].ok(item)
            return r[t.JsonPayload].fail_op(
                "extract array index",
                e.render_template(
                    c.ERR_TEMPLATE_INDEX_OUT_OF_RANGE,
                    index=int(array_match),
                ),
            )
        except (ValueError, IndexError):
            return r[t.JsonPayload].fail_op(
                "extract array index",
                e.render_template(c.ERR_TEMPLATE_INVALID_INDEX, index=array_match),
            )

    @staticmethod
    def _get_raw(
        data: p.AccessibleData | t.ConfigModelInput,
        key: str,
    ) -> t.JsonPayload | t.JsonValue:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping() if key in data:
                return FlextUtilitiesMapperAccess._normalize_accessible_value(data[key])
            case m.ConfigMap() | m.Dict() if key in data.root:
                return FlextUtilitiesMapperAccess._normalize_accessible_value(
                    data.root[key],
                )
            case _ if hasattr(data, key):
                return FlextUtilitiesMapperAccess._normalize_accessible_value(
                    getattr(data, key),
                )
            case _:
                return ""


__all__: list[str] = ["FlextUtilitiesMapperAccess"]
