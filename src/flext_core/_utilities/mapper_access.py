"""Facade for FlextUtilitiesMapperAccess.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import FlextExceptions, FlextResult, c, m, p, t
from flext_core._models import FlextModelsContainers
from flext_core._utilities import FlextUtilitiesGuardsTypeCore, FlextUtilitiesPydantic


class FlextUtilitiesMapperAccess:
    """Compose mapper access through its internal part hierarchy."""

    @staticmethod
    def _normalize_accessible_value(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec | None,
    ) -> t.JsonPayload | t.JsonValue:
        """Normalize protocol-accessible values.

        Return canonical runtime/container shapes.

        Returns:
            The resulting ``t.JsonPayload | t.JsonValue``.

        """
        if value is None:
            # Preserve nulls so the extraction contract decides fail/default policy.
            return None
        if isinstance(value, m.BaseModel):
            return value
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return FlextUtilitiesPydantic.normalize_to_container(
                m.ConfigMap.model_validate(model_dump_attr()),
            )
        return FlextUtilitiesMapperAccess._normalize_payload_value(value)

    @staticmethod
    def _normalize_payload_value(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec | None,
    ) -> t.JsonPayload | t.JsonValue:
        """Normalize the payload shapes that never expose ``model_dump``.

        Returns:
            The resulting ``t.JsonPayload | t.JsonValue``.

        """
        if isinstance(value, p.ValidatorSpec):
            return str(value)
        if isinstance(value, (*c.SCALAR_TYPES, Path)):
            return value
        return FlextUtilitiesMapperAccess._container_or_string(value)

    @staticmethod
    def _container_or_string(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec,
    ) -> t.JsonPayload | t.JsonValue:
        """Return container values as-is and everything else stringified.

        Returns:
            The resulting ``t.JsonPayload | t.JsonValue``.

        """
        if isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value):
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and all(
                FlextUtilitiesGuardsTypeCore.container_value(item) for item in value
            )
        ):
            return value
        return str(value)

    @staticmethod
    def _resolve_raw_value(
        raw: t.JsonPayload | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Wrap a raw value, preserving null as an explicit failed contract.

        Returns:
            The resulting ``p.Result[t.JsonPayload]``.

        """
        if raw is None:
            return FlextResult[t.JsonPayload].fail_op(
                "resolve extracted value",
                FlextExceptions.render_template(
                    c.ERR_TEMPLATE_PATH_IS_NONE, path=key_part
                ),
            )
        return FlextResult[t.JsonPayload].ok(
            raw if FlextUtilitiesGuardsTypeCore.container_value(raw) else str(raw),
        )

    @staticmethod
    def _extract_get_value(
        current: t.JsonPayload
        | t.JsonMapping
        | m.ConfigMap
        | Sequence[t.JsonValue | t.JsonPayload]
        | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Get a raw value from a mapping, model, or protocol object.

        Returns:
            The resulting ``p.Result[t.JsonPayload]``.

        """
        not_found_result: p.Result[t.JsonPayload] = FlextResult[t.JsonPayload].fail_op(
            "extract key",
            FlextExceptions.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )
        result: p.Result[t.JsonPayload]
        mapping_obj: Mapping[str, t.JsonValue | t.JsonPayload] | None = None
        if isinstance(current, m.ConfigMap):
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
                else FlextResult[t.JsonPayload].fail_op(
                    "extract mapping key",
                    FlextExceptions.render_template(
                        c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part
                    )
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

    @staticmethod
    def _extract_handle_array_index(
        current: t.JsonPayload
        | t.JsonMapping
        | Sequence[t.JsonValue | t.JsonPayload]
        | FlextModelsContainers.ObjectList,
        array_match: str,
    ) -> p.Result[t.JsonPayload | None]:
        """Handle array indexing with negative index support.

        Returns:
            The resulting ``p.Result[t.JsonPayload | None]``.

        """
        if isinstance(current, FlextModelsContainers.ObjectList):
            sequence: Sequence[t.JsonValue | t.JsonPayload] = current.root
        elif isinstance(current, Sequence) and not isinstance(
            current,
            c.STR_BYTES_TYPES,
        ):
            sequence = current
        else:
            return FlextResult[t.JsonPayload | None].fail_op(
                "extract array index",
                c.ERR_MAPPER_NOT_A_SEQUENCE,
            )
        index_result = FlextUtilitiesMapperAccess._normalize_array_index(
            array_match,
            len(sequence),
        )
        return index_result.map(lambda index: sequence[index])

    @staticmethod
    def _normalize_array_index(array_match: str, sequence_size: int) -> p.Result[int]:
        try:
            idx = int(array_match)
        except ValueError:
            return FlextResult[int].fail_op(
                "extract array index",
                FlextExceptions.render_template(
                    c.ERR_TEMPLATE_INVALID_INDEX, index=array_match
                ),
            )
        resolved_idx = sequence_size + idx if idx < 0 else idx
        if 0 <= resolved_idx < sequence_size:
            return FlextResult[int].ok(resolved_idx)
        return FlextResult[int].fail_op(
            "extract array index",
            FlextExceptions.render_template(
                c.ERR_TEMPLATE_INDEX_OUT_OF_RANGE, index=idx
            ),
        )

    @staticmethod
    def _get_raw(
        data: p.AccessibleData | p.ConfigModelInput,
        key: str,
    ) -> t.JsonPayload | t.JsonValue:
        """Get raw values without DSL conversion.

        Returns:
            The resulting ``t.JsonPayload | t.JsonValue``.

        """
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
