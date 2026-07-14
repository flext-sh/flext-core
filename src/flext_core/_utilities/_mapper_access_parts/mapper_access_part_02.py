"""Mapper value-access primitives.

Low-level helpers for reading properties out of mappings, Pydantic models,
and protocol objects. Consumed by the path-based ``extract`` pipeline and
by the public ``prop``/``agg`` helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
)
from flext_core._models.containers import FlextModelsContainers

from .mapper_access_part_01 import (
    FlextUtilitiesMapperAccess as FlextUtilitiesMapperAccessPart01,
)


class FlextUtilitiesMapperAccess(FlextUtilitiesMapperAccessPart01):
    @staticmethod
    def _extract_handle_array_index(
        current: t.JsonPayload
        | t.JsonMapping
        | t.SequenceOf[t.JsonValue | t.JsonPayload]
        | FlextModelsContainers.ObjectList,
        array_match: str,
    ) -> p.Result[t.JsonPayload | None]:
        """Handle array indexing with negative index support."""
        if isinstance(current, FlextModelsContainers.ObjectList):
            sequence: t.SequenceOf[t.JsonValue | t.JsonPayload] = current.root
        elif isinstance(current, Sequence) and not isinstance(
            current, t.STR_BYTES_TYPES
        ):
            sequence = current
        else:
            return r[t.JsonPayload | None].fail_op(
                "extract array index", c.ERR_MAPPER_NOT_A_SEQUENCE
            )
        index_result = FlextUtilitiesMapperAccess._normalize_array_index(
            array_match, len(sequence)
        )
        return index_result.map(lambda index: sequence[index])

    @staticmethod
    def _normalize_array_index(array_match: str, sequence_size: int) -> p.Result[int]:
        try:
            idx = int(array_match)
        except ValueError:
            return r[int].fail_op(
                "extract array index",
                e.render_template(c.ERR_TEMPLATE_INVALID_INDEX, index=array_match),
            )
        resolved_idx = sequence_size + idx if idx < 0 else idx
        if 0 <= resolved_idx < sequence_size:
            return r[int].ok(resolved_idx)
        return r[int].fail_op(
            "extract array index",
            e.render_template(c.ERR_TEMPLATE_INDEX_OUT_OF_RANGE, index=idx),
        )

    @staticmethod
    def _get_raw(
        data: p.AccessibleData | t.ConfigModelInput, key: str
    ) -> t.JsonPayload | t.JsonValue:
        """Get raw values without DSL conversion."""
        match data:
            case dict() | Mapping() if key in data:
                return FlextUtilitiesMapperAccess._normalize_accessible_value(data[key])
            case m.ConfigMap() | m.Dict() if key in data.root:
                return FlextUtilitiesMapperAccess._normalize_accessible_value(
                    data.root[key]
                )
            case _ if hasattr(data, key):
                return FlextUtilitiesMapperAccess._normalize_accessible_value(
                    getattr(data, key)
                )
            case _:
                return ""


__all__: list[str] = ["FlextUtilitiesMapperAccess"]
