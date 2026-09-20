"""Runtime metadata and JSON normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .. import t
from .._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel as ugm
from ._base import FlextRuntimeBase

if TYPE_CHECKING:
    from flext_core import m

    from .._typings.services import FlextTypesServices as ts


class FlextRuntimeMetadata(FlextRuntimeBase):
    """Normalize runtime values into metadata and JSON contracts."""

    @staticmethod
    def normalize_to_json_value(
        value: ts.JsonPayload
        | t.Scalar
        | Path
        | m.ConfigMap
        | m.Dict
        | AbstractSet[t.Scalar]
        | m.BaseModel
        | None,
    ) -> t.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``."""
        from flext_core import m

        validated_value: t.JsonValue
        if value is None:
            validated_value = t.json_value_adapter().validate_python(None)

        elif ugm.has_model_dump(value):
            validated_value = t.json_value_adapter().validate_python(
                value.model_dump(mode="json")
            )
        elif isinstance(value, m.BaseModel):
            validated_value = t.json_value_adapter().validate_python(str(value))
        else:
            validated_value = t.json_value_adapter().validate_python(
                FlextRuntimeMetadata.normalize_to_metadata(value)
            )
        return validated_value

    @staticmethod
    def normalize_to_json_mapping(
        value: t.MappingKV[str, ts.JsonPayload | t.Scalar],
    ) -> t.JsonMapping:
        """Normalize a mapping to a validated ``JsonMapping``."""
        return FlextRuntimeMetadata._normalize_dict_entries([
            (key, item) for key, item in value.items()
        ])

    @staticmethod
    def _normalize_dict_entries(
        items: t.SequenceOf[t.Pair[str, ts.JsonPayload]],
    ) -> t.JsonDict:
        """Normalize key-value pairs for container dict construction."""
        return dict(
            t.json_mapping_adapter().validate_python({
                key: FlextRuntimeMetadata.normalize_to_json_value(item)
                for key, item in items
            })
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: m.BaseModel
        | m.Dict
        | ts.ConfigModelInput
        | t.MappingKV[str, ts.JsonPayload]
        | None,
    ) -> t.JsonMapping | None:
        """Normalize model-like input to a plain mapping."""
        from flext_core import m

        if value is None:
            return None
        if isinstance(value, m.Dict):
            return FlextRuntimeMetadata._normalize_dict_entries([
                (key, item) for key, item in value.root.items()
            ])
        if isinstance(value, Mapping):
            return FlextRuntimeMetadata._normalize_dict_entries([
                (key, item) for key, item in value.items()
            ])
        return dict(
            t.json_mapping_adapter().validate_python(value.model_dump(mode="json"))
        )

    @staticmethod
    def normalize_to_metadata(
        val: ts.JsonPayload
        | t.Scalar
        | Path
        | m.ConfigMap
        | m.Dict
        | AbstractSet[t.Scalar]
        | None,
    ) -> t.JsonValue:
        """Normalize input into metadata-compatible JSON-native values."""
        from flext_core import m

        normalized_value: t.JsonValue
        if isinstance(val, (m.ConfigMap, m.Dict)):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.root.items())
            )
        elif val is None:
            normalized_value = ""
        elif isinstance(val, datetime):
            normalized_value = val.isoformat()
        elif isinstance(val, Path):
            normalized_value = str(val)
        elif isinstance(val, t.PRIMITIVES_TYPES):
            normalized_value = val
        elif ugm.has_model_dump(val):
            normalized_value = FlextRuntimeMetadata.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.items())
            )
        elif isinstance(val, AbstractSet) or (
            isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray))
        ):
            normalized_value = list(
                t.json_list_adapter().validate_python([
                    FlextRuntimeMetadata.normalize_to_json_value(item) for item in val
                ])
            )
        elif isinstance(val, (bytes, bytearray)):
            normalized_value = str(val)
        else:
            normalized_value = val
        return normalized_value


__all__: list[str] = ["FlextRuntimeMetadata"]
