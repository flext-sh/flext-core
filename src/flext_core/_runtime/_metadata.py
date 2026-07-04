"""Runtime metadata and JSON normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.typeadapters import FlextTypesTypeAdapters as tta
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel as ugm

from ._base import FlextRuntimeBase

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices as ts


class FlextRuntimeMetadata(FlextRuntimeBase):
    """Normalize runtime values into metadata and JSON contracts."""

    @staticmethod
    def normalize_to_json_value(
        value: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar]
        | mp.BaseModel
        | None,
    ) -> tb.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``."""
        validated_value: tb.JsonValue
        if value is None:
            validated_value = tta.json_value_adapter().validate_python("")
        elif ugm.has_model_dump(value):
            validated_value = tta.json_value_adapter().validate_python(
                value.model_dump(mode="json"),
            )
        elif isinstance(value, mp.BaseModel):
            validated_value = tta.json_value_adapter().validate_python(str(value))
        else:
            validated_value = tta.json_value_adapter().validate_python(
                FlextRuntimeMetadata.normalize_to_metadata(value),
            )
        return validated_value

    @staticmethod
    def normalize_to_json_mapping(
        value: tb.MappingKV[str, ts.JsonPayload | tb.Scalar],
    ) -> tb.JsonMapping:
        """Normalize a mapping to a validated ``JsonMapping``."""
        return FlextRuntimeMetadata._normalize_dict_entries(
            [(key, item) for key, item in value.items()],
        )

    @staticmethod
    def _normalize_dict_entries(
        items: tb.SequenceOf[tb.Pair[str, ts.JsonPayload]],
    ) -> tb.JsonDict:
        """Normalize key-value pairs for container dict construction."""
        return dict(
            tta.json_mapping_adapter().validate_python(
                {
                    key: FlextRuntimeMetadata.normalize_to_json_value(item)
                    for key, item in items
                },
            ),
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: mp.BaseModel
        | mc.Dict
        | ts.ConfigModelInput
        | tb.MappingKV[str, ts.JsonPayload]
        | None,
    ) -> tb.JsonMapping | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        if isinstance(value, mc.Dict):
            return FlextRuntimeMetadata._normalize_dict_entries(
                [(key, item) for key, item in value.root.items()],
            )
        if isinstance(value, Mapping):
            return FlextRuntimeMetadata._normalize_dict_entries(
                [(key, item) for key, item in value.items()],
            )
        return dict(
            tta.json_mapping_adapter().validate_python(value.model_dump(mode="json")),
        )

    @staticmethod
    def normalize_to_metadata(
        val: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar]
        | None,
    ) -> tb.JsonValue:
        """Normalize input into metadata-compatible JSON-native values."""
        normalized_value: tb.JsonValue
        if isinstance(val, (mc.ConfigMap, mc.Dict)):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.root.items()),
            )
        elif val is None:
            normalized_value = ""
        elif isinstance(val, datetime):
            normalized_value = val.isoformat()
        elif isinstance(val, Path):
            normalized_value = str(val)
        elif isinstance(val, tb.PRIMITIVES_TYPES):
            normalized_value = val
        elif ugm.has_model_dump(val):
            normalized_value = FlextRuntimeMetadata.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.items()),
            )
        elif isinstance(val, AbstractSet) or (
            isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray))
        ):
            normalized_value = list(
                tta.json_list_adapter().validate_python([
                    FlextRuntimeMetadata.normalize_to_json_value(item) for item in val
                ]),
            )
        elif isinstance(val, (bytes, bytearray)):
            normalized_value = str(val)
        else:
            normalized_value = val
        return normalized_value


__all__: list[str] = ["FlextRuntimeMetadata"]
