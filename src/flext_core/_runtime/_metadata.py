"""Runtime metadata and JSON normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import c, t
from flext_core._models import FlextModelsContainers, FlextModelsPydantic
from flext_core._runtime import FlextRuntimeBase
from flext_core._utilities import (
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesTypingAdapterJson,
)

if TYPE_CHECKING:
    from flext_core import m


class FlextRuntimeMetadata(FlextRuntimeBase):
    """Normalize runtime values into metadata and JSON contracts."""

    @staticmethod
    def normalize_to_json_value(
        value: t.JsonPayload
        | t.Scalar
        | Path
        | m.ConfigMap
        | m.Dict
        | AbstractSet[t.Scalar]
        | m.BaseModel
        | None,
    ) -> t.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``.

        Returns:
            The resulting ``t.JsonValue``.

        """
        adapter = FlextUtilitiesTypingAdapterJson.json_value_adapter()
        validated_value: t.JsonValue
        if value is None:
            validated_value = adapter.validate_python(None)

        elif FlextUtilitiesGuardsTypeModel.has_model_dump(value):
            validated_value = adapter.validate_python(
                value.model_dump(mode="json"),
            )
        elif isinstance(value, FlextModelsPydantic.BaseModel):
            validated_value = adapter.validate_python(str(value))
        else:
            validated_value = adapter.validate_python(
                FlextRuntimeMetadata.normalize_to_metadata(value),
            )
        return validated_value

    @staticmethod
    def normalize_to_json_mapping(
        value: t.MappingKV[str, t.JsonPayload | t.Scalar],
    ) -> t.JsonMapping:
        """Normalize a mapping to a validated ``JsonMapping``.

        Returns:
            The resulting ``t.JsonMapping``.

        """
        return FlextRuntimeMetadata._normalize_dict_entries([
            (key, item) for key, item in value.items()
        ])

    @staticmethod
    def _normalize_dict_entries(
        items: t.SequenceOf[t.Pair[str, t.JsonPayload]],
    ) -> t.JsonDict:
        """Normalize key-value pairs for container dict construction.

        Returns:
            The resulting ``t.JsonDict``.

        """
        return dict(
            FlextUtilitiesTypingAdapterJson.json_mapping_adapter().validate_python(
                {
                    key: FlextRuntimeMetadata.normalize_to_json_value(item)
                    for key, item in items
                },
            ),
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: m.BaseModel
        | m.Dict
        | t.ConfigModelInput
        | t.MappingKV[str, t.JsonPayload]
        | None,
    ) -> t.JsonMapping | None:
        """Normalize model-like input to a plain mapping.

        Returns:
            The resulting ``t.JsonMapping | None``.

        """
        if value is None:
            return None
        if isinstance(value, FlextModelsContainers.Dict):
            return FlextRuntimeMetadata._normalize_dict_entries([
                (key, item) for key, item in value.root.items()
            ])
        if isinstance(value, Mapping):
            return FlextRuntimeMetadata._normalize_dict_entries([
                (key, item) for key, item in value.items()
            ])
        return dict(
            FlextUtilitiesTypingAdapterJson.json_mapping_adapter().validate_python(
                value.model_dump(mode="json"),
            ),
        )

    @staticmethod
    def normalize_to_metadata(
        val: t.JsonPayload
        | t.Scalar
        | Path
        | m.ConfigMap
        | m.Dict
        | AbstractSet[t.Scalar]
        | None,
    ) -> t.JsonValue:
        """Normalize input into metadata-compatible JSON-native values.

        Returns:
            The resulting ``t.JsonValue``.

        """
        normalized_value: t.JsonValue
        if isinstance(
            val,
            (FlextModelsContainers.ConfigMap, FlextModelsContainers.Dict),
        ):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.root.items()),
            )
        elif val is None:
            normalized_value = ""
        elif isinstance(val, datetime):
            normalized_value = val.isoformat()
        elif isinstance(val, Path):
            normalized_value = str(val)
        elif isinstance(val, c.PRIMITIVES_TYPES):
            normalized_value = val
        elif FlextUtilitiesGuardsTypeModel.has_model_dump(val):
            normalized_value = FlextRuntimeMetadata.normalize_to_json_value(val)
        elif isinstance(val, Mapping):
            normalized_value = FlextRuntimeMetadata._normalize_dict_entries(
                list(val.items()),
            )
        elif isinstance(val, AbstractSet) or (
            isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray))
        ):
            normalized_value = list(
                FlextUtilitiesTypingAdapterJson.json_list_adapter().validate_python(
                    [
                        FlextRuntimeMetadata.normalize_to_json_value(item)
                        for item in val
                    ],
                ),
            )
        elif isinstance(val, (bytes, bytearray)):
            normalized_value = str(val)
        else:
            normalized_value = val
        return normalized_value


__all__: list[str] = ["FlextRuntimeMetadata"]
