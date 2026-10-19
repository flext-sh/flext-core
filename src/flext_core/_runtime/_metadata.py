"""Runtime metadata and JSON normalization helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import p, t
from flext_core._runtime import FlextRuntimeBase
from flext_core._utilities import (
    FlextUtilitiesPydantic,
    FlextUtilitiesTypingAdapterJson,
)


class FlextRuntimeMetadata(FlextRuntimeBase):
    """Normalize runtime values into metadata and JSON contracts."""

    @staticmethod
    def normalize_to_json_value[ValueT](value: ValueT) -> t.JsonValue:
        """Normalize arbitrary runtime input to one validated ``JsonValue``.

        Returns:
            The resulting ``t.JsonValue``.

        """
        if isinstance(value, p.HasModelDump):
            return FlextUtilitiesPydantic.normalize_to_json_value(
                value.model_dump(mode="json"),
            )
        return FlextUtilitiesPydantic.normalize_to_json_value(value)

    @staticmethod
    def normalize_to_json_mapping[ValueT](
        value: t.MappingKV[str, ValueT],
    ) -> t.JsonMapping:
        """Normalize a mapping to a validated ``JsonMapping``.

        Returns:
            The resulting ``t.JsonMapping``.

        """
        return FlextUtilitiesTypingAdapterJson.json_mapping_adapter().validate_python(
            FlextRuntimeMetadata.normalize_to_json_value(value), strict=True
        )

    @staticmethod
    def _normalize_dict_entries[ValueT](
        items: t.SequenceOf[t.Pair[str, ValueT]],
    ) -> t.JsonDict:
        """Normalize key-value pairs for container dict construction.

        Returns:
            The resulting ``t.JsonDict``.

        """
        return FlextUtilitiesTypingAdapterJson.json_dict_adapter().validate_python(
            FlextRuntimeMetadata.normalize_to_json_value(dict(items)), strict=True
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: p.ConfigModelInput,
    ) -> t.JsonMapping | None:
        """Normalize model-like input to a plain mapping.

        Returns:
            The resulting ``t.JsonMapping | None``.

        """
        if value is None:
            return None
        return FlextUtilitiesTypingAdapterJson.json_mapping_adapter().validate_python(
            FlextRuntimeMetadata.normalize_to_json_value(value), strict=True
        )

    @staticmethod
    def normalize_to_metadata[ValueT](val: ValueT) -> t.JsonValue:
        """Normalize input into metadata-compatible JSON-native values.

        Returns:
            The resulting ``t.JsonValue``.

        """
        return FlextRuntimeMetadata.normalize_to_json_value(val)


__all__: list[str] = ["FlextRuntimeMetadata"]
