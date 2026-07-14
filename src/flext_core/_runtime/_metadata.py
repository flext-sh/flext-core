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

    # mro-wkii.17.26 (agent: codex) — preserve JSON null and reject recursive
    # container graphs with deterministic path context before adapter validation.
    @staticmethod
    def _enter_normalization(
        value: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar]
        | mp.BaseModel
        | ts.ConfigModelInput
        | tb.MappingKV[str, ts.JsonPayload | tb.Scalar],
        active_ids: set[int],
        path: str,
    ) -> int:
        """Track one active recursive value or fail with its JSON path."""
        value_id = id(value)
        if value_id in active_ids:
            msg = (
                "Cyclic reference detected during JSON normalization "
                f"at {path} ({value.__class__.__name__})"
            )
            raise ValueError(msg)
        active_ids.add(value_id)
        return value_id

    @classmethod
    def _normalize_value(
        cls,
        value: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar]
        | mp.BaseModel
        | ts.ConfigModelInput
        | tb.MappingKV[str, ts.JsonPayload | tb.Scalar]
        | None,
        *,
        active_ids: set[int],
        path: str,
    ) -> tb.JsonValue:
        """Normalize one value while retaining the active recursion path."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tb.PRIMITIVES_TYPES):
            return value
        if isinstance(value, (bytes, bytearray)):
            return str(value)
        if isinstance(value, (mc.ConfigMap, mc.Dict)):
            value_id = cls._enter_normalization(value, active_ids, path)
            try:
                return dict(
                    tta.json_mapping_adapter().validate_python({
                        key: cls._normalize_value(
                            item, active_ids=active_ids, path=f"{path}[{key!r}]"
                        )
                        for key, item in value.root.items()
                    })
                )
            finally:
                active_ids.remove(value_id)
        if ugm.has_model_dump(value):
            value_id = cls._enter_normalization(value, active_ids, path)
            try:
                dumped = value.model_dump(mode="json")
                return cls._normalize_value(
                    dumped, active_ids=active_ids, path=f"{path}.model_dump"
                )
            finally:
                active_ids.remove(value_id)
        if isinstance(value, Mapping):
            value_id = cls._enter_normalization(value, active_ids, path)
            try:
                return dict(
                    tta.json_mapping_adapter().validate_python({
                        key: cls._normalize_value(
                            item, active_ids=active_ids, path=f"{path}[{key!r}]"
                        )
                        for key, item in value.items()
                    })
                )
            finally:
                active_ids.remove(value_id)
        if isinstance(value, AbstractSet) or (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        ):
            value_id = cls._enter_normalization(value, active_ids, path)
            try:
                return list(
                    tta.json_list_adapter().validate_python([
                        cls._normalize_value(
                            item, active_ids=active_ids, path=f"{path}[{index}]"
                        )
                        for index, item in enumerate(value)
                    ])
                )
            finally:
                active_ids.remove(value_id)
        return tta.json_value_adapter().validate_python(value)

    @classmethod
    def normalize_to_json_value(
        cls,
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
        return tta.json_value_adapter().validate_python(
            cls._normalize_value(value, active_ids=set(), path="$")
        )

    @classmethod
    def normalize_to_json_mapping(
        cls, value: tb.MappingKV[str, ts.JsonPayload | tb.Scalar]
    ) -> tb.JsonDict:
        """Normalize a mapping to a concrete validated ``JsonDict``."""
        return dict(
            tta.json_dict_adapter().validate_python(
                cls._normalize_value(value, active_ids=set(), path="$")
            )
        )

    @classmethod
    def normalize_model_input_mapping(
        cls,
        value: mp.BaseModel
        | mc.Dict
        | ts.ConfigModelInput
        | tb.MappingKV[str, ts.JsonPayload]
        | None,
    ) -> tb.JsonMapping | None:
        """Normalize model-like input to a plain mapping."""
        if value is None:
            return None
        return dict(
            tta.json_mapping_adapter().validate_python(
                cls._normalize_value(value, active_ids=set(), path="$")
            )
        )

    @classmethod
    def normalize_to_metadata(
        cls,
        val: ts.JsonPayload
        | tb.Scalar
        | Path
        | mc.ConfigMap
        | mc.Dict
        | AbstractSet[tb.Scalar]
        | None,
    ) -> tb.JsonValue:
        """Normalize input into metadata-compatible JSON-native values."""
        return cls._normalize_value(val, active_ids=set(), path="$")


__all__: list[str] = ["FlextRuntimeMetadata"]
