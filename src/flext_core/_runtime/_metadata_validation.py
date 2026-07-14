"""Runtime metadata validation helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.mixins import FlextConstantsMixins as cm
from flext_core._protocols.result import FlextProtocolsResult as prt
from flext_core._typings.typeadapters import FlextTypesTypeAdapters as tta

from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts


from ._metadata import FlextRuntimeMetadata


class FlextRuntimeMetadataValidation(FlextRuntimeMetadata):
    """Validate metadata payloads after JSON normalization."""

    @staticmethod
    def normalize_metadata_input_mapping(
        value: ts.MetadataInput | ts.JsonPayload,
    ) -> tb.MappingKV[str, ts.JsonPayload | None] | None:
        """Normalize mapping-like metadata input while preserving explicit None."""
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {
                key: (
                    None
                    if item is None
                    else FlextRuntimeMetadataValidation.normalize_to_json_value(item)
                )
                for key, item in value.items()
            }
        if not isinstance(value, prt.HasModelDump):
            raise TypeError(ce.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE)
        dumped = value.model_dump(mode="json")
        return {
            key: None
            if item is None
            else tta.json_value_adapter().validate_python(item)
            for key, item in dumped.items()
        }

    @staticmethod
    def validate_metadata_attributes(value: ts.MetadataInput) -> tb.JsonMapping:
        """Normalize and validate metadata attributes input."""
        if value is None:
            return {}
        normalized_result = (
            FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(value)
        )
        if normalized_result is None:
            return {}
        normalized_mapping = normalized_result
        for key in normalized_mapping:
            if key.startswith("_"):
                raise ValueError(
                    ce.ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED.format(key=key)
                )
        validated_metadata: tb.JsonMapping = tta.metadata_map_adapter().validate_python({
            key: item for key, item in normalized_mapping.items() if item is not None
        })
        return validated_metadata

    @staticmethod
    def validate_metadata_model_input[TModel: BaseModel](
        value: ts.MetadataInput, metadata_model: type[TModel]
    ) -> TModel:
        """Normalize metadata-like input into the provided metadata model."""
        if value is None:
            return metadata_model.model_validate({cm.FIELD_ATTRIBUTES: {}})
        if isinstance(value, metadata_model):
            return value
        if isinstance(value, Mapping):
            raw_mapping_obj: tb.MappingKV[str, ts.JsonPayload | None] = value
        else:
            raw_mapping_obj = value.model_dump(mode="json")
        return metadata_model.model_validate({
            cm.FIELD_ATTRIBUTES: dict(raw_mapping_obj)
        })


__all__: list[str] = ["FlextRuntimeMetadataValidation"]
