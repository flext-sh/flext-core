"""Runtime metadata validation helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel

from flext_core import c
from flext_core._protocols import FlextProtocolsResult
from flext_core._runtime._metadata import FlextRuntimeMetadata
from flext_core._typings.typeadapters import FlextTypesTypeAdapters

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.services import FlextTypesServices


class FlextRuntimeMetadataValidation(FlextRuntimeMetadata):
    """Validate metadata payloads after JSON normalization."""

    @staticmethod
    def normalize_metadata_input_mapping(
        value: FlextTypesServices.MetadataInput | FlextTypesServices.JsonPayload,
    ) -> FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None] | None:
        """Normalize mapping-like metadata input while preserving explicit None.

        Returns:
            The resulting ``tb.MappingKV[str, ts.JsonPayload | None] | None``.

        Raises:
            TypeError: If ``not isinstance(value, prt.HasModelDump)``.

        """
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
        if not isinstance(value, FlextProtocolsResult.HasModelDump):
            raise TypeError(c.ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE)
        dumped = value.model_dump(mode="json")
        return {
            key: None
            if item is None
            else FlextTypesTypeAdapters.json_value_adapter().validate_python(item)
            for key, item in dumped.items()
        }

    @staticmethod
    def validate_metadata_attributes(
        value: FlextTypesServices.MetadataInput,
    ) -> FlextTypingBase.JsonMapping:
        """Normalize and validate metadata attributes input.

        Returns:
            The resulting ``tb.JsonMapping``.

        Raises:
            ValueError: If ``key.startswith('_')``.

        """
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
                    c.ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED.format(key=key),
                )
        validated_metadata: FlextTypingBase.JsonMapping = (
            FlextTypesTypeAdapters.metadata_map_adapter().validate_python({
                key: item
                for key, item in normalized_mapping.items()
                if item is not None
            })
        )
        return validated_metadata

    @staticmethod
    def validate_metadata_model_input[TModel: BaseModel](
        value: FlextTypesServices.MetadataInput,
        metadata_model: type[TModel],
    ) -> TModel:
        """Normalize metadata-like input into the provided metadata model.

        Returns:
            The resulting ``TModel``.

        """
        if value is None:
            return metadata_model.model_validate({c.FIELD_ATTRIBUTES: {}})
        if isinstance(value, metadata_model):
            return value
        if isinstance(value, Mapping):
            raw_mapping_obj: FlextTypingBase.MappingKV[
                str,
                FlextTypesServices.JsonPayload | None,
            ] = value
        else:
            raw_mapping_obj = value.model_dump(mode="json")
        return metadata_model.model_validate({
            c.FIELD_ATTRIBUTES: dict(raw_mapping_obj),
        })


__all__: list[str] = ["FlextRuntimeMetadataValidation"]
