"""Exception metadata normalization.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from flext_core import c, m
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core._protocols import FlextProtocolsResult
from flext_core._runtime._metadata_validation import FlextRuntimeMetadataValidation

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.services import FlextTypesServices


class FlextBaseErrorMetadataMixin:
    @staticmethod
    def normalize_metadata(
        metadata: FlextProtocolsResult.HasModelDump | FlextTypingBase.JsonValue | None,
        merged_kwargs: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from various input types to m.Metadata model.

        Returns:
            The resulting ``m.Metadata``.

        """
        if metadata is None:
            normalized_attrs = {
                key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                for key, value in merged_kwargs.items()
            }
            resolved_metadata = m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: normalized_attrs,
            })
        else:
            metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
            if metadata_model is not None:
                merged_attrs = {
                    key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                    for key, value in metadata_model.attributes.items()
                    if value is not None
                }
                for key, value in merged_kwargs.items():
                    if value is None:
                        continue
                    merged_attrs[key] = (
                        FlextRuntimeMetadataValidation.normalize_to_metadata(value)
                    )
                resolved_metadata = m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: merged_attrs,
                })
            else:
                metadata_dict: (
                    FlextTypingBase.MappingKV[
                        str,
                        FlextTypesServices.JsonPayload | None,
                    ]
                    | None
                ) = None
                if isinstance(metadata, (Mapping, FlextProtocolsResult.HasModelDump)):
                    try:
                        validation = FlextRuntimeMetadataValidation
                        metadata_dict = validation.normalize_metadata_input_mapping(
                            metadata,
                        )
                    except c.EXC_PYDANTIC_TYPE_VALUE:
                        metadata_dict = None
                resolved_metadata = (
                    FlextBaseErrorMetadataMixin._normalize_metadata_from_dict(
                        metadata_dict,
                        merged_kwargs,
                    )
                    if metadata_dict is not None
                    else m.Metadata.model_validate({
                        c.FIELD_ATTRIBUTES: {"value": str(metadata)},
                    })
                )
        return resolved_metadata

    @staticmethod
    def _normalize_metadata_from_dict(
        metadata_dict: FlextTypingBase.MappingKV[
            str,
            FlextTypesServices.JsonPayload | None,
        ],
        merged_kwargs: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from dict-like recursive containers.

        Returns:
            The resulting ``m.Metadata``.

        """
        merged_attrs: MutableMapping[str, FlextTypingBase.JsonValue | None] = {}
        for k, v in metadata_dict.items():
            if v is None:
                continue
            merged_attrs[k] = FlextRuntimeMetadataValidation.normalize_to_metadata(v)
        if merged_kwargs:
            for k, v in merged_kwargs.items():
                if v is None:
                    continue
                merged_attrs[k] = FlextRuntimeMetadataValidation.normalize_to_metadata(
                    v,
                )
        return m.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: {
                k: FlextRuntimeMetadataValidation.normalize_to_metadata(v)
                for k, v in merged_attrs.items()
                if v is not None
            },
        })


__all__: list[str] = ["FlextBaseErrorMetadataMixin"]
