"""Exception metadata normalization.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from flext_core import c, m
from flext_core._exceptions import FlextExceptionsHelpers
from flext_core._protocols import FlextProtocolsResult
from flext_core._runtime import FlextRuntimeMetadataValidation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from flext_core._typings import FlextTypesServices, FlextTypingBase


class FlextBaseErrorMetadataMixin:
    @staticmethod
    def _normalized_all(
        entries: Iterable[
            tuple[
                str, FlextTypesServices.JsonPayload | FlextTypingBase.JsonValue | None
            ]
        ],
    ) -> FlextTypingBase.MutableMappingKV[str, FlextTypingBase.JsonValue | None]:
        """Normalize every entry, keeping ``None`` values as declared.

        Returns:
            The resulting ``dict[str, FlextTypingBase.JsonValue | None]``.

        """
        return {
            key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
            for key, value in entries
        }

    @staticmethod
    def _normalized_present(
        entries: Iterable[
            tuple[
                str, FlextTypesServices.JsonPayload | FlextTypingBase.JsonValue | None
            ]
        ],
    ) -> FlextTypingBase.MutableMappingKV[str, FlextTypingBase.JsonValue | None]:
        """Normalize only the entries that carry a value, skipping ``None``.

        Returns:
            The resulting ``dict[str, FlextTypingBase.JsonValue | None]``.

        """
        return {
            key: FlextRuntimeMetadataValidation.normalize_to_metadata(value)
            for key, value in entries
            if value is not None
        }

    @staticmethod
    def _merged_model_metadata(
        metadata_model: m.Metadata,
        merged_kwargs: FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload],
    ) -> m.Metadata:
        """Merge one metadata model with normalized keyword attributes.

        Returns:
            The resulting ``m.Metadata``.

        """
        merged_attrs = FlextBaseErrorMetadataMixin._normalized_present(
            metadata_model.attributes.items(),
        )
        merged_attrs.update(
            FlextBaseErrorMetadataMixin._normalized_present(merged_kwargs.items()),
        )
        return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: merged_attrs})

    @staticmethod
    def _normalized_metadata_input(
        metadata: FlextProtocolsResult.HasModelDump | FlextTypingBase.JsonValue,
    ) -> FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None] | None:
        """Normalize raw metadata input to a mapping when provable.

        Returns:
            The resulting mapping, or ``None`` when unprovable.

        """
        if isinstance(metadata, (Mapping, FlextProtocolsResult.HasModelDump)):
            try:
                return FlextRuntimeMetadataValidation.normalize_metadata_input_mapping(
                    metadata,
                )
            except c.EXC_PYDANTIC_TYPE_VALUE:
                return None
        return None

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
            return m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: (
                    FlextBaseErrorMetadataMixin._normalized_all(merged_kwargs.items())
                ),
            })
        metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
        if metadata_model is not None:
            return FlextBaseErrorMetadataMixin._merged_model_metadata(
                metadata_model,
                merged_kwargs,
            )
        metadata_dict = FlextBaseErrorMetadataMixin._normalized_metadata_input(metadata)
        if metadata_dict is not None:
            return FlextBaseErrorMetadataMixin._normalize_metadata_from_dict(
                metadata_dict,
                merged_kwargs,
            )
        return m.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: {"value": str(metadata)},
        })

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
        merged_attrs.update(
            FlextBaseErrorMetadataMixin._normalized_present(metadata_dict.items()),
        )
        merged_attrs.update(
            FlextBaseErrorMetadataMixin._normalized_present(merged_kwargs.items()),
        )
        return m.Metadata.model_validate({
            c.FIELD_ATTRIBUTES: {
                k: FlextRuntimeMetadataValidation.normalize_to_metadata(v)
                for k, v in merged_attrs.items()
                if v is not None
            },
        })


__all__: list[str] = ["FlextBaseErrorMetadataMixin"]
