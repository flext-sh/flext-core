"""Exception metadata normalization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from flext_core import p
from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.mixins import FlextConstantsMixins as cm
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core._models.base import FlextModelsBase as m
from flext_core._protocols.result import FlextProtocolsResult as pr
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as FlextRuntime,
)

from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts


class FlextBaseErrorMetadataMixin:
    @staticmethod
    def _normalize_metadata(
        metadata: pr.HasModelDump | tb.JsonValue | None,
        merged_kwargs: tb.MappingKV[str, ts.JsonPayload],
    ) -> p.Metadata:
        """Normalize metadata from various input types to m.Metadata model."""
        if metadata is None:
            normalized_attrs = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in merged_kwargs.items()
            }
            resolved_metadata = m.Metadata.model_validate({
                cm.FIELD_ATTRIBUTES: normalized_attrs
            })
        else:
            metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
            if metadata_model is not None:
                merged_attrs = {
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in metadata_model.attributes.items()
                }
                for key, value in merged_kwargs.items():
                    if value is None:
                        continue
                    merged_attrs[key] = FlextRuntime.normalize_to_metadata(value)
                resolved_metadata = m.Metadata.model_validate({
                    cm.FIELD_ATTRIBUTES: merged_attrs
                })
            else:
                metadata_dict: tb.MappingKV[str, ts.JsonPayload | None] | None = None
                if isinstance(metadata, (Mapping, pr.HasModelDump)):
                    try:
                        metadata_dict = FlextRuntime.normalize_metadata_input_mapping(
                            metadata
                        )
                    except ce.EXC_PYDANTIC_TYPE_VALUE:
                        metadata_dict = None
                resolved_metadata = (
                    FlextBaseErrorMetadataMixin._normalize_metadata_from_dict(
                        metadata_dict, merged_kwargs
                    )
                    if metadata_dict is not None
                    else m.Metadata.model_validate({
                        cm.FIELD_ATTRIBUTES: {"value": str(metadata)}
                    })
                )
        return resolved_metadata

    @staticmethod
    def _normalize_metadata_from_dict(
        metadata_dict: tb.MappingKV[str, ts.JsonPayload | None],
        merged_kwargs: tb.MappingKV[str, ts.JsonPayload],
    ) -> p.Metadata:
        """Normalize metadata from dict-like recursive containers."""
        merged_attrs: MutableMapping[str, tb.JsonValue | None] = {}
        for k, v in metadata_dict.items():
            if v is None:
                continue
            merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
        if merged_kwargs:
            for k, v in merged_kwargs.items():
                if v is None:
                    continue
                merged_attrs[k] = FlextRuntime.normalize_to_metadata(v)
        return m.Metadata.model_validate({
            cm.FIELD_ATTRIBUTES: {
                k: FlextRuntime.normalize_to_metadata(v)
                for k, v in merged_attrs.items()
                if v is not None
            }
        })


__all__: list[str] = ["FlextBaseErrorMetadataMixin"]
