"""Exception metadata normalization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from flext_core import (
    FlextConstants as c,
    FlextExceptionsHelpers,
    FlextModelsBase as m,
    FlextProtocols as p,
    FlextRuntime,
    FlextTypes as t,
)


class FlextBaseErrorMetadataMixin:
    @staticmethod
    def _normalize_metadata(
        metadata: p.HasModelDump | t.JsonValue | None,
        merged_kwargs: t.MappingKV[str, t.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from various input types to m.Metadata model."""
        if metadata is None:
            normalized_attrs = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in merged_kwargs.items()
            }
            resolved_metadata = m.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: normalized_attrs,
            })
        else:
            metadata_model = FlextExceptionsHelpers.safe_metadata(metadata)
            if metadata_model is not None:
                merged_attrs = {
                    key: FlextRuntime.normalize_to_metadata(value)
                    for key, value in metadata_model.attributes.items()
                    if value is not None
                }
                for key, value in merged_kwargs.items():
                    if value is None:
                        continue
                    merged_attrs[key] = FlextRuntime.normalize_to_metadata(value)
                resolved_metadata = m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: merged_attrs,
                })
            else:
                metadata_dict: t.MappingKV[str, t.JsonPayload | None] | None = None
                if isinstance(metadata, (Mapping, p.HasModelDump)):
                    try:
                        metadata_dict = FlextRuntime.normalize_metadata_input_mapping(
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
        metadata_dict: t.MappingKV[str, t.JsonPayload | None],
        merged_kwargs: t.MappingKV[str, t.JsonPayload],
    ) -> m.Metadata:
        """Normalize metadata from dict-like recursive containers."""
        merged_attrs: MutableMapping[str, t.JsonValue | None] = {}
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
            c.FIELD_ATTRIBUTES: {
                k: FlextRuntime.normalize_to_metadata(v)
                for k, v in merged_attrs.items()
                if v is not None
            },
        })


__all__: list[str] = ["FlextBaseErrorMetadataMixin"]
