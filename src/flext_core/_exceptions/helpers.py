"""Exception internal helpers — safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from flext_core import c, m, t
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core.runtime import FlextRuntime


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def safe_bool(value: t.Scalar | None, *, default: bool) -> bool:
        """Extract strict bool from dynamic values with default fallback."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return default

    @staticmethod
    def safe_settings_map(
        value: m.Metadata
        | t.ConfigMap
        | Mapping[str, t.MetadataOrValue | None]
        | t.MetadataValue
        | t.RecursiveContainer
        | None,
    ) -> Mapping[str, t.MetadataOrValue | None] | None:
        """Extract SettingsMap when value is mapping-compatible."""
        if value is None:
            return None
        try:
            return t.dict_str_metadata_adapter().validate_python(value)
        except PydanticValidationError:
            return None

    @staticmethod
    def safe_metadata(
        value: m.Metadata
        | t.ConfigMap
        | Mapping[str, t.MetadataOrValue | None]
        | t.MetadataValue
        | t.RecursiveContainer
        | None,
    ) -> m.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        if value is None:
            return None
        try:
            return m.Metadata.model_validate(value, from_attributes=True)
        except (PydanticValidationError, TypeError):
            pass
        dumped_map: Mapping[str, t.MetadataOrValue | None] | None = None
        if FlextUtilitiesGuardsTypeModel.pydantic_model(value):
            dumped_candidate = value.model_dump()
            try:
                dumped_map = t.dict_str_metadata_adapter().validate_python(
                    dumped_candidate,
                )
            except PydanticValidationError:
                dumped_map = None
        if dumped_map is not None:
            attrs_raw = dumped_map.get(c.FIELD_ATTRIBUTES)
            attrs_map = FlextExceptionsHelpers.safe_settings_map(attrs_raw)
            if attrs_map is not None:
                attrs = {
                    k: FlextRuntime.normalize_to_metadata(v)
                    for k, v in attrs_map.items()
                }
                return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: attrs})
        attrs_map = FlextExceptionsHelpers.safe_settings_map(value)
        if attrs_map is not None:
            attrs = {
                k: FlextRuntime.normalize_to_metadata(v) for k, v in attrs_map.items()
            }
            return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: attrs})
        return None

    @staticmethod
    def safe_optional_str(value: t.Container | type | None) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def build_context_map(
        context: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> t.ConfigMap:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        context_map: t.ConfigMap = t.ConfigMap(root={})
        if context:
            context_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in context.items()
                if k not in excluded
            })
        if extra_kwargs:
            context_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in extra_kwargs.items()
                if k not in excluded
            })
        return context_map

    @staticmethod
    def build_param_map(
        context: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
        keys: set[str] | frozenset[str],
    ) -> t.ConfigMap:
        """Build unnormalized parameter map for strict params validation."""
        param_map: t.ConfigMap = t.ConfigMap(root={})
        if context:
            param_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in context.items()
                if k in keys
            })
        if extra_kwargs:
            param_map.update({
                k: FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
                for k, v in extra_kwargs.items()
                if k in keys
            })
        return param_map

    @staticmethod
    def init_error_params[TParams: t.ModelCarrier](
        context: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap | None,
        extra_kwargs: Mapping[str, t.MetadataOrValue | None] | t.ConfigMap,
        named_params: Mapping[str, t.RuntimeData | None],
        params_cls: t.ModelClass[TParams],
        existing_params: TParams | None,
        param_keys: set[str] | frozenset[str],
        *,
        excluded_context_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[TParams, t.ConfigMap | None, t.MetadataValue | None, str | None]:
        """Extract, resolve and build error parameters from kwargs.

        Shared init boilerplate for all typed error subclasses.
        Returns: (resolved_params, error_context, metadata, correlation_id)
        """
        mutable_extra: MutableMapping[str, t.MetadataValue] = {
            key: FlextRuntime.normalize_to_metadata(value)
            for key, value in extra_kwargs.items()
        }
        preserved_metadata_raw = mutable_extra.pop(c.FIELD_METADATA, None)
        preserved_metadata = (
            FlextRuntime.normalize_to_metadata(preserved_metadata_raw)
            if preserved_metadata_raw is not None
            else None
        )
        correlation_id_raw = mutable_extra.pop(c.ContextKey.CORRELATION_ID, None)
        correlation_id_str = (
            FlextExceptionsHelpers.safe_optional_str(correlation_id_raw)
            if FlextUtilitiesGuardsTypeCore.scalar(correlation_id_raw)
            else None
        )
        normalized_extra_kwargs: Mapping[str, t.MetadataValue] = {
            key: FlextRuntime.normalize_to_metadata(value)
            for key, value in mutable_extra.items()
        }
        param_values: MutableMapping[str, t.ValueOrModel] = dict(
            FlextExceptionsHelpers.build_param_map(
                context,
                normalized_extra_kwargs,
                keys=param_keys,
            ),
        )
        for key, val in named_params.items():
            if val is not None:
                normalized_val = FlextRuntime.normalize_to_metadata(val)

                def to_normalized(value: t.MetadataValue) -> t.RecursiveContainer:
                    if FlextUtilitiesGuardsTypeCore.scalar(value):
                        return value
                    if FlextUtilitiesGuardsTypeCore.mapping(value):
                        return {
                            str(inner_key): to_normalized(
                                FlextRuntime.normalize_to_metadata(inner_val),
                            )
                            for inner_key, inner_val in value.items()
                        }
                    if isinstance(value, list):
                        return [
                            to_normalized(FlextRuntime.normalize_to_metadata(inner_val))
                            for inner_val in value
                        ]
                    return str(value)

                param_values[key] = to_normalized(normalized_val)
        resolved: TParams = (
            existing_params
            if existing_params is not None
            else params_cls.model_validate(dict(param_values))
        )
        error_context = FlextExceptionsHelpers.build_context_map(
            context,
            normalized_extra_kwargs,
            excluded_keys=excluded_context_keys,
        )
        for key in param_keys:
            attr_val = getattr(resolved, key, None)
            if attr_val is not None:
                error_context[key] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(attr_val),
                )
        resolved_fields: Mapping[str, FieldInfo] = resolved.__class__.model_fields
        for key in param_keys:
            field_info = resolved_fields.get(key)
            if field_info is None:
                continue
            field_help = field_info.description or field_info.title
            if isinstance(field_help, str) and field_help:
                error_context[f"{key}_description"] = field_help
        return (resolved, error_context or None, preserved_metadata, correlation_id_str)

    @staticmethod
    def build_error_context(
        correlation_id: str | None,
        metadata_obj: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None,
        kwargs: Mapping[str, t.MetadataValue] | t.ConfigMap,
    ) -> t.ConfigMap:
        """Build error context dictionary."""
        error_context: t.ConfigMap = t.ConfigMap(root={})
        if correlation_id is not None:
            error_context[c.ContextKey.CORRELATION_ID] = correlation_id
        FlextExceptionsHelpers.merge_metadata_into_context(error_context, metadata_obj)
        for k, v in kwargs.items():
            if k not in {c.ContextKey.CORRELATION_ID, c.FIELD_METADATA}:
                error_context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
        return error_context

    @staticmethod
    def merge_metadata_into_context(
        context: t.ConfigMap,
        metadata_obj: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None,
    ) -> None:
        """Merge metadata recursive-container values into the context dictionary."""
        if metadata_obj is None:
            return
        metadata_model = FlextExceptionsHelpers.safe_metadata(metadata_obj)
        if metadata_model is not None:
            for k, v in metadata_model.attributes.items():
                context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(v),
                )
            return
        metadata_map = FlextExceptionsHelpers.safe_settings_map(metadata_obj)
        if metadata_map is not None:
            for k, metadata_value in metadata_map.items():
                context[k] = FlextRuntime.normalize_to_container(
                    FlextRuntime.normalize_to_metadata(metadata_value),
                )

    @staticmethod
    def extract_common_kwargs(
        kwargs: Mapping[str, t.MetadataValue],
    ) -> tuple[str | None, m.Metadata | Mapping[str, t.MetadataOrValue] | None]:
        """Extract correlation_id and metadata from kwargs."""
        correlation_id_raw = kwargs.get(c.ContextKey.CORRELATION_ID)
        correlation_id = (
            FlextExceptionsHelpers.safe_optional_str(correlation_id_raw)
            if FlextUtilitiesGuardsTypeCore.scalar(correlation_id_raw)
            else None
        )
        metadata_raw = kwargs.get(c.FIELD_METADATA)
        metadata: m.Metadata | Mapping[str, t.MetadataOrValue | None] | None = None
        model_dump = getattr(metadata_raw, "model_dump", None)
        if callable(model_dump):
            metadata = FlextExceptionsHelpers.safe_metadata(metadata_raw)
        if metadata is None:
            metadata = FlextExceptionsHelpers.safe_settings_map(metadata_raw)
        if metadata is None:
            attrs_raw = getattr(metadata_raw, c.FIELD_ATTRIBUTES, None)
            attrs_map = FlextExceptionsHelpers.safe_settings_map(attrs_raw)
            if attrs_map is not None:
                metadata = m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: dict(attrs_map),
                })
        return (correlation_id, metadata)


__all__: list[str] = ["FlextExceptionsHelpers"]
