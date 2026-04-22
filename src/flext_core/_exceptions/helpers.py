"""Exception internal helpers — safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
)

from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from flext_core import (
    FlextModelsBase as m,
    FlextRuntime,
    c,
    p,
    t,
)


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: Mapping[str, t.MetadataData | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.MetadataData | None],
    ) -> tuple[tuple[str, t.MetadataValue], ...]:
        """Collect normalized metadata entries from context and kwargs once."""
        entries: list[tuple[str, t.MetadataValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            try:
                source_mapping = FlextRuntime.normalize_metadata_input_mapping(
                    source_value,
                )
            except (PydanticValidationError, TypeError, ValueError):
                continue
            if not source_mapping:
                continue
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((key, FlextRuntime.normalize_to_metadata(value)))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: p.HasModelDump
        | Mapping[str, t.MetadataData | None]
        | t.MetadataValue
        | None,
    ) -> m.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        if value is None:
            return None
        try:
            return m.Metadata.model_validate(value, from_attributes=True)
        except (PydanticValidationError, TypeError):
            pass
        if not isinstance(value, Mapping) and not isinstance(value, p.HasModelDump):
            return None
        try:
            attrs_map = FlextRuntime.normalize_metadata_input_mapping(value)
        except (PydanticValidationError, TypeError, ValueError):
            return None
        if attrs_map is None:
            return None
        attrs = {key: value for key, value in attrs_map.items() if value is not None}
        return m.Metadata.model_validate({c.FIELD_ATTRIBUTES: attrs})

    @staticmethod
    def safe_optional_str(value: t.MetadataData | type | None) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def build_context_map(
        context: Mapping[str, t.MetadataData | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.MetadataData | None],
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> dict[str, t.MetadataValue]:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context,
                extra_kwargs,
            )
            if key not in excluded
        }

    @staticmethod
    def build_param_map(
        context: Mapping[str, t.MetadataData | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.MetadataData | None],
        keys: set[str] | frozenset[str],
    ) -> dict[str, t.MetadataValue]:
        """Build parameter map restricted to declared param keys."""
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context,
                extra_kwargs,
            )
            if key in keys
        }

    @staticmethod
    def init_error_params[TParams: t.ModelCarrier](
        context: Mapping[str, t.MetadataData | None] | p.HasModelDump | None,
        extra_kwargs: Mapping[str, t.MetadataData | None],
        named_params: Mapping[str, t.RuntimeData | None],
        params_cls: t.ModelClass[TParams],
        existing_params: TParams | None,
        param_keys: set[str] | frozenset[str],
        *,
        excluded_context_keys: set[str] | frozenset[str] | None = None,
    ) -> tuple[
        TParams,
        dict[str, t.MetadataValue] | None,
        t.MetadataValue | None,
        str | None,
    ]:
        """Extract, resolve and build error parameters from kwargs.

        Shared init boilerplate for all typed error subclasses.
        Returns: (resolved_params, error_context, metadata, correlation_id)
        """
        mutable_extra: MutableMapping[str, t.MetadataData | None] = dict(extra_kwargs)
        preserved_metadata_raw = mutable_extra.pop(c.FIELD_METADATA, None)
        preserved_metadata = (
            FlextRuntime.normalize_to_metadata(preserved_metadata_raw)
            if preserved_metadata_raw is not None
            else None
        )
        correlation_id_raw = mutable_extra.pop(c.ContextKey.CORRELATION_ID, None)
        correlation_id_str = FlextExceptionsHelpers.safe_optional_str(
            correlation_id_raw,
        )
        remaining_extra_kwargs: Mapping[str, t.MetadataData | None] = dict(
            mutable_extra,
        )
        param_values: dict[str, t.MetadataValue] = (
            FlextExceptionsHelpers.build_param_map(
                context,
                remaining_extra_kwargs,
                keys=param_keys,
            )
        )
        for key, val in named_params.items():
            if val is None:
                continue
            normalized_val = FlextRuntime.normalize_to_metadata(val)
            if isinstance(normalized_val, t.SCALAR_TYPES):
                param_values[key] = normalized_val
            else:
                param_values[key] = str(normalized_val)
        resolved: TParams = (
            existing_params
            if existing_params is not None
            else params_cls.model_validate(dict(param_values))
        )
        error_context = FlextExceptionsHelpers.build_context_map(
            context,
            remaining_extra_kwargs,
            excluded_keys=excluded_context_keys,
        )
        for key in param_keys:
            attr_val = getattr(resolved, key, None)
            if attr_val is not None:
                error_context[key] = FlextRuntime.normalize_to_metadata(attr_val)
        resolved_fields: Mapping[str, FieldInfo] = params_cls.__pydantic_fields__
        for key in param_keys:
            field_info = resolved_fields.get(key)
            if field_info is None:
                continue
            field_help = field_info.description or field_info.title
            if isinstance(field_help, str) and field_help:
                error_context[f"{key}_description"] = field_help
        return (resolved, error_context or None, preserved_metadata, correlation_id_str)


__all__: list[str] = ["FlextExceptionsHelpers"]
