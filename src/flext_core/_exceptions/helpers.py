"""Exception internal helpers - safe type coercion and metadata normalization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError as PydanticValidationError

from flext_core import p
from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.mixins import FlextConstantsMixins as cm
from flext_core._models.base import FlextModelsBase as m
from flext_core._protocols.result import FlextProtocolsResult as pr
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as FlextRuntime,
)

from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts


class FlextExceptionsHelpers:
    """Internal helpers for exception param extraction and metadata normalization."""

    @staticmethod
    def _normalized_source_entries(
        context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
        extra_kwargs: tb.MappingKV[str, ts.JsonPayload | None],
    ) -> tuple[tuple[str, tb.JsonValue], ...]:
        """Collect normalized metadata entries from context and kwargs once."""
        entries: list[tuple[str, tb.JsonValue]] = []
        source_values = (context, extra_kwargs)
        for source_value in source_values:
            if source_value is None:
                continue
            try:
                source_mapping = FlextRuntime.normalize_metadata_input_mapping(
                    source_value
                )
            except ce.EXC_PYDANTIC_TYPE_VALUE:
                continue
            if not source_mapping:
                continue
            for key, value in source_mapping.items():
                if value is not None:
                    entries.append((key, FlextRuntime.normalize_to_metadata(value)))
        return tuple(entries)

    @staticmethod
    def safe_metadata(
        value: pr.HasModelDump
        | tb.MappingKV[str, ts.JsonPayload | None]
        | tb.JsonValue
        | None,
    ) -> p.Metadata | None:
        """Normalize supported metadata inputs to runtime metadata model."""
        metadata: p.Metadata | None = None
        if value is not None:
            try:
                metadata = m.Metadata.model_validate(value, from_attributes=True)
            except (PydanticValidationError, TypeError):
                if isinstance(value, (Mapping, pr.HasModelDump)):
                    try:
                        attrs_map = FlextRuntime.normalize_metadata_input_mapping(value)
                    except ce.EXC_PYDANTIC_TYPE_VALUE:
                        attrs_map = None
                    if attrs_map is not None:
                        attrs = {
                            key: item
                            for key, item in attrs_map.items()
                            if item is not None
                        }
                        metadata = m.Metadata.model_validate({
                            cm.FIELD_ATTRIBUTES: attrs
                        })
        return metadata

    @staticmethod
    def safe_optional_str(value: ts.JsonPayload | type | None) -> str | None:
        """Extract optional strict string from dynamic values."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def build_context_map(
        context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
        extra_kwargs: tb.MappingKV[str, ts.JsonPayload | None],
        excluded_keys: set[str] | frozenset[str] | None = None,
    ) -> tb.JsonDict:
        """Build normalized context map from context and kwargs."""
        excluded = excluded_keys or frozenset()
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context, extra_kwargs
            )
            if key not in excluded
        }

    @staticmethod
    def build_param_map(
        context: tb.MappingKV[str, ts.JsonPayload | None] | pr.HasModelDump | None,
        extra_kwargs: tb.MappingKV[str, ts.JsonPayload | None],
        keys: set[str] | frozenset[str],
    ) -> tb.JsonDict:
        """Build parameter map restricted to declared param keys."""
        return {
            key: value
            for key, value in FlextExceptionsHelpers._normalized_source_entries(
                context, extra_kwargs
            )
            if key in keys
        }


__all__: list[str] = ["FlextExceptionsHelpers"]
